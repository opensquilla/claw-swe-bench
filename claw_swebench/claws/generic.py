"""GenericAgent (lsdefine/GenericAgent) adapter for SWE-bench evaluation.

Architecture
------------
GenericAgent is a Python harness with 9 atomic tools (file_read/write/patch,
code_run, web_scan/web_execute_js, ask_user, plus 2 memory helpers). The
harness's task-mode CLI is:

    python agentmain.py --task <id> --input <prompt> --llm_no N --nobg --verbose

In task mode GA reads ``temp/<id>/input.txt``, runs the agent loop, writes
``temp/<id>/output.txt`` ending with ``[ROUND END]``, and then blocks up to
10 minutes waiting for ``temp/<id>/reply.txt`` before exiting.

For SWE-bench we:
  1. Bind-mount a per-instance host dir at ``<GA_REPO>/temp`` inside the
     container (container_run_args), so input/output files are visible to
     both host and container.
  2. Pre-write input.txt on the host, then start agentmain.py via docker exec.
  3. Poll output.txt on the host for the ``[ROUND END]`` sentinel.
  4. Once found, terminate the GA subprocess (we never want a second round).
  5. Runner-side ``git diff /testbed`` extracts the patch.

This claw uses prompts/generic.txt — identical to prompts/default.txt except
for 3 added lines naming GA's tools and banning its web tools (GA has no
config-level tool disable, so the ban is prompt-level).
"""

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from claw_swebench.config import (
    API_KEY_ENV_VARS,
    CLAW_CONFIGS_DIR,
    CLAW_PYTHON_BIN,
    CLAW_PYTHON_HOME,
    GA_ENV_PATH,
    GA_MEMORY_ROOT,
    GA_MEMORY_SRC,
    GA_REPO_PATH,
    GA_SITE_PACKAGES,
    GA_TEMP_ROOT,
    PROMPTS_DIR,
)
from claw_swebench.claws.base import BaseClawAdapter
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

# Sentinel GA writes at end of each round's output file
ROUND_END = "[ROUND END]"

GA_CONFIG_DIR = CLAW_CONFIGS_DIR / "generic"

# Forward GA_LANG in addition to API keys so mykey.py / GA run in English
FORWARDED_ENV_VARS = API_KEY_ENV_VARS + ("GA_LANG",)

# Token usage lines printed by llmcore._record_usage()
ANTHROPIC_USAGE_RE = re.compile(
    r'\[Cache\]\s*input=(\d+)\s*creation=(\d+)\s*read=(\d+)'
)
OAI_INPUT_RE = re.compile(r'\[Cache\]\s*input=(\d+)\s*cached=(\d+)')
OAI_OUTPUT_RE = re.compile(r'\[Output\]\s*tokens=(\d+)')


class GenericAgentAdapter(BaseClawAdapter):
    """Drives the GenericAgent harness inside a SWE-bench container.

    Stateless across instances (no agent registration/teardown needed); each
    invocation runs in its own container with its own temp dir, so concurrent
    instances do not share state.

    `llm_no` selects the Nth entry in claw_configs/generic/mykey.py — model
    choice lives there, not in `model` (which is metadata only).
    """

    name = "generic"

    def __init__(
        self,
        model: str,
        timeout: int,
        max_turns: int | None = None,
        llm_no: int = 0,
    ):
        super().__init__(model, timeout, max_turns)
        self.llm_no = llm_no

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        # Per-instance host dir mounted at <GA_REPO>/temp inside the
        # container; GA writes input.txt / output.txt / stdout.log there,
        # so the host can poll without docker exec.
        host_ga_temp = GA_TEMP_ROOT / instance_id
        host_ga_temp.mkdir(parents=True, exist_ok=True)

        # Per-instance writable copy of GA's memory dir. GA writes
        # file_access_stats.json on every memory-file read; with the parent
        # install dir mounted :ro that fails with OSError [Errno 30].
        host_ga_memory = GA_MEMORY_ROOT / instance_id
        if host_ga_memory.exists():
            shutil.rmtree(host_ga_memory)
        shutil.copytree(GA_MEMORY_SRC, host_ga_memory)

        return [
            "-v", f"{CLAW_PYTHON_HOME}:{CLAW_PYTHON_HOME}:ro",
            "-v", f"{GA_REPO_PATH}:{GA_REPO_PATH}:ro",
            "-v", f"{GA_ENV_PATH}:{GA_ENV_PATH}:ro",
            "-v", f"{GA_CONFIG_DIR}:/opt/generic-config:ro",
            "-v", f"{host_ga_temp}:{GA_REPO_PATH}/temp:rw",
            "-v", f"{host_ga_memory}:{GA_REPO_PATH}/memory:rw",
        ]

    def prompt_template(self) -> Path | None:
        return PROMPTS_DIR / "generic.txt"

    # ------------------------------------------------------------------
    # Session backup
    # ------------------------------------------------------------------

    def backup_session(self, agent_id: str, dest: Path) -> None:
        """Copy GA's per-task output files into the artifact dir for analysis.

        Best-effort: searches for the per-instance temp dir by walking
        GA_TEMP_ROOT looking for any subdir matching agent_id.
        """
        if not dest:
            return
        # Walk: GA_TEMP_ROOT/<instance_id>/<agent_id>/...
        host_temp = None
        if GA_TEMP_ROOT.exists():
            for d in GA_TEMP_ROOT.iterdir():
                cand = d / agent_id
                if cand.exists():
                    host_temp = cand
                    break
        if host_temp is None:
            return
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("output.txt", "stdout.log", "stderr.log", "input.txt"):
            src = host_temp / name
            if src.exists():
                (dest / name).write_bytes(src.read_bytes())
        # also copy model_responses_*.txt if present
        model_resp_dir = host_temp / "model_responses"
        if model_resp_dir.exists():
            (dest / "model_responses").mkdir(exist_ok=True)
            for f in model_resp_dir.glob("model_responses_*.txt"):
                (dest / "model_responses" / f.name).write_bytes(f.read_bytes())

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def send_task(
        self,
        prompt: str,
        agent_id: str,
        container_name: str,
        artifact_dir: Path | None = None,
        instance_id: str | None = None,
    ) -> AgentResult:
        """Run GenericAgent on `prompt` inside `container_name`.

        Polls the host-side bind-mounted temp dir for completion sentinel.
        """
        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)

        # container_run_args mounted GA_TEMP_ROOT/<instance_id> → <GA_REPO>/temp;
        # GA then creates <GA_REPO>/temp/<agent_id>/ for this task.
        # On host that path is GA_TEMP_ROOT/<instance_id>/<agent_id>/.
        if instance_id is None:
            instance_id = agent_id  # fallback if caller didn't pass through
        host_temp = GA_TEMP_ROOT / instance_id / agent_id
        host_temp.mkdir(parents=True, exist_ok=True)
        # Clean any stale output from a previous attempt
        for stale in host_temp.glob("output*.txt"):
            stale.unlink()
        for stale in (host_temp / "reply.txt",):
            if stale.exists():
                stale.unlink()

        # Pre-write input.txt so we can skip --input (avoids CLI escaping)
        (host_temp / "input.txt").write_text(prompt, encoding="utf-8")

        stdout_path = artifact_dir / "agent_stdout.log" if artifact_dir else None
        stderr_path = artifact_dir / "agent_stderr.log" if artifact_dir else None

        # Build docker exec command.
        # PYTHONPATH places mykey.py override FIRST so `import mykey` from
        # llmcore.py picks it up; then GA's site-packages so external deps
        # (httpx, openai, anthropic, etc.) import; then GA_REPO_PATH so its
        # own modules import.
        cmd = [
            "docker", "exec",
            "-w", "/testbed",
            "-e", f"PYTHONPATH=/opt/generic-config:{GA_SITE_PACKAGES}:{GA_REPO_PATH}",
            "-e", "GA_LANG=en",
        ]
        for env_name in FORWARDED_ENV_VARS:
            val = os.environ.get(env_name)
            if val:
                cmd.extend(["-e", f"{env_name}={val}"])
        cmd.extend([
            container_name,
            CLAW_PYTHON_BIN,
            f"{GA_REPO_PATH}/agentmain.py",
            "--task", agent_id,
            "--llm_no", str(self.llm_no),
            "--nobg",
            "--verbose",
        ])

        start_time = time.time()
        timed_out = False
        exit_code = 0
        stdout_text = ""
        stderr_text = ""

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output_path = host_temp / "output.txt"
        sentinel_seen = False
        deadline = start_time + self.timeout

        try:
            while time.time() < deadline:
                # Process died?
                if proc.poll() is not None:
                    break
                # Sentinel reached?
                if output_path.exists():
                    try:
                        txt = output_path.read_text(encoding="utf-8", errors="replace")
                        if ROUND_END in txt:
                            sentinel_seen = True
                            break
                    except OSError:
                        pass
                time.sleep(2)
            else:
                timed_out = True
        finally:
            # Terminate GA subprocess (it's either done writing or stuck)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            # Drain stdout/stderr
            try:
                stdout_text, stderr_text = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_text, stderr_text = proc.communicate()
            exit_code = proc.returncode if proc.returncode is not None else -1

        duration = time.time() - start_time

        # Persist captured stdout/stderr
        if stdout_path:
            stdout_path.write_text(stdout_text or "")
        if stderr_path:
            stderr_path.write_text(stderr_text or "")

        # Parse cumulative token usage from the captured stdout. (In --nobg
        # mode, GA prints its [Cache]/[Output] lines straight to stdout, which
        # we captured into agent_stdout.log.)
        usage = _parse_usage_text(stdout_text or "")

        # Determine finish reason
        if timed_out and not sentinel_seen:
            finish_reason = "timeout"
        elif not sentinel_seen:
            finish_reason = "error"
        else:
            finish_reason = "stop"

        return AgentResult(
            success=finish_reason == "stop",
            timeout=timed_out,
            exit_code=exit_code,
            finish_reason=finish_reason,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            session_id=None,
            duration_seconds=round(duration, 1),
            usage=usage,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_usage_text(text: str) -> dict:
    """Sum tokens across all LLM calls logged in `text` (GA stdout).

    GA's llmcore._record_usage() prints, per call, lines like:
        [Cache] input=N creation=M read=K        (Anthropic NativeClaudeSession)
        [Cache] input=N cached=M                 (OpenAI/NativeOAISession)
        [Output] tokens=O                        (both)

    For OpenAI-style, `input` is already the non-cached portion. For
    Anthropic-style, the regex captures cache creation and read separately.
    """
    total = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    if not text:
        return total

    # Anthropic-style takes precedence (3 fields)
    anthropic_spans = []
    for m in ANTHROPIC_USAGE_RE.finditer(text):
        total["input"] += int(m.group(1))
        total["cacheWrite"] += int(m.group(2))
        total["cacheRead"] += int(m.group(3))
        anthropic_spans.append((m.start(), m.end()))

    # OpenAI-style — skip lines already matched by Anthropic regex
    for m in OAI_INPUT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in anthropic_spans):
            continue
        total["input"] += int(m.group(1))
        total["cacheRead"] += int(m.group(2))

    # Output tokens (works for both providers)
    for m in OAI_OUTPUT_RE.finditer(text):
        total["output"] += int(m.group(1))

    return total
