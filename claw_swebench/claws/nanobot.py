"""NanoBot CLI adapter for SWE-bench evaluation.

Wraps `nanobot agent -m` CLI calls with structured result handling
and timeout management.

Architecture: NanoBot runs INSIDE the SWE-bench Docker container via
bind-mounted standalone Python + nanobot venv. Model and provider are
configured in claw_configs/nanobot/config.json (mounted read-only at
/opt/nanobot-config).
"""

import logging
import subprocess
import time
from pathlib import Path

from claw_swebench.config import (
    CLAW_CONFIGS_DIR,
    CLAW_PYTHON_BIN,
    CLAW_PYTHON_HOME,
    NANOBOT_ENV_PATH,
    NANOBOT_SITE_PACKAGES,
)
from claw_swebench.claws.base import BaseClawAdapter, decode_output
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_BUFFER = 120

NANOBOT_CONFIG_DIR = CLAW_CONFIGS_DIR / "nanobot"


class NanoBotAdapter(BaseClawAdapter):
    """Drives NanoBot CLI inside containers and returns structured results.

    NanoBot runs inside the container via bind-mounted standalone Python
    and nanobot venv. Each instance runs in its own container, so no
    extra isolation is needed.
    """

    name = "nanobot"

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        return [
            "-v", f"{CLAW_PYTHON_HOME}:{CLAW_PYTHON_HOME}:ro",
            "-v", f"{NANOBOT_ENV_PATH}:{NANOBOT_ENV_PATH}:ro",
            "-v", f"{NANOBOT_CONFIG_DIR}:/opt/nanobot-config",
        ]

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
        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = artifact_dir / "agent_stdout.log" if artifact_dir else None
        stderr_path = artifact_dir / "agent_stderr.log" if artifact_dir else None

        # Build Python code to invoke nanobot CLI inside the container.
        # NanoBot has no --model CLI flag, model is in config.json.
        nanobot_code = (
            "import sys; "
            f"sys.argv = ['nanobot', 'agent', '-m', {repr(prompt)}, "
            f"'-c', '/opt/nanobot-config/config.json', "
            f"'-w', '/testbed', "
            f"'--no-markdown', '--logs']; "
            "from nanobot.cli.commands import app; "
            "app()"
        )

        cmd = [
            "docker", "exec",
            "-e", f"PYTHONPATH={NANOBOT_SITE_PACKAGES}",
            container_name,
            CLAW_PYTHON_BIN,
            "-c", nanobot_code,
        ]

        start_time = time.time()
        timed_out = False

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + SUBPROCESS_TIMEOUT_BUFFER,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            exit_code = -1
            stdout = decode_output(e.stdout)
            stderr = decode_output(e.stderr)
            logger.warning(
                "NanoBot subprocess timed out after %ds",
                self.timeout + SUBPROCESS_TIMEOUT_BUFFER,
            )

        duration = time.time() - start_time

        # Save session JSONL before cleanup (contains full conversation with tool calls).
        if artifact_dir:
            _save_session_jsonl(container_name, artifact_dir)

        # Clean up NanoBot metadata files from /testbed before patch collection.
        # NanoBot creates AGENTS.md, SOUL.md, etc. in the workspace (-w /testbed),
        # which would pollute the git diff.
        _cleanup_nanobot_metadata(container_name)

        if stdout_path:
            stdout_path.write_text(stdout)
        if stderr_path:
            stderr_path.write_text(stderr)

        if timed_out:
            finish_reason = "timeout"
        elif exit_code != 0:
            finish_reason = "error"
        elif not stdout or not stdout.strip():
            finish_reason = "empty"
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
            usage={},
        )


def _save_session_jsonl(container_name: str, artifact_dir: Path) -> None:
    """Copy NanoBot session JSONL from container before cleanup."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "cat",
             "/testbed/sessions/cli_direct.jsonl"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            (artifact_dir / "session.jsonl").write_text(result.stdout)
    except Exception:
        pass


def _cleanup_nanobot_metadata(container_name: str) -> None:
    """Remove NanoBot-created metadata files from /testbed before patch collection."""
    subprocess.run(
        ["docker", "exec", container_name, "bash", "-c",
         "cd /testbed && rm -rf "
         "AGENTS.md HEARTBEAT.md SOUL.md TOOLS.md USER.md "
         "memory/ sessions/ .nanobot/"],
        capture_output=True, timeout=10,
    )
