"""OpenSquilla CLI adapter for Claw-SWE-Bench.

OpenSquilla runs inside the SWE-bench container through a bind-mounted
standalone Python 3.12 and OpenSquilla venv. `--model` selects an experiment
group (B* single-model baselines, G* llm_ensemble profiles), not a provider
model string passed to the agent CLI.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from claw_swebench.claws.base import BaseClawAdapter, decode_output
from claw_swebench.config import (
    API_KEY_ENV_VARS,
    CLAW_CONFIGS_DIR,
    CLAW_PYTHON_BIN,
    CLAW_PYTHON_HOME,
    OPENSQUILLA_ENV_PATH,
    OPENSQUILLA_SITE_PACKAGES,
    PROMPTS_DIR,
)
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_BUFFER = 120
DEFAULT_MAX_TURNS = 100

CONTAINER_HOME = "/tmp/opensquilla"
HOME_DIR = "/tmp/opensquilla-home"
CONFIG_PATH = f"{CONTAINER_HOME}/config.toml"
STATE_DIR = f"{CONTAINER_HOME}/state"
CACHE_DIR = f"{CONTAINER_HOME}/cache"
LOG_DIR = f"{CONTAINER_HOME}/logs"
SCRATCH_DIR = "/tmp/opensquilla-scratch"
ARTIFACTS_DIR = "/tmp/opensquilla-artifacts"
TRANSCRIPT_PATH = f"{ARTIFACTS_DIR}/transcript.jsonl"
USAGE_PATH = f"{ARTIFACTS_DIR}/usage.json"
CONFIG_TEMPLATE_DIR = CLAW_CONFIGS_DIR / "opensquilla"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_FALLBACK_MODEL = "deepseek/deepseek-v4-pro"

SINGLE_MODEL_GROUPS: dict[str, str] = {
    "B0": "anthropic/claude-opus-4.8",
    "B1": "openai/gpt-5.5",
    "B2": "z-ai/glm-5.2",
    "B4": "deepseek/deepseek-v4-pro",
    "B5": "moonshotai/kimi-k2.7-code",
    "B6": "qwen/qwen3.7-plus",
    "B7": "google/gemini-3-flash-preview",
}

ENSEMBLE_GROUPS: dict[str, str] = {
    "B3": "b3_glm_self_fusion",
    "G1": "g1_code",
    "G2": "g2_general",
    "G3": "g3_standard",
    "G4": "g4_gemini_aggregator",
    "G5": "g5_opus_aggregator",
    "G6": "g6_gpt_aggregator",
    "G7": "g7_two_proposers",
    "G8": "g8_four_proposers",
    "G9": "g9_qwen_aggregator",
    "G10": "g10_gemini_aggregator",
    "G11": "g11_deepseek_aggregator",
    "G12": "g12_k2_replace_gemini",
    "G13": "g13_five_proposers",
    "G14": "g14_k2_replace_qwen",
    "G15": "g15_g8_top3_prefilter",
    "G16": "g16_sampled_cheap_proposers",
    "G17": "g17_two_layer_moa",
    "G18": "g18_select_best_candidate",
    "G19": "g19_g12_top3_prefilter",
    "G20": "g20_g12_top2_prefilter",
    "G21": "g21_g13_top3_prefilter",
    "G22": "g22_g12_glm_top3_prefilter",
    "G23": "g23_g12_plus_gemini_sampled_top3_prefilter",
}

DENY_TOOLS = [
    "group:web",
    "group:memory",
    "group:sessions",
    "group:messaging",
    "message",
    "cron",
    "image",
    "image_generate",
    "gateway",
    "agents_list",
    "subagents",
]


@dataclass(frozen=True)
class GroupSpec:
    group: str
    kind: str
    model: str | None = None
    profile: str | None = None


def resolve_group(group: str) -> GroupSpec:
    normalized = (group or "").strip().upper()
    if normalized in SINGLE_MODEL_GROUPS:
        return GroupSpec(
            group=normalized,
            kind="single",
            model=SINGLE_MODEL_GROUPS[normalized],
        )
    if normalized in ENSEMBLE_GROUPS:
        return GroupSpec(
            group=normalized,
            kind="ensemble",
            profile=ENSEMBLE_GROUPS[normalized],
        )
    valid = ", ".join(sorted(SINGLE_MODEL_GROUPS) + sorted(ENSEMBLE_GROUPS))
    raise ValueError(f"Unknown OpenSquilla experiment group '{group}'. Valid groups: {valid}")


class OpenSquillaAdapter(BaseClawAdapter):
    """Drives `opensquilla agent` in a clean, container-local runtime.

    The adapter intentionally disables network, memory, session, cron, media,
    messaging, gateway, and subagent tools to preserve Claw-SWE-Bench's
    no-network-answers and per-instance isolation assumptions.
    """

    name = "opensquilla"

    def __init__(self, model: str, timeout: int, max_turns: int | None = None):
        super().__init__(model, timeout, max_turns)
        self.group_spec = resolve_group(model)
        self.group = self.group_spec.group
        self.max_turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        return [
            "-v", f"{CLAW_PYTHON_HOME}:{CLAW_PYTHON_HOME}:ro",
            "-v", f"{OPENSQUILLA_ENV_PATH}:{OPENSQUILLA_ENV_PATH}:ro",
            "-v", f"{CONFIG_TEMPLATE_DIR}:/opt/opensquilla-config:ro",
        ]

    def prompt_template(self) -> Path | None:
        return PROMPTS_DIR / "opensquilla.txt"

    def post_container_start(self, workspace) -> None:
        config = self._render_config()
        code = (
            "from pathlib import Path\n"
            f"for path in {repr(_container_runtime_dirs())}:\n"
            "    Path(path).mkdir(parents=True, exist_ok=True)\n"
            f"Path({CONFIG_PATH!r}).write_text({config!r}, encoding='utf-8')\n"
        )
        result = workspace.run_in_container(
            f"{CLAW_PYTHON_BIN} -c {shlex.quote(code)}",
            timeout=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "Failed to provision OpenSquilla config: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

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
        result_path = artifact_dir / "opensquilla_result.json" if artifact_dir else None

        cmd = self._build_docker_exec_command(container_name)

        start_time = time.time()
        timed_out = False

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
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
                "OpenSquilla subprocess timed out after %ds",
                self.timeout + SUBPROCESS_TIMEOUT_BUFFER,
            )

        duration = time.time() - start_time

        if stdout_path:
            stdout_path.write_text(stdout)
        if stderr_path:
            stderr_path.write_text(stderr)

        parsed = self._parse_output(stdout) or self._parse_output(stderr)
        if result_path:
            payload = parsed if parsed is not None else {"status": "unparsed"}
            result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        errors = parsed.get("errors", []) if isinstance(parsed, dict) else []
        status = parsed.get("status") if isinstance(parsed, dict) else None
        text = parsed.get("text") if isinstance(parsed, dict) else None

        if timed_out:
            finish_reason = "timeout"
        elif parsed is None:
            finish_reason = "error"
        elif exit_code != 0 or status != "ok" or errors:
            finish_reason = "error"
        elif not (text or "").strip():
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
            session_id=parsed.get("session_key") if isinstance(parsed, dict) else None,
            duration_seconds=round(duration, 1),
            usage=parsed.get("usage", {}) if isinstance(parsed, dict) else {},
        )

    def collect_usage(self, workspace, artifact_dir: Path) -> dict:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        copied_artifacts = artifact_dir / "opensquilla_artifacts"
        if copied_artifacts.exists():
            shutil.rmtree(copied_artifacts)
        workspace.copy_from_container(ARTIFACTS_DIR, str(copied_artifacts))
        copied_logs = artifact_dir / "opensquilla_logs"
        if copied_logs.exists():
            shutil.rmtree(copied_logs)
        workspace.copy_from_container(LOG_DIR, str(copied_logs))

        usage = {}
        usage_path = copied_artifacts / "usage.json"
        if usage_path.exists():
            try:
                usage = json.loads(usage_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Failed to parse OpenSquilla usage JSON at %s", usage_path)

        cli_status = None
        result_path = artifact_dir / "opensquilla_result.json"
        if result_path.exists():
            try:
                cli_status = json.loads(result_path.read_text()).get("status")
            except json.JSONDecodeError:
                cli_status = "unparsed"

        meta = {
            "claw": self.name,
            "group": self.group,
            "kind": self.group_spec.kind,
            "profile": self.group_spec.profile,
            "model": self.group_spec.model,
            "cli_status": cli_status,
            "usage": usage,
        }
        meta.update(self._usage_totals(usage))
        return meta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_docker_exec_command(self, container_name: str) -> list[str]:
        code = self._agent_invocation_code()
        cmd = [
            "docker",
            "exec",
            "-i",
            "-e",
            f"PYTHONPATH={OPENSQUILLA_SITE_PACKAGES}",
            "-e",
            f"OPENSQUILLA_GATEWAY_CONFIG_PATH={CONFIG_PATH}",
            "-e",
            f"OPENSQUILLA_STATE_DIR={CONTAINER_HOME}",
            "-e",
            f"OPENSQUILLA_LOG_DIR={LOG_DIR}",
            "-e",
            "OPENSQUILLA_TURN_CALL_LOG=1",
            "-e",
            f"OPENSQUILLA_TURN_CALL_LOG_DIR={LOG_DIR}",
            "-e",
            f"XDG_CACHE_HOME={CACHE_DIR}",
            "-e",
            f"HOME={HOME_DIR}",
            "-e",
            "OPENSQUILLA_AGENT_PERMISSIONS=bypass",
        ]
        for env_name in API_KEY_ENV_VARS:
            if os.environ.get(env_name):
                cmd.extend(["-e", env_name])
        cmd.extend([container_name, CLAW_PYTHON_BIN, "-c", code])
        return cmd

    def _agent_invocation_code(self) -> str:
        args = [
            "opensquilla",
            "agent",
            "--workspace",
            "/testbed",
            "--workspace-strict",
            "--workspace-lockdown",
            "--scratch-dir",
            SCRATCH_DIR,
            "--stateless",
            "--no-memory-capture",
            "--session-db-path",
            ":memory:",
            "--timeout",
            str(self.timeout),
            "--max-iterations",
            str(self.max_turns),
            "--thinking",
            "high",
            "--transcript-path",
            TRANSCRIPT_PATH,
            "--usage-path",
            USAGE_PATH,
            "--permissions",
            "bypass",
            "--json",
        ]
        return (
            "import sys\n"
            "message = sys.stdin.read()\n"
            f"sys.argv = {args!r}\n"
            "sys.argv.extend(['--message', message])\n"
            "from opensquilla.cli.main import app\n"
            "app()\n"
        )

    def _render_config(self) -> str:
        llm_model = self.group_spec.model or DEFAULT_FALLBACK_MODEL
        ensemble_enabled = self.group_spec.kind == "ensemble"
        active_profile = self.group_spec.profile or "g3_standard"
        return "\n".join(
            [
                "# Generated by claw-swe-bench OpenSquillaAdapter.",
                'workspace_dir = "/testbed"',
                "workspace_strict = true",
                f"state_dir = {_toml_string(STATE_DIR)}",
                "log_file_enabled = false",
                "",
                "[llm]",
                'provider = "openrouter"',
                f"model = {_toml_string(llm_model)}",
                'api_key_env = "OPENROUTER_API_KEY"',
                f"base_url = {_toml_string(OPENROUTER_BASE_URL)}",
                "max_tokens = 0",
                'thinking = "high"',
                "",
                "[squilla_router]",
                "enabled = false",
                "",
                "[memory]",
                'source = "state"',
                "session_source_enabled = false",
                "auto_capture_enabled = false",
                'capture_mode = "off"',
                "capture_user = false",
                "capture_assistant = false",
                "flush_enabled = false",
                "",
                "[tools]",
                f"deny = {_toml_list(DENY_TOOLS)}",
                "",
                "[permissions]",
                'default_mode = "bypass"',
                "",
                "[attachments]",
                "persist_transcripts = false",
                f"media_root = {_toml_string(f'{CONTAINER_HOME}/media')}",
                "",
                "[llm_ensemble]",
                f"enabled = {_toml_bool(ensemble_enabled)}",
                f"active_profile = {_toml_string(active_profile)}",
                'mode = "b5_fusion"',
                "proposer_tools = false",
                "min_successful_proposers = 1",
                'all_failed_policy = "fallback_single"',
                "",
            ]
        )

    @staticmethod
    def _parse_output(text: str) -> dict | None:
        if not text:
            return None

        decoder = json.JSONDecoder()
        idx = text.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and (
                    "status" in obj or "usage" in obj or "text" in obj
                ):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = text.find("{", idx + 1)

        return None

    @staticmethod
    def _usage_totals(usage: dict) -> dict:
        input_tokens = _as_int(usage.get("input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))
        total_tokens = _as_int(usage.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": _as_float(
                usage.get("billed_cost", usage.get("cost_usd", usage.get("total_cost_usd", 0.0)))
            ),
        }


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _container_runtime_dirs() -> list[str]:
    return [
        CONTAINER_HOME,
        HOME_DIR,
        STATE_DIR,
        CACHE_DIR,
        LOG_DIR,
        SCRATCH_DIR,
        ARTIFACTS_DIR,
    ]
