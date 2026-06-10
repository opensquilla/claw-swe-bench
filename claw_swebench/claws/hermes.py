"""Hermes Agent CLI adapter for SWE-bench evaluation.

Wraps `hermes chat -q` CLI calls with structured result handling
and timeout management.

Architecture: Hermes runs INSIDE the SWE-bench Docker container via
bind-mounted standalone Python + hermes venv. The agent directly
operates on /testbed/ without docker exec in the prompt.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

from claw_swebench.config import (
    API_KEY_ENV_VARS,
    CLAW_CONFIGS_DIR,
    CLAW_PYTHON_BIN,
    CLAW_PYTHON_HOME,
    HERMES_ENV_PATH,
    HERMES_SITE_PACKAGES,
)
from claw_swebench.claws.base import BaseClawAdapter, decode_output
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

# Extra buffer beyond agent timeout for subprocess.
SUBPROCESS_TIMEOUT_BUFFER = 120

HERMES_CONFIG_DIR = CLAW_CONFIGS_DIR / "hermes"


class HermesAdapter(BaseClawAdapter):
    """Drives Hermes Agent CLI inside containers and returns structured results.

    Hermes runs inside the container via bind-mounted standalone Python
    and hermes venv. Each invocation is stateless (--yolo).
    """

    name = "hermes"

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        return [
            "-v", f"{CLAW_PYTHON_HOME}:{CLAW_PYTHON_HOME}:ro",
            "-v", f"{HERMES_ENV_PATH}:{HERMES_ENV_PATH}:ro",
            "-v", f"{HERMES_CONFIG_DIR}:/opt/hermes-config:ro",
        ]

    def post_container_start(self, workspace) -> None:
        # HERMES_HOME doubles as Hermes' runtime state dir (auth.json,
        # state.db, sessions/...), so it must live inside the throwaway
        # container — pointing it at the host config dir would leak
        # credentials and session logs onto the host.
        config_path = HERMES_CONFIG_DIR / "config.yaml"
        if not config_path.exists():
            logger.warning(
                "Hermes config not found at %s — copy config.yaml.example "
                "and fill in your API keys.", config_path,
            )
        r = workspace.run_in_container(
            "mkdir -p /tmp/hermes-home && "
            "cp /opt/hermes-config/config.yaml /tmp/hermes-home/config.yaml"
        )
        if r.exit_code != 0:
            logger.warning("Failed to provision HERMES_HOME: %s", r.stderr)

    def collect_usage(self, workspace, artifact_dir: Path) -> dict:
        # Copy session logs out of the container before it is destroyed.
        workspace.copy_from_container(
            "/tmp/hermes-home/sessions", str(artifact_dir)
        )
        return {}

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
        """Send a task to Hermes Agent running inside a container."""
        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = artifact_dir / "agent_stdout.log" if artifact_dir else None
        stderr_path = artifact_dir / "agent_stderr.log" if artifact_dir else None

        # Build Python code to invoke hermes CLI inside the container.
        # We can't use the hermes script directly (shebang path mismatch),
        # so we import the entry point and call it.
        # Don't pass --provider; let Hermes read default from config.yaml's
        # `model.provider` field. For custom_providers (dashscope, infini-ai,
        # deepseek), --provider arg is rejected by argparse since they're not
        # in the built-in provider whitelist.
        hermes_code = (
            "import sys; "
            f"sys.argv = ['hermes', 'chat', '-q', {repr(prompt)}, "
            f"'--quiet', '--yolo', "
            f"'--max-turns', '{self.max_turns}', "
            f"'--toolsets', 'terminal,file', "
            f"'--model', '{self.model}']; "
            "from hermes_cli.main import main; "
            "sys.exit(main())"
        )

        cmd = [
            "docker", "exec",
            "-e", f"PYTHONPATH={HERMES_SITE_PACKAGES}",
            "-e", "HERMES_HOME=/tmp/hermes-home",
        ]
        # Forward API key env vars into the container
        for env_name in API_KEY_ENV_VARS:
            val = os.environ.get(env_name)
            if val:
                cmd.extend(["-e", f"{env_name}={val}"])
        cmd.extend([
            container_name,
            CLAW_PYTHON_BIN,
            "-c", hermes_code,
        ])

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
                "Hermes subprocess timed out after %ds",
                self.timeout + SUBPROCESS_TIMEOUT_BUFFER,
            )

        duration = time.time() - start_time

        # Save logs
        if stdout_path:
            stdout_path.write_text(stdout)
        if stderr_path:
            stderr_path.write_text(stderr)

        # Hermes has no JSON output. Determine finish reason from exit code + output.
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
