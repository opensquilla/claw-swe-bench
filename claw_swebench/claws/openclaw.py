"""OpenClaw CLI adapter for SWE-bench evaluation.

Wraps `openclaw agent` CLI calls with structured result handling,
timeout management, and per-instance agent isolation.

Isolation strategy: each instance gets a temporary OpenClaw agent
(via `openclaw agents add` / `openclaw agents delete`), ensuring
completely independent workspace, session store, and memory.

Container integration: the host's Node.js binary, the OpenClaw module
directory, and the OpenClaw state dir (~/.openclaw) are bind-mounted
into the SWE-bench container, so `openclaw agent` runs *inside* the
container and operates directly on /testbed.
"""

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from claw_swebench.config import (
    OPENCLAW_MODULE_DIR,
    OPENCLAW_NODE_BIN,
    OPENCLAW_STATE_DIR,
)
from claw_swebench.claws.base import BaseClawAdapter
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

# Extra buffer beyond agent timeout for subprocess (let OpenClaw handle
# its own timeout first; only kill subprocess as last resort).
SUBPROCESS_TIMEOUT_BUFFER = 60

# Base path for temporary agent workspaces
TEMP_WORKSPACE_ROOT = Path("/tmp/openclaw-swe-workspaces")

# Tools denied to every per-instance agent. Network access is the hard
# contamination boundary (no fetching the fix from upstream or the web);
# memory / cross-session / scheduling tools are banned because each
# instance must be solved in isolation from any other state.
DENY_TOOLS = [
    "memory_search", "memory_get",
    "web_search", "web_fetch",
    "sessions_list", "sessions_history", "sessions_send",
    "sessions_yield", "sessions_spawn",
    "subagents", "session_status",
    "cron", "image",
]


class OpenClawAdapter(BaseClawAdapter):
    """Drives OpenClaw agent via CLI and returns structured results.

    For each instance, the orchestrator calls create_agent() before
    send_task() and delete_agent() after collecting results. This
    ensures strict isolation between instances.
    """

    name = "openclaw"

    def __init__(self, model: str, timeout: int, max_turns: int | None = None):
        # max_turns accepted for interface uniformity; OpenClaw has no
        # turn-limit flag (its own timeout bounds the run).
        super().__init__(model, timeout, max_turns)
        self._config_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        return [
            "-v", f"{OPENCLAW_NODE_BIN}:/usr/bin/node:ro",
            "-v", f"{OPENCLAW_MODULE_DIR}:/usr/lib/node_modules/openclaw:ro",
            "-v", f"{OPENCLAW_STATE_DIR}:/root/.openclaw",
        ]

    # ------------------------------------------------------------------
    # Agent lifecycle (isolation)
    # ------------------------------------------------------------------

    def create_agent(self, agent_id: str) -> None:
        """Create a temporary isolated OpenClaw agent.

        Each agent has its own workspace, session store, and memory.
        Thread-safe: openclaw.json writes are protected by _config_lock.
        """
        # Clean up any leftover agent with the same ID
        self._force_delete_agent(agent_id)

        workspace = TEMP_WORKSPACE_ROOT / agent_id
        workspace.mkdir(parents=True, exist_ok=True)

        with self._config_lock:
            result = subprocess.run(
                [
                    "openclaw", "agents", "add", agent_id,
                    "--non-interactive",
                    "--workspace", str(workspace),
                    "--model", self.model,
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0 and "already exists" not in result.stderr:
                raise RuntimeError(
                    f"Failed to create agent {agent_id}: {result.stderr}"
                )

            self._set_agent_tools_deny(agent_id, DENY_TOOLS)

        logger.info("Created isolated agent: %s (workspace=%s)", agent_id, workspace)

    def delete_agent(self, agent_id: str) -> None:
        """Delete a temporary agent and clean up all its directories."""
        if not agent_id:
            return
        self._force_delete_agent(agent_id)

    def _force_delete_agent(self, agent_id: str) -> None:
        """Force delete an agent, its workspace, and state directories.

        Thread-safe: openclaw.json write is protected by _config_lock.
        """
        with self._config_lock:
            subprocess.run(
                ["openclaw", "agents", "delete", agent_id, "--force"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        # Remove workspace directory (independent per agent, no lock needed)
        workspace = TEMP_WORKSPACE_ROOT / agent_id
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        # Remove agent state directory (sessions, auth, etc.)
        agent_state = OPENCLAW_STATE_DIR / "agents" / agent_id
        if agent_state.exists():
            shutil.rmtree(agent_state, ignore_errors=True)
        logger.debug("Force-deleted agent %s (workspace + state)", agent_id)

    @staticmethod
    def _set_agent_tools_deny(agent_id: str, deny_list: list[str]) -> None:
        """Set tools.deny for a specific agent by editing openclaw.json directly.

        We edit the JSON file because `openclaw config set` addresses
        agents by list index, which is unreliable for temporary agents.
        """
        config_path = OPENCLAW_STATE_DIR / "openclaw.json"
        with open(config_path) as f:
            data = json.load(f)

        for agent in data.get("agents", {}).get("list", []):
            if agent.get("id") == agent_id:
                agent.setdefault("tools", {})["deny"] = deny_list
                break

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("Set tools.deny=%s for agent %s", deny_list, agent_id)

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
        """Send a task to the specified agent running inside a container."""

        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = artifact_dir / "agent_stdout.log" if artifact_dir else None
        stderr_path = artifact_dir / "agent_stderr.log" if artifact_dir else None

        cmd = [
            "docker", "exec", container_name,
            "node", "/usr/lib/node_modules/openclaw/openclaw.mjs",
            "agent",
            "--agent", agent_id,
            "--message", prompt,
            "--timeout", str(self.timeout),
            "--json",
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
            stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            logger.warning("OpenClaw subprocess timed out after %ds", self.timeout + SUBPROCESS_TIMEOUT_BUFFER)

        duration = time.time() - start_time

        # Save logs
        if stdout_path:
            stdout_path.write_text(stdout)
        if stderr_path:
            stderr_path.write_text(stderr)

        # Parse JSON output (may be in stdout or stderr depending on mode)
        parsed = self._parse_output(stdout) or self._parse_output(stderr)

        # Normalize JSON structure.
        # Gateway mode: {"status":"ok","result":{"payloads":[...],"meta":{...}}}
        # Embedded mode: {"payloads":[...],"meta":{...}}
        if parsed and "result" in parsed:
            result_obj = parsed["result"]
            status = parsed.get("status", "ok")
        elif parsed and "payloads" in parsed:
            result_obj = parsed
            status = "ok"
        else:
            result_obj = {}
            status = "error" if parsed is None else parsed.get("status", "error")

        payloads = result_obj.get("payloads", [])
        agent_meta = result_obj.get("meta", {}).get("agentMeta", {})

        # Determine finish reason
        if timed_out:
            finish_reason = "timeout"
        elif parsed is None:
            finish_reason = "error"
        elif status != "ok":
            finish_reason = "error"
        elif not payloads or all(
            "couldn't generate" in (p.get("text") or "") for p in payloads
        ):
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
            session_id=agent_meta.get("sessionId"),
            duration_seconds=round(duration, 1),
            usage=agent_meta.get("lastCallUsage", {}),
        )

    def backup_session(self, agent_id: str, dest: Path) -> None:
        """Copy session JSONL files from an agent to dest directory."""
        sessions_dir = OPENCLAW_STATE_DIR / "agents" / agent_id / "sessions"
        if not sessions_dir.exists():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for f in sessions_dir.glob("*.jsonl"):
            shutil.copy2(f, dest / f.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(text: str) -> dict | None:
        """Try to parse JSON from openclaw agent --json output.

        Tool-failure log lines may precede the JSON and contain '{'
        themselves (e.g. raw_params={...}), and the agent JSON is
        pretty-printed across many lines — so scan every '{' and accept
        the first object that looks like agent output.
        """
        if not text:
            return None

        decoder = json.JSONDecoder()
        idx = text.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and (
                    "payloads" in obj or "result" in obj or "status" in obj
                ):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = text.find("{", idx + 1)

        return None
