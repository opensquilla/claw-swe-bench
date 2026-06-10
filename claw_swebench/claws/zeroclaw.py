"""ZeroClaw CLI adapter for SWE-bench evaluation.

Wraps `zeroclaw agent -m` CLI calls with structured result handling
and timeout management.

Architecture: ZeroClaw runs INSIDE the SWE-bench Docker container via
bind-mounted Rust binary (~37MB). Single binary, no runtime dependencies.
Its config.toml is copied into the container's workspace dir after start.

Usage accounting: ZeroClaw writes per-turn token usage to
workspace/state/costs.jsonl inside the container. collect_usage() copies
that file out, optionally merges cache/reasoning token info from a
host-side proxy usage log (matched by time range + container IP), and
computes cost_usd from a static pricing table.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from claw_swebench.config import CLAW_CONFIGS_DIR, ZEROCLAW_BIN
from claw_swebench.claws.base import BaseClawAdapter, decode_output
from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_BUFFER = 120

ZEROCLAW_CONFIG_DIR = CLAW_CONFIGS_DIR / "zeroclaw"

# Host-side LLM proxy usage log (written by proxies/dashscope_cache_proxy.py
# or claw_configs/zeroclaw/tool_filter_proxy.py). Optional.
PROXY_USAGE_LOG = Path(os.environ.get("PROXY_USAGE_LOG", "/tmp/proxy_usage.jsonl"))

# Static per-token pricing (USD). cost = fresh_input*input
# + cache_read*cache_read + output*output.
MODEL_PRICING = {
    "z-ai/glm-5.1": {
        "input": 0.00000098,
        "output": 0.00000308,
        "cache_read": 0.000000182,
    },
    "glm-5.1": {
        "input": 0.00000098,
        "output": 0.00000308,
        "cache_read": 0.000000182,
    },
    # DashScope qwen3.6-flash (tier 1, single request <=256K input)
    "qwen3.6-flash": {
        "input": 0.000000181,
        "output": 0.00000104,
        "cache_read": 0.0000000361,
    },
}


class ZeroClawAdapter(BaseClawAdapter):
    """Drives ZeroClaw CLI inside containers and returns structured results.

    ZeroClaw is a single Rust binary, bind-mounted into the container.
    Each instance runs in its own container with its own workspace dir.
    """

    name = "zeroclaw"

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        return [
            "-v", f"{ZEROCLAW_BIN}:/usr/local/bin/zeroclaw:ro",
        ]

    def post_container_start(self, workspace) -> None:
        """Copy config.toml into the container's workspace directory.

        The tool-filtering proxy runs on the host (see
        claw_configs/zeroclaw/tool_filter_proxy.py); config.toml points the
        provider at host.docker.internal, which the workspace's --add-host
        flag maps to the host gateway.
        """
        config_src = ZEROCLAW_CONFIG_DIR / "config.toml"
        if not config_src.exists():
            logger.warning(
                "ZeroClaw config not found at %s — copy config.toml.example "
                "and fill in your provider settings.", config_src,
            )
            return
        workspace.run_in_container("mkdir -p /tmp/zeroclaw-workspace")
        workspace.copy_to_container(
            str(config_src), "/tmp/zeroclaw-workspace/config.toml"
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

        cmd = [
            "docker", "exec",
            "-e", "ZEROCLAW_WORKSPACE=/tmp/zeroclaw-workspace",
            container_name,
            "zeroclaw", "agent", "-m", prompt,
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
                "ZeroClaw subprocess timed out after %ds",
                self.timeout + SUBPROCESS_TIMEOUT_BUFFER,
            )

        duration = time.time() - start_time

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

    # ------------------------------------------------------------------
    # Usage accounting
    # ------------------------------------------------------------------

    def collect_usage(self, workspace, artifact_dir: Path) -> dict:
        """Copy costs.jsonl out of the container and compute token usage/cost."""
        workspace.copy_from_container(
            "/tmp/zeroclaw-workspace/workspace/state/costs.jsonl",
            str(artifact_dir / "costs.jsonl"),
        )
        container_ip = workspace.get_container_ip()
        return _parse_costs(artifact_dir, client_ip=container_ip, model=self.model)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _compute_cost(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int) -> float:
    """Compute cost in USD using static pricing table.

        cost = (input_tokens - cache_read_tokens) * input_price
             + cache_read_tokens * cache_read_price
             + output_tokens * output_price
    """
    p = MODEL_PRICING.get(model)
    if not p:
        return 0.0
    fresh_input = max(0, input_tokens - cache_read_tokens)
    cost = (
        fresh_input * p["input"]
        + cache_read_tokens * p.get("cache_read", 0.0)
        + output_tokens * p["output"]
    )
    return round(cost, 6)


def _parse_proxy_usage(start_ts: str, end_ts: str, client_ip: str = "") -> dict:
    """Extract cache/reasoning token info from proxy usage log by time range and client IP."""
    if not PROXY_USAGE_LOG.exists():
        return {}
    total_cache_read = 0
    total_reasoning = 0
    try:
        with open(PROXY_USAGE_LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ts = entry.get("timestamp", "")
                if ts < start_ts or ts > end_ts:
                    continue
                # Filter by client IP if provided (for parallel safety)
                if client_ip and entry.get("client_ip", "") != client_ip:
                    continue
                usage = entry.get("usage", {})
                # Cache read: use prompt_cache_hit_tokens (DeepSeek)
                # or prompt_tokens_details.cached_tokens — they are the same value, don't double count
                cache_hit = usage.get("prompt_cache_hit_tokens", 0)
                if not cache_hit:
                    cached_detail = usage.get("prompt_tokens_details", {})
                    cache_hit = cached_detail.get("cached_tokens", 0) if cached_detail else 0
                total_cache_read += cache_hit
                # Reasoning tokens
                comp_detail = usage.get("completion_tokens_details", {})
                if comp_detail:
                    total_reasoning += comp_detail.get("reasoning_tokens", 0)
    except (json.JSONDecodeError, KeyError):
        pass
    result = {}
    if total_cache_read > 0:
        result["cache_read_tokens"] = total_cache_read
    if total_reasoning > 0:
        result["reasoning_tokens"] = total_reasoning
    return result


def _parse_costs(artifact_dir: Path, client_ip: str = "", model: str = "") -> dict:
    """Parse costs.jsonl to extract turns and token usage.

    Merges cache/reasoning info from proxy usage log, then computes cost from
    token counts using MODEL_PRICING (formula-based, not proxy-reported).
    """
    costs_path = artifact_dir / "costs.jsonl"
    if not costs_path.exists():
        return {}
    turns = 0
    total_input = 0
    total_output = 0
    total_tokens = 0
    first_ts = ""
    last_ts = ""
    try:
        with open(costs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                turns += 1
                usage = entry.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
                ts = usage.get("timestamp", "")
                if ts and not first_ts:
                    first_ts = ts
                if ts:
                    last_ts = ts
    except (json.JSONDecodeError, KeyError):
        pass
    result = {
        "turns": turns,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
    }
    # Merge proxy usage data (cache_read, reasoning) by time range + client IP
    if first_ts and last_ts:
        proxy_data = _parse_proxy_usage(first_ts, last_ts, client_ip)
        result.update(proxy_data)
    # Compute cost from token counts using static pricing table
    if model and (total_input or total_output):
        cache_read = result.get("cache_read_tokens", 0)
        cost = _compute_cost(model, total_input, total_output, cache_read)
        if cost > 0:
            result["cost_usd"] = cost
    return result
