"""Report the effective context-window budget each claw runs with.

Context window is a model/runtime budget, not a harness capability. If claws
run with materially different budgets then part of what the benchmark measures
is the budget rather than the harness, so every run records the number it
actually found.

Each claw stores the setting in its own file under a different key:

    hermes    claw_configs/hermes/config.yaml     custom_providers[<provider>].context_length
    zeroclaw  claw_configs/zeroclaw/config.toml   agent.max_context_tokens
    nanobot   claw_configs/nanobot/config.json    agents.defaults.contextWindowTokens
    openclaw  $OPENCLAW_STATE_DIR/openclaw.json   agents.defaults.contextWindowTokens
    generic   claw_configs/generic/mykey.py       not exposed

``tokens is None`` means the claw falls back to its own runtime default, which
is not visible from here. That is a real finding, not an error: an unset budget
is exactly what makes two claws incomparable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from claw_swebench.config import CLAW_CONFIGS_DIR, OPENCLAW_STATE_DIR

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass(frozen=True)
class ContextWindow:
    """What one claw's config says about its context budget."""

    claw: str
    tokens: int | None
    key: str
    source: str | None
    note: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _missing(claw: str, key: str, path: Path, what: str = "config not found") -> ContextWindow:
    return ContextWindow(claw=claw, tokens=None, key=key, source=str(path), note=what)


def _hermes() -> ContextWindow:
    key = "custom_providers[<model.provider>].context_length"
    path = CLAW_CONFIGS_DIR / "hermes" / "config.yaml"
    if yaml is None:
        return _missing("hermes", key, path, "pyyaml not installed")
    if not path.is_file():
        return _missing("hermes", key, path)
    data = yaml.safe_load(path.read_text()) or {}
    provider = (data.get("model") or {}).get("provider")
    for entry in data.get("custom_providers") or []:
        if isinstance(entry, dict) and entry.get("name") == provider:
            return ContextWindow("hermes", entry.get("context_length"), key, str(path),
                                 None if entry.get("context_length") else "unset for this provider")
    return ContextWindow("hermes", None, key, str(path),
                         f"no custom_providers entry named {provider!r}")


def _zeroclaw() -> ContextWindow:
    key = "agent.max_context_tokens"
    path = CLAW_CONFIGS_DIR / "zeroclaw" / "config.toml"
    if tomllib is None:
        return _missing("zeroclaw", key, path, "tomllib requires Python 3.11+")
    if not path.is_file():
        return _missing("zeroclaw", key, path)
    data = tomllib.loads(path.read_text())
    value = (data.get("agent") or {}).get("max_context_tokens")
    return ContextWindow("zeroclaw", value, key, str(path), None if value else "unset")


def _json_agent_default(claw: str, path: Path, key: str) -> ContextWindow:
    if not path.is_file():
        return _missing(claw, key, path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return _missing(claw, key, path, f"invalid JSON ({exc})")
    value = ((data.get("agents") or {}).get("defaults") or {}).get("contextWindowTokens")
    return ContextWindow(claw, value, key, str(path),
                         None if value else "unset; claw falls back to its runtime default")


def _nanobot() -> ContextWindow:
    return _json_agent_default(
        "nanobot", CLAW_CONFIGS_DIR / "nanobot" / "config.json",
        "agents.defaults.contextWindowTokens")


def _openclaw() -> ContextWindow:
    return _json_agent_default(
        "openclaw", OPENCLAW_STATE_DIR / "openclaw.json",
        "agents.defaults.contextWindowTokens")


def _generic() -> ContextWindow:
    path = CLAW_CONFIGS_DIR / "generic" / "mykey.py"
    return ContextWindow("generic", None, "n/a", str(path),
                         "GenericAgent exposes max_tokens (output), not a context budget")


_READERS = {
    "hermes": _hermes,
    "zeroclaw": _zeroclaw,
    "nanobot": _nanobot,
    "openclaw": _openclaw,
    "generic": _generic,
}


def effective_context_window(claw: str) -> ContextWindow:
    """Read ``claw``'s live config and report the budget it will run with."""
    reader = _READERS.get(claw)
    if reader is None:
        return ContextWindow(claw, None, "n/a", None, "no reader for this claw")
    try:
        return reader()
    except Exception as exc:  # a broken config must not take the run down
        return ContextWindow(claw, None, "n/a", None, f"could not read config ({exc})")


def describe_all() -> list[ContextWindow]:
    return [effective_context_window(name) for name in sorted(_READERS)]


def format_table(rows: list[ContextWindow]) -> str:
    lines = [f"{'claw':<10} {'context window':>14}  source"]
    lines.append("-" * 64)
    for row in rows:
        tokens = f"{row.tokens:,}" if row.tokens else "unset"
        lines.append(f"{row.claw:<10} {tokens:>14}  {row.source or '-'}")
        if row.note:
            lines.append(f"{'':<10} {'':>14}  note: {row.note}")
    return "\n".join(lines)


def main() -> None:
    print(format_table(describe_all()))
    print()
    print("Budgets that differ across claws make a harness comparison partly a")
    print("comparison of context budget. See README, 'Context window'.")


if __name__ == "__main__":
    main()
