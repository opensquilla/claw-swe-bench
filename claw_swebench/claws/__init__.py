"""Claw adapter registry.

Add a new claw by implementing BaseClawAdapter (see base.py) and
registering the class here.
"""

from claw_swebench.config import CLAW_DEFAULTS
from claw_swebench.claws.base import BaseClawAdapter
from claw_swebench.claws.generic import GenericAgentAdapter
from claw_swebench.claws.hermes import HermesAdapter
from claw_swebench.claws.nanobot import NanoBotAdapter
from claw_swebench.claws.openclaw import OpenClawAdapter
from claw_swebench.claws.zeroclaw import ZeroClawAdapter

CLAWS: dict[str, type[BaseClawAdapter]] = {
    "openclaw": OpenClawAdapter,
    "hermes": HermesAdapter,
    "nanobot": NanoBotAdapter,
    "zeroclaw": ZeroClawAdapter,
    "generic": GenericAgentAdapter,
}


def get_adapter(
    name: str,
    model: str | None = None,
    timeout: int | None = None,
    max_turns: int | None = None,
    llm_no: int | None = None,
) -> BaseClawAdapter:
    """Construct a claw adapter, filling unset arguments from CLAW_DEFAULTS."""
    if name not in CLAWS:
        raise ValueError(f"Unknown claw '{name}'. Available: {sorted(CLAWS)}")

    defaults = CLAW_DEFAULTS[name]
    kwargs = {
        "model": model or defaults["model"],
        "timeout": timeout or defaults["timeout"],
        "max_turns": max_turns if max_turns is not None else defaults["max_turns"],
    }
    if name == "generic":
        kwargs["llm_no"] = llm_no if llm_no is not None else defaults.get("llm_no", 0)

    return CLAWS[name](**kwargs)
