"""Generate prompts for claw agents from SWE-bench instances.

All claws share the same phase-by-phase prompt template
(prompts/default.txt). A claw may override the template (see
prompts/generic.txt) when its tool names require explicit guidance —
overrides should stay as close to the default as possible.
"""

import logging
from pathlib import Path

from claw_swebench.config import PROMPTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = PROMPTS_DIR / "default.txt"


def build_prompt(
    instance: dict,
    template_path: Path | None = None,
) -> str:
    """Render a prompt for the given SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with at least
            'problem_statement' and 'base_commit'.
        template_path: Path to prompt template file. Uses default if None.

    Returns:
        Rendered prompt string.
    """
    path = template_path or DEFAULT_TEMPLATE_PATH
    template = path.read_text()

    prompt = template.format(
        problem_statement=instance["problem_statement"],
        base_commit=instance.get("base_commit", ""),
    )
    return prompt


def render_debug_prompt(instance: dict) -> str:
    """Render a prompt for debugging/review without running the agent."""
    prompt = build_prompt(instance)
    header = f"=== DEBUG PROMPT for {instance['instance_id']} ===\n"
    footer = "\n=== END ===\n"
    return header + prompt + footer
