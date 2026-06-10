"""Base interface every claw adapter implements.

The orchestrator and workspace are claw-agnostic; everything specific to a
claw (how its runtime gets into the container, how a task is launched, how
sessions/usage are collected) lives behind this interface.

Hook call order for one instance (see orchestrator.run_one_instance):

    create_agent(agent_id)                  # optional isolation setup
    container_run_args(instance_id)         # extra `docker run` args (mounts, env)
    post_container_start(workspace)         # provision config inside container
    send_task(prompt, ...)                  # run the agent, return AgentResult
    collect_usage(workspace, artifact_dir)  # claw-specific usage, container alive
    backup_session(agent_id, artifact_dir)  # save session logs
    delete_agent(agent_id)                  # teardown (always called)
"""

import logging
from pathlib import Path

from claw_swebench.types import AgentResult

logger = logging.getLogger(__name__)


class BaseClawAdapter:
    """Common no-op implementations; adapters override what they need."""

    #: Short claw identifier; used for container naming and logs.
    name = "base"

    def __init__(
        self,
        model: str,
        timeout: int,
        max_turns: int | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Container integration
    # ------------------------------------------------------------------

    def container_run_args(self, instance_id: str) -> list[str]:
        """Extra arguments for `docker run` (bind mounts, env vars)."""
        return []

    def post_container_start(self, workspace) -> None:
        """Provision the running container (e.g. copy a config file in)."""
        pass

    # ------------------------------------------------------------------
    # Agent lifecycle (no-ops for stateless claws)
    # ------------------------------------------------------------------

    def create_agent(self, agent_id: str) -> None:
        pass

    def delete_agent(self, agent_id: str) -> None:
        pass

    def backup_session(self, agent_id: str, dest: Path) -> None:
        pass

    def switch_model(self, model_name: str) -> None:
        self.model = model_name
        logger.info("Model set to %s", model_name)

    # ------------------------------------------------------------------
    # Task execution & accounting
    # ------------------------------------------------------------------

    def send_task(
        self,
        prompt: str,
        agent_id: str,
        container_name: str,
        artifact_dir: Path | None = None,
        instance_id: str | None = None,
    ) -> AgentResult:
        """Run the agent on `prompt` inside `container_name`."""
        raise NotImplementedError

    def collect_usage(self, workspace, artifact_dir: Path) -> dict:
        """Collect claw-specific usage/artifacts while the container is alive.

        The returned dict is stored as metadata.json's top-level "usage" key.
        """
        return {}

    def prompt_template(self) -> Path | None:
        """Prompt template override; None means prompts/default.txt.

        Overrides should stay as close to the default template as possible —
        only tool-name guidance may differ (see prompts/generic.txt).
        """
        return None


def decode_output(data) -> str:
    """Decode subprocess bytes output, or return string as-is."""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""
