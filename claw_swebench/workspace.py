"""Manage SWE-bench Docker container lifecycle and repo operations.

Responsibilities:
- Start / stop Docker containers from SWE-bench images
- Execute commands inside containers
- Prepare repo (git reset, future-commit cleanup, gitignore)
- Collect git diff (patch extraction)
- Check repo cleanliness

Claw-specific container integration (bind mounts, env vars, post-start
provisioning) is delegated to the claw adapter via two hooks:
- adapter.container_run_args(instance_id) -> extra `docker run` args
- adapter.post_container_start(workspace)  -> e.g. copy a config file in
"""

import logging
import subprocess
from dataclasses import dataclass

from claw_swebench.config import (
    CONTAINER_MEMORY,
    CONTAINER_PIDS_LIMIT,
    GIT_USER_EMAIL,
    GIT_USER_NAME,
    GITIGNORE_PATTERNS,
    instance_id_to_container,
    instance_id_to_image,
    instance_id_to_image_sweagent,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of a command executed inside a container."""

    stdout: str
    stderr: str
    exit_code: int


class SWEBenchWorkspace:
    """Manages a single SWE-bench Docker container for one instance."""

    def __init__(self, instance_id: str, adapter):
        self.instance_id = instance_id
        self.adapter = adapter
        self.image_name = self._resolve_image(instance_id)
        self.container_name = instance_id_to_container(adapter.name, instance_id)
        self._started = False

    @staticmethod
    def _resolve_image(instance_id: str) -> str:
        """Find available Docker image for this instance.

        Tries harness format first, then SWE-agent format.
        """
        for name_fn in (instance_id_to_image, instance_id_to_image_sweagent):
            candidate = name_fn(instance_id)
            result = subprocess.run(
                ["docker", "image", "inspect", candidate],
                capture_output=True,
            )
            if result.returncode == 0:
                return candidate
        # Default to harness format even if not found (will fail at start)
        return instance_id_to_image(instance_id)

    def start(self) -> str:
        """Start the Docker container. Returns the container name."""
        # Remove stale container with same name if exists
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )

        logger.info("Starting container %s from image %s",
                     self.container_name, self.image_name)
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            # Resource limits — protect host from agent / test framework
            # (jest etc.) fork-bombing under parallel workers.
            "--pids-limit", str(CONTAINER_PIDS_LIMIT),
            "--memory", CONTAINER_MEMORY,
            "--memory-swap", CONTAINER_MEMORY,  # no swap: cgroup-OOM stays in-container
            # Host gateway so in-container claws can reach host-side proxies
            "--add-host", "host.docker.internal:host-gateway",
        ]
        cmd.extend(self.adapter.container_run_args(self.instance_id))
        cmd.extend([self.image_name, "tail", "-f", "/dev/null"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container: {result.stderr.strip()}"
            )
        self._started = True

        self.adapter.post_container_start(self)
        logger.info("Container %s started.", self.container_name)
        return self.container_name

    def run_in_container(self, cmd: str, timeout: int = 300) -> ExecResult:
        """Execute a bash command inside the container.

        Args:
            cmd: Shell command to run.
            timeout: Timeout in seconds (default 300).

        Returns:
            ExecResult with stdout, stderr, exit_code.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Command timed out after %ds: %s", timeout, cmd[:100])
            return ExecResult(stdout="", stderr="TIMEOUT", exit_code=-1)

    def get_container_ip(self) -> str:
        """Get the container's IP address on the Docker bridge network."""
        result = subprocess.run(
            ["docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
             self.container_name],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def copy_from_container(self, container_path: str, host_path: str) -> bool:
        """Copy a file from the container to the host. Returns True on success."""
        result = subprocess.run(
            ["docker", "cp", f"{self.container_name}:{container_path}", host_path],
            capture_output=True,
        )
        return result.returncode == 0

    def copy_to_container(self, host_path: str, container_path: str) -> bool:
        """Copy a file from the host into the container. Returns True on success."""
        result = subprocess.run(
            ["docker", "cp", host_path, f"{self.container_name}:{container_path}"],
            capture_output=True,
        )
        return result.returncode == 0

    def prepare_instance(
        self, base_commit: str, setup_gitignore: bool = False
    ) -> None:
        """Reset the repo and configure the workspace.

        Args:
            base_commit: Git commit hash to reset to.
            setup_gitignore: If True, inject global gitignore for build artifacts.
        """
        logger.info("Preparing instance %s at commit %s",
                     self.instance_id, base_commit[:12])

        # Git reset
        r = self.run_in_container(f"cd /testbed && git reset --hard {base_commit}")
        if r.exit_code != 0:
            raise RuntimeError(f"git reset failed: {r.stderr}")

        # Clean untracked files
        r = self.run_in_container("cd /testbed && git clean -fd")
        if r.exit_code != 0:
            logger.warning("git clean failed: %s", r.stderr)

        # Strip future commits/tags so the agent can't `git log --all` the fix
        # commit out of the eval container. SWE-bench harness applies this for
        # Python repos (python.py) but skips it for Multilingual (utils.py) —
        # without this an agent can read the gold patch from git history.
        cleanup_script = (
            f'cd /testbed && '
            f'TARGET_TIMESTAMP=$(git show -s --format=%ci {base_commit}) && '
            'git tag -l | while read tag; do '
            '  TAG_COMMIT=$(git rev-list -n 1 "$tag" 2>/dev/null); '
            '  [ -z "$TAG_COMMIT" ] && continue; '
            '  TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT" 2>/dev/null); '
            '  if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then '
            '    git tag -d "$tag" >/dev/null 2>&1; '
            '  fi; '
            'done && '
            'git reflog expire --expire=now --all && '
            'git gc --prune=now --aggressive >/dev/null 2>&1 && '
            'AFTER=$(date -d "$TARGET_TIMESTAMP + 1 second" "+%Y-%m-%d %H:%M:%S") && '
            'COUNT=$(git log --oneline --all --since="$AFTER" 2>/dev/null | wc -l) && '
            '[ "$COUNT" -eq 0 ]'
        )
        r = self.run_in_container(cleanup_script, timeout=120)
        if r.exit_code != 0:
            logger.warning(
                "Future-commit cleanup for %s failed (exit=%d): %s",
                self.instance_id, r.exit_code, r.stderr[:300]
            )
        else:
            logger.info("Stripped future commits/tags from %s", self.instance_id)

        # Configure git user
        self.run_in_container(
            f'git config --global user.email "{GIT_USER_EMAIL}" && '
            f'git config --global user.name "{GIT_USER_NAME}"'
        )

        # Inject gitignore for multilingual (build artifact suppression)
        if setup_gitignore:
            patterns = "\n".join(GITIGNORE_PATTERNS)
            self.run_in_container(
                f"cat > /root/.gitignore_global << 'GITIGNORE'\n"
                f"{patterns}\n"
                f"GITIGNORE\n"
                f"git config --global core.excludesfile /root/.gitignore_global"
            )
            logger.info("Injected global gitignore for build artifacts.")

    def is_repo_clean(self) -> bool:
        """Check if the repo has any uncommitted changes."""
        r = self.run_in_container("cd /testbed && git status --porcelain")
        return r.exit_code == 0 and r.stdout.strip() == ""

    def reset_repo(self, base_commit: str) -> None:
        """Reset repo back to base_commit. Use after a failed run."""
        self.run_in_container(f"cd /testbed && git reset --hard {base_commit}")
        self.run_in_container("cd /testbed && git clean -fd")

    def get_git_diff(self, base_commit: str) -> str:
        """Collect patch from the container repo.

        Stages all changes, removes binary files from staging,
        then returns the cached diff against base_commit.
        """
        # Stage everything
        r = self.run_in_container("cd /testbed && git add -A")
        if r.exit_code != 0:
            logger.warning("git add -A failed: %s", r.stderr)

        # Unstage binary files
        self.run_in_container(
            'cd /testbed && '
            'for f in $(git diff --cached --name-only); do '
            '  if file "/testbed/$f" 2>/dev/null | grep -q "binary"; then '
            '    git reset HEAD -- "$f" 2>/dev/null; '
            '  fi; '
            'done'
        )

        # Get the diff
        r = self.run_in_container(
            f"cd /testbed && git diff --no-color --cached {base_commit}",
            timeout=60,
        )
        if r.exit_code != 0:
            logger.warning("git diff failed: %s", r.stderr)
            return ""

        return r.stdout

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if not self._started:
            return
        logger.info("Cleaning up container %s", self.container_name)
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )
        self._started = False
