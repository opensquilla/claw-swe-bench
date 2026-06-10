"""Global configuration and constants for the claw-swe-bench harness.

All paths, naming conventions, timeouts, and defaults are defined here.
Modules should import from this file instead of hardcoding values.

Host-machine paths (claw runtimes, SWE-bench venv) can be overridden via
environment variables so the framework is portable across machines.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASET_VERIFIED = "princeton-nlp/SWE-bench_Verified"
DATASET_MULTILINGUAL = "SWE-bench/SWE-bench_Multilingual"
DEFAULT_SPLIT = "test"

# ---------------------------------------------------------------------------
# Docker image / container naming
# ---------------------------------------------------------------------------
DOCKER_IMAGE_PREFIX = "sweb.eval.x86_64"
DOCKER_IMAGE_TAG = "latest"

# Per-container resource limits — protect the host from agent / test
# framework (jest etc.) fork-bombing under parallel workers.
CONTAINER_PIDS_LIMIT = int(os.environ.get("CLAW_PIDS_LIMIT", "300"))
CONTAINER_MEMORY = os.environ.get("CLAW_CONTAINER_MEMORY", "8g")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CLAW_CONFIGS_DIR = PROJECT_ROOT / "claw_configs"

# SWE-bench harness virtual environment (for run_eval.py)
SWEBENCH_VENV = Path(os.environ.get("SWEBENCH_VENV", "/data/swe-bench-env"))
# Working directory the harness is invoked from (its logs/ dir lands here)
SWEBENCH_WORK_DIR = Path(os.environ.get("SWEBENCH_WORK_DIR", "/data"))

# ---------------------------------------------------------------------------
# Per-claw defaults
#
# These are the settings each claw was evaluated with. `model` semantics
# differ per claw (see each adapter's docstring): for some claws the model
# is selected here, for others it lives in the claw's own runtime config
# and this value is recorded as metadata only.
# ---------------------------------------------------------------------------
CLAW_DEFAULTS = {
    "openclaw": {"model": "openrouter/anthropic/claude-opus-4.6", "timeout": 3600, "max_turns": 300},
    "hermes":   {"model": "glm-5.1",        "timeout": 3600, "max_turns": 300},
    "nanobot":  {"model": "qwen3.6-flash",  "timeout": 3600, "max_turns": 300},
    "zeroclaw": {"model": "qwen3.6-flash",  "timeout": 3600, "max_turns": 300},
    "generic":  {"model": "glm-5.1",        "timeout": 3600, "max_turns": 300, "llm_no": 0},
}

# API-key environment variables forwarded from the host into containers
# for claws that read keys from the environment (hermes, generic).
API_KEY_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "INFINI_API_KEY",
)

DEFAULT_AGENT_TIMEOUT = 3600  # seconds
DEFAULT_MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Claw runtime locations on the HOST (bind-mounted into containers).
# Override via environment variables to match your installation.
# ---------------------------------------------------------------------------
# Shared standalone Python used by hermes / nanobot / generic venv mounts
CLAW_PYTHON_HOME = os.environ.get(
    "CLAW_PYTHON_HOME",
    "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu",
)
CLAW_PYTHON_BIN = f"{CLAW_PYTHON_HOME}/bin/python3.12"

# OpenClaw (Node.js CLI installed on host)
OPENCLAW_NODE_BIN = os.environ.get("OPENCLAW_NODE_BIN", "/usr/bin/node")
OPENCLAW_MODULE_DIR = os.environ.get(
    "OPENCLAW_MODULE_DIR", "/usr/lib/node_modules/openclaw"
)
OPENCLAW_STATE_DIR = Path(os.environ.get("OPENCLAW_STATE_DIR", str(Path.home() / ".openclaw")))

# Hermes (Python venv)
HERMES_ENV_PATH = os.environ.get("HERMES_ENV_PATH", "/opt/hermes-env")
HERMES_SITE_PACKAGES = f"{HERMES_ENV_PATH}/lib/python3.12/site-packages"

# NanoBot (Python venv)
NANOBOT_ENV_PATH = os.environ.get("NANOBOT_ENV_PATH", "/opt/nanobot-env")
NANOBOT_SITE_PACKAGES = f"{NANOBOT_ENV_PATH}/lib/python3.12/site-packages"

# ZeroClaw (single Rust binary)
ZEROCLAW_BIN = os.environ.get("ZEROCLAW_BIN", "/usr/local/bin/zeroclaw")

# GenericAgent (cloned repo + uv venv)
GA_REPO_PATH = os.environ.get("GA_REPO_PATH", "/opt/genericagent")
GA_ENV_PATH = os.environ.get("GA_ENV_PATH", "/opt/genericagent-env")
GA_SITE_PACKAGES = f"{GA_ENV_PATH}/lib/python3.12/site-packages"
GA_TEMP_ROOT = PROJECT_ROOT / "ga_temp"
GA_MEMORY_ROOT = PROJECT_ROOT / "ga_memory"
GA_MEMORY_SRC = f"{GA_REPO_PATH}/memory"

# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------
DEFAULT_EVAL_TIMEOUT = 1800  # seconds per instance
DEFAULT_EVAL_WORKERS = 1

# ---------------------------------------------------------------------------
# Repo cleanup
# ---------------------------------------------------------------------------
SETUP_FILES_TO_REMOVE = (
    # Python setup files
    "pyproject.toml", "tox.ini", "setup.py",
    # Dependency lock files (auto-generated, often massive and conflict with base)
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock", "composer.lock",
    "Pipfile.lock", "go.sum",
)

GITIGNORE_PATTERNS = [
    "*.class",
    "*.jar",
    "*.war",
    "*.ear",
    "*.o",
    "*.obj",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.a",
    "*.lib",
    "*.out",
    "*.pyc",
    "*.pyo",
    "__pycache__/",
    "*.egg-info/",
    "/target/",
    "node_modules/",
    "*.exe",
    "*.bin",
]

# ---------------------------------------------------------------------------
# Git config injected into containers
# ---------------------------------------------------------------------------
GIT_USER_EMAIL = "eval@claw-swe-bench.local"
GIT_USER_NAME = "Claw SWE-bench Eval"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def instance_id_to_image(instance_id: str) -> str:
    """Convert instance_id to Docker image name.

    SWE-bench harness builds images as:
        sweb.eval.x86_64.django__django-16429:latest
    """
    return f"{DOCKER_IMAGE_PREFIX}.{instance_id}:{DOCKER_IMAGE_TAG}"


def instance_id_to_image_sweagent(instance_id: str) -> str:
    """Alternative image name used by SWE-agent (with swebench/ prefix)."""
    transformed = instance_id.replace("__", "_1776_").lower()
    return f"swebench/{DOCKER_IMAGE_PREFIX}.{transformed}:{DOCKER_IMAGE_TAG}"


def instance_id_to_container(claw_name: str, instance_id: str) -> str:
    """Convert instance_id to container name.

    (openclaw, django__django-16429) -> openclaw-swe-django__django-16429
    """
    return f"{claw_name}-swe-{instance_id}"


def get_artifact_dir(run_id: str, instance_id: str) -> Path:
    """Return artifacts/<run_id>/<instance_id>/."""
    return ARTIFACTS_ROOT / run_id / instance_id


def get_state_path(run_id: str) -> Path:
    """Return artifacts/<run_id>/state.jsonl."""
    return ARTIFACTS_ROOT / run_id / "state.jsonl"


def get_predictions_path(run_id: str) -> Path:
    """Return artifacts/<run_id>/predictions.jsonl."""
    return ARTIFACTS_ROOT / run_id / "predictions.jsonl"
