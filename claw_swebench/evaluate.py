"""Run SWE-bench official harness evaluation.

The swebench package is installed in a separate virtual environment
(default /data/swe-bench-env, override via SWEBENCH_VENV env var),
so we invoke it via subprocess.
"""

import logging
import subprocess
from pathlib import Path

from claw_swebench.config import SWEBENCH_VENV, SWEBENCH_WORK_DIR
from claw_swebench.prediction import validate_prediction_file

logger = logging.getLogger(__name__)


def run_evaluation(
    predictions_path: str | Path,
    dataset_name: str,
    run_id: str = "claw-swe-bench",
    instance_ids: list[str] | None = None,
    max_workers: int = 1,
    timeout: int = 1800,
) -> int:
    """Run SWE-bench harness evaluation.

    Args:
        predictions_path: Path to predictions JSONL file.
        dataset_name: Full dataset name (e.g. "princeton-nlp/SWE-bench_Verified").
        run_id: Evaluation run identifier.
        instance_ids: If provided, only evaluate these instances.
        max_workers: Number of parallel evaluation workers.
        timeout: Timeout per instance in seconds.

    Returns:
        Subprocess return code (0 = success).
    """
    predictions_path = Path(predictions_path)

    # Pre-validate
    errors = validate_prediction_file(predictions_path)
    if errors:
        for e in errors:
            logger.error("Prediction validation: %s", e)
        raise ValueError(f"Prediction file invalid: {len(errors)} error(s)")

    # Build command
    python_bin = SWEBENCH_VENV / "bin" / "python"
    if not python_bin.exists():
        raise FileNotFoundError(
            f"SWE-bench venv not found at {SWEBENCH_VENV}. "
            "Install swebench there or set the SWEBENCH_VENV env var."
        )

    cmd = [
        str(python_bin), "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path.resolve()),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--namespace", "",
    ]

    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)

    logger.info("Running SWE-bench evaluation: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(SWEBENCH_WORK_DIR),  # harness writes logs/ relative to cwd
        text=True,
        capture_output=True,
        timeout=timeout * max(len(instance_ids or [1]), 1) * 5 + 300,
    )

    if result.stdout:
        logger.info("Harness stdout:\n%s", result.stdout[-2000:])
    if result.stderr:
        logger.info("Harness stderr:\n%s", result.stderr[-2000:])

    if result.returncode != 0:
        logger.error("Harness exited with code %d", result.returncode)
    else:
        logger.info("Harness evaluation completed successfully.")

    return result.returncode
