#!/usr/bin/env python3
"""Entry point: evaluate predictions against the official SWE-bench harness."""

import argparse
import logging

from claw_swebench.evaluate import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench predictions")
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--dataset_name", required=True,
        help="Full dataset name (e.g. princeton-nlp/SWE-bench_Verified)",
    )
    parser.add_argument(
        "--run_id", required=True,
        help="Evaluation run ID (use a distinct ID per claw/run)",
    )
    parser.add_argument(
        "--instance_ids", nargs="+", default=None,
        help="Specific instance IDs to evaluate",
    )
    parser.add_argument(
        "--max_workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Timeout per instance in seconds (default: 1800)",
    )
    args = parser.parse_args()

    rc = run_evaluation(
        predictions_path=args.predictions,
        dataset_name=args.dataset_name,
        run_id=args.run_id,
        instance_ids=args.instance_ids,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )

    if rc == 0:
        logger.info("Evaluation finished successfully.")
    else:
        logger.error("Evaluation failed with exit code %d", rc)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
