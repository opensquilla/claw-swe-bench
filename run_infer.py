#!/usr/bin/env python3
"""Entry point: run any claw on SWE-bench instances to generate patches.

One command, any claw:

    python3 run_infer.py --claw openclaw --dataset multilingual --run_id oc-1
    python3 run_infer.py --claw hermes   --dataset verified     --run_id hm-1 \
        --instance_file config/verified_mini_50.txt
"""

import argparse
import logging
import sys

import yaml

from claw_swebench.claws import CLAWS, get_adapter
from claw_swebench.config import CLAW_DEFAULTS, CONFIG_DIR
from claw_swebench.dataset import load_instances
from claw_swebench.orchestrator import run_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a claw on SWE-bench instances")
    parser.add_argument(
        "--claw", required=True, choices=sorted(CLAWS),
        help="Which claw (agent harness) to run",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["verified", "multilingual"],
        help="Which benchmark to run (loads config/<dataset>.yaml)",
    )
    parser.add_argument(
        "--run_id", required=True,
        help="Unique run identifier (used for artifact directory)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: per-claw default; for nanobot/zeroclaw the "
             "model lives in the claw's own config and this is metadata only)",
    )
    parser.add_argument(
        "--instance_ids", nargs="+", default=None,
        help="Specific instance IDs to run",
    )
    parser.add_argument(
        "--instance_file", default=None,
        help="Path to file with instance IDs (one per line)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Agent timeout in seconds (default: per-claw default)",
    )
    parser.add_argument(
        "--max_turns", type=int, default=None,
        help="Max tool-use turns per instance (default: per-claw default; "
             "ignored by claws without a turn limit flag)",
    )
    parser.add_argument(
        "--llm_no", type=int, default=None,
        help="generic only: index into claw_configs/generic/mykey.py",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Do not resume from previous state (re-run all instances)",
    )
    args = parser.parse_args()

    # Load dataset config
    config_path = CONFIG_DIR / f"{args.dataset}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    split = config.get("split", "test")
    setup_gitignore = config.get("setup_gitignore", False)

    defaults = CLAW_DEFAULTS[args.claw]
    model = args.model or defaults["model"]
    timeout = args.timeout or config.get("timeout") or defaults["timeout"]

    logger.info("Config: claw=%s dataset=%s model=%s timeout=%ds gitignore=%s",
                args.claw, dataset_name, model, timeout, setup_gitignore)

    # Load dataset
    instances = load_instances(
        dataset_name=dataset_name,
        split=split,
        instance_ids=args.instance_ids,
        instance_file=args.instance_file,
    )
    if not instances:
        logger.error("No instances to run.")
        sys.exit(1)

    logger.info("Loaded %d instances to process.", len(instances))

    # Create adapter
    adapter = get_adapter(
        args.claw,
        model=model,
        timeout=timeout,
        max_turns=args.max_turns,
        llm_no=args.llm_no,
    )

    # Run
    records = run_batch(
        instances=instances,
        adapter=adapter,
        model_name=model,
        run_id=args.run_id,
        setup_gitignore=setup_gitignore,
        resume=not args.no_resume,
        max_workers=args.workers,
    )

    # Final report
    logger.info("=" * 60)
    logger.info("Run complete: %s", args.run_id)
    for r in records:
        status = "EMPTY" if r.patch_empty else r.state.value
        logger.info("  %s → %s (%.1fs)", r.instance_id, status, r.duration_seconds or 0)


if __name__ == "__main__":
    main()
