#!/usr/bin/env python3
"""Export Claw-SWE-Bench Lite instance lists for OpenSquilla runs.

The official runner already knows how to evaluate Multilingual and Verified
datasets separately. This helper reads the HF Lite split and writes two plain
instance-id files that can be passed to run_infer.py --instance_file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_MULTILINGUAL = PROJECT_ROOT / "config" / "opensquilla_lite_multilingual.txt"
DEFAULT_OUTPUT_VERIFIED = PROJECT_ROOT / "config" / "opensquilla_lite_verified.txt"
FULL_MULTILINGUAL = PROJECT_ROOT / "config" / "multilingual_300_instances.txt"
FULL_VERIFIED = PROJECT_ROOT / "config" / "verified_mini_50.txt"


def _read_instance_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def classify_instance(
    row: dict,
    *,
    multilingual_ids: set[str],
    verified_ids: set[str],
) -> str:
    instance_id = str(row.get("instance_id") or "").strip()
    source_dataset = str(row.get("source_dataset") or "").strip().lower()

    if "multilingual" in source_dataset:
        return "multilingual"
    if "verified" in source_dataset:
        return "verified"
    if instance_id in multilingual_ids:
        return "multilingual"
    if instance_id in verified_ids:
        return "verified"
    raise ValueError(
        f"Could not classify Lite instance {instance_id!r} "
        f"(source_dataset={source_dataset!r})"
    )


def load_lite_rows(dataset: str, config: str, split: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        return _load_lite_rows_from_dataset_server(dataset, config, split)

    return load_dataset(dataset, config, split=split)


def _load_lite_rows_from_dataset_server(
    dataset: str,
    config: str,
    split: str,
) -> list[dict]:
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": 0,
            "length": 100,
        }
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    total = int(payload.get("num_rows_total") or 0)
    rows = [item["row"] for item in payload.get("rows", [])]
    if total and len(rows) < total:
        raise RuntimeError(
            f"Dataset-server fallback returned {len(rows)} rows, expected {total}; "
            "install datasets and rerun for paginated loading."
        )
    return rows


def export_lite_lists(
    *,
    dataset: str,
    config: str,
    split: str,
    output_multilingual: Path,
    output_verified: Path,
    expected_total: int = 80,
) -> tuple[list[str], list[str]]:
    multilingual_ids = _read_instance_set(FULL_MULTILINGUAL)
    verified_ids = _read_instance_set(FULL_VERIFIED)

    output: dict[str, list[str]] = {"multilingual": [], "verified": []}
    seen: set[str] = set()
    for row in load_lite_rows(dataset, config, split):
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"Lite row is missing instance_id: {row!r}")
        if instance_id in seen:
            raise ValueError(f"Duplicate Lite instance_id: {instance_id}")
        seen.add(instance_id)
        bucket = classify_instance(
            row,
            multilingual_ids=multilingual_ids,
            verified_ids=verified_ids,
        )
        output[bucket].append(instance_id)

    total = len(output["multilingual"]) + len(output["verified"])
    if expected_total and total != expected_total:
        raise ValueError(f"Expected {expected_total} Lite instances, got {total}")

    output_multilingual.parent.mkdir(parents=True, exist_ok=True)
    output_verified.parent.mkdir(parents=True, exist_ok=True)
    output_multilingual.write_text("\n".join(output["multilingual"]) + "\n")
    output_verified.write_text("\n".join(output["verified"]) + "\n")
    return output["multilingual"], output["verified"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="TokenRhythm/Claw-SWE-Bench")
    parser.add_argument("--config", default="lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_multilingual", type=Path, default=DEFAULT_OUTPUT_MULTILINGUAL)
    parser.add_argument("--output_verified", type=Path, default=DEFAULT_OUTPUT_VERIFIED)
    parser.add_argument("--expected_total", type=int, default=80)
    args = parser.parse_args()

    multilingual, verified = export_lite_lists(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        output_multilingual=args.output_multilingual,
        output_verified=args.output_verified,
        expected_total=args.expected_total,
    )
    print(f"Wrote {len(multilingual)} multilingual IDs -> {args.output_multilingual}")
    print(f"Wrote {len(verified)} verified IDs -> {args.output_verified}")


if __name__ == "__main__":
    main()
