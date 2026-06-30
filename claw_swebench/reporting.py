"""Run report materialization for inference runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from claw_swebench.config import ARTIFACTS_ROOT, REPORTS_ROOT
from claw_swebench.types import InstanceRecord


REPORT_FILE_CANDIDATES = (
    "metadata.json",
    "git.patch",
    "git.patch.raw",
    "agent_stdout.log",
    "agent_stderr.log",
    "opensquilla_result.json",
    "opensquilla_artifacts/transcript.jsonl",
    "opensquilla_artifacts/usage.json",
    "opensquilla_logs",
    "interrupted_container/opensquilla_logs",
)


def sanitize_report_name(name: str) -> str:
    """Return a single path component safe for reports/<name>/."""
    cleaned = []
    for char in str(name or "").strip():
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    value = "".join(cleaned).strip("._")
    if not value:
        raise ValueError("report name must contain at least one letter or number")
    return value[:120]


def write_run_report(
    *,
    records: Iterable[InstanceRecord],
    run_id: str,
    report_name: str,
    claw_name: str,
    dataset_name: str,
    split: str,
    model_name: str,
    timeout: int,
    max_turns: int | None,
    reports_root: Path = REPORTS_ROOT,
    artifacts_root: Path = ARTIFACTS_ROOT,
) -> Path:
    """Copy human-debuggable agent outputs into reports/<report_name>/."""
    safe_name = sanitize_report_name(report_name)
    report_dir = reports_root / safe_name
    if report_dir.exists() or report_dir.is_symlink():
        if report_dir.is_symlink() or report_dir.is_file():
            report_dir.unlink()
        else:
            shutil.rmtree(report_dir)
    instances_dir = report_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    record_rows = [_record_row(record) for record in records]
    counts: dict[str, int] = {}
    patch_empty_count = 0
    for row in record_rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
        if row["patch_empty"] is True:
            patch_empty_count += 1

    artifact_run_dir = artifacts_root / run_id
    for row in record_rows:
        instance_id = row["instance_id"]
        src_dir = artifact_run_dir / instance_id
        dst_dir = instances_dir / instance_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in REPORT_FILE_CANDIDATES:
            src = src_dir / rel_path
            if not src.exists():
                continue
            dst = dst_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    predictions_path = artifact_run_dir / "predictions.jsonl"
    if predictions_path.exists():
        shutil.copy2(predictions_path, report_dir / "predictions.jsonl")

    summary = {
        "report_name": safe_name,
        "requested_report_name": report_name,
        "run_id": run_id,
        "claw": claw_name,
        "dataset_name": dataset_name,
        "split": split,
        "model": model_name,
        "timeout": timeout,
        "max_turns": max_turns,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_run_dir),
        "predictions_path": str(predictions_path),
        "instance_count": len(record_rows),
        "state_counts": counts,
        "patch_empty_count": patch_empty_count,
        "instances": record_rows,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(
        _render_summary_markdown(summary),
        encoding="utf-8",
    )
    return report_dir


def _record_row(record: InstanceRecord) -> dict[str, object]:
    return {
        "instance_id": record.instance_id,
        "state": record.state.value,
        "model": record.model,
        "run_id": record.run_id,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "duration_seconds": record.duration_seconds,
        "patch_empty": record.patch_empty,
        "error": record.error,
    }


def _render_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        f"# {summary['report_name']}",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- claw: `{summary['claw']}`",
        f"- model: `{summary['model']}`",
        f"- dataset: `{summary['dataset_name']}` / `{summary['split']}`",
        f"- timeout: `{summary['timeout']}`",
        f"- max_turns: `{summary['max_turns']}`",
        f"- artifacts: `{summary['artifact_root']}`",
        "",
        "## Summary",
        "",
        f"- instances: `{summary['instance_count']}`",
        f"- patch_empty: `{summary['patch_empty_count']}`",
        f"- state_counts: `{json.dumps(summary['state_counts'], sort_keys=True)}`",
        "",
        "## Instances",
        "",
        "| instance | state | patch_empty | duration_s |",
        "|---|---|---:|---:|",
    ]
    for row in summary["instances"]:  # type: ignore[index]
        lines.append(
            "| {instance_id} | {state} | {patch_empty} | {duration_seconds} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)
