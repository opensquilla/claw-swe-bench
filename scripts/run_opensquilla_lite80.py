#!/usr/bin/env python3
"""Run OpenSquilla experiment groups on the Claw-SWE-Bench Lite-80 split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from claw_swebench.claws.opensquilla import resolve_group  # noqa: E402
from claw_swebench.reporting import sanitize_report_name  # noqa: E402

REPORTS_ROOT = PROJECT_ROOT / "reports"

DEFAULT_GROUPS = ("B0", "B1", "G8", "G12", "G16", "G19")

DATASETS = (
    {
        "label": "multi",
        "dataset": "multilingual",
        "instance_file": "config/opensquilla_lite_multilingual.txt",
        "dataset_name": "SWE-bench/SWE-bench_Multilingual",
    },
    {
        "label": "verified",
        "dataset": "verified",
        "instance_file": "config/opensquilla_lite_verified.txt",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
    },
)


def default_python() -> str:
    repo_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    return sys.executable


@dataclass(frozen=True)
class Step:
    group: str
    dataset_label: str
    phase: str
    run_id: str
    command: list[str]
    log_path: Path
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class StepResult:
    step: Step
    returncode: int
    duration_seconds: float


@dataclass
class GroupResult:
    group: str
    ok: bool
    steps: list[StepResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Experiment groups to run (default: B0 B1 G8 G12 G16 G19)",
    )
    parser.add_argument(
        "--parallel-groups",
        type=int,
        default=1,
        help="How many experiment groups to run concurrently (default: 1)",
    )
    parser.add_argument(
        "--run-prefix",
        default="osq",
        help="Prefix for per-run IDs and per-run reports (default: osq)",
    )
    parser.add_argument(
        "--batch-name",
        default="",
        help="Batch report directory under reports/ (default: timestamped name)",
    )
    parser.add_argument(
        "--python",
        default=default_python(),
        help="Python executable used to call run_infer.py/run_eval.py",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Inference timeout seconds")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override OpenSquilla max turns. Omit to use adapter default.",
    )
    parser.add_argument("--workers", type=int, default=1, help="run_infer.py --workers")
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="run_eval.py --max_workers",
    )
    parser.add_argument("--eval-timeout", type=int, default=1800, help="run_eval.py timeout")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Pass --no_resume to run_infer.py",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run inference; skip run_eval.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without executing them",
    )
    parser.add_argument(
        "--allow-missing-key",
        action="store_true",
        help="Do not fail early when OPENROUTER_API_KEY is unset.",
    )
    return parser.parse_args()


def build_steps(args: argparse.Namespace, batch_dir: Path) -> list[Step]:
    steps: list[Step] = []
    logs_dir = batch_dir / "logs"
    eval_work_dir = batch_dir / "swebench_work"
    for group in args.groups:
        for dataset in DATASETS:
            label = dataset["label"]
            run_id = f"{args.run_prefix}-{group}-lite-{label}"
            infer_cmd = [
                args.python,
                "run_infer.py",
                "--claw",
                "opensquilla",
                "--dataset",
                dataset["dataset"],
                "--run_id",
                run_id,
                "--report_name",
                run_id,
                "--model",
                group,
                "--instance_file",
                dataset["instance_file"],
                "--timeout",
                str(args.timeout),
                "--workers",
                str(args.workers),
            ]
            if args.max_turns is not None:
                infer_cmd.extend(["--max_turns", str(args.max_turns)])
            if args.no_resume:
                infer_cmd.append("--no_resume")
            steps.append(
                Step(
                    group=group,
                    dataset_label=label,
                    phase="infer",
                    run_id=run_id,
                    command=infer_cmd,
                    log_path=logs_dir / f"{run_id}-infer.log",
                )
            )
            if args.skip_eval:
                continue
            eval_run_id = f"{run_id}-eval"
            eval_cmd = [
                args.python,
                "run_eval.py",
                "--predictions",
                f"artifacts/{run_id}/predictions.jsonl",
                "--dataset_name",
                dataset["dataset_name"],
                "--run_id",
                eval_run_id,
                "--max_workers",
                str(args.eval_workers),
                "--timeout",
                str(args.eval_timeout),
            ]
            steps.append(
                Step(
                    group=group,
                    dataset_label=label,
                    phase="eval",
                    run_id=eval_run_id,
                    command=eval_cmd,
                    log_path=logs_dir / f"{eval_run_id}.log",
                    env={"SWEBENCH_WORK_DIR": str(eval_work_dir)},
                )
            )
    return steps


def normalize_groups(groups: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_group in groups:
        group = raw_group.strip().upper()
        if not group:
            continue
        try:
            resolve_group(group)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if group in seen:
            raise SystemExit(f"Duplicate experiment group: {group}")
        seen.add(group)
        normalized.append(group)
    if not normalized:
        raise SystemExit("At least one experiment group is required")
    return normalized


def executable_exists(executable: str) -> bool:
    if os.sep in executable or (os.altsep and os.altsep in executable):
        return Path(executable).exists()
    return shutil.which(executable) is not None


def validate_environment(args: argparse.Namespace) -> None:
    if args.parallel_groups < 1:
        raise SystemExit("--parallel-groups must be >= 1")
    args.groups = normalize_groups(args.groups)
    try:
        args.run_prefix = sanitize_report_name(args.run_prefix)
        if args.batch_name:
            args.batch_name = sanitize_report_name(args.batch_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        not args.dry_run
        and not args.allow_missing_key
        and not os.environ.get("OPENROUTER_API_KEY")
    ):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it or pass --allow-missing-key."
        )
    if not args.dry_run and not executable_exists(args.python):
        raise SystemExit(f"Python executable not found: {args.python}")
    missing = [
        dataset["instance_file"]
        for dataset in DATASETS
        if not (PROJECT_ROOT / dataset["instance_file"]).exists()
    ]
    if missing:
        raise SystemExit(f"Missing Lite instance file(s): {', '.join(missing)}")


def validate_report_paths(batch_dir: Path, steps: list[Step]) -> None:
    infer_report_names = {step.run_id for step in steps if step.phase == "infer"}
    if batch_dir.name in infer_report_names:
        raise SystemExit(
            "--batch-name must not match a per-run report name because run_infer.py "
            f"would replace reports/{batch_dir.name}/"
        )


def run_logged(step: Step, *, dry_run: bool, print_lock: Lock) -> StepResult:
    step.log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    header = f"$ {' '.join(step.command)}\n"
    env_header = "".join(
        f"# env {key}={value}\n" for key, value in sorted(step.env.items())
    )
    with step.log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        log.write(env_header)
        log.flush()
        with print_lock:
            print(f"[{step.group} {step.dataset_label} {step.phase}] {header.strip()}")
            if env_header:
                print(
                    f"[{step.group} {step.dataset_label} {step.phase}] "
                    f"{env_header.strip()}"
                )
        if dry_run:
            return StepResult(step=step, returncode=0, duration_seconds=0.0)
        for key, value in step.env.items():
            if key == "SWEBENCH_WORK_DIR":
                Path(value).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(step.env)
        try:
            proc = subprocess.Popen(
                step.command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            log.write(f"Failed to start command: {exc}\n")
            log.flush()
            return StepResult(
                step=step,
                returncode=127,
                duration_seconds=round(time.monotonic() - start, 1),
            )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            with print_lock:
                print(f"[{step.group} {step.dataset_label} {step.phase}] {line}", end="")
        returncode = proc.wait()
    return StepResult(
        step=step,
        returncode=returncode,
        duration_seconds=round(time.monotonic() - start, 1),
    )


def run_group(group: str, steps: list[Step], *, dry_run: bool, print_lock: Lock) -> GroupResult:
    results: list[StepResult] = []
    for step in steps:
        result = run_logged(step, dry_run=dry_run, print_lock=print_lock)
        results.append(result)
        if result.returncode != 0:
            return GroupResult(group=group, ok=False, steps=results)
    return GroupResult(group=group, ok=True, steps=results)


def write_summary(batch_dir: Path, args: argparse.Namespace, results: list[GroupResult]) -> None:
    payload = {
        "batch_name": batch_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "groups": [group.strip().upper() for group in args.groups if group.strip()],
        "parallel_groups": args.parallel_groups,
        "skip_eval": args.skip_eval,
        "dry_run": args.dry_run,
        "ok": all(result.ok for result in results),
        "results": [
            {
                "group": result.group,
                "ok": result.ok,
                "steps": [
                    {
                        "dataset": step_result.step.dataset_label,
                        "phase": step_result.step.phase,
                        "run_id": step_result.step.run_id,
                        "returncode": step_result.returncode,
                        "duration_seconds": step_result.duration_seconds,
                        "log_path": str(step_result.step.log_path),
                        "env": step_result.step.env,
                    }
                    for step_result in result.steps
                ],
            }
            for result in results
        ],
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"# {batch_dir.name}",
        "",
        f"- ok: `{payload['ok']}`",
        f"- parallel_groups: `{args.parallel_groups}`",
        f"- groups: `{', '.join(payload['groups'])}`",
        "",
        "| group | ok | completed_steps |",
        "|---|---:|---:|",
    ]
    for result in results:
        lines.append(f"| {result.group} | {result.ok} | {len(result.steps)} |")
    lines.append("")
    (batch_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_environment(args)
    batch_name = args.batch_name or (
        "opensquilla-lite80-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    batch_name = sanitize_report_name(batch_name)
    batch_dir = REPORTS_ROOT / batch_name
    steps = build_steps(args, batch_dir)
    validate_report_paths(batch_dir, steps)
    by_group: dict[str, list[Step]] = {}
    for step in steps:
        by_group.setdefault(step.group, []).append(step)

    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "batch_name": batch_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": [
            {
                "group": step.group,
                "dataset": step.dataset_label,
                "phase": step.phase,
                "run_id": step.run_id,
                "command": step.command,
                "log_path": str(step.log_path),
                "env": step.env,
            }
            for step in steps
        ],
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_lock = Lock()
    results: list[GroupResult] = []
    with ThreadPoolExecutor(max_workers=args.parallel_groups) as executor:
        futures = {
            executor.submit(
                run_group,
                group,
                group_steps,
                dry_run=args.dry_run,
                print_lock=print_lock,
            ): group
            for group, group_steps in by_group.items()
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with print_lock:
                status = "ok" if result.ok else "failed"
                print(f"[{result.group}] group finished: {status}")

    results.sort(key=lambda item: item.group)
    write_summary(batch_dir, args, results)
    print(f"Batch report: {batch_dir}")
    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
