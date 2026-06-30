import argparse
import unittest
from pathlib import Path

from scripts.run_opensquilla_lite80 import (
    DEFAULT_GROUPS,
    build_steps,
    normalize_groups,
    validate_report_paths,
)


class RunOpenSquillaLite80Tests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "groups": list(DEFAULT_GROUPS),
            "run_prefix": "osq",
            "python": "python3",
            "timeout": 3600,
            "max_turns": None,
            "workers": 1,
            "eval_workers": 1,
            "eval_timeout": 1800,
            "no_resume": False,
            "skip_eval": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_default_plan_runs_each_group_on_both_lite_splits(self):
        steps = build_steps(self._args(), Path("reports/batch"))

        infer_steps = [step for step in steps if step.phase == "infer"]
        eval_steps = [step for step in steps if step.phase == "eval"]

        self.assertEqual(len(infer_steps), len(DEFAULT_GROUPS) * 2)
        self.assertEqual(len(eval_steps), len(DEFAULT_GROUPS) * 2)
        self.assertIn("osq-B0-lite-multi", {step.run_id for step in infer_steps})
        self.assertIn("osq-G19-lite-verified", {step.run_id for step in infer_steps})
        self.assertTrue(all("--report_name" in step.command for step in infer_steps))
        self.assertTrue(all("--max_turns" not in step.command for step in infer_steps))
        self.assertTrue(
            all(
                step.env == {"SWEBENCH_WORK_DIR": "reports/batch/swebench_work"}
                for step in eval_steps
            )
        )

    def test_max_turns_and_no_resume_are_forwarded_when_requested(self):
        steps = build_steps(
            self._args(groups=["G8"], max_turns=123, no_resume=True),
            Path("reports/batch"),
        )
        infer = next(step for step in steps if step.phase == "infer")

        self.assertIn("--max_turns", infer.command)
        self.assertIn("123", infer.command)
        self.assertIn("--no_resume", infer.command)

    def test_normalize_groups_validates_and_rejects_duplicates(self):
        self.assertEqual(normalize_groups(["b0", "g8"]), ["B0", "G8"])

        with self.assertRaises(SystemExit):
            normalize_groups(["G8", "g8"])

        with self.assertRaises(SystemExit):
            normalize_groups(["Z99"])

    def test_batch_report_cannot_collide_with_inference_report(self):
        steps = build_steps(self._args(groups=["G8"]), Path("reports/osq-G8-lite-multi"))

        with self.assertRaises(SystemExit):
            validate_report_paths(Path("reports/osq-G8-lite-multi"), steps)


if __name__ == "__main__":
    unittest.main()
