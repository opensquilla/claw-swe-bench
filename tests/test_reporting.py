import json
import unittest
from pathlib import Path

from claw_swebench.reporting import sanitize_report_name, write_run_report
from claw_swebench.types import InstanceRecord, InstanceState


class ReportingTests(unittest.TestCase):
    def test_sanitize_report_name_keeps_single_component(self):
        self.assertEqual(sanitize_report_name("../my report"), "my_report")
        self.assertEqual(sanitize_report_name("G8-lite"), "G8-lite")

    def test_write_run_report_copies_agent_outputs(self):
        root = Path(self.id().replace(".", "_"))
        reports_root = Path("tmp-test-reports") / root
        artifacts_root = Path("tmp-test-artifacts") / root
        run_dir = artifacts_root / "run-1"
        instance_dir = run_dir / "django__django-1"
        (instance_dir / "opensquilla_artifacts").mkdir(parents=True, exist_ok=True)
        (instance_dir / "opensquilla_logs").mkdir(parents=True, exist_ok=True)
        (instance_dir / "metadata.json").write_text('{"ok": true}', encoding="utf-8")
        (instance_dir / "agent_stdout.log").write_text("stdout", encoding="utf-8")
        (instance_dir / "agent_stderr.log").write_text("stderr", encoding="utf-8")
        (instance_dir / "git.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (instance_dir / "opensquilla_result.json").write_text(
            '{"status": "ok"}',
            encoding="utf-8",
        )
        (instance_dir / "opensquilla_artifacts" / "transcript.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (instance_dir / "opensquilla_logs" / "traces-20260630.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        record = InstanceRecord(
            instance_id="django__django-1",
            state=InstanceState.PATCH_COLLECTED,
            model="G8",
            run_id="run-1",
            duration_seconds=12.3,
            patch_empty=False,
        )
        stale = reports_root / "g8_smoke" / "instances" / "old__instance" / "agent_stdout.log"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale", encoding="utf-8")

        try:
            report_dir = write_run_report(
                records=[record],
                run_id="run-1",
                report_name="g8 smoke",
                claw_name="opensquilla",
                dataset_name="SWE-bench/SWE-bench_Multilingual",
                split="test",
                model_name="G8",
                timeout=3600,
                max_turns=100,
                reports_root=reports_root,
                artifacts_root=artifacts_root,
            )

            summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["report_name"], "g8_smoke")
            self.assertEqual(summary["state_counts"], {"patch_collected": 1})
            self.assertTrue((report_dir / "summary.md").exists())
            self.assertTrue((report_dir / "predictions.jsonl").exists())
            self.assertFalse(stale.exists())
            copied = report_dir / "instances" / "django__django-1"
            self.assertEqual((copied / "agent_stdout.log").read_text(encoding="utf-8"), "stdout")
            self.assertTrue((copied / "opensquilla_artifacts" / "transcript.jsonl").exists())
            self.assertTrue((copied / "opensquilla_logs" / "traces-20260630.jsonl").exists())
        finally:
            import shutil

            shutil.rmtree(reports_root.parent, ignore_errors=True)
            shutil.rmtree(artifacts_root.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
