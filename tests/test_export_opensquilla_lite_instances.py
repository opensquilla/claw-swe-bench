import unittest

from scripts.export_opensquilla_lite_instances import classify_instance


class ExportOpenSquillaLiteInstancesTests(unittest.TestCase):
    def test_classify_instance_uses_source_dataset_first(self):
        self.assertEqual(
            classify_instance(
                {"instance_id": "x", "source_dataset": "SWE-bench_Multilingual"},
                multilingual_ids=set(),
                verified_ids=set(),
            ),
            "multilingual",
        )
        self.assertEqual(
            classify_instance(
                {"instance_id": "x", "source_dataset": "SWE-bench_Verified"},
                multilingual_ids=set(),
                verified_ids=set(),
            ),
            "verified",
        )

    def test_classify_instance_falls_back_to_full_membership(self):
        self.assertEqual(
            classify_instance(
                {"instance_id": "repo__issue-1", "source_dataset": ""},
                multilingual_ids={"repo__issue-1"},
                verified_ids=set(),
            ),
            "multilingual",
        )
        self.assertEqual(
            classify_instance(
                {"instance_id": "repo__issue-2", "source_dataset": ""},
                multilingual_ids=set(),
                verified_ids={"repo__issue-2"},
            ),
            "verified",
        )


if __name__ == "__main__":
    unittest.main()
