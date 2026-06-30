import unittest
from types import SimpleNamespace
from unittest.mock import patch

from claw_swebench.config import (
    instance_id_to_container,
    instance_id_to_image,
    instance_id_to_image_sweagent,
)
from claw_swebench.workspace import SWEBenchWorkspace


class DummyAdapter:
    name = "opensquilla"


def _docker_result(returncode: int) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode)


class WorkspaceTests(unittest.TestCase):
    def test_container_name_includes_safe_run_id_when_provided(self):
        container_name = instance_id_to_container(
            "opensquilla",
            "apache__druid-14136",
            run_id="osq/G8 lite 01",
        )

        self.assertEqual(
            container_name,
            "opensquilla-swe-osq_G8_lite_01-apache__druid-14136",
        )

    @patch("claw_swebench.workspace.subprocess.run")
    def test_resolve_image_prefers_local_harness_image(self, mock_run):
        mock_run.return_value = _docker_result(0)

        image = SWEBenchWorkspace._resolve_image("apache__druid-16875")

        self.assertEqual(image, instance_id_to_image("apache__druid-16875"))
        self.assertEqual(mock_run.call_count, 1)

    @patch("claw_swebench.workspace.subprocess.run")
    def test_resolve_image_uses_local_sweagent_image_second(self, mock_run):
        mock_run.side_effect = [_docker_result(1), _docker_result(0)]

        image = SWEBenchWorkspace._resolve_image("apache__druid-16875")

        self.assertEqual(image, instance_id_to_image_sweagent("apache__druid-16875"))

    @patch("claw_swebench.workspace.subprocess.run")
    def test_resolve_image_falls_back_to_pullable_sweagent_name(self, mock_run):
        mock_run.side_effect = [_docker_result(1), _docker_result(1)]

        image = SWEBenchWorkspace._resolve_image("apache__druid-16875")

        self.assertEqual(image, instance_id_to_image_sweagent("apache__druid-16875"))

    @patch("claw_swebench.workspace.subprocess.run")
    def test_workspace_uses_run_scoped_container_name(self, mock_run):
        mock_run.side_effect = [_docker_result(1), _docker_result(1)]

        workspace = SWEBenchWorkspace(
            "apache__druid-16875",
            DummyAdapter(),
            run_id="osq-G8-lite",
        )

        self.assertEqual(
            workspace.container_name,
            "opensquilla-swe-osq-G8-lite-apache__druid-16875",
        )


if __name__ == "__main__":
    unittest.main()
