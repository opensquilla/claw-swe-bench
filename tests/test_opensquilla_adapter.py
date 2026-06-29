import os
import unittest
from unittest.mock import patch

from claw_swebench.claws.opensquilla import DENY_TOOLS, OpenSquillaAdapter
from claw_swebench.config import CLAW_PYTHON_HOME, OPENSQUILLA_ENV_PATH


class OpenSquillaAdapterTests(unittest.TestCase):
    def test_g12_generates_ensemble_config(self):
        adapter = OpenSquillaAdapter("G12", timeout=3600, max_turns=12)

        config = adapter._render_config()

        self.assertIn("enabled = true", config)
        self.assertIn('active_profile = "g12_k2_replace_gemini"', config)
        self.assertIn("proposer_tools = false", config)
        self.assertIn('all_failed_policy = "fallback_single"', config)
        for selector in DENY_TOOLS:
            self.assertIn(f'"{selector}"', config)

    def test_b2_generates_single_model_config(self):
        adapter = OpenSquillaAdapter("B2", timeout=3600, max_turns=12)

        config = adapter._render_config()

        self.assertIn("enabled = false", config)
        self.assertIn('model = "z-ai/glm-5.2"', config)

    def test_invalid_group_fails_early(self):
        with self.assertRaisesRegex(ValueError, "Unknown OpenSquilla experiment group"):
            OpenSquillaAdapter("Z99", timeout=3600, max_turns=12)

    def test_container_mounts_standalone_python_and_opensquilla_env(self):
        adapter = OpenSquillaAdapter("G12", timeout=3600, max_turns=12)

        args = adapter.container_run_args("django__django-16429")

        joined = " ".join(args)
        self.assertIn(f"{CLAW_PYTHON_HOME}:{CLAW_PYTHON_HOME}:ro", joined)
        self.assertIn(f"{OPENSQUILLA_ENV_PATH}:{OPENSQUILLA_ENV_PATH}:ro", joined)
        self.assertIn("/opt/opensquilla-config:ro", joined)

    def test_docker_command_forwards_key_name_without_secret_value(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-openrouter-secret"}):
            adapter = OpenSquillaAdapter("G12", timeout=3600, max_turns=12)

            cmd = adapter._build_docker_exec_command("container-name")

        self.assertIn("OPENROUTER_API_KEY", cmd)
        self.assertNotIn("fake-openrouter-secret", " ".join(cmd))
        self.assertIn("OPENSQUILLA_GATEWAY_CONFIG_PATH=/tmp/opensquilla/config.toml", cmd)
        self.assertIn("-i", cmd)

    def test_adapter_does_not_mutate_process_model_env(self):
        env = dict(os.environ)
        env.pop("OPENSQUILLA_LLM_ENSEMBLE_ACTIVE_PROFILE", None)
        with patch.dict(os.environ, env, clear=True):
            before = dict(os.environ)

            OpenSquillaAdapter("G13", timeout=3600, max_turns=12)._render_config()

            self.assertEqual(before, dict(os.environ))


if __name__ == "__main__":
    unittest.main()
