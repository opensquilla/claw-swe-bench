#!/usr/bin/env python3
"""Tests for the per-claw context-window reporter.

Run from the repository root:

    python3 -m unittest discover -s tests
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from claw_swebench import context_window as cw

HERMES_YAML = """\
model:
  default: "glm-5.1"
  provider: "openrouter"
custom_providers:
  - name: "deepseek"
    context_length: 1000000
  - name: "openrouter"
    context_length: 200000
"""

ZEROCLAW_TOML = """\
[agent]
max_tool_iterations = 300
max_context_tokens = 128000
"""


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="claw_ctx_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = mock.patch.object(cw, "CLAW_CONFIGS_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, claw, name, text):
        d = self.tmp / claw
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)

    # -- hermes ------------------------------------------------------------

    def test_hermes_reads_the_provider_the_model_uses(self):
        self.write("hermes", "config.yaml", HERMES_YAML)
        got = cw.effective_context_window("hermes")
        self.assertEqual(got.tokens, 200000)  # openrouter, not deepseek's 1000000
        self.assertIsNone(got.note)

    def test_hermes_reports_a_provider_with_no_matching_entry(self):
        self.write("hermes", "config.yaml",
                   'model:\n  provider: "nope"\ncustom_providers: []\n')
        got = cw.effective_context_window("hermes")
        self.assertIsNone(got.tokens)
        self.assertIn("nope", got.note)

    # -- zeroclaw ----------------------------------------------------------

    def test_zeroclaw_reads_max_context_tokens(self):
        self.write("zeroclaw", "config.toml", ZEROCLAW_TOML)
        self.assertEqual(cw.effective_context_window("zeroclaw").tokens, 128000)

    def test_zeroclaw_without_the_key_is_unset(self):
        self.write("zeroclaw", "config.toml", "[agent]\nmax_tool_iterations = 300\n")
        got = cw.effective_context_window("zeroclaw")
        self.assertIsNone(got.tokens)
        self.assertEqual(got.note, "unset")

    # -- nanobot -----------------------------------------------------------

    def test_nanobot_reads_context_window_tokens(self):
        self.write("nanobot", "config.json",
                   json.dumps({"agents": {"defaults": {"contextWindowTokens": 202752}}}))
        self.assertEqual(cw.effective_context_window("nanobot").tokens, 202752)

    def test_nanobot_without_the_key_says_it_falls_back(self):
        """The case reported in #1: the budget is whatever NanoBot defaults to."""
        self.write("nanobot", "config.json",
                   json.dumps({"agents": {"defaults": {"model": "qwen3.6-flash"}}}))
        got = cw.effective_context_window("nanobot")
        self.assertIsNone(got.tokens)
        self.assertIn("runtime default", got.note)

    # -- failure modes -----------------------------------------------------

    def test_missing_config_is_reported_not_raised(self):
        got = cw.effective_context_window("zeroclaw")
        self.assertIsNone(got.tokens)
        self.assertEqual(got.note, "config not found")

    def test_invalid_json_is_reported_not_raised(self):
        self.write("nanobot", "config.json", "{not json")
        got = cw.effective_context_window("nanobot")
        self.assertIsNone(got.tokens)
        self.assertIn("invalid JSON", got.note)

    def test_broken_config_never_takes_the_run_down(self):
        self.write("zeroclaw", "config.toml", "this is not toml = = =")
        got = cw.effective_context_window("zeroclaw")
        self.assertIsNone(got.tokens)
        self.assertIsNotNone(got.note)

    def test_unknown_claw(self):
        got = cw.effective_context_window("does-not-exist")
        self.assertIsNone(got.tokens)
        self.assertIn("no reader", got.note)

    def test_generic_has_no_context_budget(self):
        got = cw.effective_context_window("generic")
        self.assertIsNone(got.tokens)
        self.assertIn("output", got.note)

    # -- reporting ---------------------------------------------------------

    def test_describe_all_covers_every_claw_with_a_reader(self):
        rows = cw.describe_all()
        self.assertEqual({r.claw for r in rows},
                         {"hermes", "zeroclaw", "nanobot", "openclaw", "generic"})

    def test_table_shows_the_number_and_flags_unset(self):
        self.write("hermes", "config.yaml", HERMES_YAML)
        table = cw.format_table(cw.describe_all())
        self.assertIn("200,000", table)
        self.assertIn("unset", table)

    def test_as_dict_is_json_serialisable_for_metadata(self):
        self.write("zeroclaw", "config.toml", ZEROCLAW_TOML)
        blob = cw.effective_context_window("zeroclaw").as_dict()
        json.dumps(blob)
        self.assertEqual(blob["tokens"], 128000)
        self.assertEqual(blob["key"], "agent.max_context_tokens")


class ShippedExampleTests(unittest.TestCase):
    """The examples users are told to copy should say what they run with."""

    ROOT = Path(__file__).resolve().parents[1] / "claw_configs"

    def test_examples_parse(self):
        json.loads((self.ROOT / "nanobot" / "config.json.example").read_text())
        import yaml
        yaml.safe_load((self.ROOT / "hermes" / "config.yaml.example").read_text())

    def test_hermes_and_zeroclaw_examples_pin_a_budget(self):
        hermes = (self.ROOT / "hermes" / "config.yaml.example").read_text()
        zeroclaw = (self.ROOT / "zeroclaw" / "config.toml.example").read_text()
        self.assertIn("context_length", hermes)
        self.assertIn("max_context_tokens", zeroclaw)

    def test_nanobot_example_documents_the_missing_budget(self):
        """It ships unset, so the comment has to say so."""
        cfg = json.loads((self.ROOT / "nanobot" / "config.json.example").read_text())
        self.assertNotIn("contextWindowTokens", cfg["agents"]["defaults"])
        self.assertIn("contextWindowTokens", cfg["_comment"])


if __name__ == "__main__":
    unittest.main()
