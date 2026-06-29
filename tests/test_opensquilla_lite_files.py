from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OpenSquillaLiteFilesTests(unittest.TestCase):
    def test_lite_files_have_expected_counts_and_no_duplicates(self):
        multilingual = _read_ids(PROJECT_ROOT / "config" / "opensquilla_lite_multilingual.txt")
        verified = _read_ids(PROJECT_ROOT / "config" / "opensquilla_lite_verified.txt")

        self.assertEqual(70, len(multilingual))
        self.assertEqual(10, len(verified))
        self.assertEqual(80, len(set(multilingual) | set(verified)))
        self.assertFalse(set(multilingual) & set(verified))


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


if __name__ == "__main__":
    unittest.main()
