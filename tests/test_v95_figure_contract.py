"""Contract checks for the manuscript-facing V95 figure map."""

from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class V95FigureContractTests(unittest.TestCase):
    def test_v95_manifest_has_current_figure_families(self) -> None:
        manifest = json.loads(
            (ROOT / "manifests" / "v95_figure_entry_points.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(manifest["figures"]),
            {"11", "12", "13", "14", "15", "16", "17a", "17b", "18"},
        )
        for figure_key, spec in manifest["figures"].items():
            entry = ROOT / spec["entry_point"]
            self.assertTrue(entry.is_file(), figure_key)
            self.assertTrue(spec["output_stems"], figure_key)
            self.assertTrue(spec["source_bundles"], figure_key)

    def test_legacy_map_is_retained_separately(self) -> None:
        namespace: dict[str, object] = {
            "__name__": "v95_contract",
            "__file__": str(ROOT / "reproduce.py"),
        }
        exec((ROOT / "reproduce.py").read_text(encoding="utf-8"), namespace)
        self.assertEqual(set(namespace["FIGURE_MAP"]), set(range(3, 15)))
        self.assertEqual(namespace["FIGURE_MAP"][5], "A")
        self.assertEqual(namespace["FIGURE_MAP"][8], "B")

    def test_new_public_files_do_not_contain_personal_paths_or_credentials(self) -> None:
        files = [
            ROOT / "reproduce.py",
            ROOT / "README.md",
            ROOT / "docs" / "V95_FIGURE_REPRODUCTION.md",
            ROOT / "manifests" / "v95_figure_entry_points.json",
            *sorted((ROOT / "scripts").glob("render_*composite*.py")),
        ]
        forbidden = ("D:\\OneDrive", "E:\\UserData", "C:\\Users", "Lazqh", "ghp_")
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(token in text for token in forbidden), path)


if __name__ == "__main__":
    unittest.main()
