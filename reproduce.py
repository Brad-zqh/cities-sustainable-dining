"""Reproduce the legacy figure map or the current V95 manuscript figures."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parent
# Keep the preceding release stable: downstream users may still request its
# 3–14 numbering. The V95 map is loaded from a separate manifest below.
FIGURE_MAP = {3:3, 4:4, 5:'A', 6:6, 7:7, 8:'B', 9:10,
              10:11, 11:12, 12:13, 13:14, 14:16}
LEGACY_FIGURE_MAP = FIGURE_MAP
V95_MANIFEST = ROOT / "manifests" / "v95_figure_entry_points.json"


def load_v95_manifest() -> dict[str, dict[str, object]]:
    """Load the manuscript-facing V95 entry-point contract."""
    payload = json.loads(V95_MANIFEST.read_text(encoding="utf-8"))
    figures = payload.get("figures")
    if not isinstance(figures, dict) or not figures:
        raise ValueError("V95 figure manifest has no figure entries")
    return figures


V95_FIGURE_MAP = load_v95_manifest()


def source_data_available() -> bool:
    """Return whether any authorized local source file is present."""
    data_root = ROOT / "source_data"
    return data_root.is_dir() and any(path.is_file() for path in data_root.rglob("*"))


def legacy_spec(number: int) -> dict[str, object]:
    """Convert the historical integer map to the shared runner schema."""
    original = FIGURE_MAP[number]
    if isinstance(original, str):
        return {
            "entry_point": "scripts/render_evidence_composites.py",
            "arguments": [original],
            "output_subdir": "",
        }
    return {
        "entry_point": "scripts/render_final.py",
        "arguments": [str(original)],
        "output_subdir": "",
    }


def run_spec(spec: dict[str, object], output_dir: Path) -> int:
    """Execute one entry point with a per-figure output directory."""
    entry_value = spec.get("entry_point")
    if not isinstance(entry_value, str):
        raise ValueError("figure entry point must be a string")
    entry = ROOT / entry_value
    if not entry.is_file():
        raise FileNotFoundError(entry)
    raw_arguments = spec.get("arguments", [])
    if not isinstance(raw_arguments, list) or not all(isinstance(x, str) for x in raw_arguments):
        raise ValueError(f"invalid arguments for {entry_value}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-X", "utf8", str(entry), *raw_arguments]
    # Composite renderers expose --output-dir; the older single-figure scripts
    # receive the same location through the documented environment variable.
    if Path(entry_value).name.startswith("render_") and Path(entry_value).name != "render_final.py":
        command.extend(["--output-dir", str(output_dir)])
    env = os.environ.copy()
    env["CITIES_FIGURE_DIR"] = str(output_dir)
    return subprocess.call(command, cwd=ROOT, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edition', choices=['legacy', 'v95'], default='legacy',
                        help='Use the historical 3–14 map or the current V95 map.')
    parser.add_argument('--figure',
                        help='One figure number (V95 also accepts 17a or 17b).')
    parser.add_argument('--output-dir', type=Path,
                        help='Root for generated figure outputs; defaults to figures/current or figures/v95.')
    args = parser.parse_args()
    if not source_data_available():
        parser.error('Empirical data are not included in the code-only release. '
                     'Supply the cleared source_data bundle described in DATA_AVAILABILITY.md. '
                     'No synthetic values will be substituted.')
    if args.edition == 'legacy':
        specs = {str(number): legacy_spec(number) for number in FIGURE_MAP}
        default_root = ROOT / 'figures' / 'current'
        audit = ROOT / 'audit' / 'current_figure_manifest.json'
    else:
        specs = V95_FIGURE_MAP
        default_root = ROOT / 'figures' / 'v95'
        audit = ROOT / 'audit' / 'v95_figure_manifest.json'
    requested = [args.figure] if args.figure else list(specs)
    unknown = [value for value in requested if value not in specs]
    if unknown:
        parser.error(f"unknown {args.edition} figure(s): {', '.join(unknown)}")
    output_root = (args.output_dir or default_root).resolve()
    runs = []
    for figure_key in requested:
        spec = specs[figure_key]
        output_dir = output_root / str(spec.get('output_subdir', figure_key))
        try:
            status = run_spec(spec, output_dir)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        runs.append(dict(edition=args.edition, manuscript_figure=figure_key,
                         entry_point=spec['entry_point'], output_dir=str(output_dir),
                         returncode=status))
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(runs, indent=2), encoding='utf-8')
        if status:
            return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
