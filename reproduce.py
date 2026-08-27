"""Reproduce the current manuscript data figures (3–16), not old previews."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=int, choices=range(3, 17))
    args = parser.parse_args()
    if not any((ROOT / 'source_data').rglob('*.csv')):
        parser.error('Empirical data are not included in the code-only release. '
                     'Supply the cleared source_data bundle described in DATA_AVAILABILITY.md. '
                     'No synthetic values will be substituted.')
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/render_final.py')]
    if args.figure is not None:
        command.append(str(args.figure))
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
