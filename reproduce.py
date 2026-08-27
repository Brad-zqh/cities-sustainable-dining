"""Reproduce the twelve current main data figures (3–14), including composites."""
from pathlib import Path
import argparse
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parent
FIGURE_MAP = {3:3, 4:4, 5:'A', 6:6, 7:7, 8:'B', 9:10,
              10:11, 11:12, 12:13, 13:14, 14:16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=int, choices=sorted(FIGURE_MAP))
    args = parser.parse_args()
    if not any((ROOT / 'source_data').rglob('*.csv')):
        parser.error('Empirical data are not included in the code-only release. '
                     'Supply the cleared source_data bundle described in DATA_AVAILABILITY.md. '
                     'No synthetic values will be substituted.')
    runs=[]
    for number in ([args.figure] if args.figure else FIGURE_MAP):
        original=FIGURE_MAP[number]
        if isinstance(original,str):
            command=[sys.executable,'-X','utf8',str(ROOT/'scripts/render_evidence_composites.py'),
                     original,'--output-dir',str(ROOT/'figures/current')]
        else:
            command=[sys.executable,'-X','utf8',str(ROOT/'scripts/render_final.py'),str(original)]
        status=subprocess.call(command,cwd=ROOT)
        runs.append(dict(current_figure=number,source_figure_or_composite=original,returncode=status))
        audit=ROOT/'audit/current_figure_manifest.json'
        audit.parent.mkdir(parents=True,exist_ok=True)
        audit.write_text(json.dumps(runs,indent=2),encoding='utf-8')
        if status: return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
