"""Reproduce the eight-panel LSBG–DCCA MAUP map audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from figures.v4_plot_four_year_results import load_frames, plot_maup_figure  # noqa: E402


if __name__ == "__main__":
    plot_maup_figure(load_frames("LSBG"), load_frames("DCCA"))
