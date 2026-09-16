"""Reproduce the current manuscript's eight analytical figures (Figs. 12–19).

Figures 1–11 are conceptual or method illustrations and are deliberately not
presented as numerical reproductions. Figure 18 contains two separately
exported parts, 18a and 18b.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "current"
AUDIT = ROOT / "audit" / "current_figure_manifest.json"

# This registry follows the numbering and captions in the 16 September 2026
# manuscript. The ``source`` field documents reused evidence-locked analysis
# modules; it is not an alternative figure number.
MANUSCRIPT_FIGURES = {
    12: {
        "caption": "Restaurant quality SDI, 2011 to 2024",
        "source": "source-native SDI evolution and coverage composite",
        "commands": (("scripts/render_evidence_composites.py", "A"),),
        "stems": ("Fig12_Restaurant_Quality_SDI_Evolution_Coverage",),
    },
    13: {
        "caption": "Four-year spatial distributions of six restaurant quality components",
        "source": "four-year component atlas",
        "commands": (("scripts/render_final.py", "6"),),
        "stems": ("Fig13_Six_Component_FourYear_Atlas",),
    },
    14: {
        "caption": "Restaurant quality inequality",
        "source": "structural and component-level inequality composite",
        "commands": (("scripts/render_quality_inequality_composite.py",),),
        "stems": ("Fig14_Restaurant_Quality_Inequality",),
    },
    15: {
        "caption": "Price composition and adjusted socioeconomic associations",
        "source": "four-year price-market decomposition",
        "commands": (("scripts/render_final.py", "7"),),
        "stems": ("Fig15_Price_Composition_Socioeconomic_Associations",),
        "required": ("Fig15_Price_Composition_Socioeconomic_Associations_600dpi.png",),
    },
    16: {
        "caption": "Joint quality, walking and price opportunity",
        "source": "source-native joint-opportunity evidence composite",
        "commands": (("scripts/render_evidence_composites.py", "B"),),
        "stems": ("Fig16_Joint_Quality_Walking_Price_Opportunity",),
    },
    17: {
        "caption": "Same-year socioeconomic differences in zero joint opportunity",
        "source": "same-year subgroup analysis with fixed 2021 composition in 2024",
        "commands": (("scripts/render_final.py", "10"),),
        "stems": ("Fig17_SameYear_Socioeconomic_Zero_Joint_Opportunity",),
    },
    18: {
        "caption": "Spatial-scale sensitivity and weighting diagnostics",
        "source": "LSBG-DCCA scale and weighting sensitivity",
        "commands": (
            ("scripts/render_scale_weight_composite_single_page.py", "--split-part", "maps"),
            ("scripts/render_scale_weight_composite_single_page.py", "--split-part", "diagnostics"),
        ),
        "stems": (
            "Fig18a_Spatial_Scale_Sensitivity_Maps",
            "Fig18b_Weighting_And_Cross_Scale_Diagnostics",
        ),
    },
    19: {
        "caption": "Public-housing food-provision siting strategies",
        "source": "conditional planning stress-test composite",
        "commands": (("scripts/render_planning_composite.py",),),
        "stems": ("Fig19_Planning_Strategies",),
    },
}

# Backwards-compatible import name used by external checks.
FIGURE_MAP = MANUSCRIPT_FIGURES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in sorted((ROOT / "source_data").rglob("*"))
        if path.is_file()
    }


def command_for(parts: tuple[str, ...]) -> list[str]:
    command = [sys.executable, "-X", "utf8", str(ROOT / parts[0]), *parts[1:]]
    if parts[0] in {
        "scripts/render_evidence_composites.py",
        "scripts/render_quality_inequality_composite.py",
        "scripts/render_scale_weight_composite_single_page.py",
        "scripts/render_planning_composite.py",
    }:
        command.extend(["--output-dir", str(OUT)])
    return command


def exported_files(stems: tuple[str, ...]) -> list[Path]:
    return sorted({path for stem in stems for path in OUT.glob(f"{stem}*") if path.is_file()})


def write_manifest(payload: dict) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=sorted(MANUSCRIPT_FIGURES))
    args = parser.parse_args()
    before = source_hashes()
    if not before:
        parser.error(
            "Empirical data are not included in the code-only release. Supply the "
            "cleared source_data bundle described in DATA_AVAILABILITY.md. No "
            "synthetic values will be substituted."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    selected = [args.figure] if args.figure else list(MANUSCRIPT_FIGURES)
    report = {
        "manuscript_version": "Submission 0916",
        "scope": "analytical main-text figures 12–19; Figure 18 exports parts a and b",
        "conceptual_method_figures": "Figures 1–11 are outside numerical reproduction",
        "source_files_before": len(before),
        "runs": [],
    }
    write_manifest(report)

    for number in selected:
        spec = MANUSCRIPT_FIGURES[number]
        run = {
            "manuscript_figure": number,
            "caption": spec["caption"],
            "source_analysis": spec["source"],
            "commands": [],
        }
        report["runs"].append(run)
        started = time.time_ns()
        for parts in spec["commands"]:
            command = command_for(parts)
            process = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
            )
            run["commands"].append({
                "argv": command,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            })
            write_manifest(report)
            if process.returncode:
                return process.returncode

        required_png = [
            OUT / name for name in spec.get(
                "required", tuple(f"{stem}.png" for stem in spec["stems"])
            )
        ]
        missing = [path.name for path in required_png if not path.is_file()]
        stale = [path.name for path in required_png if path.is_file() and path.stat().st_mtime_ns < started]
        if missing or stale:
            run["returncode"] = 1
            run["output_contract_error"] = {"missing": missing, "not_rewritten": stale}
            write_manifest(report)
            return 1
        outputs = exported_files(spec["stems"])
        run["returncode"] = 0
        run["output_sha256"] = {
            path.relative_to(ROOT).as_posix(): digest(path) for path in outputs
        }
        write_manifest(report)

    after = source_hashes()
    if before != after:
        report["source_data_unchanged"] = False
        write_manifest(report)
        raise RuntimeError("Source data changed during rendering")
    report["source_data_unchanged"] = True
    report["source_files_after"] = len(after)
    report["status"] = "PASS"
    write_manifest(report)
    print(
        f"PASS: {len(selected)} manuscript figure entries; "
        f"{len(after)} source files unchanged; manifest={AUDIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
