# Sustainable dining: analysis and figure reproduction

## Latest author-requested figure refinements

The 5 October 2026 code-only update adds the current Fig. 15 six-domain
coefficient display and the requested spacing, subgroup-legend and square-panel
refinements for Figs. 15, 17 and 3. See
[the figure refinement guide](docs/V204_FIGURE_COSMETICS.md).
Existing V180 entry points and the empirical-data sharing boundary are unchanged.

## Current V180 manuscript

The current branch includes the eight-task numerical replay and current Figs.
12–19. Word manuscript preparation, editing and document checks remain private. Follow
[the coauthor reproduction guide](docs/V180_REPRODUCTION.md).

```sh
python verify_current.py --help
python reproduce.py --edition v180 --help
```

Supply authorized inputs outside the clone and use fresh output directories.
The scripts record source/input hashes, preserve source files and report
differences explicitly. The data-sharing boundary remains separate from this
code release. The sections below document historical entry points.

Code for the Cities revision on sustainable dining, walking opportunity and
social inequality in Hong Kong.

**Release status: code-only; the empirical data bundle is not publicly released.**
This repository must not be cited as evidence that all data or all experiments
are openly reproducible. Redistribution clearance for platform-derived
aggregates and spatial inputs is still being checked. No restricted raw records,
manuscripts, reviewer correspondence or credentials are included.

The V135 revision adds synthetic-testable utilities for annual dish mentions,
nutrition scoring and deletion audits, and external benchmark bookkeeping.
This is a **partial code-only update**, not a release of V135 empirical inputs,
final figure assets or a clean-clone reproduction of every revised result.
See `docs/V135_CODE_ONLY_STATUS.md` for the boundary.

The latest V148 result-figure code lineage is included as code only. Its
layered entry points and required separately cleared inputs are described in
[`docs/V148_VISUALIZATION_CODE.md`](docs/V148_VISUALIZATION_CODE.md).

## What is included

- The preceding-release main-text data-figure entry points, Figs. 3–14, and
  shared styling. The current V95 manuscript-facing map is recorded separately
  in `manifests/v95_figure_entry_points.json` and covers Figs. 11–18.
- Two source-native composites: Fig. 5 combines SDI change and coverage;
  Fig. 8 combines joint-opportunity maps, distributions and sensitivity checks.
- Joint quality/price/walking computation and subgroup/planning computation,
  with their required estimator modules. These require separately authorized
  inputs; raw acquisition and multimodal model execution are not reproduced here.
- Dependency specifications, synthetic unit tests, source-file hashes and a
  data dictionary/availability inventory (metadata, not empirical observations).
- Licensed TeX Gyre Heros fonts; these are Helvetica-compatible, not Helvetica.

Word manuscript writing, revision, layout and document-comparison scripts
are outside this public repository.

The historical conceptual and method illustrations are outside these plotting
entry points. Current V180 reproduction covers Figs. 12–19; other main-text
figures and supplementary model-comparison figures have separate source
lineages. Supplementary archived image bytes are not included in this release.

## Install and test

Python 3.12 is the tested target. Create an isolated environment:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on POSIX. Then:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Alternatively, `conda env create -f environment.yml` creates the environment
using the same pip requirements. Tests use explicitly synthetic fixtures only.
Passing them does not validate the empirical findings.

## Reproduce figures when authorized data are supplied

Place the cleared data bundle in `source_data/`, retaining the relative layout
listed in `manifests/source_inventory.json`.

```sh
python reproduce.py
python reproduce.py --figure 10
python scripts/qc_joint_quality_access.py
python scripts/qc_joint_downstream.py
```

For the current V95 manuscript numbering, use the edition flag. The runner
accepts `17a` and `17b` for the two parts of the scale/weighting result:

```powershell
python reproduce.py --edition v95
python reproduce.py --edition v95 --figure 15
python reproduce.py --edition v95 --figure 17b
```

The V95 figure-level contract and source-bundle requirements are documented in
[`docs/V95_FIGURE_REPRODUCTION.md`](docs/V95_FIGURE_REPRODUCTION.md). The
legacy 3–14 map remains unchanged for historical reruns; do not infer current
manuscript numbering from that table.

The current entry point is **reproduce.py**, not a legacy preview script.
Outputs go to `figures/current/`; the current-number execution record is
`audit/current_figure_manifest.json`. Legacy source/artist checks go to
`audit/render_final/`; composite checks are saved beside their exports.
Missing data cause a failure, never synthetic substitution.
Legacy filenames are retained for unmerged figures; use the mapping below.

| Current figure | Source in the preceding 16-figure layout |
| --- | --- |
| 3, 4 | 3, 4 |
| 5 | 5 + 15, redrawn as an eight-panel composite |
| 6, 7 | 6, 7 |
| 8 | 8 + 9, redrawn as an eleven-panel composite |
| 9–14 | 10, 11, 12, 13, 14, 16, respectively |

The quality checks distinguish an unavailable restricted-input check from a
passed check. SHA-256 verifies file identity, not validity or ownership.

## Scientific boundaries

- SDI snapshots: 2011, 2016, 2021 and 2024. Walking networks: 2016, 2021 and 2024.
- LSBG is the primary spatial unit; DCCA is used for sensitivity/block resampling.
- The joint outcome counts reachable low-price opportunities in destination
  areas meeting the SDI threshold. It is not a validated individual-restaurant
  SDI or a measure of household expenditure or realized dining.
- The 2024 subgroup analysis holds 2021 census composition fixed; it is not a
  contemporaneous 2024 census estimate.
- Planning is a hypothetical stress test, not a forecast or causal evaluation.
- Same-year subgroup stars follow BH-adjusted values from paired DCCA-block
  replicates. Styling does not add significance or change estimates.

See [data availability](DATA_AVAILABILITY.md),
[reproduction scope](REPRODUCIBILITY.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).

Code is MIT-licensed. Fonts retain their own licence. No licence for withheld
third-party data is granted by this repository. An archived DOI and final
manuscript citation will be added only when they exist.
