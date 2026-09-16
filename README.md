# Sustainable dining: figure-reproduction code

Code for the Cities revision on restaurant quality, walking opportunity and
socioeconomic inequality in Hong Kong.

**Release status: code-only; the empirical data bundle is not publicly released.**
This repository must not be cited as evidence that all data or all experiments
are openly reproducible. Redistribution clearance for platform-derived
aggregates and spatial inputs is still being checked. No restricted raw records,
manuscripts, reviewer correspondence or credentials are included.

## What is included

- Current manuscript analytical-figure entry points, Figs. 12–19, and shared
  styling. Figure 18 exports two separately captioned parts (18a and 18b).
- Two source-native composites: Fig. 12 combines SDI change and coverage;
  Fig. 16 combines joint-opportunity maps, distributions and sensitivity checks.
- Joint quality/price/walking computation and subgroup/planning computation,
  with their required estimator modules. These require separately authorized
  inputs; raw acquisition and multimodal model execution are not reproduced here.
- Dependency specifications, synthetic unit tests, source-file hashes and a
  data dictionary/availability inventory (metadata, not empirical observations).
- Licensed TeX Gyre Heros fonts; these are Helvetica-compatible, not Helvetica.
- The 16 September submission snapshot also retains the source-native composite
  renderers and the Times New Roman visual-system variant listed in
  [SUBMISSION_0916_CODE_MAP.md](SUBMISSION_0916_CODE_MAP.md). These scripts are
  optional publication-layout entry points and do not change empirical values.

Figures 1–11 are conceptual, data-processing or method illustrations. They are
not numerical reproductions and are therefore intentionally outside the
analytical batch runner. Supplementary archived image bytes are not included in
this release.

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
python reproduce.py --figure 16
python scripts/qc_joint_quality_access.py
python scripts/qc_joint_downstream.py
```

The current entry point is **reproduce.py**, not a legacy preview script.
Outputs go to `figures/current/`; the manuscript-number execution record is
`audit/current_figure_manifest.json`. Legacy source/artist checks go to
`audit/render_final/`; composite checks are saved beside their exports.
Missing data cause a failure, never synthetic substitution.
Every output stem now starts with the figure number used in the current
manuscript. The registry in `reproduce.py` is the machine-readable crosswalk.

| Manuscript figure | Analytical content | Canonical output stem |
| --- | --- | --- |
| 12 | Four-year SDI, specification sensitivity and coverage | `Fig12_Restaurant_Quality_SDI_Evolution_Coverage` |
| 13 | Four-year maps of six SDI components | `Fig13_Six_Component_FourYear_Atlas` |
| 14 | Citywide and component-level socioeconomic inequality | `Fig14_Restaurant_Quality_Inequality` |
| 15 | Price composition and adjusted socioeconomic associations | `Fig15_Price_Composition_Socioeconomic_Associations` |
| 16 | Joint quality, walking and price opportunity | `Fig16_Joint_Quality_Walking_Price_Opportunity` |
| 17 | Same-year socioeconomic differences in zero joint opportunity | `Fig17_SameYear_Socioeconomic_Zero_Joint_Opportunity` |
| 18a–b | Spatial-scale sensitivity and weighting diagnostics | `Fig18a_...`, `Fig18b_...` |
| 19 | Conditional public-housing siting stress test | `Fig19_Planning_Strategies` |

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

## Submission snapshot

This branch was refreshed for the 16 September manuscript package. The public
repository remains code-only: empirical source tables, raw platform records,
restaurant identifiers, coordinates and origin-destination pairs are excluded.
The matching private submission folder contains the authorized local source
bundle and should not be treated as a public redistribution package.
