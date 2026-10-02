# Current manuscript reproduction (V180)

This guide is for a coauthor checking the current manuscript and Supplementary
Information. Python 3.12 and the versions in `requirements.txt` are the tested
environment. The repository supplies code; empirical inputs are supplied
separately to authorized collaborators.

## 1. Install from the current branch

```sh
git clone --branch codex/v95-public-update https://github.com/Brad-zqh/cities-sustainable-dining.git
cd cities-sustainable-dining
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on macOS/Linux, then run:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Unit tests use synthetic fixtures. The following steps use the empirical inputs.

## 2. Obtain authorized inputs

Keep these outside the clone. The prepared bundle has two directories:

```text
prepared_bundle/
  source_data/                         # frozen aggregate figure/table bundles
  outputs/restricted/
    v134_fixed_nutrition_full_chain/   # four-year, six-component LSBG/DCCA inputs
    v134_fixed_nutrition_joint_access/
    v134_fixed_nutrition_joint_downstream/
    v135_fixed_nutrition_aggregation_variants/
    v135_fixed_nutrition_temporal_uncertainty/
    v135_fixed_nutrition_structural_inequality_999/
    v135_fixed_nutrition_variant_inequality_999/
    v135_fixed_nutrition_price_market/
    v135_fixed_nutrition_component_social/
    v135_fixed_nutrition_subgroup_contrasts/
    v135_no_nutrition_opportunity_sensitivity_v3_parameterized_replay/
```

Analysis additionally uses an authorized project tree containing eligible-outlet
tables, audited outlet–LSBG walking pairs and planning-node walking pairs;
the restaurant master CSV; and a census tree. `--census-root` contains:

- `2016/5. 2016LargeStreetBlockGroups(Large Subunit Group)_1622_SHP/LSBG_16BC_percent.gpkg`
- `2021/5. 2021LargeSubunitGroups(Large Street Block Group)_1746_SHP/LSUG_21C_percent.gpkg`

`--project-root` contains the three `outputs/t3_count_candidate*` eligible-outlet
folders, three `outputs/restricted/t3_reachable_pairs_*` folders,
`source_data/fig04_network_price_v4/lsbg_network_price_opportunity.csv`, and
`outputs/restricted/v7_public_housing_planning_node_coverage_202608/public_housing_site_lsbg_reachable_pairs_15min.csv`.

The prepared bundle retains the spatial units, population weights, missing
states and fixed-result references used in the manuscript. Permissions for
using and sharing it follow the original providers and the study's data agreements.

## 3. Recalculate results

Substitute your local paths. The example is one command, valid on Windows,
macOS and Linux; choose a new output folder outside all input trees.

```sh
python verify_current.py --data-root "prepared_bundle" --project-root "authorized_project" --census-root "authorized_census" --master "authorized_restaurant_master.csv" --output-dir "recalculated_v180" --authorized
```

The eight tasks rebuild aggregation/weighting/MAUP, temporal uncertainty,
primary SDI inequality, variant inequality, price-market comparisons, joint
opportunity, social-group/planning results and component comparisons.
They retain the archived estimator definitions, bootstrap seeds and 999 draws.
Each worker computes fresh outputs before comparing every numeric column,
missingness state and selected-site identifier with the frozen references.

Expected receipt: `recalculated_v180/verification.json` has
`status: MATCHES_FROZEN_REFERENCE` and `inputs_unchanged: true`. Per-task logs
explain any failure. Numeric comparison uses 1e-10 absolute/relative tolerance;
identifiers and labels match exactly. Source and input hashes bind the run.

For a shorter independent check, use `--tasks aggregation,temporal,structural`.
Dependency order follows the full eight-task sequence printed in `--help`.
When an upstream task is omitted, the corresponding frozen intermediate is used.

## 4. Regenerate current result figures

```sh
python reproduce.py --edition v180 --data-root "prepared_bundle" --census-root "authorized_census" --output-dir "figures_v180"
```

This produces current Figs. 12–19 as PNG, PDF and SVG in numbered folders.
`--figure 19` selects one figure. Fig. 18's plotting lineage also exports an
additional scale-map page; the current Word Fig. 18 uses the diagnostics page.
`figure_reproduction.json` records actual outputs and renderer commands.
The explicit `--edition legacy` and `--edition v95` preserve historical numbering.

## 5. Compare with the actual Word files

```sh
python verify_documents.py --manuscript "01_Manuscript_Bilingual_V180_A4.docx" --supplement "02_Supplementary_Information_Bilingual_V180_Indexed.docx" --data-root "prepared_bundle" --results-root "recalculated_v180" --figures-root "figures_v180"
```

This V180-specific mapping checks 51 main-text numeric anchors and 698
supplementary-table values, with display rounding respected. Expected
`document_comparison.json` numeric status is `PASS`. Every source is labelled:
518 SI values use recalculated results and 180 use frozen source tables.
Figure comparison reads the embedded Word images and compares RGB pixels,
excluding PNG metadata. Pixel equality can vary with renderer versions;
dimensions and channel differences are reported alongside the numerical checks.

## Scope of this check

The replay begins with frozen multimodal area components and audited walking
pairs. It checks downstream calculation and agreement with the manuscript.
Platform acquisition, fresh LLM inference, original network construction,
the conceptual/method illustrations, recipe-reference benchmark inference and
the source-only SI comparisons retain their separate evidence records.
The replay does not establish independent validation of the SDI construct.
Receipts containing local paths belong with the private collaborator outputs.
