# V203 processed display-data release

## Target and scope

This deposit corresponds to the author-supplied V203 bilingual manuscript and
supplement of 5 October 2026. Document captions were read to confirm numbering;
the Word files and their preparation/check tools are not deposited. It supports
main Figs. 12–19, SI Fig. S9 and the selected numerical results below. Later SI
edits may change labels/numbering. All values retain the frozen result lineage.
It does not independently validate the original constructs or rerun models.

| V203 figure | Display input folders | Native plot lineage |
| --- | --- | --- |
| 12: SDI evolution/coverage | area, temporal | V147 composite |
| 13: six-component atlas | area, atlas | V138 atlas; fixed V134/V135 scores |
| 14: structural/component inequality | structure, component_social | V148; frozen weighted box statistics, no census/raw rebuild |
| 15: income and market composition | market | V203 six-domain primary coefficients; V204 layout |
| 16: joint walking/price/quality opportunity | joint, network | V147 composite |
| 17: census-group zero joint opportunity | subgroups, contrasts | V138/V204; archived contrasts, no replicate re-estimation |
| 18: weighting/scale diagnostics | sensitivity | V148 diagnostics only; historical companion maps excluded |
| 19: siting outcomes/overlap | planning | V148; anonymised node labels preserve sets, locations and overlaps |
| SI S9: income specifications | regression_comparison | V203 six-/five-domain display |

The latest cosmetic script's historical `03` option is not mapped to V203
main Fig. 3, which is a method illustration. It is outside this deposit.

## Use

Install the existing pinned `requirements.txt` in a separate Python 3.12
environment. Windows users can invoke their environment's `python.exe`; the
same relative-path commands work on Linux. Do not create environments inside
OneDrive. From the repository root:

```sh
python scripts/reproduce_display.py --validate-only
python scripts/reproduce_display.py --output-dir outputs/display-v203
python scripts/reproduce_display.py --figure 15 --output-dir outputs/figure15-v203
python scripts/build_display_tables.py --output-dir outputs/tables-v203
```

Every output directory must be fresh. A missing/changed input, wrong schema,
invalid geometry or external empirical read fails instead of substituting
data. Each figure receives a fresh subprocess; native numeric/layout checks
still run. The new receipt records consumed input hashes and three export
hashes. Internal legacy audit wording can refer to the old private workflow;
`public_display_receipt.json` defines the public run scope. Rendered assets,
logs and caches are local outputs, excluded from the repository.

## Numerical table mapping (V203 snapshot)

`build_display_tables.py` exports descriptive filenames. These identify
numerical contents, not a final numbered/revised supplementary document.

| V203 SI reference | Public numerical output/content |
| --- | --- |
| S3 | component_weights |
| S4 | nutrition_omission_zero_opportunity and nutrition_omission_thresholds |
| S5 | lsbg_dcca_agreement (primary equal-six rows among archived variants) |
| S6 | network_opportunity_15min |
| S8 | sdi_structural_inequality_intervals |
| S9 | alternative_sdi_income_concentration |
| S10 | income_coefficients_2024 (primary and sensitivity explicitly labelled) |
| S12 | planning_outcomes (including baseline row) |
| S13 | joint_opportunity_estimates and joint_opportunity_intervals |
| S14 | census_group_joint_opportunity_2024 |
| S15 | six_census_domain_contrasts_2024 |

The exports retain unrounded values and additional rows for the relevant
figure comparisons; they are not copies of formatted Word tables. S1/S2/S7/S11
and the literature/method tables are not promised as complete formatted-table
reproductions. SI S14 reference-row q is blank and `is_reference=true`.

## Source/version, units and missing values

The per-file manifest records historical prepared-output lineage, original
source SHA-256, public-file SHA-256, columns, row counts, CRS and geometry
operations. Most sources are the fixed-Nutrition V134/V135 result chain. Market
primary coefficients and the six-/five-domain comparison come from the V203
income-results tables. The historical five-domain market table is not used as
the primary model. No later source tables or manuscript values are fabricated.

CSV is UTF-8, comma separated, with blank numeric values denoting unavailable
or undefined values. Preserve them; do not impute zero. True zero opportunity,
missing SDI and `network_available=false` have distinct meanings. Fractions
remain 0–1 unless a column explicitly says percentage points. Reproduction
uses archived intervals; it does not bootstrap or refit. The field dictionary
provides units/definitions for every released column.

LSBG is primary; DCCA provides scale/uncertainty comparisons. SDI years are
2011/2016/2021/2024; audited walking years are 2016/2021/2024. Geometry is
EPSG:2326 (Hong Kong 1980 Grid), in metres; anonymous candidate drawing points
are longitude/latitude in EPSG:4326. The simplified polygons are plotting
geometries, unsuitable for routing, exact boundary-area work or record linkage.
The manifest records the renderer's native 10/18-m simplification; the public
adapter avoids a second simplification. Atlas and planning polygons retain
their existing drawing geometry. Fig. 18 reads only its three aggregate
statistical tables; the older companion maps and their clipping mask are excluded.
No released geometry requires a validity repair.

SDI excludes recorded price. Practice remains an undated snapshot. Joint
opportunity is potential network reach under the price and destination-area
SDI gates, not observed visits, restaurant certification or household spending.
2024 census-group results project fixed 2021 composition. Household-income
groups use domestic-household counts; other groups use person counts. The
`ma_hh` field is an area census median used by the Fig. 16 distribution plot,
not an individual's/household's income record. Planning is a hypothetical
fixed-pool siting comparison, with no causal/forecast claim and no invented CI.

## Disclosure and licence boundary

Only actual display fields and aggregate result tables are deposited. There
are no names/addresses, raw reviews/images, restaurant master/source IDs,
personal profiles, record-level hashes, routing OD pairs, credentials, rendered
caches, private absolute paths, Word files or their workflow/check scripts.
Anonymous planning identifiers are display keys; area codes are geometry keys.
The code licence is unchanged. No new data licence, raw-source redistribution
right, archival DOI or claim that all original analysis is open is asserted.

Use manifest hashes to review or revert this version. A changed result bundle
requires an explicitly versioned manifest/dictionary and renewed field review;
do not silently replace the primary CSV or rename private intermediates as public.

## Target-document numerical check

The release receipt in `manifests/v203_document_result_match.json` records 195
checks against the supplied V203 SI S10/S12/S14/S15 numeric cells, including
intervals and household/person denominators. All selected cells match at the
document display precision. This is a bounded correspondence check, not an
independent scientific or complete manuscript audit. No document is edited.
