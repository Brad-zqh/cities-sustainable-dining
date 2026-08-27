# Reproduction scope and verification

The current runner covers 12 main-text data figures, numbered 3–14. Each is
rendered from the local source bundle rather than from an AI-generated image.
Source hashes are compared before and after rendering. A presentation audit
also compares plotted numeric primitives before and after adding borders,
changing text style and moving legends.

These checks are limited: unchanged files do not prove that the underlying
study design, measurements, licences or data provenance are correct.

The public code-only checkout can run synthetic unit tests. It **cannot**
reproduce empirical figures without the withheld data. The complete local
bundle can be used for private reconstruction by authorized researchers; this
does not make it an open-data release.

The shared computation code calculates joint opportunity and subgroup/planning
results from the documented prepared inputs. It does not include or reproduce
the original data acquisition, model API runs or network preparation. Those
stages and their source-specific permissions remain outside this milestone.

The optional restricted-input hash check is skipped, with an explicit message,
when CITIES_RESTRICTED_PROJECT is not supplied. Do not interpret that skip as
verification of the raw origin–destination records.

The default figure entry point is `python reproduce.py`. The source bundle
layout is in `manifests/source_inventory.json`. The final export uses the same
numeric data and classification rules as the manuscript candidate. A clean
run writes its current-number status to `audit/current_figure_manifest.json`.
Legacy rendering writes to `audit/render_final/export_log.json`; composites
write source hashes and numerical checks alongside their exports.

## Recorded local verification: V44, 27 August 2026

- A fresh Python 3.12 virtual environment installed `requirements.txt`.
- A separate checkout containing the code and the privately held source bundle
  regenerated all 14 data figures. All 174 input-file hashes remained unchanged.
- All 15 PNG exports (including one additional-resolution export) were
  byte-identical to the corresponding manuscript-candidate exports.
- Seven synthetic unit tests passed. The local released-table checks replayed
  51 same-year subgroup contrasts and found 11 BH-adjusted significant contrasts.
- The restricted origin–destination input hash check was **skipped** because the
  restricted project was not supplied to this clean-checkout check.

These are recorded local results, not a claim that a public code-only checkout
can reproduce the empirical results. Automated public CI runs only code tests.

## Composite verification: V45, 27 August 2026

The preceding 16-figure layout has been consolidated into 14 main figures,
including the two unchanged method illustrations. The two composites were
independently regenerated in the fresh Python 3.12 environment using the private
bundle. Both PNGs were byte-identical and all 174 source hashes were unchanged.
The joint composite additionally checked 30 weighted box summaries (five
statistics each) against the native estimator and checked 18 point estimates
against their interval-source estimates. No extra observations, significance
tests or uncertainty bands were synthesized for the layouts. Seven synthetic
unit tests passed with the updated current-number mapping.
