# Reproduction scope and verification

## Current V180 entry points

Follow [the V180 reproduction guide](docs/V180_REPRODUCTION.md) for the input
layout and complete commands. The public repository contains code. Authorized
coauthors receive the empirical input bundle separately.

- `verify_current.py` recalculates eight groups of downstream analyses and
  compares 29 result tables with frozen references. Input hashes are checked
  before and after calculation; source hashes and task logs are retained.
- `reproduce.py --edition v180` renders current main-text Figs. 12–19 as PNG,
  PDF and SVG. The current Word Fig. 18 uses the diagnostics export (18b);
  the same lineage also exports an additional scale-map page (18a).
- `verify_documents.py` compares 51 main-text numeric anchors and 698 SI-table
  values with the named outputs. It separates 518 values supported by this
  recalculation from 180 values checked against source tables. Optional image
  comparison checks the embedded Word images against the renderer exports.

Use fresh output folders outside the inputs. Numerical receipts are written
to `verification.json`, Word comparisons to `document_comparison.json` and
`word_table_comparison.json`, and renderer commands/output hashes to
`figure_reproduction.json` in their respective output folders.

The replay starts from prepared multimodal area components and audited walking
pairs. Original platform acquisition, fresh model inference, pedestrian-network
construction, recipe-reference inference and independent construct validation
have separate evidence requirements. Agreement in this downstream replay
supports the calculations and document correspondence within that scope.

The sections below retain historical verification records. Their figure
numbers, commands, test counts and receipts refer to those earlier releases.

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

## Current-manuscript verification: Submission 0916, 17 September 2026

The entry-point registry was aligned to the current manuscript rather than the
preceding consolidated layout. `python reproduce.py` regenerated analytical
Figures 12–19 (with separate Fig. 18a and Fig. 18b exports) from 174 authorized
prepared input files. Every figure command returned 0, every required output
was rewritten during that run, and all input SHA-256 values were unchanged.
Eight contract tests passed. The joint-opportunity and downstream checks also
passed, replaying 999-block intervals, 51 paired contrasts and BH adjustment,
237 planning candidates, 1,744 demand LSBGs and 12 non-baseline scenarios. The
optional hash check for the restricted origin–destination pair file remained
explicitly skipped because `CITIES_RESTRICTED_PROJECT` was not supplied.

See `MANUSCRIPT_CODE_CORRESPONDENCE.md` for the caption-level crosswalk and the
machine-readable local record at `audit/current_figure_manifest.json` for exact
commands and output hashes.

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
