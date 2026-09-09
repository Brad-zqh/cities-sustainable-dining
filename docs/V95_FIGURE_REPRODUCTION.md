# V95 figure reproduction map

This document freezes the public entry-point mapping for the V95 Cities
manuscript. It is deliberately separate from the legacy fourteen-figure map in
`manifests/figure_entry_points.json`: the legacy map remains available for
historical reproduction, while the V95 map follows the current manuscript's
numbering and captions.

## Scope and data boundary

The repository is code-only. The aggregate and spatial source bundles listed in
`manifests/source_inventory.json` are not redistributed here. A local rerun is
allowed only after the user has supplied the cleared bundles in the exact
relative paths recorded by that inventory. Restricted review text, menu images,
record-level platform data, credentials and reviewer correspondence are never
inputs to this public runner.

The V95 scripts reproduce the estimands and presentation logic; they do not
turn potential opportunity into observed household affordability, realised
dining, need, welfare or a causal planning effect. Figure 16 uses the fixed
2021 census composition for the 2024 subgroup comparison. Figure 18 is a
conditional stress test over the audited candidate set and site budgets, not a
forecast.

## Command

After installing the pinned environment and supplying authorized source data:

```powershell
python reproduce.py --edition v95
python reproduce.py --edition v95 --figure 15
```

The runner writes a machine-readable receipt to
`audit/v95_figure_manifest.json`. Each V95 figure is rendered into its own
subdirectory under `figures/v95/` so that a map/diagnostic pair cannot silently
overwrite another figure's audit.

## V95 entry points

| Manuscript figure | Scientific role | Entry point |
| --- | --- | --- |
| 11 | SDI change and coverage | `render_evidence_composites.py A` |
| 12 | Six-component spatial atlas | `fig05_component_atlas.py` |
| 13 | Structural inequality and component contrasts | `render_quality_inequality_composite.py` |
| 14 | Price composition and adjusted associations | `fig06_price_market.py` |
| 15 | Joint quality–walking–price opportunity | `render_evidence_composites.py B` |
| 16 | Zero-joint-opportunity subgroup contrasts | `fig09_social_within_year.py` |
| 17a | LSBG/DCCA scale-sensitivity maps | `render_scale_weight_composite_single_page.py --split-part maps` |
| 17b | Weighting and cross-scale diagnostics | `render_scale_weight_composite_single_page.py --split-part diagnostics` |
| 18 | Conditional siting stress test | `render_planning_composite.py` |

The source-native composites are presentation consolidations only. Their audit
files retain source hashes and the panel map; they do not introduce new
estimators or unreported observations.
