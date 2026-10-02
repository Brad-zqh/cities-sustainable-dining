# V148 visualization code lineage

The current manuscript result figures retain the established V134 figure designs. The
V135 fixed-nutrition analytical outputs replace the earlier figure inputs; V138,
V140, V147 and V148 scripts make presentation refinements without recomputing
the estimands. The V148 wrapper is the latest code for the four panels it names.

| Layer | Code | Purpose |
| --- | --- | --- |
| Historical renderers | `scripts/fig05_component_atlas.py`, `scripts/fig06_price_market.py`, `scripts/fig09_social_within_year.py`, `scripts/render_*_composite.py` | Original result layouts and source-data plotting logic |
| Fixed-result bridge | `scripts/v138_reproduce_visual_polish.py`, `scripts/v138_visual_polish/` | Route revised V135 inputs into those layouts |
| Reader refinements | `scripts/v140_reader_figure_touchups.py`, `scripts/v147_reader_figure_refinements.py` | Typography, layout and colour-key changes |
| Latest panel refinements | `scripts/v148_panel_precision.py` | Figs. 14, 15, 18b and 19 only |

Run from the repository root with Python 3.12 after separately authorized V135
aggregate and geometry inputs have been placed under the `outputs/restricted/`
paths checked by `assert_inputs()`. For example:

```sh
python scripts/v148_panel_precision.py --only 14
python scripts/v148_panel_precision.py --only 15
python scripts/v148_panel_precision.py --only 18b
python scripts/v148_panel_precision.py --only 19
```

Other V138/V140/V147 entry points remain available for their respective figure
sets. The default historical-renderer path is this repository's `scripts/`;
`CITIES_V134_VISUAL_SCRIPTS` can select another authorized script directory.
The bridge checks V135 point estimates and hashes all required empirical inputs
before rendering. Missing inputs stop execution. This code-only branch does not
contain the restricted inputs, finished figure images, source data, or the full
multimodal extraction chain. Consequently a clean clone can inspect and test
the code but cannot regenerate the empirical figures.

The planning-map display in V148 uses the observed 0%, 34.3%, and 100%
classes. The implementation asserts the underlying distribution before applying
that display. It does not change the source values or the planning comparison.
