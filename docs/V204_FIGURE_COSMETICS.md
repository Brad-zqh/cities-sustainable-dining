# Author-requested figure refinements, 5 October 2026

This update publishes code only. It does not publish the rendered empirical
figures, platform-derived tables, spatial inputs, manuscripts or reviewer
correspondence. The existing data-sharing boundary remains unchanged.

The author requested three display changes:

- Fig. 15: add 6 mm between the b/c row and the d row. Keep the physical panel
  sizes, text sizes, plotted values and colours unchanged.
- Fig. 17: enlarge the six subgroup legends from 4.8 to 6.25 pt (titles 6.6 pt).
  Find an empty lane for the Ethnicity legend rather than covering a comparison
  bracket. The bars, confidence intervals, reference labels and stars are unchanged.
- Fig. 3: reduce the bottom diagnostic row to three aligned 45.75-mm square
  plotting rectangles. Retain clearance from the map row above.

`scripts/v203_fig15_six_primary.py` retains the delivered Fig. 15 typography
and reads the six-domain primary coefficients from an explicitly supplied CSV.
It does not fit a model. Do not substitute the historical five-domain
sensitivity coefficients: the CSV schema cannot certify how an outcome was
constructed. Authorized users must supply the corresponding six-domain table.

`scripts/v204_cosmetic_figure_refinements.py` applies the three cosmetic changes
through the existing native Matplotlib sources. It records before/after
fingerprints of numerical artists and colours and checks that they agree.
Exports include 600-dpi PNG and vector PDF/SVG. The row crops are convenience
exports; the complete figures retain the other panels. No document is edited.

## Run with separately authorized inputs

Use the fixed input-tree layout documented in `V180_REPRODUCTION.md`. Keep the
input tree outside the clone. Each command requires a fresh output directory.
Fig. 3 additionally uses the external historical visualization tree containing
`source_data/fig03_study_area_annual_distribution_v1`, `figS_openrice_fehd_v4`
and `fig_v4_four_year`. Supply it through `--visual-data-root` when it differs
from the fixed result tree.

```sh
python scripts/v204_cosmetic_figure_refinements.py --figure 03 \
  --data-root /authorized/input-tree --output /local/new-fig03 \
  --visual-data-root /authorized/historical-visualization-tree
python scripts/v204_cosmetic_figure_refinements.py --figure 15 \
  --data-root /authorized/input-tree --output /local/new-fig15 \
  --six-domain-regression /authorized/six-domain/quality_market_decomposition.csv
python scripts/v204_cosmetic_figure_refinements.py --figure 17 \
  --data-root /authorized/input-tree --output /local/new-fig17
```

The commands are examples with placeholder paths, not available datasets.
Missing files or an existing output directory cause an error; synthetic data
are never substituted. Existing V180 entry points remain unchanged.
