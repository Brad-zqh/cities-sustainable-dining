# Manuscript–code correspondence audit (Submission 0916)

This audit uses the figure numbering and English captions extracted from the
current manuscript, not from an earlier layout. The analytical batch contains
Figures 12–19. Figures 1–11 are conceptual or method illustrations and are not
represented as numerical reproductions.

| Manuscript figure | Executed analytical route | Principal prepared inputs | Canonical output |
| --- | --- | --- | --- |
| 12 | `render_evidence_composites.py A` | `fig_v4_four_year`; `figS_temporal_uncertainty_v4` | `Fig12_Restaurant_Quality_SDI_Evolution_Coverage` |
| 13 | `render_final.py 6` → `fig05_component_atlas.py` | `fig_v4_four_year` | `Fig13_Six_Component_FourYear_Atlas` |
| 14 | `render_quality_inequality_composite.py` | `figS_sdi_structural_inequality_v4`; `fig13_component_social_decomposition_v1` | `Fig14_Restaurant_Quality_Inequality` |
| 15 | `render_final.py 7` → `fig06_price_market.py` | `figS_price_market_v4` | `Fig15_Price_Composition_Socioeconomic_Associations` |
| 16 | `render_evidence_composites.py B` | `fig13_joint_quality_affordable_access_v1`; `fig04_network_price_v4` | `Fig16_Joint_Quality_Walking_Price_Opportunity` |
| 17 | `render_final.py 10` → `fig09_social_within_year.py` | `fig14_joint_subgroup_planning_v1` | `Fig17_SameYear_Socioeconomic_Zero_Joint_Opportunity` |
| 18a–b | `render_scale_weight_composite_single_page.py` with `maps` and `diagnostics` | `fig_v4_four_year`; `figS_weight_sensitivity_v4`; audited land mask | `Fig18a_Spatial_Scale_Sensitivity_Maps`; `Fig18b_Weighting_And_Cross_Scale_Diagnostics` |
| 19 | `render_planning_composite.py` | `fig04_equity_siting_v7`; `fig14_joint_subgroup_planning_v1` | `Fig19_Planning_Strategies` |

## Verification completed on 17 September 2026

- A clean full run of `python reproduce.py` completed all eight manuscript
  entries (nine exported figure parts) with return code 0.
- All 174 prepared input files had identical SHA-256 values before and after
  rendering. Exact commands and output SHA-256 values are recorded locally in
  `audit/current_figure_manifest.json`.
- Eight code-contract tests passed, including the current Fig. 12–19 registry,
  command-path checks, output-name checks, weighted-Gini invariance, population-
  weighted zero-access calculation and value-preserving heatmap finishing.
- Joint-opportunity QC passed for 2016, 2021 and 2024; the minimum valid
  bootstrap count was 999, price linkage was at least 0.998 and destination-SDI
  linkage was at least 0.968.
- Downstream QC replayed 69 subgroup-year estimates, 51 paired contrasts and
  their BH adjustments, all 999 paired replicates per subgroup, 237 candidate
  sites, 1,744 demand LSBGs and 12 non-baseline planning scenarios.

## Boundary of the claim

The public repository is code-only. Its continuous integration tests the code
contract but cannot reproduce empirical figures without the separately
authorized `source_data` bundle. The restricted origin–destination pair file
was not supplied through `CITIES_RESTRICTED_PROJECT`, so its optional raw-file
hash check was explicitly skipped; released aggregate tables and all downstream
calculations were checked. This is a disclosed verification boundary, not a
passed raw-data check.
