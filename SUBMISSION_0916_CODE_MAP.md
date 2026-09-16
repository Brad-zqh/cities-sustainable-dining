# Submission 0916 code map

This note identifies the additional publication-layout scripts retained in the
16 September code snapshot. They read the existing evidence-locked aggregate
inputs and do not create, impute or alter empirical observations.

| Script | Current manuscript figure | Default pipeline status |
| --- | --- | --- |
| `scripts/render_study_benchmark_composite.py` | Optional expanded Fig. 2 analytical benchmark | Optional publication composite |
| `scripts/fig05_component_endpoints_v61.py` | Optional endpoint-only view related to Fig. 13 | Optional endpoint plate |
| `scripts/fig07_joint_quality_access_tnr.py` | Optional standalone view related to Fig. 16 | Optional typography variant |
| `scripts/render_scale_weight_composite.py` | Alternate combined rendering of Fig. 18a–b | Optional publication composite |
| `scripts/render_scale_weight_composite_single_page.py` | Canonical split renderings for Fig. 18a and Fig. 18b | Default analytical pipeline |
| `scripts/render_quality_inequality_composite.py` | Fig. 14 restaurant-quality inequality | Default analytical pipeline |
| `scripts/render_planning_composite.py` | Fig. 19 planning strategies and overlap diagnostics | Default analytical pipeline |
| `figures/v5_cities_visual_system_tnr.py` | Times New Roman configuration for its corresponding optional renderer | Shared style module |

The canonical batch entry point remains `reproduce.py`, whose registry maps
the current manuscript's analytical Figs. 12–19 to their exact scripts and
output stems. It fails when the
authorized `source_data/` bundle is absent and never substitutes synthetic
values. The public GitHub repository intentionally excludes that empirical
bundle pending source-specific redistribution clearance.
