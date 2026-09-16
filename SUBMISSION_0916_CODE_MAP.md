# Submission 0916 code map

This note identifies the additional publication-layout scripts retained in the
16 September code snapshot. They read the existing evidence-locked aggregate
inputs and do not create, impute or alter empirical observations.

| Script | Purpose | Default pipeline status |
| --- | --- | --- |
| `scripts/render_study_benchmark_composite.py` | Integrates the four annual restaurant maps, the OpenRice-FEHD benchmark and agreement diagnostics | Optional publication composite |
| `scripts/fig05_component_endpoints_v61.py` | Presents 2011 and 2024 component endpoints while retaining class breaks pooled across all four study years | Optional endpoint plate |
| `scripts/fig07_joint_quality_access_tnr.py` | Renders jointly qualified walking opportunity with the Times New Roman visual system | Optional typography variant |
| `scripts/render_scale_weight_composite.py` | Combines spatial-scale and weighting sensitivity evidence | Optional publication composite |
| `scripts/render_scale_weight_composite_single_page.py` | Places the scale and weighting evidence on one manuscript page | Optional single-page layout |
| `scripts/render_quality_inequality_composite.py` | Integrates structural inequality and component-level socioeconomic evidence | Optional publication composite |
| `scripts/render_planning_composite.py` | Presents the four planning objectives, population reach and overlap diagnostics | Optional publication composite |
| `figures/v5_cities_visual_system_tnr.py` | Provides the Times New Roman plotting configuration used by the corresponding renderer | Shared style module |

The canonical batch entry point remains `reproduce.py`. It fails when the
authorized `source_data/` bundle is absent and never substitutes synthetic
values. The public GitHub repository intentionally excludes that empirical
bundle pending source-specific redistribution clearance.
