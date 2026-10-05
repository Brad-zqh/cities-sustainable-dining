"""Export selected V203 SI numerical results from the public display bundle.

Filenames identify estimands, not mutable supplementary table numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from reproduce_display import DATA, sha256, validate_bundle


def build_tables() -> dict[str, pd.DataFrame]:
    """Read frozen result statistics; do not rerun empirical models."""
    tables = {}
    tables['component_weights'] = pd.read_csv(DATA/'sensitivity/weight_schemes.csv')
    tables['nutrition_omission_zero_opportunity'] = pd.read_csv(DATA/'nutrition_sensitivity/point_estimates.csv')
    tables['nutrition_omission_thresholds'] = pd.read_csv(DATA/'nutrition_sensitivity/threshold_equating.csv')
    tables['lsbg_dcca_agreement'] = pd.read_csv(DATA/'sensitivity/maup_agreement.csv')
    tables['network_opportunity_15min'] = pd.read_csv(DATA/'network/network_price_inequality_point_estimates.csv').query('threshold_min == 15')
    tables['sdi_structural_inequality_intervals'] = pd.read_csv(DATA/'structure/sdi_inequality_block_intervals.csv')
    tables['alternative_sdi_income_concentration'] = pd.read_csv(DATA/'variant_inequality/sdi_variant_income_inequality_intervals.csv').query('metric == "concentration_index"')
    tables['income_coefficients_2024'] = pd.read_csv(DATA/'regression_comparison/income_models_six_and_five.csv').query('year == 2024 and specification != "Five-domain same-six-sample"')
    tables['planning_outcomes'] = pd.read_csv(DATA/'planning/scenario_results.csv')
    tables['joint_opportunity_estimates'] = pd.read_csv(DATA/'joint/joint_access_point_estimates.csv')
    tables['joint_opportunity_intervals'] = pd.read_csv(DATA/'joint/joint_access_dcca_block_intervals.csv')
    points = pd.read_csv(DATA/'subgroups/joint_subgroup_point_estimates.csv').query('year == 2024')
    intervals = pd.read_csv(DATA/'subgroups/joint_subgroup_dcca_block_intervals.csv').query('year == 2024 and metric == "zero_joint_access_share"')
    contrasts = pd.read_csv(DATA/'contrasts/within_year_contrasts.csv').query('year == 2024')
    keys = ['year','domain','group']
    groups = points.merge(intervals,on=keys,validate='one_to_one')
    q = contrasts.rename(columns={'comparison_group':'group'})[keys+['q_bh_within_domain_year']]
    groups = groups.merge(q,on=keys,how='left',validate='one_to_one')
    groups['is_reference'] = groups.q_bh_within_domain_year.isna()
    tables['census_group_joint_opportunity_2024'] = groups
    component = pd.read_csv(DATA/'component_social/component_domain_reference_contrasts_2024.csv')
    matrix = component.pivot(index=['domain','focal_group','reference_group'],columns='component',values='difference').reset_index()
    zero = contrasts.rename(columns={'comparison_group':'focal_group'})
    # SI S15 uses Female-minus-Male; the zero-opportunity forest archive uses
    # Male-minus-Female. Reverse both the point and interval endpoints; q/p stay.
    reverse = zero.copy()
    reverse['focal_group'], reverse['reference_group'] = zero.reference_group, zero.focal_group
    reverse['difference_percentage_points'] = -zero.difference_percentage_points
    reverse['lower_95_percentage_points'] = -zero.upper_95_percentage_points
    reverse['upper_95_percentage_points'] = -zero.lower_95_percentage_points
    oriented = pd.concat([zero,reverse],ignore_index=True)
    result = matrix.merge(oriented,on=['domain','focal_group','reference_group'],validate='one_to_one')
    if len(result) != 6:
        raise ValueError('All six focal-minus-reference domains must be retained')
    tables['six_census_domain_contrasts_2024'] = result
    return tables


def main() -> None:
    """Write fresh CSV tables and a path-neutral provenance receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(DATA) or DATA.is_relative_to(out):
        parser.error('Use a fresh directory outside the input bundle')
    before = validate_bundle()
    tables = build_tables()
    out.mkdir(parents=True)
    outputs = {}
    for label, table in tables.items():
        path = out/f'{label}.csv'
        table.to_csv(path,index=False,encoding='utf-8')
        outputs[path.name] = dict(rows=len(table),sha256=sha256(path))
    if before != validate_bundle():
        raise AssertionError('Input bundle changed')
    receipt = dict(version='V203 numeric snapshot, 2026-10-05',
                   scope='Selected numerical SI results; later SI numbering/layout may change',
                   empirical_model_rerun=False,input_sha256=before,outputs=outputs)
    (out/'table_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(f'Exported {len(tables)} numerical result tables')


if __name__ == '__main__':
    main()
