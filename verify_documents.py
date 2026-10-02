"""Compare the final Word files with recalculated results, not just saved tables.

Every input is read-only. Recalculated and source-only comparisons are labelled
separately so that matching an old CSV is never counted as re-estimation.
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib, io, json, re
import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree as E

import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--figures-root', type=Path)
    args = parser.parse_args()
    BASE = RECALC = args.results_root.resolve()
    PROJECT = args.data_root.resolve()
    N = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    ROUTES = {
        'v135_fixed_nutrition_aggregation_variants/weight_schemes.csv': RECALC / 'aggregation/weight_schemes.csv',
        'v135_fixed_nutrition_subgroup_contrasts/within_year_contrasts.csv': RECALC / 'planning/within_year_contrasts.csv',
        'v135_fixed_nutrition_price_market/quality_market_decomposition.csv': RECALC / 'market/quality_market_decomposition.csv',
        'v135_fixed_nutrition_component_social/component_domain_reference_contrasts_2024.csv': RECALC / 'components/component_domain_reference_contrasts_2024.csv',
        'v135_fixed_nutrition_aggregation_variants/maup_agreement.csv': RECALC / 'aggregation/maup_agreement.csv',
    }
    if (RECALC/'variants/sdi_variant_income_inequality_intervals.csv').exists():
        ROUTES['v135_fixed_nutrition_variant_inequality_999/sdi_variant_income_inequality_intervals.csv']=RECALC/'variants/sdi_variant_income_inequality_intervals.csv'
    for folder, newfolder, files in [
        ('v135_fixed_nutrition_structural_inequality_999', 'structural', ['sdi_inequality_summary.csv', 'sdi_inequality_block_intervals.csv']),
        ('v134_fixed_nutrition_joint_downstream', 'planning', ['scenario_results.csv', 'joint_subgroup_point_estimates.csv', 'joint_subgroup_dcca_block_intervals.csv']),
        ('v134_fixed_nutrition_joint_access', 'joint', ['joint_access_point_estimates.csv', 'joint_access_dcca_block_intervals.csv']),
    ]:
        for file in files:
            ROUTES[folder + '/' + file] = RECALC / newfolder / file

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def root_of(path):
        with ZipFile(path) as z:
            return E.fromstring(z.read('word/document.xml'))

    def text_of(node):
        return ''.join(node.xpath('.//w:t/text()', namespaces=N))

    supplement, main = args.supplement.resolve(), args.manuscript.resolve()
    report = {'documents': {str(p): digest(p) for p in [main, supplement]},
              'scope': 'Word values vs newly recalculated results, with source-only coverage explicitly separated',
              'table_checks': {}, 'main_claims': [], 'visual_checks': [], 'issues': []}

    from scripts.audit_v180_supplement_tables import audit_tables
    variant_summary = pd.read_csv(RECALC / 'aggregation/variant_population_summary.csv')
    area_report = {'area_sdi': [
        {'year': int(row.year), 'weighted_mean': float(row.population_weighted_mean)}
        for row in variant_summary.loc[
            variant_summary.scale.eq('LSBG') & variant_summary.variant.eq('Equal six (strict)')
        ].itertuples()
    ]}
    def load_table(rel):
        key = rel.removeprefix('outputs/restricted/')
        if key == 'v134_fixed_nutrition_full_chain/population_weighted_summary.csv':
            p = RECALC / 'aggregation/variant_population_summary.csv'
            frame = pd.DataFrame([
                {'year': r['year'], 'scale': 'LSBG', 'population_weighted_sdi_equal_arithmetic': r['weighted_mean']}
                for r in area_report['area_sdi']
            ])
            return frame, {'path': str(p), 'comparison_kind': 'recalculated', 'sha256': digest(p)}
        p = ROUTES.get(key, PROJECT / rel)
        return pd.read_csv(p), {'path': str(p), 'comparison_kind': 'recalculated' if key in ROUTES else 'source-only, not rerun', 'sha256': digest(p)}
    word_root = root_of(supplement)
    for path in ROUTES.values():
        if not path.is_file():
            raise FileNotFoundError(f'Recalculated table required: {path}')
    tables = [[[text_of(cell) for cell in row.findall('w:tc', N)]
               for row in table.findall('w:tr', N)]
              for table in word_root.xpath('//w:body/w:tbl', namespaces=N)]
    receipt = audit_tables(tables, load_table)
    for cell in receipt['checks']:
        label=cell['cell']
        cell['comparison_kind']='recalculated' if label.startswith(('S1 SDI ','S3 ','S4 ','S9 ','S11 ','S12 ','S14 ','S15 ','table 5,','table 8,')) else 'source-only, not rerun'
        if label.startswith('table 9,') and 'v135_fixed_nutrition_variant_inequality_999/sdi_variant_income_inequality_intervals.csv' in ROUTES:
            cell['comparison_kind']='recalculated'
    report['si_value_coverage']={kind:sum(c['comparison_kind']==kind for c in receipt['checks']) for kind in ['recalculated','source-only, not rerun']}
    (BASE/'word_table_comparison.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    report['table_checks'] = {k: receipt[k] for k in ['status','checked_numeric_values','mismatches','sources']}
    report['table_mapping_script_sha256'] = digest(Path(__file__).parent / 'scripts/audit_v180_supplement_tables.py')

    paragraphs = [text_of(p) for p in root_of(main).xpath('//w:body/w:p', namespaces=N)]

    def claim(label, selector, values, kind='recalculated', note=''):
        matches = [(i,t) for i,t in enumerate(paragraphs) if selector in t]
        assert len(matches)==1, (label, selector, len(matches))
        index, text = matches[0]
        checks=[]
        for printed, expected in values:
            present=bool(re.search(r'(?<![\d.])'+re.escape(printed)+r'(?!\d|\.\d)',text))
            value=float(printed.replace(',',''))
            decimals=len(printed.split('.')[1]) if '.' in printed else 0
            matched=abs(value-float(expected)) <= .50001*10**(-decimals)
            checks.append({'displayed': printed, 'recalculated_or_source_value': float(expected),
                           'present_in_paragraph': present, 'rounding_matches': matched})
        report['main_claims'].append({'label':label,'paragraph_index':index,'paragraph':text,
                                      'comparison_kind':kind,'checks':checks,'note':note,
                                      'pass':all(c['present_in_paragraph'] and c['rounding_matches'] for c in checks)})

    area = area_report
    means = {r['year']: r['weighted_mean'] for r in area['area_sdi']}
    market_summary=pd.read_csv(RECALC/'market/price_market_summary.csv').set_index('year')
    claim('4.1: active restaurant counts','To trace how sustainable dining changed',
          [('13,055',market_summary.loc[2011,'active_restaurant_n']),('15,214',market_summary.loc[2016,'active_restaurant_n']),
           ('25,647',market_summary.loc[2021,'active_restaurant_n']),('24,888',market_summary.loc[2024,'active_restaurant_n'])])
    claim('4.1: four population-weighted SDI means','In areas with complete evidence',
          [(f'{means[y]:.3f}',means[y]) for y in [2011,2016,2021,2024]])
    temporal=pd.read_csv(RECALC/'temporal/change_from_2011_intervals.csv')
    change=temporal.loc[temporal.year.eq(2024)&temporal.metric.eq('equal_six_strict')].iloc[0]
    claim('4.1: SDI change and its 999-block interval','In areas with complete evidence',
          [('0.049',change.difference),('0.037',change.ci_low),('0.060',change.ci_high)])
    coverage=pd.read_csv(RECALC/'temporal/metric_coverage_audit.csv')
    cov=coverage.loc[coverage.year.eq(2011)&coverage.metric.eq('equal_six_strict')].iloc[0]
    claim('4.1: complete-evidence population coverage','To trace how sustainable dining changed',[('51.4',100*cov.population_coverage_share)])
    structural = pd.read_csv(RECALC/'structural/sdi_inequality_summary.csv').set_index('year')
    claim('4.2: income concentration and citywide Gini','We next asked whether improvements',
          [('0.0254',structural.loc[2011,'concentration_index']),('0.0286',structural.loc[2024,'concentration_index']),
           ('0.123',structural.loc[2011,'weighted_gini']),('0.102',structural.loc[2021,'weighted_gini']),('0.105',structural.loc[2024,'weighted_gini'])])
    component=pd.read_csv(RECALC/'components/component_domain_reference_contrasts_2024.csv')
    def gap(label, dimension):
        f=component.loc[component.contrast_label.eq(label)&component.component.eq(dimension)]
        assert len(f)==1,(label,dimension)
        return float(f.iloc[0].difference)
    claim('4.2: environment and hygiene social-group gaps','The component comparisons make these social differences',
          [('0.06',-gap(label,dim)) for label in ['<HK$10k − ≥HK$40k','Elementary − Managers'] for dim in ['Environmental sustainability','Hygiene']]+
          [('0.07',-gap('Primary − Post-secondary','Environmental sustainability')),('0.08',-gap('Primary − Post-secondary','Hygiene'))]+
          [('0.11',gap('Filipino − Chinese',dim)) for dim in ['Environmental sustainability','Hygiene']])
    market=pd.read_csv(RECALC/'market/quality_market_decomposition.csv')
    def beta(model,predictor):
        return float(market.loc[market.year.eq(2024)&market.model.eq(model)&market.predictor.eq(predictor),'standardized_beta'].iloc[0])
    ratio=beta('Income + market composition','log_income')/beta('Income only','log_income')
    report['main_claims'].append({'label':'4.2: adjusted income association about half the unadjusted value',
        'paragraph':next(t for t in paragraphs if 'We then adjusted the income comparison' in t),
        'comparison_kind':'recalculated','checks':[], 'recalculated_ratio':ratio,
        'pass':.45<ratio<.55 and beta('Income + market composition','log_supply')>0 and beta('Income + market composition','low_price_share')<0})
    scenario = pd.read_csv(RECALC/'planning/scenario_results.csv').set_index(['scenario','budget_sites'])
    equity=scenario.loc[('zero_affordability_gap',10)]
    population=scenario.loc[('population_reach',10)]
    joint=scenario.loc[('equity_joint',10)]
    claim('4.5: ten-restaurant simulation, shared denominator','We then compared four ways to locate ten',
          [('289,343',equity.zero_affordability_priority_population_reached),('582,181',equity.zero_affordability_priority_population_denominator),
           ('49.70',100*equity.zero_affordability_priority_reached_share),('25,065',population.zero_affordability_priority_population_reached),
           ('4.31',100*population.zero_affordability_priority_reached_share),('1,513,257',population.population_within_selected_catchments),
           ('649,316',equity.population_within_selected_catchments),('42.05',100*joint.joint_priority_reached_share),
           ('33.01',100*joint.zero_affordability_priority_reached_share)],
          note='49.70% and 4.31% use the same baseline-zero low-income denominator; 42.05% uses a different joint-disadvantage denominator.')
    claim('4.5: twenty-restaurant simulation','When the simulated programme expanded',
          [('55.60',100*scenario.loc[('zero_affordability_gap',20),'zero_affordability_priority_reached_share']),
           ('61.76',100*scenario.loc[('equity_joint',20),'joint_priority_reached_share'])])
    point=pd.read_csv(RECALC/'joint/joint_access_point_estimates.csv')
    def access(year, outcome):
        f=point.loc[point.year.eq(year)&point.outcome.eq(outcome)&point.threshold_min.eq(15)&point.price_ceiling_hkd.eq(100)&point.destination_area_sdi_threshold.eq(.45)]
        assert len(f)==1,(year,outcome,len(f))
        return f.iloc[0]
    claim('4.3: qualifying-option mean and retention','To see whether higher SDI scores translated',
          [(str(round(100*access(y,'joint_access').population_weighted_mean/access(y,'low_price_access').population_weighted_mean)),
            100*access(y,'joint_access').population_weighted_mean/access(y,'low_price_access').population_weighted_mean) for y in [2016,2021,2024]]+
          [('22.5',access(2016,'joint_access').population_weighted_mean),('106.8',access(2024,'joint_access').population_weighted_mean),
           ('109.7',access(2016,'low_price_access').population_weighted_mean),('186.9',access(2024,'low_price_access').population_weighted_mean)],
          note='Retention is the ratio of population-weighted means, not an unweighted share of distinct restaurants.')
    claim('4.3: baseline-zero population shares','We compared the share of residents with no nearby option',
          [('10.03',100*access(2024,'low_price_access').zero_access_population_share),
           ('19.64',100*access(2024,'joint_access').zero_access_population_share),
           ('35.98',100*access(2016,'joint_access').zero_access_population_share)])
    groups=pd.read_csv(RECALC/'planning/joint_subgroup_point_estimates.csv')
    def ethnic(group):
        f=groups.loc[groups.year.eq(2024)&groups.domain.eq('Ethnicity')&groups.group.eq(group)]
        assert len(f)==1,(group,groups.domain.unique(),groups.group.unique())
        return f.iloc[0]
    claim('4.3: ethnic-group zero-opportunity shares','Adding the SDI threshold changed which income groups',
          [('19.38',100*ethnic('Chinese').zero_joint_access_share),('32.92',100*ethnic('White').zero_joint_access_share)])
    contrasts=pd.read_csv(RECALC/'planning/within_year_contrasts.csv')
    row=contrasts.loc[contrasts.year.eq(2024)&contrasts.reference_group.eq('Chinese')&contrasts.comparison_group.eq('Filipino')].iloc[0]
    claim('4.3: Filipino–Chinese paired zero-opportunity gap','Supplementary Table S15 brings two sides',
          [('4.78',row.difference_percentage_points)])
    maup=pd.read_csv(RECALC/'aggregation/maup_agreement.csv')
    strict_maup=maup.loc[maup.variant.eq('Equal six (strict)')]
    claim('4.4: LSBG–DCCA rank agreement range','To examine whether the citywide pattern depended',
          [('0.71',strict_maup.spearman.min()),('0.77',strict_maup.spearman.max())])
    for i,text in enumerate(paragraphs):
        if '表6呈现' in text:
            report['issues'].append({'kind':'stale Chinese cross-reference','paragraph_index':i,'text':text,
                                     'correct_target':'Supplementary Table S15 / 补充表S15'})
        if 'Among restaurants priced at HK$100 or less' in text:
            report['issues'].append({'kind':'retention-denominator wording','paragraph_index':i,'text':text,
                                     'required_clarification':'20/46/57% are ratios of population-weighted option counts.'})

    # Figure identities follow the V180 document contract; captions are checked separately.
    if args.figures_root:
        figure_specs = json.loads((Path(__file__).parent / 'manifests/v180_figure_entry_points.json').read_text(encoding='utf-8'))['figures']
        with ZipFile(main) as z:
            for figure, spec in figure_specs.items():
                path = args.figures_root / figure / (spec['output_stem'] + '.png')
                original = np.asarray(Image.open(io.BytesIO(z.read(f'word/media/image{int(figure)}.png'))).convert('RGB'))
                regenerated = np.asarray(Image.open(path).convert('RGB'))
                equal_shape = original.shape == regenerated.shape
                diff = np.abs(original.astype(np.int16) - regenerated.astype(np.int16)) if equal_shape else None
                report['visual_checks'].append({
                    'main_figure': figure, 'renderer_output': str(path),
                    'same_dimensions': equal_shape, 'rgb_pixels_identical': bool(equal_shape and not np.any(diff)),
                    'same_rgb_channel_fraction': float(np.mean(diff == 0)) if equal_shape else None,
                    'mean_rgb_channel_absolute_difference': float(np.mean(diff)) if equal_shape else None,
                    'maximum_rgb_channel_difference': int(diff.max()) if equal_shape else None,
                    'scope': 'Rendering from prepared inputs; recomputation is recorded in verification.json'
                })
                del original, regenerated, diff
    report['main_numeric_checks']=sum(len(c['checks']) for c in report['main_claims'])
    report['numeric_status']='PASS' if receipt['status']=='PASS' and all(c['pass'] for c in report['main_claims']) else 'MISMATCH'
    report['complete_end_to_end_public_reproduction']=False
    report['unrerun_scope']=['Raw platform acquisition and LLM assessment','Historical pedestrian-network construction',
                           'Source-only SI tables (see per-source labels)','Main methodological diagrams and SI model-comparison figures']
    (BASE/'document_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'numeric_status':report['numeric_status'],'si_values':receipt['checked_numeric_values'],
                      'main_values':report['main_numeric_checks'],'identical_rgb_figures':sum(c['rgb_pixels_identical'] for c in report['visual_checks']),
                      'wording_or_reference_issues':len(report['issues'])},ensure_ascii=False,indent=2))


    return int(report['numeric_status'] != 'PASS')

if __name__ == '__main__':
    raise SystemExit(main())
