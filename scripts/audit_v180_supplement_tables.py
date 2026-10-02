"""V180 supplementary-table cell mappings; comparison only, no data editing."""
import re
import pandas as pd

def audit_tables(tables, loader):
    """Compare every numeric cell using a caller-supplied, provenance-aware loader."""
    receipt = {'checks': [], 'mismatches': [], 'sources': {}}
    def load(rel):
        frame, metadata = loader(rel)
        receipt['sources'][rel] = metadata
        return frame
    def one(frame,**filters):
        x=frame
        for k,v in filters.items():x=x.loc[x[k].eq(v)]
        assert len(x)==1,(filters,len(x))
        return x.iloc[0]
    def check(label,text,expected):
        vals=re.findall(r'(?<![A-Za-z])[-−+]?\d[\d,]*(?:\.\d+)?',text)
        expected=expected if isinstance(expected,(list,tuple)) else [expected]
        assert len(vals)==len(expected),(label,text,expected)
        for s,e in zip(vals,expected):
            v=float(s.replace(',','').replace('−','-'))
            places=len(s.split('.')[1]) if '.' in s else 0
            ok=abs(v-float(e))<=0.50001*10**(-places)
            receipt['checks'].append({'cell':label,'displayed':s,'source_value':float(e),'match':ok})
            if not ok:receipt['mismatches'].append(receipt['checks'][-1])
    def numeric_row(table,index,record,mapping):
        row=tables[table][index]
        for col,field in mapping.items():check(f'table {table+1}, row {index+1}, col {col+1}',row[col],record[field])

    s=load('source_data/supplement_tables_v4/Table_S1_temporal_membership_and_coverage.csv')
    f=load('outputs/restricted/v134_fixed_nutrition_full_chain/population_weighted_summary.csv')
    for j,row in enumerate(tables[0][1:],1):
        y=int(row[0]);r=one(s,year=y)
        numeric_row(0,j,r,{0:'year',1:'population_reference_year',2:'unit_n',3:'restaurant_n',4:'population_weighted_supply_per_1000',5:'population_without_restaurant_share',6:'nutrition_population_coverage',7:'carbon_population_coverage',8:'high_coverage_population_share'})
        a=f.loc[f.year.eq(y)&f.scale.eq('LSBG')]
        assert len(a)==1,a
        check(f'S1 SDI {y}',row[9],a.iloc[0]['population_weighted_sdi_equal_arithmetic'])

    w=load('outputs/restricted/v135_fixed_nutrition_aggregation_variants/weight_schemes.csv')
    for j,row in enumerate(tables[2][1:],1):
        for col,scheme in [(1,'Absolute PC1 loading'),(2,'Entropy'),(3,'Equal six')]:
            label={'Diversity':'Cuisine diversity','Environment':'Sustainability signal','Practice':'Practice tag'}.get(row[0],row[0])
            r=one(w,scheme=scheme,component_label=label);check(f'S3 {row[0]} {scheme}',row[col],r.weight)

    a=load('outputs/restricted/v135_no_nutrition_opportunity_sensitivity_v3_parameterized_replay/point_estimates.csv')
    q=load('outputs/restricted/v135_no_nutrition_opportunity_sensitivity_v3_parameterized_replay/threshold_equating.csv')
    for row in tables[3][1:]:
        y=int(row[0])
        for col,outcome in [(1,'six_joint'),(2,'five_fixed_045'),(3,'five_prevalence_matched')]:
            r=one(a,year=y,outcome=outcome)
            exp=[r.zero_access_population_share*100]
            if col==3:exp.append(one(q,year=y).five_matched_cutoff)
            check(f'S13 {y} {outcome}',row[col],exp)

    a=load('outputs/restricted/v135_fixed_nutrition_aggregation_variants/maup_agreement.csv')
    for j,row in enumerate(tables[4][1:],1):
        r=one(a,year=int(row[0]),variant='Equal six (strict)')
        numeric_row(4,j,r,{0:'year',2:'dcca_n',3:'pearson',4:'spearman',5:'mae',6:'direct_population_weighted_mean',7:'aggregated_population_weighted_mean'})

    for ti,file,filters,mapping in [
     (5,'Table_S5_network_inequality_15min.csv',lambda r:{'year':int(r[0]),'outcome':r[1]},{0:'year',2:'lsbg_n',3:'population_weighted_mean',4:'weighted_gini',5:'theil_t',6:'zero_access_population_share',7:'concentration_index',8:'income_quintile_relative_ratio'}),
     (6,'Table_S6_price_threshold_sensitivity_point_estimates.csv',lambda r:{'year':int(r[0]),'price_ceiling_hkd':int(r[1])},{0:'year',1:'price_ceiling_hkd',2:'population_weighted_mean',3:'weighted_gini',4:'theil_t',5:'zero_access_population_share',6:'concentration_index',7:'income_quintile_relative_ratio'}),
     (10,'Table_S8_OpenRice_FEHD_count_pattern_benchmark.csv',lambda r:{'threshold_min':float(r[0])},{0:'threshold_min',1:'LSBG_n',2:'Pearson',3:'Spearman',4:'population_weighted_low_access_agreement',5:'low_access_Jaccard'})]:
        a=load('source_data/supplement_tables_v4/'+file)
        for j,row in enumerate(tables[ti][1:],1):numeric_row(ti,j,one(a,**filters(row)),mapping)

    a=load('outputs/restricted/v135_fixed_nutrition_structural_inequality_999/sdi_inequality_summary.csv')
    c=load('outputs/restricted/v135_fixed_nutrition_structural_inequality_999/sdi_inequality_block_intervals.csv')
    for j,row in enumerate(tables[7][1:],1):
        y=int(row[0]);r=one(a,year=y);numeric_row(7,j,r,{0:'year',1:'lsbg_n',2:'population_analyzed'})
        for col,metric in [(3,'weighted_gini'),(4,'theil_t'),(5,'concentration_index'),(6,'income_quintile_relative_ratio')]:
            ci=one(c,year=y,metric=metric);check(f'S4 {y} {metric}',row[col],[r[metric],ci.lower_95,ci.upper_95])

    a=load('outputs/restricted/v135_fixed_nutrition_variant_inequality_999/sdi_variant_income_inequality_intervals.csv')
    variant_map={'Equal weights, six complete components':'equal_six_strict','Equal five (excluding Practice)':'equal_five_no_practice','Equal six (≥50% coverage)':'equal_six_highcoverage','Available components (≥4/6)可用 组成维度 (≥4/6)':'equal_six_available4','Geometric six':'geometric_six','Absolute-PC1 weights':'pca_six','Entropy weights':'entropy_six'}
    for j,row in enumerate(tables[8][1:],1):
        r=one(a,year=int(row[0]),variant_field=variant_map[row[1]],metric='concentration_index')
        numeric_row(8,j,r,{0:'year',2:'estimate',3:'lower_95',4:'upper_95',5:'lsbg_n',6:'population_analyzed'})

    a=load('outputs/restricted/v135_fixed_nutrition_price_market/quality_market_decomposition.csv')
    for row,model,predictor in zip(tables[9][1:],['Income only']+['Income + market composition']*3,['log_income','log_income','log_supply','low_price_share']):
        r=one(a,year=2024,model=model,predictor=predictor)
        check(f'S14 {model} {predictor}',row[1],r.standardized_beta);check(f'S14 interval {predictor}',row[2],[r.ci_low,r.ci_high])

    a=load('outputs/restricted/v134_fixed_nutrition_joint_downstream/scenario_results.csv')
    for row in tables[11][1:]:
        name=('zero_affordability_gap' if row[0].startswith('Low-income') else 'equity_joint' if row[0].startswith('Joint') else 'access_deficit' if row[0].startswith('Low joint') else 'population_reach')
        r=one(a,scenario=name,budget_sites=int(row[1]))
        check(f'S9 {name} {row[1]} population',row[2],r.population_within_selected_catchments/1e6)
        check(f'S9 {name} {row[1]} low-income',row[3],r.zero_affordability_priority_reached_share*100)
        check(f'S9 {name} {row[1]} joint',row[4],r.joint_priority_reached_share*100)

    a=load('outputs/restricted/v134_fixed_nutrition_joint_access/joint_access_point_estimates.csv')
    c=load('outputs/restricted/v134_fixed_nutrition_joint_access/joint_access_dcca_block_intervals.csv')
    for row in tables[12][1:]:
        y=int(row[0]);r=one(a,year=y,outcome='joint_access');low=one(a,year=y,outcome='low_price_access')
        check(f'S11 {y} low',row[1],low.population_weighted_mean);check(f'S11 {y} qualifying',row[2],r.population_weighted_mean)
        check(f'S11 {y} retention',row[3],100*r.population_weighted_mean/low.population_weighted_mean)
        for col,metric,scale in [(4,'zero_access_population_share',100),(5,'weighted_gini',1),(6,'concentration_index',1)]:
            ci=one(c,year=y,outcome='joint_access',metric=metric);check(f'S11 {y} {metric}',row[col],[r[metric]*scale,ci.ci_low*scale,ci.ci_high*scale])

    a=load('outputs/restricted/v134_fixed_nutrition_joint_downstream/joint_subgroup_point_estimates.csv')
    c=load('outputs/restricted/v134_fixed_nutrition_joint_downstream/joint_subgroup_dcca_block_intervals.csv')
    q=load('outputs/restricted/v135_fixed_nutrition_subgroup_contrasts/within_year_contrasts.csv')
    for row in tables[13][1:]:
        r=one(a,year=2024,domain=row[0],group=row[1]);ci=one(c,year=2024,domain=row[0],group=row[1],metric='zero_joint_access_share')
        check(f'S12 {row[0]} {row[1]} denominator',row[2],r.group_denominator)
        check(f'S12 {row[1]} mean',row[3],r.mean_joint_options);check(f'S12 {row[1]} zero',row[4],r.zero_joint_access_share*100)
        check(f'S12 {row[1]} interval',row[5],[ci.lower_95*100,ci.upper_95*100])
        if row[6]!='Reference':check(f'S12 {row[1]} q',row[6],one(q,year=2024,domain=row[0],comparison_group=row[1]).q_bh_within_domain_year)

    a=load('outputs/restricted/v135_fixed_nutrition_component_social/component_domain_reference_contrasts_2024.csv')
    for row,label in zip(tables[14][1:],['Filipino − Chinese','Elementary − Managers','Primary − Post-secondary','<HK$10k − ≥HK$40k','65+ − 25–44','Female − Male']):
        r=a.loc[a.contrast_label.eq(label)]
        assert len(r)==6,(label,len(r),a.contrast_label.unique())
        for col,component in [(1,'Environmental sustainability'),(2,'Hygiene')]:check(f'S15 {label} {component}',row[col],one(r,component=component).difference)
        rr=r.iloc[0];qq=q.loc[q.year.eq(2024)&q.domain.eq(rr.domain)]
        t=qq.loc[qq.comparison_group.eq(rr.focal_group)&qq.reference_group.eq(rr.reference_group)]
        sign=1
        if len(t)==0:t=qq.loc[qq.reference_group.eq(rr.focal_group)&qq.comparison_group.eq(rr.reference_group)];sign=-1
        assert len(t)==1,label
        check(f'S15 {label} zero',row[3],t.iloc[0].difference_percentage_points*sign);check(f'S15 {label} q',row[4],t.iloc[0].q_bh_within_domain_year)

    receipt['checked_numeric_values']=len(receipt['checks'])
    receipt['status']='PASS' if not receipt['mismatches'] else 'MISMATCH'

    return receipt
