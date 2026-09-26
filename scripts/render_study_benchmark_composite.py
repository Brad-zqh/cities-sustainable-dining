"""Nine-panel source-native study-area and external count-benchmark composite.

The benchmark is not a validation of SDI quality or of historical price tiers.
The four distribution panels use recorded activity intervals and fixed 200-m
cells, not individual restaurant coordinates. No acquisition is run here.
"""
from pathlib import Path
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import render_evidence_composites as shared
import fig03_study_area_restaurant_distribution as study
import fig03_openrice_fehd as benchmark
from v5_cities_visual_system import north_arrow,segmented_scale_bar

STEM='Fig03_Study_Area_External_Benchmark_Integrated_Evidence'

def heading(ax,letter,title):
    ax.text(-.035,1.035,letter,transform=ax.transAxes,fontsize=9,
            fontweight='bold',ha='right',va='bottom')
    ax.set_title(title,loc='left',fontsize=7.5,pad=8,fontweight='normal')

def draw():
    shared.style()
    cells=pd.read_csv(study.SOURCE/'annual_restaurant_density_200m.csv')
    counts=pd.read_csv(study.SOURCE/'annual_restaurant_coordinate_coverage.csv').set_index('year')
    contract=json.loads((study.SOURCE/'analysis_contract.json').read_text())
    polygons=gpd.read_file(study.GEOMETRY)
    roads=gpd.read_file(study.SOURCE/'osm_major_walkable_context_2025_01_01.gpkg')
    assert polygons.crs.to_epsg()==roads.crs.to_epsg()==2326
    fig=plt.figure(figsize=(183/25.4,242/25.4),facecolor='white')
    audit={'annual_counts':{},'benchmark':{},'crs':'EPSG:2326',
           'panels':9,'basemap':False,'font':mpl.rcParams['font.family'],
           'osm_context':'2025-01-01 fixed orientation reference only, not historical networks'}
    norm=BoundaryNorm(study.COUNT_BREAKS,256,clip=True)
    for idx,year in enumerate(study.YEARS):
        row,col=divmod(idx,2)
        ax=fig.add_axes([.052+col*.448,[.786,.566][row],.402,.194])
        study.decorate_map(ax,polygons,roads)
        d=cells.loc[cells.year.eq(year)].sort_values('restaurant_n')
        total=int(d.restaurant_n.sum())
        assert total==int(counts.loc[year,'valid_coordinate_restaurant_n'])==contract['annual_restaurant_counts'][str(year)]
        size=np.clip(.65+np.log2(d.restaurant_n.to_numpy(float)+1)*.70,1.05,5.2)
        dots=ax.scatter(d.easting_m,d.northing_m,c=d.restaurant_n,s=size,
            cmap=study.single_hue_map(study.COLORS[year]),norm=norm,
            alpha=1,lw=0,zorder=10,rasterized=True)
        north_arrow(ax,x=.075,y=.94,height=.052)
        segmented_scale_bar(ax,length_km=10,x=.06,y=.022)
        heading(ax,chr(97+idx),str(year))
        # Keep the count in the title band.  Placing it below the map caused
        # the second-row counts to collide with the OpenRice/FEHD headings.
        ax.text(.995,1.035,f'{total:,} restaurants',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=6.2)
        cax=fig.add_axes([.459+col*.448,.804,.010,.153])
        cb=fig.colorbar(dots,cax=cax,ticks=[1,5,20,80,320],spacing='uniform')
        cb.set_ticklabels(['1','5','20','80','320+'])
        cb.ax.set_title('n',fontsize=6.2,pad=4)
        shared.map_key_height(fig,ax,cax,polygons)
        audit['annual_counts'][str(year)]={'restaurants':total,'cells':len(d)}
    data=pd.read_csv(benchmark.SOURCE_BUNDLE/'lsbg_openrice_fehd_count_access.csv',dtype={'lsbg_id':str})
    summary=json.loads((benchmark.SOURCE_BUNDLE/'comparison_summary.json').read_text())['summary']
    geo=gpd.read_file(benchmark.RAW_LSBG)[['lsbg','geometry']].copy()
    assert geo.crs.to_epsg()==2326
    geo['lsbg_id']=geo.lsbg.astype(str).str.replace(r'\.0$','',regex=True)
    geo=geo.merge(data.loc[data.threshold_min.eq(15)],on='lsbg_id',how='left',validate='one_to_one')
    columns=['openrice_per_1000_source','fehd_calibrated_per_1000_source']
    pooled=np.concatenate([geo[c].dropna().to_numpy(float) for c in columns])
    breaks=np.quantile(pooled,np.linspace(0,1,7))
    breaks[0]=min(0,float(breaks[0]));breaks[-1]=np.nextafter(breaks[-1],np.inf)
    assert np.all(np.diff(breaks)>0)
    bnorm=BoundaryNorm(breaks,6,clip=True)
    benchmark.draw_north_arrow=lambda ax:north_arrow(ax,x=.075,y=.94,height=.052)
    benchmark_map_palettes = {
        # Restrained red/blue pairing requested for the external benchmark.
        # The four annual panels retain their year colours; only the paired
        # OpenRice--FEHD comparison uses this diverging visual language.
        'OpenRice': ['#F8E8EA','#EEC4CA','#E29AA5','#CF6675','#B33E52','#84263A'],
        'FEHD': ['#E7F1F7','#C8DEEC','#9BC2DA','#66A0C6','#347CAD','#18527E'],
    }
    for col,source in enumerate(['OpenRice','FEHD']):
        ax=fig.add_axes([.052+col*.448,.346,.402,.194])
        cmap=ListedColormap(benchmark_map_palettes[source])
        benchmark.draw_map(ax,geo,columns[col],'',bnorm,cmap)
        heading(ax,chr(101+col),f'2021 · {source}')
        cax=fig.add_axes([.459+col*.448,.361,.010,.160])
        cb=fig.colorbar(mpl.cm.ScalarMappable(norm=bnorm,cmap=cmap),cax=cax,
                       ticks=breaks,boundaries=breaks,spacing='uniform')
        cb.ax.set_yticklabels([f'{v:.1f}' if v<20 else f'{v:.0f}' for v in breaks])
        cb.set_label('per 1,000 outlets',fontsize=6.2,labelpad=2)
        shared.map_key_height(fig,ax,cax,geo)
    audit['benchmark']['map_breaks']=breaks.tolist()
    # Equal-width, aligned diagnostics: compactness comes from larger axes,
    # not from smaller text.
    # Equal-width diagnostics with a little more left margin for the two-line y label.
    axg=fig.add_axes([.078,.058,.228,.230])
    axh=fig.add_axes([.365,.058,.228,.230])
    # A wider gutter protects the metric labels in panel i from panel h.
    axi=fig.add_axes([.720,.058,.230,.230])
    # One colour pair encodes the thresholds consistently across all three
    # diagnostics. A muted blue/red pair matches the benchmark maps and keeps
    # the two walking thresholds distinct without introducing a third hue.
    threshold_10, threshold_15 = "#347EAE", "#BC4254"
    benchmark.TEAL, benchmark.VIOLET = threshold_10, threshold_15
    benchmark.draw_scatter(axg,data,summary)
    axg.set_xticks([0,50,100]);axg.set_yticks([0,50,100])
    axg.set_xlabel('OpenRice opportunity\n(per 1,000 outlets)',fontsize=6.5)
    axg.set_ylabel('FEHD opportunity\n(per 1,000 outlets)',fontsize=6.5)
    benchmark.TEAL, benchmark.VIOLET = threshold_10, threshold_15
    benchmark.draw_bland_altman(axh,data)
    axh.set_xlabel('Mean normalized\naccessibility',fontsize=6.5)
    axh.set_ylabel('FEHD minus OpenRice',fontsize=6.5)
    axh.set_xticks([0,40,80]);axh.set_yticks([-10,0,10,20])
    axh.set_ylim(-10.5,20)
    benchmark.TEAL, benchmark.VIOLET = threshold_10, threshold_15
    benchmark.draw_robustness(axi,summary)
    axi.set_ylim(-.4,3.9)
    axi.set_xlim(.78,1.055)
    original_legend=axi.get_legend()
    handles=original_legend.legend_handles
    labels=[t.get_text() for t in original_legend.get_texts()]
    axi.legend(handles,labels,loc='upper left',bbox_to_anchor=(0.01,.91),ncol=2,fontsize=6.2,
               columnspacing=.5,handletextpad=.2)
    for ax,letter,title in [(axg,'g','LSBG agreement'),(axh,'h','Bias and agreement limits'),(axi,'i','Availability diagnostics')]:
        heading(ax,letter,title)
        ax.tick_params(labelsize=6.2)
    for threshold in [10,15]:
        d=data.loc[data.threshold_min.eq(threshold)]
        diff=d.fehd_calibrated_per_1000_source-d.openrice_per_1000_source
        bias=float(diff.mean());sd=float(diff.std(ddof=1))
        audit['benchmark'][str(threshold)]={'rows':len(d),'bias':bias,
            'limits':[bias-1.96*sd,bias+1.96*sd],
            'spearman_recomputed':float(d.openrice_per_1000_source.corr(d.fehd_calibrated_per_1000_source,method='spearman')),
            'pearson_recomputed':float(d.openrice_per_1000_source.corr(d.fehd_calibrated_per_1000_source)),
            'archived_summary':summary[str(threshold)]}
        assert np.isclose(audit['benchmark'][str(threshold)]['spearman_recomputed'],summary[str(threshold)]['correlation']['fehd_calibrated']['spearman'])
        assert np.isclose(audit['benchmark'][str(threshold)]['pearson_recomputed'],summary[str(threshold)]['correlation']['fehd_calibrated']['pearson'])
        assert diff.min()>axh.get_ylim()[0] and diff.max()<axh.get_ylim()[1]
    shared.export(fig,STEM,audit)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'figures/v48_study_benchmark')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    shared.OUT=args.output_dir;before=shared.source_hashes()
    draw();assert before==shared.source_hashes()
    (args.output_dir/'study_benchmark_audit.json').write_text(json.dumps(
        {'source_files':len(before),'source_sha256':before,'figures':shared.AUDIT['figures']},indent=2),encoding='utf-8')

if __name__=='__main__':main()
