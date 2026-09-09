"""Combine structural SDI inequality and its component evidence from source tables.

Presentation-only consolidation of prior main Figs 13 and 14. No new tests,
estimates, interpolation, sample exclusions or source-data writes.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import render_evidence_composites as shared
import fig13_component_social_decomposition as comp
from v5_cities_visual_system import YEAR_COLORS, clean_axis

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT/'source_data/figS_sdi_structural_inequality_v4'
DETAIL = ROOT/'source_data/fig13_component_social_decomposition_v1'
STEM = 'Fig13_Quality_Inequality_Integrated_Evidence'
YEARS = [2011, 2016, 2021, 2024]
MARKERS = ['o', 's', 'D', '^']
NAMES = ['Nutrition', 'Carbon', 'Diversity', 'Environment', 'Hygiene', 'Practice']
COLORS = ['#D58236', '#367FAD', '#7858A4', '#428B54', '#B74652', '#287F79']


def heading(ax, letter, title):
    ax.set_title(title, loc='left', fontsize=7.7, fontweight='normal', pad=7)
    ax.text(-.075, 1.04, letter, transform=ax.transAxes,
            fontsize=8.8, fontweight='bold', ha='right', va='bottom')


def frame_heatmap(ax, colorbar):
    """Use one quiet outline; retain seamless cells inside the matrix."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#202020')
        spine.set_linewidth(.55)
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor('#202020')
    colorbar.outline.set_linewidth(.55)


def render():
    shared.style()
    intervals = pd.read_csv(STRUCT/'sdi_inequality_block_intervals.csv')
    curves = pd.read_csv(STRUCT/'sdi_income_concentration_curves.csv')
    quintiles = pd.read_csv(STRUCT/'sdi_income_quintile_means.csv')
    concentration = pd.read_csv(STRUCT/'component_income_concentration.csv')
    boxes = pd.read_csv(DETAIL/'weighted_component_income_distributions.csv')
    contrasts = pd.read_csv(DETAIL/'component_domain_reference_contrasts_2024.csv')
    means = pd.read_csv(DETAIL/'component_domain_group_means_2024.csv')
    # Recalculate all 90 weighted boxes and all 36 ecological contrasts using
    # the existing estimators. Verify rather than merely copy old artwork.
    fresh = comp.build_distributions()
    keys = ['year', 'component', 'income_quintile']
    a, b = [d.sort_values(keys).reset_index(drop=True) for d in (boxes, fresh)]
    assert a[keys].equals(b[keys])
    fields = ['q05', 'q25', 'median', 'q75', 'q95']
    np.testing.assert_allclose(a[fields], b[fields], rtol=1e-12, atol=1e-12)
    fresh_contrasts = comp.build_contrasts(means)
    keys2 = ['domain', 'component']
    a, b = [d.sort_values(keys2).reset_index(drop=True) for d in (contrasts, fresh_contrasts)]
    assert a[keys2].equals(b[keys2])
    np.testing.assert_allclose(a.difference, b.difference, rtol=1e-12, atol=1e-12)

    fig = plt.figure(figsize=(183/25.4, 242/25.4), facecolor='white')
    xs, widths = [.085, .410, .735], [.235, .235, .235]
    # Overall inequality leads; component distributions and contrasts follow.
    left = fig.add_axes([.075, .785, .105, .157])
    right = fig.add_axes([.223, .785, .105, .157])
    recorded = []
    for ax, metric, label in [(left, 'weighted_gini', 'Gini'), (right, 'theil_t', 'Theil T')]:
        d = intervals.loc[intervals.metric.eq(metric)].set_index('year').loc[YEARS]
        for idx, year in enumerate(YEARS):
            r = d.loc[year]; y = 3-idx
            ax.plot([r.lower_95, r.upper_95], [y,y], color=YEAR_COLORS[year], lw=4.5, alpha=.15)
            ax.plot([r.lower_95, r.upper_95], [y,y], color=YEAR_COLORS[year], lw=1.0)
            ax.plot(r.estimate,y,marker=MARKERS[idx],color=YEAR_COLORS[year],ms=3.7,mec='white',mew=.4)
            recorded.append(dict(metric=metric,year=year,estimate=float(r.estimate),
                                 lower=float(r.lower_95),upper=float(r.upper_95)))
        ax.set(yticks=range(4),yticklabels=YEARS[::-1] if ax is left else ['']*4,
               ylim=(-.35,3.6),xlabel=label)
        ax.spines[['left','top','right']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.set_xticks([.10,.12] if metric=='weighted_gini' else [.02,.03])
    heading(left,'a','Citywide inequality')
    handles = [Line2D([0],[0],color=YEAR_COLORS[y],marker=m,ms=3,lw=0,label=str(y))
               for y,m in zip(YEARS,MARKERS)]
    ax = fig.add_axes([.410, .785, .235, .157])
    curve_rows=[]
    for y in YEARS:
        d=curves.loc[curves.year.eq(y)].sort_values('population_fraction')
        gap=100*(d.cumulative_sdi_fraction-d.population_fraction).to_numpy(float)
        curve_rows.append(gap)
        # The pale stroke follows the observed line itself; it is visual
        # separation, not an uncertainty interval or area-to-zero fill.
        ax.plot(d.population_fraction,gap,color=YEAR_COLORS[y],lw=4.2,
                alpha=.14,solid_capstyle='round',zorder=1)
        ax.plot(d.population_fraction,gap,color=YEAR_COLORS[y],lw=1.25,
                label=str(y),zorder=3)
    curve_matrix=np.vstack(curve_rows)
    ax.set(xlim=(0,1),ylim=(min(-2.45,float(np.nanmin(curve_matrix))-.08),.32),
           xticks=[0,.25,.5,.75,1],xlabel='Income-ranked population share',
           ylabel='Gap (pp)')
    ax.legend(handles=handles,loc='upper center',ncol=2,fontsize=5.4,
              columnspacing=.65,handlelength=1.2,handletextpad=.3,frameon=False)
    clean_axis(ax); heading(ax,'b','Concentration gap')
    ax = fig.add_axes([.750, .785, .220, .157])
    for y,m in zip(YEARS,MARKERS):
        d=quintiles.loc[quintiles.year.eq(y)]
        # Restore the original trajectory view. The pale stroke follows the
        # observed series and does not imply an uncertainty interval.
        ax.plot(d.income_quintile,d.mean_sdi,color=YEAR_COLORS[y],lw=4.2,
                alpha=.14,solid_capstyle='round',zorder=1)
        ax.plot(d.income_quintile,d.mean_sdi,color=YEAR_COLORS[y],
                marker=m,lw=1.25,ms=3.8,mec='white',mew=.4,
                label=str(y),zorder=3)
    ax.set(xticks=range(1,6),xticklabels=['Q1','Q2','Q3','Q4','Q5'],
           xlabel='Income quintile',ylabel='Population-weighted SDI',ylim=(.35,.48))
    ax.legend(handles=handles,loc='upper left',ncol=2,fontsize=6.2,
              columnspacing=.7,handlelength=1.1,handletextpad=.4)
    clean_axis(ax); heading(ax,'c','Income gradient')

    box_stats=[]
    for i, ((_, component, _), name, color) in enumerate(zip(comp.COMPONENTS,NAMES,COLORS)):
        row,col=divmod(i,3)
        ax=fig.add_axes([xs[col], [.565,.353][row], widths[col], .139])
        positions=[]; statistics=[]; faces=[]
        for q in range(1,6):
            for j,y in enumerate(comp.YEARS):
                r=boxes.loc[boxes.component.eq(component)&boxes.year.eq(y)&boxes.income_quintile.eq(q)].iloc[0]
                positions.append(q+(j-1)*.24)
                faces.append(comp.blend_with_white(color,[.32,.60,.92][j]))
                stats=dict(med=r['median'],q1=r.q25,q3=r.q75,whislo=r.q05,whishi=r.q95,fliers=[])
                statistics.append(stats)
                box_stats.append(dict(component=component,year=y,quintile=q,**stats))
        artists=ax.bxp(statistics,positions=positions,widths=.20,patch_artist=True,
            showfliers=False,manage_ticks=False,medianprops=dict(color='#111111',lw=.7),
            whiskerprops=dict(color='#6B7479',lw=.48),capprops=dict(color='#6B7479',lw=.48))
        for box,face in zip(artists['boxes'],faces): box.set(facecolor=face,edgecolor=face,linewidth=.5)
        for median,stats in zip(artists['medians'],statistics):
            np.testing.assert_allclose(median.get_ydata(),stats['med'])
        ax.set(xlim=(.5,5.5),ylim=(-.04,1.16),xticks=range(1,6),
               xticklabels=['Q1','Q2','Q3','Q4','Q5'],yticks=[0,.5,1],xlabel='Income quintile')
        if col==0: ax.set_ylabel('Component score')
        handles2=[Patch(facecolor=comp.blend_with_white(color,s),edgecolor='none',label=str(y))
                  for y,s in zip(comp.YEARS,[.32,.60,.92])]
        ax.legend(handles=handles2,loc='upper center',ncol=3,fontsize=6.2,
                  handlelength=.65,handletextpad=.3,columnspacing=.65,borderaxespad=.1)
        clean_axis(ax);heading(ax,'defghi'[i],name)

    # Signed component evidence is encoded once. This replaces the duplicated
    # concentration dot display in former Fig13h and matrix in former Fig14d.
    ax=fig.add_axes([.215,.095,.370,.176])
    labels=[x[3] for x in comp.FOCAL_CONTRASTS]
    cnames=[x[1] for x in comp.COMPONENTS]
    matrix=contrasts.pivot(index='contrast_label',columns='component',values='difference').loc[labels,cnames]
    soft_bwr=LinearSegmentedColormap.from_list(
        'soft_bwr',['#78A9C7','#D3E4EC','#FAFAFA','#F1D0D2','#E1848C'])
    im=ax.imshow(matrix,cmap=soft_bwr,norm=TwoSlopeNorm(0,-.12,.12),aspect='auto',interpolation='none')
    ax.set(xticks=range(6),xticklabels=NAMES,yticks=range(6),yticklabels=[
        'Filipino − Chinese','Elementary − Managers','Primary − Post-secondary',
        r'<HK\$10k − ≥HK\$40k','65+ − 25–44','Female − Male'])
    ax.tick_params(length=0,pad=3,labelsize=6.2)
    ax.set_xticklabels(NAMES,rotation=25,ha='right',rotation_mode='anchor')
    for iy in range(6):
        for ix in range(6): ax.text(ix,iy,f'{matrix.iloc[iy,ix]:+.02f}',ha='center',va='center',fontsize=6.2)
    heading(ax,'j','2024 subgroup contrasts')
    cax=fig.add_axes([.215,.048,.370,.008])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.12,-.06,0,.06,.12])
    cb.set_label('Focal minus reference mean',fontsize=6.5,labelpad=2)
    frame_heatmap(ax,cb)

    ax=fig.add_axes([.685,.095,.285,.176])
    order=['Nutrition','Carbon','Cuisine diversity','Sustainability text','Hygiene','Practice tag*']
    cm=concentration.pivot(index='component',columns='year',values='concentration_index').loc[order,YEARS]
    lim=max(.1,float(abs(cm).max().max()))
    im=ax.imshow(cm,cmap=soft_bwr,norm=TwoSlopeNorm(0,-lim,lim),aspect='auto',interpolation='none')
    ax.set(xticks=range(4),xticklabels=YEARS,yticks=range(6),yticklabels=NAMES)
    ax.tick_params(length=0,pad=3,labelsize=6.2)
    for iy in range(6):
        for ix in range(4): ax.text(ix,iy,f'{cm.iloc[iy,ix]:+.02f}',ha='center',va='center',fontsize=6.2)
    heading(ax,'k','Component concentration')
    cax=fig.add_axes([.685,.048,.285,.008])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.1,0,.1])
    cb.set_label('Income concentration index',fontsize=6.5,labelpad=2)
    frame_heatmap(ax,cb)
    shared.export(fig,STEM,dict(panels=11,replaces=[13,14],
        weighted_boxes_recomputed=90,box_statistics=box_stats,
        subgroup_contrasts_recomputed=36,intervals=recorded,
        duplicate_removed='Same 24 component concentration values previously drawn twice',
        panel_map={'a-c':'former Fig14a-c','d-i':'former Fig13a-f',
                   'j':'former Fig13g','k':'former Fig13h and Fig14d'},
        definitions='Boxes: population-weighted median, IQR and 5th–95th percentiles; '
        'a: existing 95% DCCA-block bootstrap intervals; b: observed four-year concentration-gap curves with a visual stroke halo but no uncertainty ribbon; c: observed quintile means connected in income-quintile order with a visual stroke halo but no uncertainty ribbon; '
        'j: descriptive 2024 component contrasts at fixed 2021 population composition; '
        'Practice is an undated snapshot, not historical programme adoption.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'figures/current')
    args=p.parse_args();shared.OUT=args.output_dir.resolve();shared.OUT.mkdir(parents=True,exist_ok=True)
    before=shared.source_hashes();render()
    assert before==shared.source_hashes(),'Source data changed'
    report=dict(source_files_unchanged=len(before),source_hashes=before,**shared.AUDIT)
    (shared.OUT/'quality_inequality_composite_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
