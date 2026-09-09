# Academic Figure Skill Asset Confirmation (verified against production script)
# (a-h) LSBG/DCCA choropleths -> render_scale_weight_composite.py -> visual adapt
# (i-l) weight/MAPU diagnostics -> render_scale_weight_composite.py -> visual adapt
# RULE: all source rows, class breaks, estimates, labels and audit assertions are unchanged.
"""Single-page, two-maps-per-row composite of archived scale/weight results.

No source estimator, map class boundary, or observation is changed. Halos are
presentation strokes, not intervals. Document assembly is a separate step.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'figures'))
import render_evidence_composites as shared
import v4_plot_four_year_results as maps
import fig11_weight_maup as sensitivity
from v5_cities_visual_system import segmented_scale_bar, north_arrow, clean_axis

YEARS = [2011, 2016, 2021, 2024]
YEAR_COLORS = {2011:'#C94157', 2016:'#347FC0', 2021:'#419B58', 2024:'#E69736'}
MARKERS = {2011:'o', 2016:'s', 2021:'D', 2024:'^'}
SOURCE = ROOT / 'source_data/figS_weight_sensitivity_v4'
STEM = 'Fig10_Spatial_Scale_Weighting_Integrated_Evidence'
PAIRED_RAMPS = {
    2011: ['#F1BFC6', '#E797A4', '#D96A7C', '#BC3E56', '#862439'],
    2016: ['#BCDCEB', '#8EBFDC', '#5A9EC6', '#327BB0', '#205377'],
    2021: ['#C7E5C6', '#9ACC9D', '#6AB276', '#3D8F51', '#25613B'],
    2024: ['#F5D6B0', '#EDB777', '#E39845', '#C47525', '#925017'],
}


def heading(ax, letter, title):
    ax.set_title(title, loc='left', pad=7, fontsize=7.6, fontweight='normal')
    ax.text(-.085, 1.02, letter, transform=ax.transAxes, ha='right',
            va='bottom', fontsize=9, fontweight='bold')


def points(ax, x, y, color, marker='o', label=None):
    ax.scatter(x, y, s=68, color=color, alpha=.12, marker=marker,
               lw=0, zorder=1)
    return ax.scatter(x, y, s=19, color=color, marker=marker, label=label,
                      edgecolor='white', lw=.45, zorder=3)


def build(paired_pages=False, split_part=None):
    shared.style()
    lsbg, dcca = maps.load_frames('LSBG'), maps.load_frames('DCCA')
    mask = gpd.read_file(maps.LAND_MASK_FILE).to_crs(2326).geometry.union_all()
    for year in YEARS:
        dcca[year] = dcca[year].copy()
        dcca[year]['geometry'] = dcca[year].geometry.intersection(mask)
        dcca[year] = dcca[year].loc[~dcca[year].geometry.is_empty].copy()
    frames = list(lsbg.values()) + list(dcca.values())
    values = pd.concat([f.sdi_equal_arithmetic for f in frames]).dropna()
    values = values.loc[values.gt(0)]
    breaks = np.unique(values.quantile(np.linspace(0, 1, 6)).to_numpy(float))
    breaks[0] = np.nextafter(breaks[0], -np.inf)
    breaks[-1] = np.nextafter(breaks[-1], np.inf)
    all_bounds = np.array([f.total_bounds for f in frames])
    bounds = np.r_[all_bounds[:,:2].min(axis=0), all_bounds[:,2:].max(axis=0)]
    dx, dy = bounds[2:] - bounds[:2]
    xlim = [bounds[0]-.025*dx, bounds[2]+.025*dx]
    ylim = [bounds[1]-.11*dy, bounds[3]+.06*dy]
    if split_part == 'maps':
        fig_height_mm = 210
    elif split_part == 'diagnostics':
        fig_height_mm = 165
    else:
        fig_height_mm = 234 if not paired_pages else 247
    fig = plt.figure(figsize=(183/25.4, fig_height_mm/25.4), facecolor='white')
    audit = {'panel_map': {}, 'map_breaks':breaks.tolist(), 'crs':'EPSG:2326',
             'map_common_xlim':xlim, 'map_common_ylim':ylim,
             'quantitative_values':{}, 'halos':'cosmetic, not uncertainty'}
    if split_part != 'diagnostics':
      for row, (scale, source) in enumerate([('LSBG',lsbg), ('DCCA',dcca)]):
        for col, year in enumerate(YEARS):
            x, y = .04 + col*.241, [.792,.602][row]
            if split_part == 'maps':
                x, y = .110 + row*.340, .755-col*.220
            elif paired_pages:
                # Four compact rows; each row compares the same year's LSBG
                # map (left) with DCCA sensitivity (right). This preserves the
                # requested two-maps-per-row reading order on one A4 page.
                # The map itself is narrower than its nominal axes box after
                # the equal-aspect constraint is applied.  Bring the two axes
                # centres inward so the paired maps read as a comparison,
                # while retaining a safe gutter for the left colour key.
                x, y = .110 + row*.340, .818-col*.154
            if split_part == 'maps':
                ax = fig.add_axes([x,y,.425,.180])
            else:
                ax = fig.add_axes([x,y,.425,.132] if paired_pages else [x,y,.185,.160])
            cmap = (ListedColormap(PAIRED_RAMPS[year], name=f'paired_{year}')
                    if (paired_pages or split_part == 'maps')
                    else maps.sequential_map(YEAR_COLORS[year], f'scale_{year}'))
            norm = BoundaryNorm(breaks, cmap.N)
            maps.plot_choropleth(ax, source[year], 'sdi_equal_arithmetic', cmap,
                                norm, positive_only=True, show_cartography=False)
            if paired_pages:
                # Solid thematic fill: no hidden layer or alpha blending changes
                # the five class colours. Geometry and class membership are fixed.
                for collection in ax.collections:
                    collection.set_alpha(1.0)
            ax.set(xlim=xlim, ylim=ylim)
            north_arrow(ax, x=.11, y=.91, height=.065)
            segmented_scale_bar(ax, length_km=10, x=.075, y=.015)
            letter=chr(97+(col*2+row if (paired_pages or split_part == 'maps') else row*4+col))
            ax.text(-.03, 1.025, letter, transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='bottom')
            ax.set_title(f'{year} · {scale}' if (paired_pages or split_part == 'maps') else str(year),
                         pad=7, fontsize=8.3 if (paired_pages or split_part == 'maps') else 7.8, fontweight='normal')
            if split_part == 'maps':
                cax = fig.add_axes([x+.433,y+.016,.008,.145])
            else:
                cax = fig.add_axes([x+(.433 if paired_pages else .19),y+(.012 if paired_pages else .02),.008,(.108 if paired_pages else .13)])
            cb=mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,boundaries=breaks,
                                        ticks=breaks,orientation='vertical')
            cb.ax.set_yticklabels([maps.format_break(b) for b in breaks])
            cb.ax.set_title('SDI', fontsize=6.2, pad=4)
            shared.map_key_height(fig,ax,cax,source[year])
            if paired_pages or split_part == 'maps':
                cb.ax.tick_params(labelsize=6.6, pad=2.4)
                cax.set_position([cax.get_position().x0,cax.get_position().y0,
                                  .010,cax.get_position().height])
            checks=shared.map_check(source[year],'sdi_equal_arithmetic',breaks)
            audit['panel_map'][letter]={'scale':scale,'year':year,'rows':len(source[year]),
                                        'three_feature_check':checks}
        if not paired_pages and split_part != 'maps':
            fig.text(.015, y+.086, scale, rotation=90, ha='center', va='center', fontsize=7.3)
      fig.legend(handles=[Patch(facecolor=maps.COLORS['missing'],edgecolor='#BAC4C9',label='Incomplete components'),
                          Patch(facecolor=maps.COLORS['no_restaurant'],edgecolor='#C8D0D4',label='No listed outlet')],
                 loc='center',bbox_to_anchor=(.51,.965 if split_part == 'maps' else (.972 if paired_pages else .563)),ncol=2,
                 fontsize=6.8 if (paired_pages or split_part == 'maps') else 6.2,
                 handlelength=1.1,frameon=False)

    if paired_pages or split_part == 'maps':
        audit['map_palette'] = PAIRED_RAMPS
        audit['layout'] = ('Single A4 page: four same-year rows, LSBG left and '
                           'DCCA right; diagnostics i-l below; no continuation')
    if split_part == 'maps':
        audit['layout'] = 'Standalone map figure: four same-year rows, LSBG left and DCCA right.'
        shared.export(fig,'Fig18_Spatial_Scale_Maps',audit)
        return

    weights=pd.read_csv(SOURCE/'weight_schemes.csv')
    agreement=pd.read_csv(SOURCE/'variant_agreement.csv')
    maup=pd.read_csv(SOURCE/'maup_agreement.csv')
    variants=sensitivity.VARIANTS
    short=[sensitivity.SHORT[v] for v in variants]
    ax=fig.add_axes([.14,.331,.32,.170])
    if split_part == 'diagnostics': ax.set_position([.120,.565,.360,.320])
    elif paired_pages:
        # The geographic axes are narrowed by their fixed equal aspect inside
        # the nominal map boxes.  Use 75% of the former diagnostic width and
        # centre it on the actual left-map column rather than the page margin.
        ax.set_position([.195,.205,.255,.125])
    components=['Nutrition','Carbon','Cuisine diversity','Hygiene','Practice tag','Sustainability signal']
    display=['Nutrition','Carbon','Diversity','Hygiene','Practice','Environment']
    yy=np.arange(6)[::-1]
    for method,color,marker,offset,label in [('Equal six','#222222','o',.18,'Equal six'),
            ('Absolute PC1 loading',shared.BLUE,'s',0,'Absolute PC1'),
            ('Entropy',shared.ORANGE,'^',-.18,'Entropy')]:
        vals=weights.loc[weights.scheme.eq(method)].set_index('component_label').loc[components,'weight'].to_numpy(float)
        points(ax,vals,yy+offset,color,marker,label)
        audit['quantitative_values']['weights_'+method]=vals.tolist()
        assert np.isclose(vals.sum(),1)
    ax.set(yticks=yy,yticklabels=display,xlim=(0,.70),ylim=(-.7,5.7),xlabel='Weight')
    ax.legend(loc='upper right',fontsize=6.2,handletextpad=.35,labelspacing=.3)
    clean_axis(ax); ax.set_xticks([0,.2,.4,.6]); heading(ax,'i','Component weights')
    assert max(max(v) for k,v in audit['quantitative_values'].items() if k.startswith('weights_')) < ax.get_xlim()[1]

    ax=fig.add_axes([.655,.331,.32,.170]); yy=np.arange(4)[::-1]
    if split_part == 'diagnostics': ax.set_position([.570,.565,.360,.320])
    elif paired_pages: ax.set_position([.535,.205,.255,.125])
    for year,offset in zip(YEARS,[.24,.08,-.08,-.24]):
        vals=agreement.loc[agreement.year.eq(year)&agreement.scale.eq('LSBG')].set_index('variant').loc[variants,'spearman_vs_equal'].to_numpy(float)
        points(ax,vals,yy+offset,YEAR_COLORS[year],MARKERS[year],str(year))
        audit['quantitative_values'][f'rank_{year}']=vals.tolist()
    ax.set(yticks=yy,yticklabels=short,xlim=(.35,1.02),ylim=(-.65,3.8),xlabel='Spearman ρ (LSBG)')
    ax.axvline(1,color='#879299',ls='--',lw=.65)
    ax.legend(loc='upper left',ncol=2,fontsize=6.2,handletextpad=.3,columnspacing=.7)
    clean_axis(ax); ax.set_xticks([.4,.6,.8,1]); heading(ax,'j','Rank agreement')

    ax=fig.add_axes([.14,.079,.32,.168])
    if split_part == 'diagnostics': ax.set_position([.120,.115,.360,.300])
    elif paired_pages: ax.set_position([.195,.055,.255,.095])
    frame=agreement.loc[agreement.year.eq(2021)&agreement.scale.eq('LSBG')].set_index('variant').loc[variants]
    for y,(_,r) in zip(yy,frame.iterrows()):
        lo,hi=r.low_quintile_jaccard,r.high_quintile_jaccard
        ax.plot([lo,hi],[y,y],color='#AAB5BA',lw=4.5,alpha=.13,zorder=1)
        ax.plot([lo,hi],[y,y],color='#AAB5BA',lw=1.1,zorder=2)
        points(ax,lo,y,shared.BLUE,'o')
        points(ax,hi,y,shared.RED,'s')
    ax.scatter([],[],color=shared.BLUE,s=19,marker='o',label='Low quintile')
    ax.scatter([],[],color=shared.RED,s=19,marker='s',label='High quintile')
    ax.set(yticks=yy,yticklabels=short,xlim=(0,1.03),ylim=(-.6,3.6),xlabel='Jaccard overlap with equal weighting')
    ax.legend(loc='lower right',fontsize=6.2,handletextpad=.3)
    clean_axis(ax); ax.set_xticks([0,.25,.5,.75,1]); heading(ax,'k','2021 tail stability')
    audit['quantitative_values']['tail_2021']=frame[['low_quintile_jaccard','high_quintile_jaccard']].to_numpy(float).tolist()

    ax=fig.add_axes([.655,.113,.32,.134])
    if split_part == 'diagnostics': ax.set_position([.570,.155,.360,.260])
    elif paired_pages: ax.set_position([.535,.055,.255,.095])
    all_variants=['Equal six (strict)',*variants]
    matrix=np.array([maup.loc[maup.variant.eq(v)].set_index('year').loc[YEARS,'spearman'].to_numpy(float) for v in all_variants])
    cmap=LinearSegmentedColormap.from_list('agreement_blue_red',
        ['#78A9C7','#D3E4EC','#FAFAFA','#F1D0D2','#E1848C'])
    im=ax.imshow(matrix,aspect='auto',vmin=.58,vmax=.84,cmap=cmap)
    ax.set(xticks=np.arange(4),xticklabels=YEARS,yticks=np.arange(5),yticklabels=['Equal six',*short])
    ax.tick_params(axis='both',length=0,pad=4)
    for r in range(5):
        for c in range(4):
            ax.text(c,r,f'{matrix[r,c]:.2f}',ha='center',va='center',fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.55)
        spine.set_color('#222222')
    heading(ax,'l','Cross-scale agreement')
    cax=fig.add_axes([.655,.071,.32,.008])
    if split_part == 'diagnostics': cax.set_position([.570,.070,.360,.022])
    elif paired_pages: cax.set_position([.535,.032,.255,.007])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[.60,.65,.70,.75,.80])
    cb.set_label('Spearman ρ (LSBG–DCCA)',fontsize=6.5,labelpad=3)
    cb.ax.tick_params(labelsize=6.2,length=2,pad=2)
    cb.outline.set_linewidth(.55)
    cb.outline.set_edgecolor('#222222')
    audit['quantitative_values']['cross_scale']=matrix.tolist()
    diagnostic_letters = 'ijkl'
    audit['panel_map'].update({k:v for k,v in zip(diagnostic_letters,
        ['component weights','rank agreement','2021 tail membership','cross-scale agreement'])})
    if split_part == 'diagnostics':
        export_stem = 'Fig17b_Weighting_And_Cross_Scale_Diagnostics'
    elif split_part == 'maps':
        export_stem = 'Fig17a_Spatial_Scale_Sensitivity_Maps'
    else:
        export_stem = 'Fig18_Scale_Weight_Two_Per_Row_Single_Page' if paired_pages else STEM
    shared.export(fig,export_stem,audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/v48_scale_weight')
    parser.add_argument('--paired-pages',action='store_true',
                        help='Arrange same-year LSBG/DCCA maps two per row and retain diagnostics on one A4 page')
    parser.add_argument('--split-part',choices=['maps','diagnostics'],
                        help='Render only the standalone map or diagnostic figure')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    shared.OUT=args.output_dir
    before=shared.source_hashes()
    build(paired_pages=args.paired_pages,split_part=args.split_part)
    assert before==shared.source_hashes(),'Source inputs changed'
    report={'source_data_unchanged':True,'source_files':len(before),'source_sha256':before,
            'figures':shared.AUDIT['figures']}
    (args.output_dir/'scale_weight_composite_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
