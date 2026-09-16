"""Eight-panel, source-native planning comparison; no tiles or new estimators.

The four maps separate overlapping K=10 selections. Three aligned horizontal
bar panels show the original K=5/10/20 outcomes in their actual units. Map
priority shares retain zero and unavailable values as different states.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
import numpy as np

import fig12_planning as source
from v5_cities_visual_system import configure, north_arrow, segmented_scale_bar, FONT_FAMILY

ROOT = Path(__file__).resolve().parents[1]
STEM = 'Fig10_Planning_Selection_Reach_Tradeoffs'
K = (5, 10, 20)
SHADES = (.40, .70, 1.0)
INK = '#111111'
COLOURS = dict(zip(source.STRATEGIES, ['#BC4254', '#D78A2C', '#347EAE', '#268B68']))
MAP_RAMPS = {
    source.STRATEGIES[0]: ['#FFFFFF', '#F2C6CC', '#D97987', '#A92E45'],
    source.STRATEGIES[1]: ['#FFFFFF', '#F6D9B5', '#E7A04E', '#B96B16'],
    source.STRATEGIES[2]: ['#FFFFFF', '#C7DEEE', '#6FA8CF', '#2C6D9C'],
    source.STRATEGIES[3]: ['#FFFFFF', '#C9E4D5', '#73B691', '#247451'],
}
NAMES = ['Low-income zero-gap · LZG', 'Joint disadvantage · JD', 'Low access · LA', 'Population reach · POP']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT / 'source_data').rglob('*')) if p.is_file()}


def blend(colour, strength):
    return (1-strength) + strength*np.array(mpl.colors.to_rgb(colour))


def heading(fig, x, y, letter, title):
    fig.text(x-.028, y, letter, fontsize=9.5, fontweight='bold', va='bottom')
    fig.text(x, y, title, fontsize=8.2, fontweight='normal', va='bottom')


def draw_map(fig, ax, cax, frame, sites, selections, strategy, index):
    population = frame.t_pop.to_numpy(float)
    share = frame.zero_affordability_priority_population.to_numpy(float) / np.where(population > 0, population, np.nan)
    valid = np.isfinite(share)
    assert (share[valid] >= 0).all() and (share[valid] <= 1).all()
    work = frame.copy()
    work['priority_share'] = share
    # This is one common baseline field, not a strategy-specific outcome.
    cmap = LinearSegmentedColormap.from_list(f'priority_{strategy}', MAP_RAMPS[strategy])
    norm = Normalize(0, 1)
    work.loc[valid].plot(ax=ax, column='priority_share', cmap=cmap, norm=norm,
                        edgecolor='#DBE0E1', linewidth=.13, alpha=1, zorder=1, rasterized=True)
    if (~valid).any():
        work.loc[~valid].plot(ax=ax, color='#E4E7E8', edgecolor='#71787C',
                             hatch='////', linewidth=.2, zorder=2, rasterized=True)
    work.dissolve().boundary.plot(ax=ax, color='#68747B', linewidth=.32, zorder=3, rasterized=True)
    ax.scatter(sites.geometry.x, sites.geometry.y, s=2.7, c='#CED3D5',
               linewidth=0, alpha=1, zorder=4)
    selected_ids = set(selections.loc[(selections.scenario == strategy) &
                                      (selections.budget_sites == 10), 'site_id'])
    selected = sites.loc[sites.site_id.isin(selected_ids)]
    assert len(selected) == 10 and set(selected.site_id) == selected_ids
    coords = np.column_stack((selected.geometry.x, selected.geometry.y))
    ax.scatter(coords[:, 0], coords[:, 1], s=30, marker=source.MARKERS[strategy],
               color='white', linewidth=0, zorder=5)
    layer = ax.scatter(coords[:, 0], coords[:, 1], s=18,
                       marker=source.MARKERS[strategy], color=COLOURS[strategy],
                       edgecolor='white', linewidth=.4, alpha=1, zorder=6)
    assert np.array_equal(np.asarray(layer.get_offsets()), coords)
    xmin, ymin, xmax, ymax = work.total_bounds
    dx, dy = xmax-xmin, ymax-ymin
    ax.set(xlim=(xmin-.02*dx, xmax+.02*dx), ylim=(ymin-.105*dy, ymax+.02*dy))
    ax.set_aspect('equal'); ax.set_axis_off(); ax.set_facecolor('white')
    north_arrow(ax, x=.085, y=.88, height=.057)
    segmented_scale_bar(ax, x=.065, y=.024)
    ax.legend(handles=[Line2D([], [], marker=source.MARKERS[strategy], linestyle='none',
                              color=COLOURS[strategy], markersize=4, label='10 selected nodes')],
              loc='lower right', bbox_to_anchor=(1.00,.005), fontsize=6.2,
              handletextpad=.3, borderpad=0)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, .25, .5, .75, 1], labels=['0', '25', '50', '75', '100'])
    cb.ax.set_title('%', fontsize=6.4, pad=3)
    cb.ax.tick_params(labelsize=6.2, length=2, width=.5, pad=1.5)
    cb.outline.set_edgecolor('black'); cb.outline.set_linewidth(.5)
    fig.canvas.draw()
    low, high = [fig.transFigure.inverted().transform(ax.transData.transform((xmin,y)))[1]
                 for y in (ymin,ymax)]
    pos = ax.get_position()
    cax.set_position([pos.x1+.007, low, .009, high-low])
    heading(fig, pos.x0+.01, pos.y1+.007, 'abcd'[index], NAMES[index])
    samples = []
    for i in [np.flatnonzero(valid & (share==0))[0], np.nanargmax(share),
              np.flatnonzero(valid & (share>0) & (share<1))[0]]:
        samples.append(dict(lsbg_id=str(work.iloc[i].lsbg_id), share=float(share[i]),
                            rgba=list(cmap(norm(share[i])))))
    return dict(strategy=strategy, selected_ids=sorted(selected_ids), map_samples=samples,
                missing_share_n=int((~valid).sum()), positive_share_n=int((share[valid]>0).sum()),
                map_extent=[*ax.get_xlim(), *ax.get_ylim()], colour_limits=[0,1])


def metric(fig, ax, active, field, scale, xmax, ticks, letter, title, unit):
    ax.set_xlim(0, xmax); ax.set_ylim(-.55, 3.55)
    ax.set_yticks([3,2,1,0], labels=['LZG','JD','LA','POP'])
    ax.set_xticks(ticks); ax.set_xlabel(unit, labelpad=1.5)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.tick_params(axis='x', length=2.6, pad=2)
    ax.axvline(0, color='#AAB4B9', linewidth=.55)
    records=[]
    for i, strategy in enumerate(source.STRATEGIES):
        rows = active.loc[active.scenario == strategy].set_index('budget_sites')
        for k, offset, shade in zip(K, [.22,0,-.22], SHADES):
            value = float(rows.loc[k, field])*scale
            bars = ax.barh(3-i+offset, value, height=.16,
                           color=blend(COLOURS[strategy],shade), edgecolor='none', zorder=2)
            assert np.isclose(bars.patches[0].get_width(),value,rtol=0,atol=1e-12)
            ax.text(value+.025*xmax, 3-i+offset,
                    f'{value:.2f}' if scale<1 else f'{value:.1f}',
                    fontsize=6.2, ha='left', va='center')
            records.append(dict(strategy=strategy, K=k, field=field, value=value))
    pos=ax.get_position()
    # Keep the metric headings close to their own axes so the shared K legend
    # above the row cannot collide with the panel letter or title.
    heading(fig,pos.x0,pos.y1+.006,letter,title)
    return records


def overlap(fig, ax, cax, selections):
    sets=[set(selections.loc[(selections.scenario==s)&(selections.budget_sites==10),'site_id'])
          for s in source.STRATEGIES]
    arr=np.array([[len(a&b)/len(a|b) for b in sets] for a in sets])
    # Lightly muted SHAP-like blue/red ramp.  The endpoint contrast remains
    # readable at manuscript size, but the cells no longer dominate the page.
    cmap=LinearSegmentedColormap.from_list(
        'overlap_blue_red',['#8FB8CF','#DCEAF0','#FBFBFB','#F4DADD','#E99CA3'])
    im=ax.imshow(arr,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    assert np.array_equal(im.get_array(),arr)
    labels=[source.SHORT[s] for s in source.STRATEGIES]
    ax.set_xticks(range(4),labels=labels); ax.set_yticks(range(4),labels=labels)
    ax.tick_params(length=0,pad=4)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=7)
    for s in ax.spines.values():
        s.set_visible(True);s.set_color('black');s.set_linewidth(.5)
    cb=fig.colorbar(im,cax=cax)
    cb.set_ticks([0,.25,.5,.75,1]);cb.ax.tick_params(labelsize=6.2,length=2,width=.5)
    cb.outline.set_edgecolor('black');cb.outline.set_linewidth(.5)
    pos=ax.get_position()
    heading(fig,pos.x0,pos.y1+.014,'h','Selection overlap · K=10')
    ax.set_xlabel('Jaccard overlap',labelpad=5)
    return arr.tolist()


def text_audit(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    ignored=set()
    for ax in fig.axes:
        if not ax.axison:
            ignored.update(ax.get_xticklabels()+ax.get_yticklabels())
            ignored.update([ax.xaxis.label,ax.yaxis.label,ax.xaxis.offsetText,ax.yaxis.offsetText])
    off=[]
    for t in fig.findobj(Text):
        t.set_color(INK)
        if t.get_text()=='N':t.set_fontweight('normal')
        if t in ignored or not t.get_visible() or not t.get_text().strip():continue
        box=t.get_window_extent(renderer)
        if not fig.bbox.padded(1).contains(box.x0,box.y0) or not fig.bbox.padded(1).contains(box.x1,box.y1):
            off.append(t.get_text())
    assert not off, off
    return off


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/v51_planning')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    before=hashes()
    results,active,selections,frame,sites=source.load_data()
    assert frame.crs.to_epsg()==2326 and frame.geometry.is_valid.all()
    configure()
    mpl.rcParams.update({'font.size':8.0,'axes.labelsize':8.0,'xtick.labelsize':7.2,
                         'ytick.labelsize':7.4,'legend.fontsize':7.0,
                         'savefig.bbox':None,'axes.linewidth':.55})
    fig=plt.figure(figsize=(183/25.4,252/25.4),facecolor='white')
    maps=[]
    for i,(x,y) in enumerate([(.035,.748),(.530,.748),(.035,.500),(.530,.500)]):
        ax=fig.add_axes([x,y,.405,.228]);cax=fig.add_axes([x+.410,y,.010,.228])
        maps.append(draw_map(fig,ax,cax,frame,sites,selections,source.STRATEGIES[i],i))
    fig.legend(handles=[Patch(facecolor='#89BFBA',edgecolor='#9EAAAA',linewidth=.3,
                              label='Strict-priority population share (%)'),
                        Line2D([],[],marker='o',color='#CED3D5',linestyle='none',markersize=3,
                               label='Candidate node'),
                        Patch(facecolor='#E4E7E8',edgecolor='#71787C',hatch='////',linewidth=.3,
                              label='Unavailable')],
               loc='center',bbox_to_anchor=(.5,.491),ncol=3,frameon=False,fontsize=7.0,
               handlelength=1.2,handletextpad=.4,columnspacing=1.5)
    records=[]
    specs=[('population_within_selected_catchments',1e-6,3.2,[0,1,2,3],'e',
            'Residents within catchments','Residents (million)',[.055,.275,.365,.165]),
           ('zero_affordability_priority_reached_share',100,57,[0,15,30,45],'f',
            'Strict-priority reach','Priority residents reached (%)',[.550,.275,.365,.165]),
           ('joint_priority_reached_share',100,74,[0,20,40,60],'g',
            'Joint-priority reach','Priority residents reached (%)',[.055,.045,.365,.155])]
    for field,scale,xmax,ticks,letter,title,unit,bounds in specs:
        records+=metric(fig,fig.add_axes(bounds),active,field,scale,xmax,ticks,letter,title,unit)
    fig.legend(handles=[Patch(facecolor=blend('#414B50',v),label=f'K={k}')
                        for k,v in zip(K,SHADES)], loc='center',
               bbox_to_anchor=(.5,.469), ncol=3, frameon=False, fontsize=6.8,
               handlelength=.9, handletextpad=.3, columnspacing=.8)
    matrix=overlap(fig,fig.add_axes([.550,.045,.335,.155]),
                   fig.add_axes([.895,.045,.010,.155]),selections)
    off=text_audit(fig)
    base=args.output_dir/STEM
    for ext in ['pdf','svg','png']:
        fig.savefig(base.with_suffix('.'+ext),dpi=600,facecolor='white',bbox_inches=None)
    # Keep the empirical values and coordinate audit next to each export.
    active.to_csv(args.output_dir/'planning_outcomes_source.csv',index=False)
    assert hashes()==before,'Input changed'
    report=dict(figure='10',panels=8,basemap=False,source_sha256=before,
                sources_unchanged=len(before),font=FONT_FAMILY,width_mm=183,height_mm=242,
                map_panels=maps,bar_values_verified=records,selection_jaccard=matrix,
                heatmap_internal_lines=False,off_canvas_text=off,
                note='No uncertainty added. Shades identify K, not confidence intervals.',
                output_sha256={p.name:sha(p) for p in args.output_dir.glob(STEM+'.*')})
    (args.output_dir/'planning_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.close(fig)
    print(base.with_suffix('.png'))


if __name__=='__main__':main()
