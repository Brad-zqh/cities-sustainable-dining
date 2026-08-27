"""Full manuscript presentation pass, with immutable input/source checks.

V44 uses equal 0.5-pt inner/outer heatmap rules. Source estimators, intervals,
exclusions, bins, colours and limits are not changed by the finishing pass.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import render_base as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'figures/current'
AUDIT = ROOT/'audit/render_final'


def numeric_signature(fig):
    """Record the empirical primitives, not decorative grid/halo artists."""
    values=[]
    for ax in fig.axes:
        values.append(repr((ax.get_xlim(), ax.get_ylim())))
        for im in ax.images:
            values.append(np.asarray(im.get_array()).tobytes().hex())
            values.append(repr((im.get_clim(), im.get_extent())))
        for line in ax.lines:
            values.append(repr((list(line.get_xdata()), list(line.get_ydata()))))
        for collection in ax.collections:
            if collection.get_gid() in ('heatmap-column-dividers','heatmap-row-dividers'):
                continue
            values.append(np.asarray(collection.get_offsets()).tobytes().hex())
            if collection.get_array() is not None:
                values.append(np.asarray(collection.get_array()).tobytes().hex())
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def render(number):
    sys.path.insert(0, str(ROOT/'scripts'))
    import matplotlib as mpl
    import presentation_finish
    original_finish=presentation_finish.finish
    def finish(fig, **kwargs):
        before=numeric_signature(fig)
        original_finish(fig, **kwargs, heatmap_grid=True, heatmap_grid_linewidth=.50)
        if not getattr(fig,'_v44_layout_finished',False):
            # Only typography/legend placement is adjusted here. Keep the
            # existing common limits and all plotted empirical values.
            for ax in fig.axes:
                for title in (ax.title, ax._left_title, ax._right_title):
                    title.set_fontweight('normal')
            if number==4:
                for ax in fig.axes:
                    if ax.get_xlabel()=='Agreement':
                        ax.get_legend().set_bbox_to_anchor((.06,.85))
                    if ax.get_ylabel()=='FEHD minus OpenRice':
                        for txt in ax.get_legend().get_texts(): txt.set_fontsize(6.2)
            if number==9:
                for txt in fig.findobj(mpl.text.Text):
                    if txt.get_text() in ('Joint','≤HK$100'):
                        txt.set_fontweight('normal')
            if number==10:
                for ax in fig.axes[:6]:
                    legend=ax.get_legend()
                    if legend:
                        for txt in legend.get_texts(): txt.set_fontsize(5.7)
                        legend.get_title().set_fontsize(6.0)
                        if ax.get_title(loc='left')=='Ethnicity':
                            legend.set_bbox_to_anchor((.71,.995))
            fig._v44_layout_finished=True
        after=numeric_signature(fig)
        assert before==after, f'Empirical artists changed in Fig {number}'
        fig.canvas.draw()
        report={
            'figure':number, 'source_data_sha256':legacy.hashes(),
            'presentation_preserves_empirical_artists':True,
            'numeric_artist_sha256':after,
            'heatmaps':[a._revision_cell_border_report for a in fig.axes
                        if hasattr(a,'_revision_cell_border_report')],
            'axes':[{'bounds':list(a.get_position().bounds),
                     'title':a.get_title(loc='left') or a.get_title(),
                     'xlabel':a.get_xlabel(), 'ylabel':a.get_ylabel()} for a in fig.axes],
            'nonblack_text':[t.get_text() for t in fig.findobj(mpl.text.Text)
                             if t.get_text() and max(mpl.colors.to_rgb(t.get_color()))-
                             min(mpl.colors.to_rgb(t.get_color()))>1e-6],
        }
        assert not report['nonblack_text']
        (AUDIT/f'fig{number:02d}_artists.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    presentation_finish.finish=finish
    legacy.OUT=OUT
    legacy.render(number)


def main():
    OUT.mkdir(parents=True,exist_ok=True); AUDIT.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:
        render(int(sys.argv[1])); return
    before=legacy.hashes(); runs=[]
    for number in legacy.JOBS:
        p=subprocess.run([sys.executable,'-X','utf8',__file__,str(number)],
                         capture_output=True,text=True,encoding='utf-8',cwd=ROOT)
        runs.append(dict(figure=number,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        (AUDIT/'export_log.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        print(f'Fig. {number}: {p.returncode}',flush=True)
        if p.returncode: raise RuntimeError(p.stderr)
    assert before==legacy.hashes(), 'Source data changed'
    (AUDIT/'source_and_export_checks.json').write_text(json.dumps({
        'source_files_unchanged':len(before), 'source_sha256':before,
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},
        'heatmap_inner_outer_linewidth_pt':.50,
    },indent=2),encoding='utf-8')
    print(f'PASS: {len(runs)} figures, {len(before)} unchanged source files',flush=True)


if __name__=='__main__': main()
