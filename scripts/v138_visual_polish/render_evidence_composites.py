"""Source-native composites for current manuscript Figs. 12 and 16.

Redraws from immutable source data, not resized screenshots. Existing estimators,
map breaks and distribution helpers are reused. Document integration is separate;
this entry point writes only plots, statistics and an audit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v5_cities_visual_system import configure, FONT_FAMILY, clean_axis
from presentation_finish import finish
import fig04_four_year_sdi as sdi
import fig07_joint_quality_access as joint
import fig08_weighted_rainclouds as rain

OUT = ROOT / "figures/v45_evidence_composites"
HEATMAP_DIVIDERS = False
HEATMAP_GRID = True
YEARS = [2011, 2016, 2021, 2024]
ACCESS_YEARS = [2016, 2021, 2024]
INK = "#111111"
RED, BLUE, GREEN, ORANGE = "#B74652", "#367FAD", "#428B54", "#D58236"
PURPLE, TEAL = "#7858A4", "#287F79"
SPEC = [
    ("equal_six_strict", "Six components", RED, "o", "-"),
    ("equal_five_no_practice", "Without practice", BLUE, "s", "--"),
    ("equal_six_highcoverage", "High-coverage subset", GREEN, "D", "-."),
]
COMPONENTS = ["nutrition_score", "carbon_score", "diversity_score",
              "sustainability_score", "hygiene_score", "practice_score"]
NAMES = ["Nutrition", "Carbon", "Diversity", "Environment", "Hygiene", "Practice"]
AUDIT = {"status": "V45_SOURCE_DATA_REDRAW", "font_family": FONT_FAMILY,
         "source_data_unchanged": None, "figures": {}}


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "source_data").rglob("*")) if p.is_file()}


def style():
    configure()
    mpl.rcParams.update({"font.size": 7.3, "axes.labelsize": 7.3,
        "axes.titlesize": 8.1, "axes.titleweight": "normal",
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "savefig.bbox": None,
        "axes.linewidth": .6, "xtick.major.width": .6,
        "ytick.major.width": .6, "pdf.fonttype": 42, "svg.fonttype": "none"})


def heading(ax, letter, title):
    ax.set_title(title, loc="left", pad=7, fontsize=8.1, fontweight="normal")
    ax.text(-.09, 1.025, letter, transform=ax.transAxes, va="bottom",
            ha="right", fontsize=9.2, fontweight="bold", clip_on=False)


def series(frame, metric):
    d = frame.loc[frame.metric.eq(metric)].set_index("year")
    assert not d.index.duplicated().any(), metric
    return d.loc[YEARS]


def map_key_height(fig, ax, cax, frame):
    """Align colour key to the geographic extent, excluding furniture margins."""
    fig.canvas.draw()
    xmin, ymin, xmax, ymax = frame.total_bounds
    low = fig.transFigure.inverted().transform(ax.transData.transform((xmin, ymin)))[1]
    high = fig.transFigure.inverted().transform(ax.transData.transform((xmax, ymax)))[1]
    pos = ax.get_position()
    old = cax.get_position()
    cax.set_position([pos.x1 + .009, low, old.width, high - low])
    cax.tick_params(labelsize=5.9, pad=1.8, length=2, width=.5)
    cax.title.set_fontsize(6.2)
    cax.title.set_fontweight("normal")


def map_check(frame, field, breaks):
    d = frame.loc[pd.to_numeric(frame[field], errors="coerce").notna()].copy()
    if "restaurant_n" in d:
        d = d.loc[d.restaurant_n.gt(0)]
    else:
        d = d.loc[d[field].gt(0) & d.network_available.astype(bool)]
    order = np.argsort(d[field].to_numpy(float))
    rows = d.iloc[order[[0, len(order)//2, len(order)-1]]]
    records = []
    for idx, row in rows.iterrows():
        value = float(row[field])
        records.append({"row": str(idx), "value": value,
                        "class_index": int(np.clip(np.searchsorted(breaks, value, side="right")-1,
                                                   0, len(breaks)-2))})
    return records


def export(fig, stem, content):
    finish(fig, heatmap_dividers=HEATMAP_DIVIDERS, heatmap_grid=HEATMAP_GRID,
           heatmap_grid_linewidth=.5)
    # Reserve 6.2 pt as the floor at the actual 183 mm export width, including
    # scales and colour keys. Do not achieve consolidation by shrinking text.
    for text in fig.findobj(Text):
        if text.get_text().strip() and text.get_fontsize() < 6.2:
            text.set_fontsize(6.2)
    for ax in fig.axes:
        ax.grid(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    # Tick Text objects can have axes=None even when their parent map Axis is
    # switched off. They are not rendered and must not enter the clipping test.
    undrawn_axis_text = set()
    for ax in fig.axes:
        if not ax.axison:
            undrawn_axis_text.update(ax.get_xticklabels(which="both"))
            undrawn_axis_text.update(ax.get_yticklabels(which="both"))
            undrawn_axis_text.update([ax.xaxis.label, ax.yaxis.label,
                                      ax.xaxis.offsetText, ax.yaxis.offsetText])
    for t in fig.findobj(Text):
        if t in undrawn_axis_text or not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:
            continue
        box = t.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            clipped.append({"text": t.get_text(), "bbox": list(box.bounds)})
    if clipped:
        raise RuntimeError(f"Off-canvas text in {stem}: {clipped}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches=None, facecolor="white")
    content.update({"dimensions_mm": list(fig.get_size_inches()*25.4),
                    "heatmap_column_dividers": [ax._revision_column_divider_report
                        for ax in fig.axes if hasattr(ax, "_revision_column_divider_report")],
                    "heatmap_cell_borders": [ax._revision_cell_border_report
                        for ax in fig.axes if hasattr(ax, "_revision_cell_border_report")],
                    "off_canvas_visible_text": clipped,
                    "axes": [{"title": ax.get_title(loc="left"),
                              "bounds": list(ax.get_position().bounds)} for ax in fig.axes],
                    "outputs": {ext: hashlib.sha256((OUT/f"{stem}.{ext}").read_bytes()).hexdigest()
                                for ext in ("pdf", "svg", "png")}})
    AUDIT["figures"][stem] = content
    plt.close(fig)
    print(OUT / f"{stem}.png", flush=True)


def sdi_composite():
    style()
    base = sdi.base
    base.YEAR_RAMPS.update(sdi.MAP_RAMPS)
    frames = base.load_maps()
    breaks = base.pooled_quintiles(frames)
    datadir = ROOT / "source_data/figS_temporal_uncertainty_v4"
    annual = pd.read_csv(datadir / "annual_block_bootstrap_intervals.csv")
    changes = pd.read_csv(datadir / "change_from_2011_intervals.csv")
    coverage = pd.read_csv(datadir / "metric_coverage_audit.csv")
    bounds = np.array([frames[y].total_bounds for y in YEARS])
    common = np.array([bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()])
    center = (common[:2] + common[2:])/2
    half = (common[2:] - common[:2])/2/.9
    scaled = np.r_[center-half, center+half]
    fig = plt.figure(figsize=(183/25.4, 245/25.4), facecolor="white")
    map_audit = {}
    for index, year in enumerate(YEARS):
        row, col = divmod(index, 2)
        x, y = [.075, .555][col], [.758, .525][row]
        ax = fig.add_axes([x, y, .345, .210])
        cax = fig.add_axes([x+.36, y+.03, .012, .16])
        base.draw_map(ax, cax, frames[year], year, chr(97+index), breaks, scaled)
        for t in list(ax.texts):
            if t.get_text() == chr(97+index):
                t.set_position((-.035, 1.035))
                t.set_va("bottom")
            if t.get_text() == str(year):
                t.set_position((.5, 1.035)); t.set_va("bottom")
                t.set_fontweight("normal"); t.set_fontsize(8.4)
        map_key_height(fig, ax, cax, frames[year])
        map_audit[str(year)] = map_check(frames[year], base.FIELD, breaks)
    fig.legend(handles=[
        Patch(facecolor="#F3F5F6", edgecolor="#BAC4C9", label="Incomplete components"),
        Patch(facecolor="white", edgecolor="#C8D0D4", label="No listed outlet")],
        loc="center", bbox_to_anchor=(.51, .512), ncol=2, fontsize=6.3,
        handlelength=1.1, frameon=False)

    # One temporal panel replaces the overlapping V40 5e and 15a.
    ax = fig.add_axes([.105, .294, .345, .164])
    for metric, label, color, marker, ls in SPEC:
        d = series(annual, metric)
        ax.fill_between(YEARS, d.ci_low, d.ci_high, color=color, alpha=.12, lw=0)
        ax.plot(YEARS, d.population_weighted_mean, color=color, marker=marker, ls=ls,
                lw=1.25, ms=3.7, mec="white", mew=.45, label=label)
    ax.set(xticks=YEARS, xlim=(2010.3, 2024.7), ylim=(.34, .54),
           ylabel="Population-weighted SDI", xlabel="Year")
    ax.legend(loc="upper left", fontsize=6.1, labelspacing=.3, handlelength=1.8)
    clean_axis(ax); heading(ax, "e", "SDI across specifications")

    # Same two coverage series as V40 5g/15b; display once.
    ax = fig.add_axes([.605, .294, .33, .164])
    for metric, label, color, marker, ls in (SPEC[0], SPEC[2]):
        d = series(coverage, metric)
        val = 100*d.population_coverage_share.to_numpy(float)
        ax.plot(YEARS, val, color=color, lw=7.5, alpha=.12, ls=ls, zorder=1)
        ax.plot(YEARS, val, color=color, lw=1.25, marker=marker, ms=3.7, ls=ls,
                mec="white", mew=.45, label=label, zorder=2)
    ax.set(xticks=YEARS, xlim=(2010.3, 2024.7), ylim=(0, 100), yticks=[0,25,50,75,100],
           ylabel="Population covered (%)", xlabel="Year")
    ax.legend(loc="lower right", fontsize=6.1, handlelength=1.8)
    clean_axis(ax); heading(ax, "f", "Observation coverage")

    ax = fig.add_axes([.185, .065, .265, .165])
    order = COMPONENTS + [s[0] for s in SPEC]
    names = NAMES + [s[1] for s in SPEC]
    work = changes.loc[changes.year.eq(2024)].set_index("metric").loc[order]
    colors = [ORANGE, BLUE, PURPLE, GREEN, RED, TEAL] + [s[2] for s in SPEC]
    ypos = np.arange(9)[::-1]
    for yy, (_, r), color in zip(ypos, work.iterrows(), colors):
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=color, lw=4.0, alpha=.15,
                solid_capstyle="round", zorder=1)
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=color, lw=.9, zorder=2)
        ax.scatter(r.difference, yy, s=17, c=color, ec="white", lw=.4, zorder=3)
    ax.axvline(0, color="#879299", lw=.6, ls="--", zorder=0)
    ax.axhline(2.5, color="#D5DADD", lw=.5)
    ax.set(yticks=ypos, yticklabels=names, xlim=(-.065,.19), ylim=(-.6,8.6),
           xticks=[-.05,0,.05,.10,.15], xlabel="Change, 2024 minus 2011")
    ax.tick_params(axis="y", length=0, pad=3, labelsize=6.3)
    ax.spines["left"].set_visible(False)
    clean_axis(ax); heading(ax, "g", "Component and index changes")

    ax = fig.add_axes([.605, .065, .29, .165])
    matrix = np.array([100*series(coverage, m).population_coverage_share.to_numpy(float)
                       for m in COMPONENTS])
    cmap = LinearSegmentedColormap.from_list("coverage_soft", ["#B7D5E7", "#F5F6F5", "#E6A29C"])
    im = ax.imshow(matrix, aspect="auto", vmin=50, vmax=85, cmap=cmap, interpolation="nearest")
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=6.8)
    ax.set(xticks=range(4), xticklabels=YEARS, yticks=range(6), yticklabels=NAMES)
    ax.tick_params(length=0, pad=4)
    heading(ax, "h", "Component coverage (%)")
    cax = fig.add_axes([.91,.065,.012,.165])
    cb = fig.colorbar(im, cax=cax, ticks=[50,60,70,80,85]); cb.ax.tick_params(labelsize=6.2)
    export(fig, "Fig12_Restaurant_Quality_SDI_Evolution_Coverage", {
        "manuscript_figure": 12, "legacy_source_figures": [5,15], "panels": 8,
        "panel_map": {"a-d":"V40 Fig5a-d", "e":"Fig5e + Fig15a (all three specifications)",
                      "f":"Fig5g + Fig15b (identical coverage series)",
                      "g":"Fig5f + Fig15c (six components and three index specifications)",
                      "h":"Fig15d"},
        "map_breaks": breaks.tolist(), "map_crs":"EPSG:2326", "map_samples":map_audit,
        "interpretation":"Area quality only; 2011 is not a walking-network observation.",
        "intervals":"Existing 999-replicate DCCA-block-bootstrap intervals; no new estimates.",
        "coverage_values":matrix.tolist()})


def joint_composite():
    style()
    # Keep year semantics aligned with the four-year SDI composite.
    joint.YEAR_CMAPS.update({
        2016: ListedColormap(["#B3D5EF", "#83BCE1", "#529DCD", "#3277AA", "#174C75"]),
        2021: ListedColormap(["#B9DFB8", "#89CA93", "#58AE72", "#348C58", "#175E40"]),
        2024: ListedColormap(["#F7D3AC", "#F2B775", "#E99446", "#C76A29", "#984019"]),
    })
    rain.YEAR_FAMILIES.update({2016:("#9FC7E5", "#285F8C"),
                              2021:("#A7D6AC", "#337343"),
                              2024:("#F0C395", "#BC682B")})
    points, intervals, network, maps = joint.load_data()
    access = pd.read_csv(joint.DATA / "lsbg_joint_quality_affordable_access.csv")
    # Exact equality of the input estimates used by the two old displays.
    check = []
    for year in ACCESS_YEARS:
        for outcome in ["low_price_access", "joint_access"]:
            for metric in ["weighted_gini", "zero_access_population_share", "concentration_index"]:
                value = joint.select_point(points, year, outcome, metric)
                ivalue, lo, hi = joint.interval(intervals, year, outcome, metric)
                assert np.isclose(value, ivalue, atol=1e-12, rtol=1e-12), (year,outcome,metric,value,ivalue)
                check.append(dict(year=year,outcome=outcome,metric=metric,value=value,low=lo,high=hi))
    fig = plt.figure(figsize=(183/25.4, 225/25.4), facecolor="white")
    xs = [.095, .405, .715]
    map_audit = {}
    for i, year in enumerate(ACCESS_YEARS):
        ax = fig.add_axes([xs[i]-.050,.755,.265,.223])
        cax = fig.add_axes([xs[i]+.225,.763,.011,.170])
        joint.map_panel(ax,cax,maps[year],year)
        ax.set_title(str(year),loc="center",y=1.015,pad=0,fontsize=8.5,fontweight="normal")
        ax.text(-.025,1.015,chr(97+i),transform=ax.transAxes,ha="right",va="bottom",
                fontsize=9.2,fontweight="bold")
        map_key_height(fig,ax,cax,maps[year])
        map_audit[str(year)] = map_check(maps[year],"joint_access",joint.BREAKS)
    fig.legend(handles=[
        Patch(facecolor="#E7EEF2",ec="#CDD6DB",label="Quality gate unmet"),
        Patch(facecolor="#E9EDF0",ec="#C9D0D4",label="No low-price access"),
        Patch(facecolor="white",ec="#AEB8BE",hatch="////",label="Network unavailable")],
        loc="center",bbox_to_anchor=(.51,.735),ncol=3,fontsize=6.2,
        columnspacing=.9,handlelength=1.1)

    box_stats=[]
    for i,year in enumerate(ACCESS_YEARS):
        ax=fig.add_axes([xs[i],.518,.245,.185])
        box_stats.extend(rain.box_panel(ax,access,year,chr(100+i)))
        for t in list(ax.texts):
            if t.get_text()==chr(100+i): t.remove()
        ax.set_title("")
        heading(ax,chr(100+i),str(year))
        ax.set_xlabel("Income population quintile",fontsize=6.8,labelpad=4)
        ax.tick_params(labelsize=6.3)
        if i==0:
            ax.set_ylabel("Reachable opportunities\n(15 min; log-scaled axis)",fontsize=7.0)
        else:
            ax.set_yticklabels([])
        light,dark=rain.YEAR_FAMILIES[year]
        # Each panel gets a horizontal two-item key in its own reserved lane.
        ax.legend(handles=[Patch(fc=light,ec=light,label="≤HK$100"),
                           Patch(fc=dark,ec=dark,label="Joint")],loc="lower right",
                  bbox_to_anchor=(1.02,1.025),ncol=2,fontsize=6.0,handlelength=1,
                  handletextpad=.4,columnspacing=.7,borderaxespad=0)

    # Distribution and inequality are complementary: retain both, but do not
    # repeat these same three metrics as another forest plot elsewhere.
    families=[("weighted_gini","Gini",1.,False,ORANGE,"o"),
              ("zero_access_population_share","Zero access (%)",100.,False,PURPLE,"s"),
              ("concentration_index","Income concentration",1.,True,TEAL,"D")]
    for i,(metric,title,scale,zero,color,marker) in enumerate(families):
        ax=fig.add_axes([xs[i],.278,.245,.150])
        for outcome,ls,shade in [("low_price_access","--",.5),("joint_access","-",1.)]:
            d=[joint.interval(intervals,year,outcome,metric) for year in ACCESS_YEARS]
            vals=np.array(d)*scale
            c=tuple(np.array(mpl.colors.to_rgb(color))*shade + (1-shade)*np.ones(3))
            ax.fill_between(ACCESS_YEARS,vals[:,1],vals[:,2],color=c,alpha=.16,lw=0)
            ax.plot(ACCESS_YEARS,vals[:,0],c=c,ls=ls,marker=marker,ms=3.6,lw=1.25,
                    mec="white",mew=.45,label="Joint" if outcome=="joint_access" else "≤HK$100")
        if zero: ax.axhline(0,c="#879299",lw=.6,ls="--",zorder=0)
        ax.set(xticks=ACCESS_YEARS,xlim=(2015.5,2024.5),xlabel="Year")
        ax.tick_params(labelsize=6.3)
        clean_axis(ax); heading(ax,chr(103+i),title)
        ax.legend(loc="upper right",fontsize=6.0,handlelength=1.5,labelspacing=.25)

    ax=fig.add_axes([.105,.055,.350,.160])
    ybase=np.arange(3)[::-1]
    for yy,year in zip(ybase,ACCESS_YEARS):
        nd=network.loc[network.year.eq(year)&network.outcome.eq("opportunity_n_all_valid_tiers")]
        assert len(nd)==1
        all_mean=float(nd.iloc[0].population_weighted_mean)
        low=joint.select_point(points,year,"low_price_access","population_weighted_mean")
        value=joint.select_point(points,year,"joint_access","population_weighted_mean")
        ax.plot([value,all_mean],[yy,yy],c="#CCD3D7",lw=1.8,zorder=0)
        for val,c,m in [(all_mean,"#87929A","o"),(low,BLUE,"s"),(value,RED,"D")]:
            ax.scatter(val,yy,s=24,c=c,marker=m,ec="white",lw=.4,zorder=2)
        ax.text(value,yy-.30,f"{100*value/low:.0f}% retained",fontsize=6.3,ha="left",va="center")
    ax.set(yticks=ybase,yticklabels=ACCESS_YEARS,ylim=(-.5,2.9),xlim=(0,290),
           xlabel="Population-weighted mean opportunities")
    ax.tick_params(axis="y",length=0)
    clean_axis(ax); heading(ax,"j","Sequential opportunity screen")
    ax.legend(handles=[Line2D([],[],marker=m,c=c,ls="none",label=lab,ms=3.6)
                       for m,c,lab in [("o","#87929A","All valid-price"),("s",BLUE,"≤HK$100"),("D",RED,"Joint")]],
              loc="upper right",ncol=3,fontsize=5.9,handletextpad=.15,columnspacing=.7)

    ax=fig.add_axes([.610,.055,.275,.160])
    subset=points.loc[points.outcome.eq("joint_access_sensitivity") & points.threshold_min.eq(15.)
                      & points.price_ceiling_hkd.eq(100)]
    matrix=subset.pivot(index="year",columns="destination_area_sdi_threshold",
                        values="zero_access_population_share").loc[ACCESS_YEARS,[.4,.45,.5]]*100
    means=subset.pivot(index="year",columns="destination_area_sdi_threshold",
                       values="population_weighted_mean").loc[ACCESS_YEARS,[.4,.45,.5]]
    cmap=LinearSegmentedColormap.from_list(
        "unsigned_blue_red", ["#78A9C7","#D3E4EC","#FAFAFA","#F1D0D2","#E1848C"]
    )
    im=ax.imshow(matrix.to_numpy(),aspect="auto",vmin=10,vmax=70,cmap=cmap,interpolation="nearest")
    for i in range(3):
        for j in range(3):
            ax.text(j,i-.12,f"{matrix.iloc[i,j]:.0f}%",ha="center",va="center",fontsize=7.0)
            ax.text(j,i+.22,f"n̄={means.iloc[i,j]:.0f}",ha="center",va="center",fontsize=6.0)
    ax.set(xticks=range(3),xticklabels=["0.40","0.45","0.50"],yticks=range(3),
           yticklabels=ACCESS_YEARS,xlabel="Destination-area SDI threshold")
    ax.tick_params(length=0,pad=4)
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=.25,alpha=.75)
    ax.tick_params(which="minor",bottom=False,left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#202020")
        spine.set_linewidth(.55)
    heading(ax,"k","Threshold sensitivity")
    cax=fig.add_axes([.901,.055,.012,.160])
    cb=fig.colorbar(im,cax=cax,ticks=[10,30,50,70]); cax.set_title("Zero (%)",fontsize=6.1,pad=4)
    cb.outline.set_edgecolor("#202020")
    cb.outline.set_linewidth(.55)
    cb.ax.tick_params(labelsize=6.2)
    stats=pd.DataFrame(box_stats).drop(columns=["label","fliers"])
    # Recalculate original box summaries directly; a clean checkout does not
    # require prior image outputs to reproduce this figure.
    old_rows=[]
    for year in ACCESS_YEARS:
        slices=rain.population_quintile_slices(access.loc[access.year.eq(year)])
        for outcome in ["low_price_access", "joint_access"]:
            for q in range(1,6):
                old_rows.append({"year":year,"income_quintile":q,"outcome":outcome,
                                 **rain.weighted_box_stats(slices[q],outcome)})
    old=pd.DataFrame(old_rows)
    keys=["year","income_quintile","outcome"]
    a=stats.sort_values(keys).reset_index(drop=True)
    b=old.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys])
    for col in ["whislo","q1","med","q3","whishi"]:
        np.testing.assert_allclose(a[col],b[col],atol=1e-12,rtol=1e-12)
    stats.to_csv(OUT/"Fig16_weighted_box_statistics.csv",index=False)
    export(fig,"Fig16_Joint_Quality_Walking_Price_Opportunity",{
        "manuscript_figure":16,"legacy_source_figures":[8,9],"panels":11,
        "panel_map":{"a-c":"Fig8a-c maps", "d-f":"Fig9a-c weighted distributions",
                     "g-i":"Fig9d-f; same metrics as Fig8e shown only once",
                     "j":"Fig8d sequential screen", "k":"Fig8f thresholds"},
        "map_breaks":joint.BREAKS.tolist(),"map_crs":"EPSG:2326","map_samples":map_audit,
        "weighted_box_statistics_match_native_estimator":True,"weighted_box_rows":len(stats),
        "point_and_interval_agreement":check,
        "interpretation":"15-min network + platform price <=HK$100 + destination LSBG SDI >=0.45; not outlet-level SDI or observed household affordability.",
        "intervals":"Existing DCCA-block-bootstrap intervals. Distribution widths use unchanged weighted Scott KDE in log10(1+count) space."})


def main():
    global OUT, HEATMAP_DIVIDERS, HEATMAP_GRID
    p=argparse.ArgumentParser(); p.add_argument("figure",choices=["A","B","both"],default="both",nargs="?")
    p.add_argument("--output-dir", type=Path)
    treatment=p.add_mutually_exclusive_group()
    treatment.add_argument("--heatmap-dividers", action="store_true")
    treatment.add_argument("--heatmap-grid", action="store_true", default=True)
    args=p.parse_args()
    if args.output_dir is not None:
        OUT=args.output_dir.resolve()
    HEATMAP_DIVIDERS=args.heatmap_dividers
    HEATMAP_GRID=args.heatmap_grid
    OUT.mkdir(parents=True,exist_ok=True)
    before=source_hashes()
    if args.figure in ("A","both"): sdi_composite()
    if args.figure in ("B","both"): joint_composite()
    assert before==source_hashes(),"Source data changed"
    AUDIT["source_data_unchanged"]=len(before)
    AUDIT["source_sha256"]=before
    (OUT/f"composite_audit_{args.figure}.json").write_text(json.dumps(AUDIT,indent=2),encoding="utf-8")


if __name__=="__main__": main()
