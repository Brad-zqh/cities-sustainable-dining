"""Presentation-only finishing pass; never changes values or estimators.

Called explicitly by the revision exporter immediately before serialization.
Legacy source-column names stay intact; only displayed component labels change.
"""
import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

COMPONENT_NAMES = {
    'Cuisine diversity': 'Diversity', 'Cuisine\ndiversity': 'Diversity',
    'Environmental sustainability': 'Environment',
    'Environmental\nsustainability': 'Environment',
    'Sustainability text': 'Environment', 'Sustainability signal': 'Environment',
    'Practice tag': 'Practice', 'Practice tag*': 'Practice', 'Practice*': 'Practice',
}

def heatmap_column_dividers(ax, *, linewidth=.30, alpha=.40):
    """Add boundary-only vertical rules to a regular matrix, not its colourbar.

    Opt-in presentation treatment. Data, limits, normalization and annotations
    are untouched. Image extents and nonuniform rectangular QuadMesh cell edges
    are respected; RGB images and curvilinear meshes are not treated as tables.
    Repeated export calls update the same artist rather than darkening the rules.
    """
    if getattr(ax, '_colorbar', None) is not None:
        return None
    images = [im for im in ax.images if np.asarray(im.get_array()).ndim == 2]
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    if len(images) + len(meshes) != 1:
        return None
    if images:
        artist = images[0]
        nrows, ncols = np.shape(artist.get_array())
        left, right, bottom, top = artist.get_extent()
        edges = np.linspace(left, right, ncols + 1)[1:-1]
        source = 'AxesImage'
    else:
        artist = meshes[0]
        coords = artist.get_coordinates()
        if (not np.allclose(coords[:, :, 0], coords[0:1, :, 0]) or
                not np.allclose(coords[:, :, 1], coords[:, 0:1, 1])):
            return None
        nrows, ncols = coords.shape[0]-1, coords.shape[1]-1
        edges = coords[0, 1:-1, 0]
        bottom, top = coords[0, 0, 1], coords[-1, 0, 1]
        source = 'QuadMesh'
    segments = [[(float(x), float(bottom)), (float(x), float(top))] for x in edges]
    divider = getattr(ax, '_revision_column_divider_artist', None)
    if divider is None:
        divider = LineCollection(segments, colors='black', linewidths=linewidth,
                                 alpha=alpha, zorder=artist.get_zorder()+.1,
                                 capstyle='butt', clip_on=True)
        divider.set_gid('heatmap-column-dividers')
        ax.add_collection(divider, autolim=False)
        ax._revision_column_divider_artist = divider
    else:
        divider.set_segments(segments)
        divider.set_linewidth(linewidth)
        divider.set_alpha(alpha)
    report = dict(source=source, rows=int(nrows), columns=int(ncols),
                  vertical_rules=len(segments), x_boundaries=list(map(float, edges)),
                  linewidth_pt=linewidth, alpha=alpha)
    ax._revision_column_divider_report = report
    return report


def heatmap_cell_borders(ax, *, linewidth=.25):
    """Close all four sides of each matrix cell with opaque thin black rules.

    Shared cell boundaries are drawn once; the existing outer frame closes the
    perimeter. No colourbar internals or annotation values are modified.
    """
    columns = heatmap_column_dividers(ax, linewidth=linewidth, alpha=1.0)
    if columns is None:
        return None
    if columns['source'] == 'AxesImage':
        artist = next(im for im in ax.images if np.asarray(im.get_array()).ndim == 2)
        left, right, bottom, top = artist.get_extent()
        edges = np.linspace(bottom, top, columns['rows']+1)[1:-1]
    else:
        artist = next(c for c in ax.collections if isinstance(c, QuadMesh))
        coords = artist.get_coordinates()
        left, right = coords[0, 0, 0], coords[0, -1, 0]
        edges = coords[1:-1, 0, 1]
    segments = [[(float(left), float(y)), (float(right), float(y))] for y in edges]
    divider = getattr(ax, '_revision_row_divider_artist', None)
    if divider is None:
        divider = LineCollection(segments, colors='black', linewidths=linewidth,
                                 alpha=1.0, zorder=artist.get_zorder()+.1,
                                 capstyle='butt', clip_on=True)
        divider.set_gid('heatmap-row-dividers')
        ax.add_collection(divider, autolim=False)
        ax._revision_row_divider_artist = divider
    else:
        divider.set_segments(segments)
        divider.set_linewidth(linewidth)
    report = {**columns, 'horizontal_rules':len(segments),
              'y_boundaries':list(map(float, edges)), 'outer_frame_pt':.5,
              'treatment':'all four cell sides; shared edges drawn once'}
    ax._revision_cell_border_report = report
    return report


def finish(fig, *, halos=False, heatmap_dividers=False, heatmap_grid=False,
           heatmap_grid_linewidth=.25):
    """Black typography, thin outer frames, and optional heat-map column rules."""
    for text in fig.findobj(mpl.text.Text):
        text.set_text(COMPONENT_NAMES.get(text.get_text(), text.get_text()))
        text.set_color('#111111')
        if text.get_text() == 'N':
            text.set_fontweight('normal')
    for ax in fig.findobj(mpl.axes.Axes):
        is_colourbar = hasattr(ax, '_colorbar') and ax._colorbar is not None
        if is_colourbar:
            ax._colorbar.outline.set_visible(True)
            ax._colorbar.outline.set_edgecolor('black')
            ax._colorbar.outline.set_linewidth(.5)
        is_heatmap = (not is_colourbar and (
            any(np.asarray(im.get_array()).ndim == 2 for im in ax.images)
            or any(isinstance(c, QuadMesh) for c in ax.collections)))
        if is_heatmap and not getattr(ax, '_revision_outer_frame', False):
            # An axes-space frame also works when a heat-map has axis('off').
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor='none', edgecolor='black', lw=.5,
                                   clip_on=False, zorder=50))
            ax._revision_outer_frame = True
        if is_heatmap and heatmap_grid:
            heatmap_cell_borders(ax, linewidth=heatmap_grid_linewidth)
        elif is_heatmap and heatmap_dividers:
            heatmap_column_dividers(ax)
        if halos and not is_heatmap and not is_colourbar:
            for item in ax.collections:
                if isinstance(item, PathCollection) and len(item.get_facecolors()):
                    colour = tuple(item.get_facecolors()[0, :3]) + (.14,)
                    item.set_path_effects([pe.withStroke(linewidth=3.8, foreground=colour)])
            for line in ax.lines:
                if line.get_linestyle() in ('-', 'solid') and len(line.get_xdata()) > 1:
                    colour = mpl.colors.to_rgba(line.get_color(), .13)
                    line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth()+3.0,
                                                     foreground=colour), pe.Normal()])
