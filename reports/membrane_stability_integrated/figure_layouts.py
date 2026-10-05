#!/usr/bin/env python3
"""Portrait layouts of Figures 34 and 36 from archived arrays; no simulations.

Run from a repository checkout with numpy and matplotlib installed. Only the
report's figures/layout directory is written; original research PNGs stay intact.
Rendering, triangulations, fields, colour maps, limits, and annotations follow
Ruben/membrane_stability/plot_flow_results.py. Original colour-bar and axis ticks
are retained; only panel positions, font size, spacing, and title wrapping vary.
"""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[1]
RESULTS = REPO / 'Ruben/membrane_stability/results'
OUTPUT = REPORT / 'figures/layout'
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'savefig.dpi': 280, 'savefig.bbox': 'tight'})


def triangulation(data, hole=False):
    if 'triangles' in data:
        return mtri.Triangulation(data['x'], data['y'], data['triangles'])
    tri = mtri.Triangulation(data['x'], data['y'])
    if hole:
        xc = data['x'][tri.triangles].mean(1)
        yc = data['y'][tri.triangles].mean(1)
        tri.set_mask(np.hypot(xc - 50, yc - 50) < 1)
    return tri


def field(fig, ax, tri, values, title, cmap, limit, shrink, xlabel=False):
    options = {} if cmap == 'viridis' else {'vmin': -limit, 'vmax': limit}
    image = ax.tripcolor(tri, values, shading='gouraud', cmap=cmap, **options)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, pad=7)
    if xlabel:
        ax.set_xlabel('x / r0')
    return fig.colorbar(image, ax=ax, shrink=shrink, pad=.04, fraction=.047)


def archived_ticks(fig, axes, tri, fields, shrink, xlabel=False):
    """Read ticks from the original wide grid; retain the displayed values."""
    rendered = []
    for ax, (values, title, cmap, limit) in zip(axes, fields):
        # The original code used matplotlib's default colour-bar fraction/pad.
        options = {} if cmap == 'viridis' else {'vmin': -limit, 'vmax': limit}
        image = ax.tripcolor(tri, values, shading='gouraud', cmap=cmap, **options)
        ax.set_aspect('equal')
        if xlabel:
            ax.set_xlabel('x / r0')
        cb = fig.colorbar(image, ax=ax, shrink=shrink)
        ax.tick_params(labelsize=11)
        cb.ax.tick_params(labelsize=11)
        rendered.append((ax, cb))
    # Aspect-ratio adjustment happens on draw and sets the available tick spacing.
    fig.canvas.draw()
    specs = [(ax.get_xticks(), ax.get_yticks(), ax.get_xlim(), ax.get_ylim(), cb.get_ticks())
             for ax, cb in rendered]
    plt.close(fig)
    return specs


def apply_ticks(ax, cb, spec):
    xticks, yticks, xlim, ylim, cticks = spec
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cb.set_ticks(cticks)
    cb.ax.set_ylim(cb.vmin, cb.vmax)
    ax.tick_params(labelsize=8)
    cb.ax.tick_params(labelsize=8)


def mechanism():
    with np.load(RESULTS / 'flow_pre_t0_meshA/mechanism_at_threshold.npz') as data:
        tri = triangulation(data)
        items = [('sigma', 'surface tension sigma', .01, 'RdBu_r'),
                 ('T_min', 'smallest principal stress of sigma g + 2 eta d', .01, 'RdBu_r'),
                 ('mode', "unstable mode z' at SL_c", None, 'RdBu_r'),
                 ('vx', 'velocity v^1', None, 'viridis')]
        fields = [(data[key], title, cmap, limit or np.abs(data[key]).max())
                  for key, title, limit, cmap in items]
        oldfig, oldaxes = plt.subplots(1, 4, figsize=(21, 4.6))
        ticks = archived_ticks(oldfig, oldaxes, tri, fields, .8, xlabel=True)
        fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.65))
        fig.subplots_adjust(left=.065, right=.955, bottom=.07, top=.845,
                            wspace=.40, hspace=.47)
        for ax, (values, title, cmap, limit), spec in zip(axes.flat, fields, ticks):
            cb = field(fig, ax, tri, values, textwrap.fill(title, 34), cmap, limit, .86, xlabel=True)
            apply_ticks(ax, cb, spec)
            ax.title.set_fontsize(10)
            ax.xaxis.label.set_fontsize(10)
            ax.tick_params(labelsize=9)
            cb.ax.tick_params(labelsize=9)
        title = (f"Mechanism at the threshold (t = 0, SL = {float(data['v0']) * 100:.2f}): the drag on the PI is carried by "
                 "a tension gradient, the membrane upstream is compressed (sigma < 0, blue) and buckles")
        fig.text(.5, .98, textwrap.fill(title, 93), ha='center', va='top', fontsize=9)
        fig.savefig(OUTPUT / 'membrane_flow_pre_mechanism.png', facecolor='white', pad_inches=.06)
        plt.close(fig)


def shapes():
    with np.load(RESULTS / 'flow_pre_t-0.3_meshA/leading_modes.npz') as data:
        tri = triangulation(data, hole=True)
        keys = [key for key in ['0', '0.04', '0.08', '0.1', '0.16'] if f'base_{key}' in data]
        assert len(keys) == 5
        # Original ordering: steady shapes across the top, modes across the bottom.
        fields = []
        for row in range(2):
            for key in keys:
                values = data[f'base_{key}' if row == 0 else key]
                title = ('steady shape z*' if row == 0 else "slowest mode z'") + f', SL = {float(key) * 100:g}'
                fields.append((values, title, 'RdBu_r', np.abs(values).max()))
        oldfig, oldaxes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7.5))
        ticks = archived_ticks(oldfig, list(oldaxes.flat), tri, fields, .75)
        fig, axes = plt.subplots(len(keys), 2, figsize=(7.15, 9.25))
        fig.subplots_adjust(left=.065, right=.96, bottom=.035, top=.91,
                            wspace=.37, hspace=.47)
        for row, key in enumerate(keys):
            for col in range(2):
                index = col * len(keys) + row
                values, title, cmap, limit = fields[index]
                ax = axes[row, col]
                cb = field(fig, ax, tri, values, title, cmap, limit, .85)
                apply_ticks(ax, cb, ticks[index])
        title = ('PRE setup with contact angle t = -0.3 (imperfect bifurcation): the flow-induced deformation is the '
                 'buckling mode, growing smoothly')
        fig.text(.5, .985, textwrap.fill(title, 95), ha='center', va='top', fontsize=9)
        fig.savefig(OUTPUT / 'membrane_flow_pre_imperfect_shapes.png', facecolor='white', pad_inches=.06)
        plt.close(fig)


if __name__ == '__main__':
    mechanism()
    shapes()
    print('Two portrait figure layouts rendered from archived arrays.')
