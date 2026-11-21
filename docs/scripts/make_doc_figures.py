#!/usr/bin/env python3
"""The figures and animations of ELISHA's tutorial, from the programs of docs/examples/src.

    bash scripts/docs_figures.sh            builds the examples, runs each with `dump` and calls this script

or by hand, on the data a run of the programs with `dump` wrote into DIR:

    python3 docs/scripts/make_doc_figures.py --data DIR [--out docs/public/tutorial] [--only NAME ...]

Every figure is drawn twice, on the identity's two grounds (elisha_palette.py): `<name>.png` on washi and
`<name>-dark.png` on Kōyō night; the docs theme shows the twin of the reader's theme. Every animation is encoded per
theme as `<name>-<theme>.webm` (VP9) and `.mp4` (H.264), with the last frame as the poster `<name>-<theme>.png`.

Figures (the program and its dump):

    t01-box         t01_box-slice.dat and docs/examples/output/t01_box.txt: the potential, and its error against h
    t02-faces       t02_faces-<pair>.dat: phi along the line for each pair of faces, the ghost cells marked
    t03-isolated    t03_isolated-{line,slice}.dat: the free-space and the periodic potential against the analytic one
    t04-amr         t04_amr-slice.dat: the AMR mesh around the star, the potential's rings over it
    t05-values      t05_values-slice.dat: the box with its heated lid, the mesh refined towards it
    t06-history     docs/examples/output/t06_iterations.txt: the residual per FCG iteration, both V-cycles

Animations:

    t06-iterates    t06_iterations-frames.dat: the error of the k-th iterate of both V-cycles side by side, k = 1
                    to convergence, on a log scale
    t08-embers      t08_embers-frames.dat: the two embers turning, the mesh following them

Needs numpy, scipy, matplotlib, and ffmpeg on PATH (animations). The data are deterministic for a given build, and the
PNGs carry no time stamp.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elisha_palette as pal  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'docs' / 'examples' / 'output'
THEMES = ('light', 'dark')
DPI = 160


# ----------------------------------------------------------------------------------------------------------------------
# style and I/O
# ----------------------------------------------------------------------------------------------------------------------
@contextmanager
def style(theme: str):
    """Matplotlib defaults for one ground: text, axes, ticks and grid in the theme's inks."""
    ink, muted, ground = pal.INK[theme], pal.MUTED[theme], pal.GROUND[theme]
    rc = {
        'figure.facecolor': ground, 'savefig.facecolor': ground, 'axes.facecolor': pal.PANEL[theme],
        'axes.edgecolor': muted, 'axes.labelcolor': ink, 'text.color': ink, 'xtick.color': muted,
        'ytick.color': muted, 'xtick.labelcolor': ink, 'ytick.labelcolor': ink, 'grid.color': pal.MESH[theme],
        'grid.alpha': 0.35, 'grid.linewidth': 0.6, 'legend.frameon': False, 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'lines.markersize': 6,
        'font.family': 'DejaVu Sans', 'svg.hashsalt': 'elisha', 'path.simplify': False,
    }
    with mpl.rc_context(rc):
        yield


def save(fig, out: Path, name: str, theme: str) -> None:
    """Write the figure of one theme: <name>.png (light) or <name>-dark.png."""
    path = out / (f'{name}.png' if theme == 'light' else f'{name}-dark.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.08, metadata={'Software': None})
    plt.close(fig)
    print(f'  {path.relative_to(ROOT)}')


def read_blocks(path: Path):
    """Blocks of a slice dump: a header line (level, x0, y0, h), then the nc x nc values (i fastest)."""
    lines = path.read_text().split('\n')
    blocks = []
    for head, vals in zip(lines[0::2], lines[1::2]):
        if not head.strip():
            continue
        lev, x0, y0, h = head.split()
        v = np.array(vals.split(), dtype=float)
        nc = int(round(np.sqrt(v.size)))
        blocks.append((int(lev), float(x0), float(y0), float(h), v.reshape(nc, nc)))  # i fastest: [j, i]
    return blocks


def field_extent(blocks):
    """The common colour scale of a set of blocks."""
    vals = np.concatenate([b[4].ravel() for b in blocks])
    return vals.min(), vals.max()


def draw_blocks(ax, blocks, theme: str, vmin: float, vmax: float, outlines: bool = True, rings: int = 9,
                levels_alpha: float = 1.0):
    """The potential of a set of blocks as images, its contours (the rings in the ash) and the block outlines."""
    tiny = 1e-6 * max(abs(vmin), abs(vmax))
    if vmin < -tiny and vmax > tiny:  # both signs: diverging, the wells (phi < 0) crimson, zero the ground
        sign, cmap, lim = -1.0, pal.DIVERGING[theme], max(-vmin, vmax)
        lo_c, hi_c = -lim, lim
    else:                             # one sign: sequential on |phi|, the largest the most saturated
        sign = -1.0 if vmax <= tiny else 1.0
        cmap, lo_c, hi_c = pal.EMBER[theme], 0.0, max(abs(vmin), abs(vmax))
    for lev, x0, y0, h, v in blocks:
        n = v.shape[0]
        ax.imshow(sign * v, extent=(x0, x0 + n * h, y0, y0 + n * h), origin='lower', cmap=cmap, vmin=lo_c,
                  vmax=hi_c, interpolation='nearest', zorder=1)
    # contours on the cell centres of every block, interpolated onto the finest spacing
    pts, val = [], []
    for lev, x0, y0, h, v in blocks:
        n = v.shape[0]
        c = x0 + (np.arange(n) + 0.5) * h
        d = y0 + (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(c, d)
        pts.append(np.column_stack([X.ravel(), Y.ravel()]))
        val.append(v.ravel())
    pts, val = np.concatenate(pts), np.concatenate(val)
    hmin = min(b[3] for b in blocks)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + hmin / 2, hmin / 2), np.arange(lo[1], hi[1] + hmin / 2, hmin / 2))
    gz = griddata(pts, val, (gx, gy), method='linear')
    if rings:
        lv = np.linspace(vmin, vmax, rings + 2)[1:-1]
        ax.contour(gx, gy, gz, levels=lv, colors=pal.INK[theme], linewidths=0.8, alpha=0.75, zorder=3,
                   linestyles='solid')
    if outlines:
        for lev, x0, y0, h, v in blocks:
            n = v.shape[0]
            ax.add_patch(Rectangle((x0, y0), n * h, n * h, fill=False, lw=0.5 + 0.35 * lev,
                                   ec=pal.LEVELS[theme][min(lev, 2)], alpha=levels_alpha, zorder=4))


def level_legend(ax, theme: str, levels=(0, 1, 2)) -> None:
    """A legend of the block outlines per level."""
    handles = [Rectangle((0, 0), 1, 1, fill=False, lw=0.5 + 0.35 * lev, ec=pal.LEVELS[theme][lev]) for lev in levels]
    ax.legend(handles, [f'level {lev} blocks' for lev in levels], loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=9, handlelength=1.2)


def table_rows(name: str, pattern: str):
    """The rows of a program's output (docs/examples/output/<name>.txt) that match a pattern, as tuples of groups."""
    text = (OUTPUTS / f'{name}.txt').read_text()
    return [m.groups() for m in re.finditer(pattern, text, re.M)]


def square(ax, lo=0.0, hi=1.0, label=True):
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    if label:
        ax.set_xlabel('x')
        ax.set_ylabel('y')


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def fig_t01(data: Path, out: Path) -> None:
    blocks = read_blocks(data / 't01_box-slice.dat')
    rows = table_rows('t01_box', r'^\s*(\d+)\^3\s+T\s+([0-9.E+-]+)')
    n = np.array([int(r[0]) for r in rows], dtype=float)
    err = np.array([float(r[1]) for r in rows])
    for th in THEMES:
        with style(th):
            fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={'width_ratios': [1, 1.15]})
            vmin, vmax = field_extent(blocks)
            draw_blocks(a, blocks, th, vmin, vmax, outlines=True, rings=8)
            square(a)
            a.set_title('phi on the plane z = 1/2, 32$^3$ cells, 2$^3$ blocks', fontsize=10)
            h = 1.0 / n
            b.loglog(h, err, marker=pal.MARKERS[0], color=pal.LINES[th][0], label='max |phi - phi$_{exact}$| / max |phi$_{exact}$|')
            b.loglog(h, err[0] * (h / h[0]) ** 2, ls='--', lw=1.2, color=pal.MUTED[th], label='slope 2')
            for hi, ei, ni in zip(h, err, n):
                b.annotate(f'{int(ni)}$^3$', (hi, ei), textcoords='offset points', xytext=(6, -12), fontsize=9,
                           color=pal.MUTED[th])
            b.set_xlabel('cell spacing h')
            b.set_ylabel('relative error')
            b.grid(True, which='major')
            b.legend(loc='upper left', fontsize=9)
            b.set_title('second order: the error falls fourfold per halving of h', fontsize=10)
            b.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            fig.tight_layout()
            save(fig, out, 't01-box', th)


def fig_t02(data: Path, out: Path) -> None:
    pairs = [('periodic / periodic', lambda x: np.sin(2 * np.pi * x + 1), True),
             ('Dirichlet / Dirichlet', lambda x: np.sin(np.pi * x), False),
             ('Neumann / Neumann', lambda x: np.cos(np.pi * x), True),
             ('Dirichlet / Neumann', lambda x: np.sin(np.pi * x / 2), False)]
    for th in THEMES:
        with style(th):
            fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
            for p, ((label, u, mean_free), ax) in enumerate(zip(pairs, axes.ravel()), start=1):
                d = np.loadtxt(data / f't02_faces-{p}.dat')
                x, phi = d[:, 0], d[:, 1]
                xs = np.linspace(-0.05, 1.05, 400)
                ax.plot(xs, u(xs), color=pal.MUTED[th], lw=1.2, label='continuum u')
                inner = (x > 0) & (x < 1)
                ax.plot(x[inner], phi[inner], ls='none', marker=pal.MARKERS[0], ms=4, color=pal.LINES[th][0],
                        label='phi, cells')
                ax.plot(x[~inner], phi[~inner], ls='none', marker=pal.MARKERS[1], ms=7, mfc='none', mew=1.6,
                        color=pal.LINES[th][1], label='phi, ghost cells')
                for xf in (0.0, 1.0):
                    ax.axvline(xf, color=pal.MESH[th], lw=0.8, ls=':')
                ax.set_title(label, fontsize=10)
                ax.grid(True)
                if p == 1:
                    ax.legend(loc='lower left', fontsize=8.5)
            for ax in axes[1]:
                ax.set_xlabel('x')
            fig.tight_layout()
            save(fig, out, 't02-faces', th)


def fig_t03(data: Path, out: Path) -> None:
    d = np.loadtxt(data / 't03_isolated-line.dat')
    x, exact, iso, iso_h, per = d.T
    blocks = read_blocks(data / 't03_isolated-slice.dat')
    for th in THEMES:
        with style(th):
            fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={'width_ratios': [1.2, 1]})
            a.plot(x, exact, color=pal.MUTED[th], lw=1.4, label='analytic, -M erf(r/√2σ)/r')
            a.plot(x, iso, ls='none', marker=pal.MARKERS[0], ms=5, color=pal.LINES[th][0],
                   label='isolated (both paths)')
            a.plot(x, per, ls='none', marker=pal.MARKERS[1], ms=5, mfc='none', mew=1.4, color=pal.LINES[th][1],
                   label='periodic')
            a.set_xlabel('x (y = z = 1/2)')
            a.set_ylabel('phi')
            a.grid(True)
            a.legend(loc='lower right', fontsize=9)
            a.set_title('a star alone, and a star among its images', fontsize=10)
            vmin, vmax = field_extent(blocks)
            draw_blocks(b, blocks, th, vmin, vmax, outlines=False, rings=10)
            square(b)
            b.set_title('isolated phi, z = 1/2', fontsize=10)
            fig.tight_layout()
            save(fig, out, 't03-isolated', th)


def fig_t04(data: Path, out: Path) -> None:
    blocks = read_blocks(data / 't04_amr-slice.dat')
    for th in THEMES:
        with style(th):
            fig, a = plt.subplots(figsize=(7.2, 5.6))
            vmin, vmax = field_extent(blocks)
            draw_blocks(a, blocks, th, vmin, vmax, outlines=True, rings=10)
            square(a)
            level_legend(a, th)
            a.set_title('the star on three levels, z = 1/2', fontsize=10)
            fig.tight_layout()
            save(fig, out, 't04-amr', th)


def fig_t05(data: Path, out: Path) -> None:
    blocks = read_blocks(data / 't05_values-slice.dat')
    for th in THEMES:
        with style(th):
            fig, a = plt.subplots(figsize=(7.2, 5.6))
            vmin, vmax = field_extent(blocks)
            draw_blocks(a, blocks, th, vmin, vmax, outlines=True, rings=8)
            square(a, label=False)
            a.set_xlabel('x')
            a.set_ylabel('z')
            level_legend(a, th, levels=(0, 1))
            a.set_title('phi on the plane y = 1/2: the lid at z = 1 holds sin(πx) sin(πy)', fontsize=10)
            fig.tight_layout()
            save(fig, out, 't05-values', th)


def fig_t06(data: Path, out: Path) -> None:
    rows = table_rows('t06_iterations', r'^\s*(\d+)\s+([0-9.E+-]+)?\s*([0-9.E+-]+)?\s*$')
    k, sym, adam = [], [], []
    for r in rows:
        k.append(int(r[0]))
        sym.append(float(r[1]) if r[1] else np.nan)
        adam.append(float(r[2]) if r[2] else np.nan)
    # a row with one value left blank: the regex puts the value in the first group; the symmetric column ends first
    k, sym, adam = np.array(k), np.array(sym), np.array(adam)
    nsym = int(np.sum(~np.isnan(adam) & ~np.isnan(sym)))
    adam_full = np.where(np.isnan(adam), sym, adam)
    sym = np.where(np.arange(k.size) < nsym, sym, np.nan)
    for th in THEMES:
        with style(th):
            fig, a = plt.subplots(figsize=(7.6, 4.4))
            a.semilogy(k, sym, marker=pal.MARKERS[0], color=pal.LINES[th][0], label='symmetric V-cycle (default)')
            a.semilogy(k, adam_full, marker=pal.MARKERS[1], color=pal.LINES[th][1], label="ADAM's V-cycle")
            a.axhline(1e-10, ls='--', lw=1.1, color=pal.MUTED[th])
            a.annotate('tolerance 1e-10', (k[-1], 1e-10), textcoords='offset points', xytext=(-4, 5), ha='right',
                       fontsize=9, color=pal.MUTED[th])
            a.set_xlabel('FCG iteration k')
            a.set_ylabel('max |r| / max |f|')
            a.grid(True)
            a.legend(loc='upper right', fontsize=9)
            fig.tight_layout()
            save(fig, out, 't06-history', th)


# ----------------------------------------------------------------------------------------------------------------------
# animations
# ----------------------------------------------------------------------------------------------------------------------
def encode(frames: Path, out: Path, name: str, theme: str, fps: int, width: int, last: int) -> None:
    """The PNG frames of one theme to webm, mp4 and the poster (the last frame)."""
    scale = f'scale={width}:-2:flags=lanczos'
    base = out / f'{name}-{theme}'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(frames / 'f%04d.png'),
                    '-vf', scale, '-c:v', 'libvpx-vp9', '-crf', '36', '-b:v', '0', '-pix_fmt', 'yuv420p',
                    f'{base}.webm'], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(frames / 'f%04d.png'),
                    '-vf', scale, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-crf', '23',
                    f'{base}.mp4'], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(frames / f'f{last:04d}.png'), '-vf', scale,
                    '-frames:v', '1', f'{base}.png'], check=True)
    for ext in ('webm', 'mp4', 'png'):
        print(f'  {(out / f"{name}-{theme}.{ext}").relative_to(ROOT)}')


def anim_t06(data: Path, out: Path, fps: int, width: int) -> None:
    lines = (data / 't06_iterations-frames.dat').read_text().split('\n')
    nb, nc, it_sym, it_adam = (int(t) for t in lines[0].split())
    heads = [ln.split() for ln in lines[1:1 + nb]]
    body = lines[1 + nb:]

    def frame(c: int, k: int):
        start = (0 if c == 0 else it_sym * nb) + (k - 1) * nb
        return [np.array(body[start + b].split(), dtype=float).reshape(nc, nc) for b in range(nb)]

    # the error of an iterate: |phi_k - phi| / max |phi|, phi the converged symmetric solve (the two converged solves
    # agree to their tolerance), on a log scale from 1e-10 (the tolerance: the ground) to 1 (saturated)
    ref = frame(0, it_sym)
    scale = max(np.abs(v).max() for v in ref)
    geo = [(int(h[0]), float(h[1]), float(h[2]), float(h[3])) for h in heads]
    nk = max(it_sym, it_adam)
    for th in THEMES:
        with style(th), tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for k in range(1, nk + 1):
                fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.4))
                for c, (ax, label, its) in enumerate(zip(axes, ('symmetric V-cycle', "ADAM's V-cycle"),
                                                         (it_sym, it_adam))):
                    kk = min(k, its)
                    worst = 0.0
                    for (lev, x0, y0, h), v, r in zip(geo, frame(c, kk), ref):
                        e = np.abs(v - r) / scale
                        worst = max(worst, e.max())
                        img = ax.imshow(np.log10(np.maximum(e, 1e-12)), extent=(x0, x0 + nc * h, y0, y0 + nc * h),
                                        origin='lower', cmap=pal.EMBER[th], vmin=-10, vmax=0,
                                        interpolation='nearest', zorder=1)
                        ax.add_patch(Rectangle((x0, y0), nc * h, nc * h, fill=False, lw=0.5 + 0.35 * lev,
                                               ec=pal.LEVELS[th][min(lev, 2)], alpha=0.6, zorder=4))
                    square(ax, label=False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    state = 'converged' if k >= its else f'max error {worst:.0e}'
                    ax.set_title(f'{label}\niterate {kk}: {state}', fontsize=11,
                                 color=pal.LINES[th][c] if k >= its else pal.INK[th])
                fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.17, wspace=0.04)
                cb = fig.colorbar(img, cax=fig.add_axes([0.08, 0.08, 0.84, 0.03]), orientation='horizontal')
                cb.set_label('log10 |phi_k - phi| / max |phi|', fontsize=9)
                cb.outline.set_edgecolor(pal.MUTED[th])
                fig.savefig(tmp / f'f{k - 1:04d}.png', dpi=110, bbox_inches=None)
                plt.close(fig)
            # hold the converged state a while before the loop restarts
            last = nk - 1
            for extra in range(1, 2 * fps):
                shutil.copy(tmp / f'f{last:04d}.png', tmp / f'f{last + extra:04d}.png')
            encode(tmp, out, 't06-iterates', th, max(4, fps // 3), width, last)


def anim_t08(data: Path, out: Path, fps: int, width: int) -> None:
    text = (data / 't08_embers-frames.dat').read_text().split('\n')
    frames, i = [], 0
    while i < len(text) and text[i].strip():
        head = text[i].split()
        embers = np.array(head[1:5], dtype=float).reshape(2, 2)
        ns = int(head[5])
        blocks = []
        for b in range(ns):
            lev, x0, y0, h = text[i + 1 + 2 * b].split()
            v = np.array(text[i + 2 + 2 * b].split(), dtype=float)
            nc = int(round(np.sqrt(v.size)))
            blocks.append((int(lev), float(x0), float(y0), float(h), v.reshape(nc, nc)))
        frames.append((embers, blocks))
        i += 1 + 2 * ns
    vmin = min(field_extent(f[1])[0] for f in frames)
    vmax = max(field_extent(f[1])[1] for f in frames)
    for th in THEMES:
        with style(th), tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for n, (embers, blocks) in enumerate(frames):
                fig, ax = plt.subplots(figsize=(6.4, 6.4))
                draw_blocks(ax, blocks, th, vmin, vmax, outlines=True, rings=12, levels_alpha=0.7)
                ax.plot(embers[:, 0], embers[:, 1], ls='none', marker='o', ms=5, color=pal.LINES[th][0],
                        mec=pal.GROUND[th], mew=1.0, zorder=5)
                square(ax, 0.17, 0.83, label=False)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
                fig.savefig(tmp / f'f{n:04d}.png', dpi=110)
                plt.close(fig)
            encode(tmp, out, 't08-embers', th, fps, width, 0)


FIGURES = {'t01-box': fig_t01, 't02-faces': fig_t02, 't03-isolated': fig_t03, 't04-amr': fig_t04,
           't05-values': fig_t05, 't06-history': fig_t06}
ANIMATIONS = {'t06-iterates': anim_t06, 't08-embers': anim_t08}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, required=True, help='the directory the programs dumped into')
    ap.add_argument('--out', type=Path, default=ROOT / 'docs' / 'public' / 'tutorial')
    ap.add_argument('--only', nargs='*', default=None, help='names of figures or animations to draw')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--width', type=int, default=720)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'casts').mkdir(exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only is None or name in args.only:
            fn(args.data, args.out)
    for name, fn in ANIMATIONS.items():
        if args.only is None or name in args.only:
            fn(args.data, args.out / 'casts', args.fps, args.width)


if __name__ == '__main__':
    main()
