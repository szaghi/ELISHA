#!/usr/bin/env python3
# Provenance: the figure functions fig_selfgravity_poisson, fig_selfgravity_isolated, fig_selfgravity_amr and
# fig_selfgravity_amr_isolated of ADAM's src/tests/flume/verification/make_doc_figures.py (ADAM 669d0bdd), ported by
# ADAM issue #136, D2: the same panels, reading ELISHA's drivers (the logs of scripts/verification/poisson.sh and its
# --dump potentials) instead of ADAM's test_poisson_fft / test_poisson_mg runs.
"""The figures of ELISHA's verification pages (docs/verification/), from a run of the verification legs.

    scripts/verification/poisson.sh --legs "sv2 sv6 sv9 sv13" --np-list 2 --work WORK --keep --dump DUMP
    python3 scripts/verification/make_verification_figures.py --work WORK --dump DUMP [--out docs/public/verification]

The figures are drawn from the np-2 logs (`WORK/log-<leg>-np2.txt`, the drivers' per-level lines) and the potentials
of the dump (`DUMP/phi-r4-nk1-l3-p0.bin`, the 2-D SV-2 potential at N = 64; `DUMP/phi-r8-nk8-l3-p1.bin`, the SV-6
polytrope's padded transform array at 64^3); the SV-9 figure also runs the operator study of seam_operators.py. The
drivers' logs and dumps are deterministic at a fixed rank count, and the PNGs carry no time stamp: a rerun on the same
data writes the same bytes (same matplotlib and numpy).

Writes sv2-fft-poisson.png, sv6-isolated.png, sv9-multigrid-amr.png and sv13-james.png.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import poisson_oracle  # noqa: E402
import seam_operators  # noqa: E402

DPI = 110
META = {"Software": None}  # no matplotlib version in the PNG: the bytes depend on the data only


def log_lines(work: Path, leg: str) -> list[str]:
    """The np-2 log of a leg."""
    path = work / f"log-{leg}-np2.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run poisson.sh --legs {leg} --np-list 2 --work {work} --keep")
    return path.read_text(errors="replace").splitlines()


def save(fig: plt.Figure, out: Path, name: str) -> None:
    """Write a figure and close it."""
    fig.savefig(out / name, dpi=DPI, metadata=META)
    plt.close(fig)
    print(f"wrote {out / name}")


def fig_sv2(out: Path, work: Path, dump: Path) -> None:
    """SV-2: the spectral solver on the anisotropic periodic box [0,1] x [0,2] x [0,3], rho = rho0 + sin(k1 x)
    cos(k1 y) cos(k1 z) + 0.5 cos(k2 . x) (elisha_sv_fft_mpi --periodic, 2 ranks). Left: against N, the max error of Phi
    against the continuum solution (second order, 3-D octree and 2-D quadtree), against the discrete solution and the
    discrete residual (round-off). Middle and right: the 2-D potential at N = 64 and its error against the continuum
    solution."""
    rows: dict[str, list[tuple[int, float, float, float]]] = {}
    pattern = re.compile(r"(3-D octree|2-D quadtree|3-D one block),\s+N = (\d+): \|Phi - Phi_h\| (\S+), "
                         r"residual (\S+), \|Phi - Phi_exact\| (\S+),")
    for line in log_lines(work, "sv2"):
        m = pattern.search(line)
        if m:
            rows.setdefault(m[1], []).append((int(m[2]), float(m[3]), float(m[4]), float(m[5])))
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True, width_ratios=(1.6, 1.0, 1.0))
    for name, color in (("3-D octree", "tab:blue"), ("2-D quadtree", "tab:red")):
        n, disc, res, cont = (np.array(v) for v in zip(*rows[name], strict=True))
        order = math.log(cont[-2] / cont[-1]) / math.log(n[-1] / n[-2])
        axs[0].loglog(n, cont, "o-", color=color, label=f"{name}: |Phi - Phi_exact|, order {order:.2f}")
        axs[0].loglog(n, disc, "s--", color=color, alpha=0.7, label=f"{name}: |Phi - Phi_h| / max|Phi_h|")
        axs[0].loglog(n, res, "^:", color=color, alpha=0.7, label=f"{name}: residual of L_h, relative")
    ns = np.array([16.0, 64.0])
    axs[0].loglog(ns, 1.5e-3 * (ns[0] / ns) ** 2, "-", color="0.6", lw=0.8)
    axs[0].annotate("dx^2", (32.0, 1.5e-3 * (ns[0] / 32.0) ** 2), xytext=(0, -12), textcoords="offset points",
                    fontsize=8, color="0.4")
    axs[0].set_xlabel("N (cells along x)")
    axs[0].set_ylabel("max error")
    axs[0].set_ylim(1e-16, 1e-1)
    axs[0].legend(fontsize=7)
    axs[0].set_title("second order against the PDE, round-off against L_h", fontsize=10)
    phi = poisson_oracle.load(dump / "phi-r4-nk1-l3-p0.bin")[:, :, 0]
    nx, ny = phi.shape
    x = (np.arange(nx) + 0.5) / nx
    y = 2.0 * (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    k1 = 2.0 * np.pi * np.array([1.0, 1.0])
    k2 = 2.0 * np.pi * np.array([2.0, 0.5])
    exact = 4.0 * np.pi * (np.sin(k1[0] * xx) * np.cos(k1[1] * yy) / -(k1 @ k1)
                           + 0.5 * np.cos(k2[0] * xx + k2[1] * yy) / -(k2 @ k2))
    extent = (0.0, 1.0, 0.0, 2.0)
    im = axs[1].imshow(phi.T, origin="lower", extent=extent, cmap="RdBu_r")
    fig.colorbar(im, ax=axs[1], shrink=0.85)
    axs[1].set_title("2-D, N = 64: Phi", fontsize=10)
    im = axs[2].imshow((phi - exact).T, origin="lower", extent=extent, cmap="PuOr")
    fig.colorbar(im, ax=axs[2], shrink=0.85)
    axs[2].set_title(f"Phi - Phi_exact (max {np.max(np.abs(phi - exact)):.1e})", fontsize=10)
    for ax in axs[1:]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("SV-2: the spectral solver, L_h Phi = 4 pi (rho - mean rho) on a periodic box (gfortran, 2 ranks)")
    save(fig, out, "sv2-fft-poisson.png")


def fig_sv6(out: Path, work: Path, dump: Path) -> None:
    """SV-6: isolated boundaries on the cube [-1, 1]^3 (elisha_sv_fft_mpi --isolated, 2 ranks), two spheres of radius
    1/2, G = 1. Left: against N, the max error of Phi against the exact potential over every cell and ghost and the
    residual of L_h. Middle: Phi of the polytrope at N = 64 along the line through the centre, over the whole padded
    array, against the exact Phi (-G M / r outside). Right: the error on the plane through the centre."""
    rows: dict[str, list[tuple[int, float, float]]] = {}
    pattern = re.compile(r"(polytrope n=1|uniform sphere),\s+N = (\d+): \|Phi - Phi_h\| \S+, residual (\S+), "
                         r"\|Phi - Phi_exact\| (\S+),")
    for line in log_lines(work, "sv6"):
        m = pattern.search(line)
        if m:
            rows.setdefault(m[1], []).append((int(m[2]), float(m[3]), float(m[4])))
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True, width_ratios=(1.3, 1.3, 1.0))
    for name, color in (("polytrope n=1", "tab:blue"), ("uniform sphere", "tab:red")):
        n, res, cont = (np.array(v) for v in zip(*rows[name], strict=True))
        order = -np.polyfit(np.log(n), np.log(cont), 1)[0]
        axs[0].loglog(n, cont, "o-", color=color, label=f"{name}: |Phi - Phi_exact|, least-squares order {order:.2f}")
        axs[0].loglog(n, res, "^:", color=color, alpha=0.7, label=f"{name}: residual of L_h, relative")
    ns = np.array([16.0, 64.0])
    axs[0].loglog(ns, 1.0e-2 * (ns[0] / ns) ** 2, "-", color="0.6", lw=0.8)
    axs[0].annotate("dx^2", (32.0, 1.0e-2 * (ns[0] / 32.0) ** 2), xytext=(0, -12), textcoords="offset points",
                    fontsize=8, color="0.4")
    axs[0].set_xlabel("N (cells per side)")
    axs[0].set_ylabel("max error")
    axs[0].set_ylim(1e-15, 1e-1)
    axs[0].legend(fontsize=7)
    axs[0].set_title("second order against the PDE; the residual is the kernel's", fontsize=10)
    phi = poisson_oracle.load(dump / "phi-r8-nk8-l3-p1.bin")
    m, n, h = phi.shape[0], 64, 2.0 / 64
    a, kk = 0.5, math.pi / 0.5
    mass = 4.0 * a**3 / math.pi

    def exact(r: np.ndarray) -> np.ndarray:
        kr = np.maximum(kk * r, 1e-300)
        inside = -mass / a - 4.0 * math.pi / kk**2 * np.sin(kr) / kr
        return np.where(r < a, inside, -mass / np.maximum(r, 1e-300))

    # the padded array holds the free-space potential at the cell offsets -(m - n)/2 .. n - 1 + (m - n)/2 (wrapped)
    t = np.arange(m)
    t = np.where(t >= n + (m - n) // 2, t - m, t)
    order_t = np.argsort(t)
    x = -1.0 + (t[order_t] + 0.5) * h
    line = phi[order_t, n // 2, n // 2]
    yc = -1.0 + (n // 2 + 0.5) * h
    r_line = np.sqrt(x**2 + 2.0 * yc**2)
    axs[1].plot(x, line, ".", ms=3, color="tab:blue", label="Phi, the padded array (N = 64)")
    axs[1].plot(x, exact(r_line), "-", color="0.3", lw=0.8, label="exact, -G M / r outside")
    axs[1].axvspan(-1.0, 1.0, color="0.92", zorder=0, label="the domain")
    axs[1].set_xlabel("x (y = z = h/2)")
    axs[1].set_ylabel("Phi")
    axs[1].legend(fontsize=7)
    axs[1].set_title("the polytrope's potential and its free-space continuation", fontsize=10)
    xs = -1.0 + (np.arange(n) + 0.5) * h
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    err = phi[:n, :n, n // 2] - exact(np.sqrt(xx**2 + yy**2 + yc**2))
    im = axs[2].imshow(err.T, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0), cmap="PuOr")
    fig.colorbar(im, ax=axs[2], shrink=0.85)
    axs[2].set_title(f"Phi - Phi_exact, z = h/2 (max {np.max(np.abs(err)):.1e})", fontsize=10)
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    fig.suptitle("SV-6: the spectral solver with isolated boundaries, zero padding and the lattice Green's function "
                 "(gfortran, 2 ranks)")
    save(fig, out, "sv6-isolated.png")


def fig_sv9(out: Path, work: Path) -> None:
    """SV-9: the composite Laplacian across 2:1 faces and the hierarchical solver on it. Left and middle: the 2-D model
    of seam_operators.py (doubly periodic, coarse x < 1/2, fine x > 1/2, two seams), N = 8 to 48, three seam couplings:
    flux matching (B), its symmetrisation (S1) and the mimetic mean flux (M, the one the solver assembles); left the max
    truncation error, middle the seam face gradient error and the potential error. Right: the hierarchical solver on
    the octree and the quadtree of elisha_sv_mg_mpi --sv9 (2 ranks; the central box refined once, base levels 2 and
    3): the potential error and the 2:1 face gravity error, the growth of the max truncation error."""
    pattern = re.compile(r"SV-9b, AMR (\S+) \S+,\s+base level (\d+), leaves \d+ \(levels \d+-\d+\): iterations (\d+), "
                         r"factor \S+, symmetry (\S+), truncation (\S+), \|Phi - Phi_exact\| (\S+), "
                         r"2:1 face gravity error (\S+),")
    amr: dict[str, list[tuple[int, int, float, float, float, float]]] = {}
    for line in log_lines(work, "sv9"):
        m = pattern.search(line)
        if m:
            amr.setdefault(m[1], []).append((int(m[2]), int(m[3]), float(m[4]), float(m[5]), float(m[6]),
                                             float(m[7])))
    if not amr:
        raise FileNotFoundError("no SV-9b line in the sv9 log")
    rows = seam_operators.ladder([8, 16, 32, 48])
    style = {"B": ("tab:orange", "s", "B, flux matching (not symmetric)"),
             "S1": ("tab:red", "^", "S1 = (B + B^T) / 2 (symmetric, inconsistent)"),
             "M": ("tab:blue", "o", "M, mimetic mean flux (symmetric, consistent)")}
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for kind, (color, marker, label) in style.items():
        n, e_phi, tau, e_face, _ = (np.array(v) for v in zip(*rows[kind], strict=True))
        axs[0].loglog(n, tau, marker + "-", color=color, label=label)
        axs[1].loglog(n, e_face, marker + "-", color=color, label=f"{kind}: seam face gradient")
        axs[1].loglog(n, e_phi, marker + ":", color=color, alpha=0.6, label=f"{kind}: potential")
    ns = np.array([8.0, 48.0])
    axs[0].loglog(ns, 30.0 * ns / ns[0], "-", color="0.6", lw=0.8)
    axs[0].annotate("1/h", (24.0, 30.0 * 3.0), xytext=(-18, 2), textcoords="offset points", fontsize=8, color="0.4")
    axs[0].set_xlabel("N (coarse cells per side)")
    axs[0].set_ylabel("max |A Phi_exact / V - Lap Phi_exact|")
    axs[0].legend(fontsize=7)
    axs[0].set_title("truncation: bounded (B, M), growing as 1/h (S1)", fontsize=10)
    for p, y0, lab in ((1, 2.0, "h"), (2, 0.2, "h^2")):
        axs[1].loglog(ns, y0 * (ns[0] / ns) ** p, "-", color="0.6", lw=0.8)
        axs[1].annotate(lab, (20.0, y0 * (8.0 / 20.0) ** p), xytext=(0, -12), textcoords="offset points", fontsize=8,
                        color="0.4")
    axs[1].set_xlabel("N (coarse cells per side)")
    axs[1].set_ylabel("max error")
    axs[1].legend(fontsize=6, ncol=2)
    axs[1].set_title("the seam gradient converges for B and M, not for S1", fontsize=10)
    for dim, color in (("3-D", "tab:blue"), ("2-D", "tab:green")):
        lev, _, _, tau, e_phi, e_g = (np.array(v) for v in zip(*sorted(amr[dim]), strict=True))
        nf = 8.0 * 2.0 ** (lev + 1)  # fine cells along x: 8 per block side, the box refined once

        def order(e: np.ndarray, nf: np.ndarray = nf) -> float:
            return float(np.log(e[0] / e[-1]) / np.log(nf[-1] / nf[0]))

        axs[2].loglog(nf, e_phi, "o-", color=color,
                      label=f"{dim}: potential, order {order(e_phi):.2f} (truncation growth {tau[-1] / tau[0]:.3f})")
        axs[2].loglog(nf, e_g, "s--", color=color, label=f"{dim}: 2:1 face gravity, order {order(e_g):.2f}")
    axs[2].set_xticks([64, 128], ["64", "128"])
    axs[2].minorticks_off()
    axs[2].set_xlim(56, 146)
    axs[2].set_ylim(2e-4, 2e-1)
    axs[2].set_xlabel("fine cells along x")
    axs[2].set_ylabel("max error")
    axs[2].legend(fontsize=7, loc="upper right")
    axs[2].set_title("the hierarchical solver on M (box refined once)", fontsize=10)
    fig.suptitle("SV-9: a symmetric, consistent composite Laplacian across 2:1 faces and the hierarchical solver "
                 "(gfortran, 2 ranks)")
    save(fig, out, "sv9-multigrid-amr.png")


def fig_sv13(out: Path, work: Path) -> None:
    """SV-13: isolated boundaries on AMR, the symmetric James step of the hierarchical solver, from the log of
    elisha_sv_mg_mpi --isolated (2 ranks): SV-6's stars (radius 1/2) on the cube [-1,1]^3, uniform (16^3, 32^3) or with
    the star's box refined once over base levels 2 and 3. Left: the max error of the potential on the leaves (solid)
    and of the James ghost values against the exact potential (dashed) against the finest cells per side. Right: the
    two exactness measures against their bounds, the hierarchical against the spectral solver's isolated mode
    (uniform) and the symmetry defect of the solution map in situ (the first grid of each leg)."""
    pattern = re.compile(r"(SV-13[ab]), isolated, (uniform|AMR), (polytrope n=1|uniform sphere),\s+base level (\d+), "
                         r"leaves \d+: iterations (\d+) \+ (\d+)(.*?), \|Phi - Phi_exact\| (\S+), "
                         r"boundary ghosts (\S+),")
    legs: dict[str, list[tuple[int, float, float]]] = {}
    exact: list[tuple[str, float, str]] = []
    for line in log_lines(work, "sv13"):
        m = pattern.search(line)
        if not m:
            continue
        key = f"{m[2]}, {m[3]}"
        legs.setdefault(key, []).append((int(m[4]), float(m[8]), float(m[9])))
        fft = re.search(r"\|Phi_mg - Phi_fft\| / max\|Phi\| ([^,\s]+)", m[7])
        sym = re.search(r"symmetry of V K ([^,\s]+)", m[7])
        tick = f"{m[2]}\n{'polytrope' if m[3].startswith('poly') else 'sphere'}\nlevel {m[4]}"
        if fft:
            exact.append((tick, float(fft[1]), "hierarchical - spectral"))
        if sym:
            exact.append((tick, float(sym[1]), "symmetry of V K"))
    if not legs:
        raise FileNotFoundError("no SV-13 line in the sv13 log")
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.9), constrained_layout=True, width_ratios=(1.1, 1.0))
    colors = {"uniform, polytrope n=1": "tab:gray", "AMR, polytrope n=1": "tab:blue",
              "AMR, uniform sphere": "tab:orange"}
    for key, rows in legs.items():
        lev, e_phi, e_gh = (np.array(v) for v in zip(*sorted(rows), strict=True))
        nf = 8.0 * 2.0 ** (lev + (1 if key.startswith("AMR") else 0))  # finest cells per side
        order = float(np.log(e_phi[0] / e_phi[-1]) / np.log(nf[-1] / nf[0]))
        axs[0].loglog(nf, e_phi, "o-", color=colors[key], label=f"{key}: potential, order {order:.2f}")
        axs[0].loglog(nf, e_gh, "s--", color=colors[key], alpha=0.7, label=f"{key}: boundary ghosts")
    ns = np.array([16.0, 128.0])
    axs[0].loglog(ns, 2e-2 * (ns[0] / ns) ** 2, "-", color="0.6", lw=0.8)
    axs[0].annotate("h^2", (40.0, 2e-2 * (16.0 / 40.0) ** 2), xytext=(0, -12), textcoords="offset points", fontsize=8,
                    color="0.4")
    axs[0].set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
    axs[0].minorticks_off()
    axs[0].set_xlabel("finest cells per side")
    axs[0].set_ylabel("max |Phi - Phi_exact|")
    axs[0].set_ylim(1e-5, 1e-1)
    axs[0].legend(fontsize=7, loc="upper right")
    axs[0].set_title("second order on AMR; the James ghosts converge with the potential", fontsize=10)
    ax = axs[1]
    kinds = {"hierarchical - spectral": "tab:green", "symmetry of V K": "tab:purple"}
    for i, (_, val, kind) in enumerate(exact):
        ax.bar(i, val, color=kinds[kind], label=kind if kind not in [e[2] for e in exact[:i]] else None)
    ax.axhline(1e-9, color="tab:red", lw=1.0, ls="--", label="bound, 1e-9")
    ax.axhline(1e-12, color="0.5", lw=0.8, ls=":", label="FCG tolerance, 1e-12")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-7)
    ax.set_xticks(range(len(exact)), [e[0] for e in exact], fontsize=8)
    ax.set_ylabel("relative defect")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("the free-space solution of the spectral solver; a symmetric map", fontsize=10)
    fig.suptitle("SV-13: isolated boundaries on AMR, the symmetric James step (gfortran, 2 ranks)")
    save(fig, out, "sv13-james.png")


def main() -> int:
    """Parse the command line and draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="poisson.sh --work directory (kept with --keep)")
    parser.add_argument("--dump", type=Path, required=True, help="poisson.sh --dump directory")
    parser.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "public" / "verification")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_sv2(args.out, args.work, args.dump)
    fig_sv6(args.out, args.work, args.dump)
    fig_sv9(args.out, args.work)
    fig_sv13(args.out, args.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
