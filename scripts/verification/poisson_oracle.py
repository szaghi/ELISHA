#!/usr/bin/env python3
# Provenance: ADAM's src/tests/flume/verification/selfgravity/poisson_oracle.py (ADAM 669d0bdd), copied into ELISHA by
# ADAM issue #136, D2 (ELISHA owns the elliptic solvers' verification). Adapted: the docstring names ELISHA's drivers,
# and `compare --pattern GLOB` (default every dump, as ADAM's) selects the files; the logic and the dump format are
# ADAM's, unchanged.
"""Oracle of the FFT Poisson solver (issue #81, P2 SV-2 and P4 SV-6): two builds agree on the potential.

    poisson_oracle.py compare <dirA> <dirB> [--tol T] [--pattern GLOB]

`compare` reads the potential dumps that ELISHA's `elisha_sv_fft_mpi --dump DIR` writes for every configuration and
level (`phi-r<ratio>-nk<nk>-l<level>-p<profile>.bin`: 3 int32 sizes, then the float64 values, x fastest; profile 0
the periodic modes, 1 and 2 the isolated polytrope and uniform sphere, whose arrays are the padded ones), and those of
`elisha_sv_mg_mpi --dump DIR` (`phi-mg-<leg>-<case>-l<level>.bin`, the same format: sizes [cells per block, leaves,
1], the leaves in Morton order), from two directories, e.g. the host (FFTW) and the device (`--device`: cuFFT,
OpenACC) builds, or two rank counts, and reports `max |phi_A - phi_B| / max |phi_A|` per file.
Exit 0 when every file of A has its twin in B with the same sizes and every difference is within `--tol` (default
1e-13). `--pattern` (default `phi-*.bin`, every dump) selects the files: the direct spectral solves and the FCG
solutions of the hierarchical solver need different bounds (`scripts/verification/poisson.sh --compare`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    """Read a potential dump.

    Parameters
    ----------
    path : Path
        Dump written by `elisha_sv_fft_mpi --dump` or `elisha_sv_mg_mpi --dump`.

    Returns
    -------
    np.ndarray
        The potential, indexed [x, y, z].
    """
    raw = path.read_bytes()
    n = np.frombuffer(raw, dtype=np.int32, count=3)
    values = np.frombuffer(raw, dtype=np.float64, offset=12, count=int(np.prod(n)))
    return values.reshape(tuple(int(v) for v in n), order="F")


def compare(args: argparse.Namespace) -> int:
    """Compare the dumps of two directories, file by file."""
    files = sorted(args.a.glob(args.pattern))
    if not files:
        print(f"no dump in {args.a}")
        return 1
    ok = True
    for fa in files:
        fb = args.b / fa.name
        if not fb.is_file():
            print(f"{fa.name}: missing in {args.b}  FAIL")
            ok = False
            continue
        pa, pb = load(fa), load(fb)
        if pa.shape != pb.shape:
            print(f"{fa.name}: sizes {pa.shape} against {pb.shape}  FAIL")
            ok = False
            continue
        diff = float(np.max(np.abs(pa - pb)) / np.max(np.abs(pa)))
        verdict = "PASS" if diff <= args.tol else "FAIL"
        ok = ok and diff <= args.tol
        print(f"{fa.name}: {pa.shape}, max |phi_A - phi_B| / max |phi_A| {diff:.2e}  {verdict}")
    print(f"compare {args.a} {args.b} {args.pattern}: {'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    return 0 if ok else 1


def main() -> int:
    """Parse the command line and run the check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="check", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tol", type=float, default=1.0e-13)
    p.add_argument("--pattern", default="phi-*.bin", help="the dumps compared (a glob; ELISHA's addition)")
    args = parser.parse_args()
    return {"compare": compare}[args.check](args)


if __name__ == "__main__":
    raise SystemExit(main())
