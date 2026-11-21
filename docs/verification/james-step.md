# SV-13: the James step

| ID | Solver | Program | Leg | Figure |
|---|---|---|---|---|
| SV-13 (ADAM #87 P2) | `elisha_hierarchical_solver_object`, `isolated=.true.` | `elisha_sv_mg_mpi --isolated` | `sv13` | yes |

## Case and scope

Isolated boundaries on AMR, by a symmetric James method ([the hierarchical solver](/guide/hierarchical-solver#the-method)):
the zero-ghost solve $\Phi_0 = A^{-1}4\pi GV\rho$; the screening charge $s = B^T\Phi_0$ on the ghost cells; their
values $g = -Gs$, $G$ the inverse of the $V$-weighted coarsest operator on the infinite lattice, by a zero-padded FFT
on the coarsest box widened by the ghost layer; then $\Phi = A^{-1}(4\pi GV\rho - Bg)$. The map
$K = A^{-1} + A^{-1}BGB^TA^{-1}$ is symmetric because $A$ and $G$ are, which the exact energy exchange of self-gravity
needs. On a uniform grid it is the discrete free-space solution, the spectral solver's isolated mode.

What the case catches:
- an isolated solve that is not the free-space solution on a uniform grid;
- a solution map that is not symmetric ($VK \ne (VK)^T$);
- a ghost value or a potential that does not converge on AMR;
- an isolated solve accepted with a refined leaf on the domain boundary (the ghost cells must form the coarse lattice
  $G$ is exact on).

## Setup

| Leg | Source | Grid | Asserted |
|---|---|---|---|
| SV-13a | SV-6's $n = 1$ polytrope, radius 1/2, on $[-1,1]^3$ | uniform octree at levels 1 and 2 ($16^3$, $32^3$) | against the spectral solver's isolated mode $\le 10^{-9}$; each of the two solves in at most one iteration; order $\ge 1.7$ |
| SV-13b | the polytrope, and the uniform sphere (cell averages of $8^3$ samples) | base levels 2 and 3 with the star's box $[-1/2, 1/2]^3$ refined once (120 and 960 leaves; finest $64^3$ and $128^3$ equivalent) | FCG residual below its tolerance ($10^{-12}$); order $\ge 1.7$ (polytrope), $\ge 1.2$ (sphere) |
| both | | | no ghost unfilled; on the first grid of each leg the symmetry of the solution map in situ, $\lvert x\cdot VKy - y\cdot VKx\rvert / \lvert x\cdot VKy\rvert \le 10^{-9}$ for two pseudo-random sources; the James ghost values against the exact potential (reported) |

## Results

![SV-13: the isolated solve on AMR, errors against the finest cells per side, and its exactness measures](/verification/sv13-james.png)

*Left: the potential (solid) and the James ghost values (dashed) against the exact potential, uniform and with the
star refined: second order, the ghost values converging with the potential. Right: the hierarchical solver against the
spectral solver's isolated mode on uniform grids, and the symmetry defect of the solution map, against the bound.*

ELISHA (gfortran 16.2, np 2) next to ADAM's documented numbers (`multigrid-amr.md`):

| Asserted | Bound | ELISHA | ADAM (CPU) |
|---|---|---|---|
| SV-13a: against the spectral solver's isolated mode | $10^{-9}$ | $1.6\cdot10^{-15}$ ($16^3$), $2.63\cdot10^{-13}$ ($32^3$) | $\le 2.6\cdot10^{-13}$ |
| SV-13a: iterations of the two solves; order | $\le 1$; $\ge 1.7$ | 1 + 1; 2.140 | 1 + 1; 2.14 |
| SV-13b: order of $\Phi$, polytrope; uniform sphere | $\ge 1.7$; $\ge 1.2$ | 2.015; 1.862 | 2.01; 1.86 |
| SV-13b: iterations (zero-ghost + second solve) | | 24 + 23, 25 + 23 | 24–25 + 23 |
| symmetry of the solution map, first grid of each leg | $10^{-9}$ | $\le 2.0\cdot10^{-12}$ | $\le 1.7\cdot10^{-11}$ |
| James ghost values against the exact potential (SV-13a; SV-13b polytrope; sphere) | | $9.11\cdot10^{-4} \to 6.24\cdot10^{-5}$; $1.00\cdot10^{-4} \to 2.53\cdot10^{-5}$; $2.77\cdot10^{-4} \to 8.88\cdot10^{-5}$ | $9.1\cdot10^{-4} \to 6.2\cdot10^{-5}$; $1.0\cdot10^{-4} \to 2.5\cdot10^{-5}$; $2.8\cdot10^{-4} \to 8.9\cdot10^{-5}$ |
| AMR against the uniform grid at the same base level ($32^3$, polytrope) | | $1.18\cdot10^{-3}$ against $2.31\cdot10^{-3}$ | $1.18\cdot10^{-3}$ against $2.31\cdot10^{-3}$ |
| ghost cells unfilled | 0 | 0 | 0 |
| a refined leaf on the boundary | refused | `ELISHA_ERR_ISOLATED_BOUNDARY_LEAF` | refused (stop) |

The symmetry defect depends on the rank and the leaves' order (ADAM's pseudo-random vectors), the other numbers are
the same on 1, 2 and 3 ranks. CPU = device: see [the multigrid page](./multigrid-amr#results) (job 1369178).

## Commands

```bash
mpirun -np 2 exe/elisha_sv_mg_mpi --isolated
mpirun -np 2 exe/elisha_sv_mg_mpi --refuse-boundary-refined
scripts/verification/poisson.sh --legs sv13
python3 scripts/verification/james_proto.py      # the P0 prototype of the method (numpy)
```

## Files

| File | Role |
|---|---|
| [`src/tests/elisha_sv_mg_mpi.F90`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_mg_mpi.F90) | the driver |
| [`scripts/verification/james_proto.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/james_proto.py) | ADAM's prototype: the symmetry, the uniform check against the lattice convolution, the AMR order, and the textbook form as a negative control |
| [`scripts/verification/make_verification_figures.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/make_verification_figures.py) | the figure (`fig_sv13`) |

## Findings

- The James step is the discrete free-space solution on a uniform grid (the spectral solver's, to
  $2.6\cdot10^{-13}$, the lattice Green's function's own accuracy), second order on AMR (2.01 for the polytrope), and
  a symmetric map to the solve tolerance.
- The cost is two composite solves and one padded FFT of the coarsest box per Poisson solve, and the constraint that
  every leaf on the domain boundary sits at the coarsest level.
