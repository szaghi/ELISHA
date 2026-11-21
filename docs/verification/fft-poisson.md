# SV-2: the spectral solver, periodic

| ID | Solver | Program | Leg | Figure |
|---|---|---|---|---|
| SV-2 (ADAM #81 P2) | `elisha_spectral_solver_object` | `elisha_sv_fft_mpi --periodic` | `sv2`, `slab` | yes |

## Case and scope

Self-gravity needs a Poisson solver that is exact for the *discrete* equation $L_h\Phi = 4\pi G(\rho - \bar\rho)$,
with $L_h$ the 7-point Laplacian (5-point in 2-D). The spectral solver divides each Fourier mode by the eigenvalue of
$L_h$ itself, so its residual is at round-off.

What the case catches:
- a wrong eigenvalue of $L_h$ (a missing factor, a swapped axis, a wrong spacing): the error against the discrete
  solution leaves round-off. The box is anisotropic, $[0,1]\times[0,2]\times[0,3]$, so an axis mix-up cannot cancel;
- a misplaced block in the slab transform, or a wrong ghost in the scatter: the error over every cell and ghost, and
  the residual computed from the returned ghosts;
- a ghost layer left unfilled (a sentinel value);
- a potential that depends on the rank count: the slab transform's results must be bitwise the same on every count;
- a solver that accepts a grid without periodic boundaries.

## Setup

| Item | Value |
|---|---|
| Equation | $L_h\Phi = 4\pi G(\rho - \bar\rho)$, $G = 1$ |
| Domain | $[0,1]\times[0,2]\times[0,3]$, periodic on every face |
| Source | $\rho = 1.5 + \sin(k_1x)\cos(k_1y)\cos(k_1z) + 0.5\cos(\mathbf k_2\cdot\mathbf x)$, $\mathbf k_1 = 2\pi(1, 1, \tfrac13)$, $\mathbf k_2 = 2\pi(2, \tfrac12, \tfrac13)$; in 2-D the z wave numbers are 0 |
| Grids | one root block of $8^3$ cells (3 ghost layers), refined uniformly: a 3-D octree at levels 1–3 ($16^3$ to $64^3$ cells), a 2-D quadtree (null z) at levels 1–3, and the single block at level 0 (on 2 or more ranks, ranks without blocks) |
| Ranks | 1, 2, 3 (the harness's default), blocks in contiguous chunks |

**Exact solutions.** Each mode is an eigenfunction of $L_h$. The discrete solution divides it by its eigenvalue
$\lambda_k = \sum_d (2\cos(k_d h_d) - 2)/h_d^2$ (the mean dropped), the continuum one by $-\lvert\mathbf k\rvert^2$.

**Asserted.** Per configuration and level:
- $\max\lvert\Phi - \Phi_h\rvert / \max\lvert\Phi_h\rvert$ over every cell, ghosts included, at most $10^{-12}$;
- the residual $\max\lvert L_h\Phi - 4\pi(\rho - \rho_0)\rvert / \max\lvert 4\pi(\rho - \rho_0)\rvert$ over the
  interior, from the returned ghosts, at most $10^{-10}$;
- the order against the continuum solution on the finest pair ($N = 32\to64$) at least 1.9;
- no ghost left unfilled;
- on more than one rank, rank 0 solves each level again alone and its transform array (images and padding included)
  is bitwise the distributed one's.

## Results

![SV-2: the errors against N, the 2-D potential and its error against the continuum solution](/verification/sv2-fft-poisson.png)

*Left: against N, the error against the continuum solution falls as $h^2$; the error against the discrete solution
and the residual stay at round-off. Middle and right: the 2-D potential at $N = 64$ and its error against the
continuum solution, $4.3\cdot10^{-4}$.*

ELISHA's numbers (gfortran 16.2, np 2; the same on np 1 and 3) next to ADAM's documented ones (ADAM 669d0bdd,
`fft-poisson.md`):

| Asserted | Bound | ELISHA | ADAM (CPU) |
|---|---|---|---|
| $\max\lvert\Phi - \Phi_h\rvert / \max\lvert\Phi_h\rvert$, every cell and ghost | $10^{-12}$ | $2.78\cdot10^{-15}$ (one block), $2.31\cdot10^{-15}$ (3-D), $1.77\cdot10^{-15}$ (2-D) | $\le 2.8\cdot10^{-15}$ |
| residual of $L_h$, relative | $10^{-10}$ | $3.67\cdot10^{-14}$ (3-D, $N = 64$) | $\le 3.7\cdot10^{-14}$ |
| order against the continuum solution, $N = 32\to64$ | $\ge 1.9$ | 1.982 (3-D), 1.984 (2-D) | 1.98, 1.98 |
| 2-D error against the continuum solution, $N = 64$ | | $4.33\cdot10^{-4}$ | $4.3\cdot10^{-4}$ |
| ghost cells unfilled | 0 | 0 | 0 |
| rank 0 alone against the distributed solve, the transform array | bitwise | bitwise, every level | |
| `slab` leg: the dumps of np 1, 2, 3 | byte for byte | 26 of 26 identical (13 per rank count) | 13 of 13 |
| a grid without periodic boundaries | refused | `ELISHA_ERR_BC_MISSING` | refused (stop) |

nvfortran gives the same continuum errors and orders; its round-off numbers differ in the last digits (discrete error
$\le 2.73\cdot10^{-15}$, residual $\le 3.79\cdot10^{-14}$). CPU = device: the 13 spectral dumps of job 1369211 agree
with the host's within $8.93\cdot10^{-16}$ ([the harness](/verification/#the-harness)).

## Commands

```bash
fobis build --mode tests-gnu
mpirun -np 2 exe/elisha_sv_fft_mpi --periodic                 # SV-2
mpirun -np 2 exe/elisha_sv_fft_mpi --no-periodic              # the refusal: ELISHA_ERR_BC_MISSING, exit 0
scripts/verification/poisson.sh --legs "sv2 slab"             # np 1, 2, 3 and the byte comparison
```

## Files

| File | Role |
|---|---|
| [`src/tests/elisha_sv_fft_mpi.F90`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_fft_mpi.F90) | the driver (ADAM's `test_poisson_fft`, ported) |
| [`scripts/verification/poisson.sh`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/poisson.sh) | legs `sv2` and `slab` |
| [`scripts/verification/poisson_oracle.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/poisson_oracle.py) | `compare`: two dump directories |
| [`scripts/verification/make_verification_figures.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/make_verification_figures.py) | the figure (`fig_sv2`) |

## Findings

- The solver is exact for $L_h$, and the scheme is second order through $L_h$.
- The residual grows from $10^{-15}$ to $4\cdot10^{-14}$ between $N = 16$ and 64: $L_h$ amplifies the round-off of
  $\Phi$ by $(k h)^{-2}$.
- The potential does not depend on the rank count, bitwise, and every printed number is the same on 1, 2 and 3 ranks.
