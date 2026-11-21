# SV-9, SV-12: the hierarchical solver on AMR

| ID | Solver | Program | Legs | Figure |
|---|---|---|---|---|
| SV-9, SV-12 (ADAM #81 P5a, #87 P1); E1b, E5c, E1c (ADAM #84) | `elisha_hierarchical_solver_object`; the facade `elisha_solver_object` | `elisha_sv_mg_mpi` | `sv9`, `bcmg`, `sub`, `flail` | yes |

## Case and scope

On an AMR grid the exact exchange of the kinetic and gravitational energies needs the composite Laplacian $A$ (finite
volume, rows weighted by the cell volume $V$) to be **symmetric**, and accuracy needs it **consistent**: a bounded
truncation error at the 2:1 faces and a converging face gradient there. The hierarchical solver assembles the
**mimetic mean flux** across 2:1 faces: every fine face carries $\tfrac23 w(\bar\Phi_f - \Phi_C)$, $\bar\Phi_f$ the
mean of the face's fine cells, and the coarse cell their sum ([the method](/guide/hierarchical-solver#the-method)).
The choice comes from ADAM's operator study, `seam_operators.py`, which the figure reruns.

What the cases catch:
- a non-symmetric assembled operator: the symmetry defect of the assembled rows, which the assembly does not
  symmetrise;
- an inconsistent seam coupling: a truncation error that grows as $1/h$ under refinement;
- a multigrid that is not the spectral solve on a uniform grid, a ghost the fill leaves empty, a stalled FCG;
- a boundary kind whose row, right-hand side or ghost is wrong (E1b): the order against a manufactured potential, the
  truncation, the agreement with the spectral solver;
- a broken level inside the blocks (E5c): the iterations against the exact coarsest level;
- a facade that does not call the solver it chooses exactly (E1c), or instances that interfere;
- a configuration ELISHA must refuse: mixed periodic and bounded axes without `bc`, `bc` against the grid's
  periodicity, a ghost-Dirichlet face against another kind on one axis.

## Setup

| Leg | Domain and source | Grid | Asserted |
|---|---|---|---|
| SV-9a | SV-2's box and source, periodic | uniform octree and quadtree at level 2 | against the spectral solver $\le 10^{-8}$; at most 1 FCG iteration |
| SV-9b | the same | the central box $[L/4, 3L/4]$ refined once over base levels 2 and 3: octree (120 and 960 leaves), quadtree (28 and 112); $8^3$ cells per block, 3 ghosts | FCG residual below its tolerance; symmetry defect $\le 10^{-12}$; no ghost unfilled; order of $\Phi$ $\ge 1.7$; growth of $\max\lvert\tau\rvert \le 1.5$ |
| SV-12a | the same source without periodic boundaries; the exact potential at the ghost centres as the boundary values: $A\Phi = 4\pi V(\rho - \rho_0) - Bg$ | uniform octree at levels 1 and 2 | at most 1 iteration; order $\ge 1.7$; the truncation $\lvert A\Phi_{exact} + Bg - f\rvert$ bounded |
| SV-12b | the same | SV-9b's grids, octree and quadtree | as SV-9b, with the Dirichlet rows |
| E1b, uniform | SV-2's source, the kinds per axis (x, y, z) P/P D/N N/D; D/D N/N G/G; P/P N/N N/N (singular); 2-D N/D P/P | uniform, level 1 (2-D: level 2) | against the spectral solver with the same kinds $\le 10^{-8}$; 1 iteration; singular exactly without a Dirichlet face |
| E1b, AMR | the manufactured $\Phi_m = \prod_d\cos(2\pi x_d/L_d + \theta_d)$, $\theta = (0.3, 0.7, 1.1)$; its Laplacian the source, its boundary values of each kind the ghost values | the corner octant $[0, L/2]$ refined once over base levels 2 and 3 (leaves of both levels on the low faces) | as SV-9b, plus the order of the domain-face ghosts $\ge 1.7$ |
| E5c | SV-9's and SV-12's sources (periodic, D/D N/N G/G, P/P D/N N/D) and SV-13's polytrope (isolated) | base level 2: uniform octree, the box refined (octree, quadtree) | levels forced to one cell per block (`root_cells = 1`: 3 levels for $8^3$ blocks) against none (`root_cells = huge`): both below their tolerance, $\Phi$ within $10^{-7}$, the iterations at most 1.5 times the exact ones (20 where those are 1) |
| E1c | SV-2's source; the manufactured boundary values (through the facade's `boundary_faces`) | uniform (spectral path) and the corner refined (hierarchical path), 3-D and 2-D; the wide D(G) $p = 4$ | the facade against the solver it chooses called directly, every cell and ghost, bit for bit; three facades of a vector (a PEC face normal to z) solved interleaved against three alone, bit for bit |

The leaves are numbered in Morton order and dealt in contiguous chunks to the ranks.

## Results

![SV-9: three 2:1 couplings against N, and the hierarchical solver's errors on AMR](/verification/sv9-multigrid-amr.png)

*Left: the truncation error of S1 grows as $1/h$, M's stays bounded at about 40, an $O(1)$ seam truncation that is
still consistent. Middle: the potentials of the three couplings coincide at second order; the seam face gradient
converges at first order for M and stalls for S1. Right: the hierarchical solver on M in 3-D and 2-D: the potential
second order, the 2:1 face gravity first order, the truncation bounded.*

ELISHA (gfortran 16.2, np 2; iterations, orders and errors the same on np 1 and 3) next to ADAM's documented
numbers (`multigrid-amr.md`). ELISHA's V-cycle is the symmetric one ([the hierarchical
solver](/guide/hierarchical-solver#the-symmetric-v-cycle)), so its FCG iterations on AMR are fewer than ADAM's; every
order, error and truncation is the converged solution's and unchanged:

| Asserted | Bound | ELISHA | ADAM (CPU) |
|---|---|---|---|
| SV-9a: against the spectral solver; FCG iterations | $10^{-8}$; $\le 1$ | $1.5\cdot10^{-15}$ (3-D), $1.1\cdot10^{-15}$ (2-D); 1 | $\le 3.0\cdot10^{-15}$; 1 |
| SV-9b: symmetry defect of the assembled rows | $10^{-12}$ | $\le 4.7\cdot10^{-18}$ | $\le 7.9\cdot10^{-18}$ |
| SV-9b: order of $\Phi$, base level 2 → 3 | $\ge 1.7$ | 1.923 (3-D), 1.920 (2-D) | 1.92, 1.92 |
| SV-9b: growth of $\max\lvert\tau\rvert$ | $\le 1.5$ | 1.006 (3-D), 1.005 (2-D) | 1.006, 1.005 |
| SV-9b: 2:1 face gravity order (reported) | | 1.032 (3-D), 1.113 (2-D) | 1.03, 1.11 |
| SV-9b: FCG iterations (mean factor) | | 26, 28 (0.406, 0.431); 2-D 17, 17 (0.245, 0.258) | 36–39 (0.52–0.54); 24–25 (0.37–0.40) |
| SV-12a: iterations; order; truncation | $\le 1$; $\ge 1.7$; $\le 1.5$ growth | 1, 1; 1.952; 0.638 → 0.170 (growth 0.266) | 1; 1.95; 0.64 → 0.17 (0.27) |
| SV-12b: order of $\Phi$; iterations; symmetry | $\ge 1.7$; ; $10^{-12}$ | 1.907 (3-D), 1.926 (2-D); 24, 25 and 16, 16; $\le 3.1\cdot10^{-18}$ | 1.91, 1.93; 35–36, 22–23; $\le 6.7\cdot10^{-18}$ |
| E1b uniform: against the spectral solver; iterations | $10^{-8}$; $\le 1$ | $9.7\cdot10^{-15}$, $6.5\cdot10^{-16}$, $1.3\cdot10^{-15}$, $1.2\cdot10^{-14}$; 1 | $\le 1.2\cdot10^{-14}$; 1 |
| E1b AMR: order of $\Phi$ (P/P D/N N/D; D/D N/N G/G; P/P N/N N/N; 2-D N/D P/P) | $\ge 1.7$ | 1.951, 2.004, 1.948, 2.001 | 1.95, 2.00, 1.95, 2.00 |
| E1b AMR: order of the domain-face ghosts | $\ge 1.7$ | 1.955, 2.028, 1.952, 2.002 | 1.95, 2.03, 1.95, 2.00 |
| E1b AMR: $\max\lvert\tau\rvert$; growth; iterations | $\le 1.5$ | 11.95 (3-D), 10.32 → 10.53 (2-D); 1.003, 1.021; 23–29, 15 | 12.0, 10.3 → 10.5; 1.00–1.02; 31–40, 21 |
| E5c: iterations with the levels forced (exact coarsest) | | 14 (1), 32 (26), 19 (17), 31 (27), 38 (30), 36 (32) | 14 (1), 47 (36), 25 (24), 41 (39), 53 (40), 52 (47) |
| E5c: $\max\lvert\Phi_{sub} - \Phi_{exact}\rvert / \max\lvert\Phi\rvert$ | $10^{-7}$ | $1.6\cdot10^{-14}$ to $1.1\cdot10^{-11}$ | $1.6\cdot10^{-14}$ to $1.1\cdot10^{-11}$ |
| E1c: values differing, facade against direct; interleaved against alone | 0 | 0 in all six checks | 0 |
| E1c: the direct solve converges (AMR 3-D D/D N/N G/G; 2-D N/D P/P) | residual $\le$ tolerance | 24, 15 iterations ($4.9\cdot10^{-11}$, $6.3\cdot10^{-11}$) | 2-D: stalls, 200 iterations at 0.1157 (not asserted) |
| ghost cells unfilled | 0 | 0 | 0 |
| mixed axes without `bc`; `bc` against the grid; a ghost-Dirichlet face against a kind | refused | `ELISHA_ERR_BC_MISSING`; `_BC_PERIODICITY`; `_BC_GHOST_PAIR` | refused (stop) |
| facade: boundary values on the spectral path; isolated with `bc`; D(G) on AMR | refused | `_GHOST_VALUES`; `_ISOLATED_BC`; `_DG_AMR` | refused (stop) |

The symmetry defects depend on the rank and on the leaves' order (the two vectors are ADAM's pseudo-random ones,
indexed by the local cell), so they differ from ADAM's in the digits and across rank counts; every other number above
is the same on 1, 2 and 3 ranks, up to round-off in the last digit of the round-off-level ones.

CPU = device (job 1369178): the 18 hierarchical dumps (SV-9, SV-12, SV-13 grids) agree with the host's within
$7.85\cdot10^{-12}$. These are FCG solutions stopped at a relative residual of $10^{-10}$, so host and device agree
to about that tolerance, not to round-off: the bound is $10^{-8}$ (the device oracle test's), against $10^{-13}$ for
the direct spectral solves. Every device leg passed but `sub`, a crash in the device library under repair.

## Commands

```bash
mpirun -np 2 exe/elisha_sv_mg_mpi --sv9          # SV-9, SV-12 and their refusals
mpirun -np 2 exe/elisha_sv_mg_mpi --bc           # E1b
mpirun -np 2 exe/elisha_sv_mg_mpi --sub          # E5c
mpirun -np 2 exe/elisha_sv_mg_mpi --flail        # E1c and the facade's refusals
mpirun -np 2 exe/elisha_sv_mg_mpi --mixed        # one refusal (or --refuse-bc-mismatch, --refuse-bc-mixed-ghost, ...)
scripts/verification/poisson.sh --legs "sv9 bcmg sub flail"
python3 scripts/verification/seam_operators.py   # the operator study, N = 8 16 32 48
```

## Files

| File | Role |
|---|---|
| [`src/tests/elisha_sv_mg_mpi.F90`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_mg_mpi.F90) | the driver (ADAM's `test_poisson_mg`, ported) |
| [`scripts/verification/seam_operators.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/seam_operators.py) | the three couplings across 2:1 faces |
| [`scripts/prototypes/mg_subblock_e5c.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/prototypes/mg_subblock_e5c.py) | the levels inside the blocks on cubic cells |
| [`scripts/verification/make_verification_figures.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/make_verification_figures.py) | the figure (`fig_sv9`) |

## Findings

- **S1 is refuted, M is chosen.** The symmetrisation of flux matching has a second-order potential but a truncation
  that grows as $1/h$ and a seam gravity that does not converge. M is symmetric and consistent; it pays one order of
  seam gravity (first order, on the one layer of 2:1 faces), the potential stays second order.
- **Dirichlet boundaries cost nothing in symmetry or order** (SV-12): the coarsest level's odd extension is exact, one
  iteration on a uniform grid.
- **The levels inside the blocks trade iterations for the root's gather** (E5c): 1 to 14 iterations on a uniform box
  of 1:2:3 cells, 2 to 8 more on AMR with the symmetric V-cycle (1 to 13 with ADAM's)
  ([the hierarchical solver](/guide/hierarchical-solver#the-method)).
- **ADAM's V-cycle stalled FCG on the E1c 2-D N/D P/P solve; the symmetric one converges in 15 iterations.** The
  stall was 200 iterations with the residual at 0.1157, and ADAM's own multigrid stalled identically, so the defect
  was inherited from ADAM. Its cause: the V-cycle read the coarser correction across a 2:1 face on the way up but never
  applied the adjoint on the way down. That left the preconditioner nonsymmetric on AMR, with a symmetric part
  indefinite on some seam modes ([the symmetric V-cycle](/guide/hierarchical-solver#the-symmetric-v-cycle)). The E1c
  leg now asserts that the direct solve converges, on every AMR case.
- **A convergence that depends on the right-hand side was a preconditioner defect.** Before the fix, E1b's AMR
  2-D N/D P/P case (the same mesh, operator and V-cycle as E1c's) converged in 21 iterations while E1c's stalled. For a
  symmetric positive definite pair of operator and preconditioner, FCG converges from any right-hand side; a stall
  that a change of data cures points at the preconditioner. `vcycle_symmetry_defect` now guards it
  (`elisha_test_hierarchical_solver_mpi`: at most $10^{-12}$, about $10^{-17}$ measured).
