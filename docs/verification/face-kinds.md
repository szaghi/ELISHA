# Face kinds and the wide D(G)

| ID | Solver | Program | Leg | Figure |
|---|---|---|---|---|
| E1a, E2a, E2b, E3a, E3c (ADAM #84) | `elisha_spectral_solver_object` (`initialize_blocks`, `solve_blocks`) | `elisha_sv_bc_mpi` | `bc` | no |

## Case and scope

PRISM's elliptic solves need more than periodic and isolated boundaries: Dirichlet (PEC) and Neumann faces, often
different at the two ends of an axis, zero-ghost and free-space faces, and the wide operator D(G), the order-$p$
centred first derivative composed with itself, which PRISM applies. The spectral solver extends the source per axis
(odd about a Dirichlet face, even about a Neumann one, over $2n$ cells, or $4n$ when the faces differ; odd about the
ghost centres, $2(n+1)$ cells, for ghost-Dirichlet) and transforms the extension periodically; zero-ghost and
free-space faces with D(G) are solved by CG on rank 0, preconditioned by the $p = 2$ operator inverted exactly
([the spectral solver](/guide/spectral-solver#problems-and-methods)). The driver applies the operators itself, from
sources it shares no code with: the D1 coefficients come from GIDEON's generated rows (`gideon_tables`), PRISM's
gradient and divergence from GIDEON's centred FD and FV operators, never from ELISHA's own stencil.

What the case catches:
- a wrong extension, length or eigenvalue, a wrong ghost in the scatter, images leaking into the energy (E1a, E2a);
- a D(G) that is not PRISM's: the eigenvalue is computed in real space, D1 applied twice to the eigenfunction (E2a);
- a zero-ghost solve that is not D(G)'s, a preconditioner that does not bound the iterations in the grid (E2b);
- a solution that PRISM's own operators do not map back onto the source (E3a);
- a free-space ghost that is not the direct Green's sum, or depends on the decomposition (E3c);
- a configuration ELISHA must refuse.

## Setup

| Leg | Case | Asserted |
|---|---|---|
| E1a | per axis a discrete eigenfunction that satisfies the axis's faces, at the cell centres: P/P $\cos(2\pi kx/L)$, D/D $\sin(\pi kx/L)$, N/N $\cos(\pi kx/L)$, D/N $\sin(\pi(k + \tfrac12)x/L)$, N/D $\cos(\pi(k + \tfrac12)x/L)$, G/G $\sin(\pi ki/(n + 1))$; their product the source (plus a constant without a Dirichlet axis, which the solve must drop); 8 configurations, 6 in 3-D on $4^3$ blocks of $4^3$ cells, 2 in 2-D; the box $[0,1]\times[0,2]\times[0,3]$ | $\max\lvert\phi - \phi_h\rvert / \max\lvert\phi_h\rvert$ over the interior and the ghost layer, and the energy against the exact sum, $\le 10^{-12}$ |
| E2a | the same with `dg_order` = 2, 4, 6, 8, the ghost layers read back to depth $p$ | the same bounds |
| E2b | zero-ghost faces with D(G) $p = 2..8$ (Z/Z P/P P/P; Z/Z D/D N/N; Z/Z Z/Z Z/Z; 2-D Z/Z N/N), $16^3$ cells: a manufactured $u$, its source $D(G)u$ computed here | $u$ recovered to $10^{-9}$; the ghosts returned equal to the rules applied to the interior; CG iterations $\le 40$; at $p = 4$ the iterations on 16 and 32 cells per axis within one; the compact operator with zero ghosts equal to ghost-Dirichlet, bit for bit |
| E3a | $f = \nabla\cdot\nabla u$ by PRISM's centred FD and FV operators of order $2s$, $s = 1..4$ (periodic, D/D, N/N D/D, D/N P/P N/D, Z/Z, Z/Z D/D N/N), one $16^3$ block | $\nabla\cdot\nabla\phi$ from the solution and its returned ghosts equal to $f$: $\le 10^{-11}$ (direct), $\le 10^{-9}$ (zero ghosts, CG) |
| E3c | free-space faces with D(G) $p = 2, 4, 6$ (F everywhere; F next to D/D and P/P; F/Z N/N F/F), $2^3$ blocks of $4^3$ cells | the ghost slabs against the direct sum $-\sum f\,dV/(4\pi r)$, $\le 10^{-12}$; $D(G)\phi = $ factor $f$ with them, $\le 10^{-10}$; CG iterations $\le 40$ |
| order | D/D D/D D/D on $16^3$, $32^3$, $64^3$ cells against the continuum | compact $\ge 1.9$; wide $p = 4$ $\ge 3.8$ |

## Results

ELISHA (gfortran 16.2, np 2; iterations and orders the same on np 1 and 3, and on nvfortran):

| Asserted | Bound | ELISHA | ADAM |
|---|---|---|---|
| E1a, compact: discrete error; energy | $10^{-12}$ | $\le 3.7\cdot10^{-15}$; $\le 2.6\cdot10^{-14}$ | round-off |
| E2a, D(G) $p = 2, 4, 6, 8$: discrete error; energy | $10^{-12}$ | $\le 8.8\cdot10^{-15}$; $\le 2.1\cdot10^{-14}$ | round-off |
| E2b, zero ghosts: $u$ recovered; ghosts against the rules | $10^{-9}$; round-off | $\le 2.3\cdot10^{-12}$; 0 | |
| E2b: CG iterations, $p = 2, 4, 6, 8$ | $\le 40$ | 1, 18–19, 26, 31–32 | the prototype's 1, 17, 24, 30 (another problem) |
| E2b: iterations at $p = 4$, 16 and 32 cells per axis | within 1 | 19, 19 | |
| E2b: compact, Z/Z against G/G | bitwise | 0 values differ | bitwise |
| E2b: the CG stopped at `max_iterations` = 1 (p = 4, Z/Z D/D N/N); then 200 | `ELISHA_ERR_NOT_CONVERGED`, the last iterate finite; converged | as expected, np 1 to 3 | a rank-0 warning (not ADAM's status) |
| E3a: PRISM's FD and FV, direct solves | $10^{-11}$ | $5\cdot10^{-16}$ to $9.9\cdot10^{-16}$ | $10^{-15}$ |
| E3a: zero-ghost (CG) solves | $10^{-9}$ | $\le 8.4\cdot10^{-13}$ | |
| E3c: free ghosts against the direct sum | $10^{-12}$ | $\le 2.9\cdot10^{-16}$ | |
| E3c: $D(G)\phi$ against factor $f$; CG iterations $p = 2, 4, 6$ | $10^{-10}$; $\le 40$ | $\le 4.0\cdot10^{-12}$; 1, 17–18, 24 | |
| D/D order against the continuum, compact ($16\to32\to64$) | $\ge 1.9$ | 1.976, 1.994 | 1.99 |
| D/D order, wide $p = 4$ | $\ge 3.8$ | 3.950, 3.987 | |
| refusals: P against D on an axis; G against a kind; D(G) isolated; Z against D; F with the compact operator | refused | `ELISHA_ERR_BC_PERIODIC_PAIR`; `_BC_GHOST_PAIR`; `_ISOLATED_WIDE`; `_BC_ZERO_GHOST_PAIR`; `_BC_FREE_GHOSTS` | refused (stop) |

ADAM published no per-configuration numbers for this test beyond the D/D order and the $10^{-15}$ of PRISM's
operators; its prototypes measured the zero-ghost iterations on their own problems
([the spectral solver](/guide/spectral-solver#problems-and-methods)).

## Commands

```bash
mpirun -np 2 exe/elisha_sv_bc_mpi                          # every leg
mpirun -np 2 exe/elisha_sv_bc_mpi --zero --free            # some legs
mpirun -np 2 exe/elisha_sv_bc_mpi --refuse-zero-mixed      # one refusal
scripts/verification/poisson.sh --legs bc
```

## Files

| File | Role |
|---|---|
| [`src/tests/elisha_sv_bc_mpi.F90`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_bc_mpi.F90) | the driver (ADAM's `test_flail_bc`, ported; GIDEON for the operators) |
| [`scripts/prototypes/wide_dg_zero_ghost.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/prototypes/wide_dg_zero_ghost.py), [`wide_dg_e2b.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/prototypes/wide_dg_e2b.py), [`wide_dg_proto.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/prototypes/wide_dg_proto.py) | ADAM's prototypes of the zero-ghost CG and its preconditioner |

## Findings

- Every face kind, with the compact and the wide operator, is exact for its operator at round-off, the ghost layers
  included, and the energy carries no image.
- The zero-ghost CG takes a number of iterations set by $p$ alone, the same at 16 and 32 cells per axis.
- The solver inverts exactly the operator PRISM applies, FD and FV, every order.
