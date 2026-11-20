# The hierarchical solver

`elisha_hierarchical_solver_object` solves `A phi = factor V f - B g` on the leaves of a 2:1-balanced octree of
uniform blocks (a quadtree with a null axis), periodic, with a kind per face, or isolated. `A` is the composite
second-order Laplacian in finite-volume form, `V` the cell volumes and `B` the map of the boundary values `g` onto
the boundary rows. The method is flexible conjugate gradients (Notay's FCG(1)), preconditioned by one V-cycle whose
root is the spectral solver. It is ADAM's `poisson_mg_object` (ADAM 669d0bdd) moved onto ELISHA's mesh,
communicator and status. Its arithmetic is ADAM's, held bitwise by a test on every path (below).

```fortran
type(elisha_hierarchical_solver_object) :: mg
call mg%setup(mesh, comm, stat, errmsg, bc=bc)                           ! tolerance, max_iterations, coarse, isolated, root_cells
call mg%solve_packed(mesh, buf, factor, phi, stat, errmsg, energy=e, ghost=g)
```

| Procedure | Role |
|-----------|------|
| `setup(mesh, comm, stat, errmsg[, tolerance][, max_iterations][, coarse][, isolated][, bc][, root_cells])` | Validate the mesh, check the arguments, assemble the operators and the level hierarchy, size the root (a spectral solver of the class of `coarse`) |
| `solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy][, ghost])` | FCG to `tolerance`, warm-started while the mesh is unchanged; `phi` with every ghost layer; `ghost`: the value of each boundary face |
| `symmetry_defect(x, y, stat)`, `apply_a(x, y, stat)` | Verification: `abs(x.Ay - y.Ax) / (abs(x) abs(Ay))`, `y = A x` |
| `apply_b(g, y)`, `apply_bt(x, s)` | `y = B g`, `s = B^T x` over the boundary faces (`nbnd`, `bnd_row`, `bnd_w`, `bnd_kind`, `bnd_dir`, `bnd_x`) |

`MG_SEAM_GHOST_SCALE` (1.5) is the composite operator's face gradient across a 2:1 face: the fine-side ghost is
`phi_f - (mean_f - phi_C) / 1.5`.

## The method

| Part | Choice |
|------|--------|
| Operator | The 7-point stencil between leaves of one level; across a 2:1 face the mimetic mean flux, exact for linear `phi` face by face and symmetric in the V-weighted form (the exact energy exchange of self-gravity) |
| Krylov | FCG(1), relative max-norm residual: `tolerance` (default 1e-10) or the source's round-off floor, whichever is larger |
| V-cycle | Leaves plus one shadow block per refined node; red-black Gauss-Seidel, 2 sweeps down and 2 up; restriction by sum, piecewise-constant prolongation |
| Below the coarsest tree level | The blocks coarsen inside themselves while their cell counts are even and above `root_cells` |
| Root | The spectral solver's `solve_blocks`: on every rank (redundant) up to 32^3 cells, distributed above |
| Isolated | The James step: a zero-ghost solve, the screening charge, its free-space ghost values by the lattice Green's function, the solve with them |
| Distribution | CSR rows over global cell keys, one halo plan per operator |

## What changed from ADAM, and what did not

The plumbing changed as in the spectral solver:
- the mesh descriptor replaces ADAM's `grid` and `field`, and the leaf list is the mesh's validated gather;
- every collective goes through `elisha_comm_object`, so the serial build runs on one rank;
- failures return a status;
- the root is the class of `elisha_spectral_solver_object`.

Three changes leave every result bit unchanged:
- **Persistent halo buffers.** The host `apply` and `smooth` reuse halo buffers allocated once; ADAM allocated them at every call.
- **Asserted colouring.** After each level operator is assembled, every off-diagonal entry must link the two red-black colours, which the device smoother relies on to update a colour in parallel.
- **Mixed lookup keys.** The cell-key table mixes each key before its modulo. ADAM's `modulo(key, cap)` with linear probing clusters on the key lattice: about 51 000 probes per insert on a halo of 131 072 cells. The table maps a key to its value and ids are numbered in first-seen order, so no result depends on where a key sits.

`elisha_test_hierarchical_parity_mpi` runs ELISHA's solver and ADAM's (compiled unchanged in the tests) on the same
source. It compares the bit patterns of:
- the potential with every ghost layer, and the energy;
- the iterations, the residual, the tolerance used and the mean convergence factor, after a cold and a warm-started
  solve;
- the boundary-face list, the symmetry defect, and `A`, `B`, `B^T` on fixed vectors.

It runs 17 cases:
- uniform, with levels inside the blocks;
- 2:1 AMR, at a corner across the periodic wrap and inside;
- the exact root;
- ghost-Dirichlet AMR;
- Dirichlet, mixed and ghost-Dirichlet faces with boundary values;
- Neumann-only (singular);
- isolated, uniform and AMR;
- 2-D AMR;
- a distributed root.

All 17 are bitwise at np 1, 2 and 3 on gfortran (debug and optimised), nvfortran and ifx. Reassociating `beta`, or
swapping the red-black order, breaks 14 of the 17; the three exact-root cases never smooth.

## Performance

`exe-bench/elisha_bench_hierarchical_mpi [reps [AE|EA [cases]]]` (modes `bench-hierarchical-gnu`, `-nvf`) times both
solvers from zero on a Gaussian on [-1, 1]^3. The AMR cases have 491 520 cells with the central box refined.

| Case | Iterations | Solve E/A, median | Setup ADAM, np 1 / 2 / 4 | Setup ELISHA, np 1 / 2 / 4 |
|------|------------|-------------------|--------------------------|----------------------------|
| AMR periodic | 19 | 0.87 to 1.01 | 3.1 / 51 / 115 s | 0.93 / 0.47 / 0.24 s |
| AMR ghost-Dirichlet | 21 | 0.91 to 1.00 | 3.0 / 51 / 107 s | 0.88 / 0.45 / 0.24 s |
| AMR isolated | 39 (two solves) | 0.90 to 0.97 | 3.0 / 51 / 107 s | 0.95 / 0.49 / 0.26 s |
| Uniform 64^3, D/N faces, redundant root | 8 | 0.89 to 1.01 | | |

These are gfortran and nvfortran on the CPU, at np 1, 2 and 4, in both orders of the pair; every row is bitwise. The
setup speed-up is the key mixing. ADAM's setup above one rank is dominated by its table, and the fix carries over to
it unchanged.
