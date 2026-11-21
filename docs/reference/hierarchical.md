# The hierarchical solver

`elisha_hierarchical_solver_object` solves `A phi = factor V f - B g` on the leaves of a 2:1-balanced octree of
uniform blocks (a quadtree with a null axis), periodic, with a kind per face, or isolated. `A` is the composite
second-order Laplacian in finite-volume form, `V` the cell volumes and `B` the map of the boundary values `g` onto
the boundary rows. The method is flexible conjugate gradients (Notay's FCG(1)), preconditioned by one V-cycle whose
root is the spectral solver. It is ADAM's `poisson_mg_object` (ADAM 669d0bdd) moved onto ELISHA's mesh,
communicator and status. Its arithmetic is ADAM's except for one fix: the V-cycle is symmetric
([below](#the-symmetric-v-cycle)). With `vcycle=ELISHA_VCYCLE_ADAM` the arithmetic is ADAM's exactly, and a test holds
it bitwise on every path.

```fortran
type(elisha_hierarchical_solver_object) :: mg
call mg%setup(mesh, comm, stat, errmsg, bc=bc)                           ! tolerance, max_iterations, coarse, isolated, root_cells, vcycle
call mg%solve_packed(mesh, buf, factor, phi, stat, errmsg, energy=e, ghost=g)
```

| Procedure | Role |
|-----------|------|
| `setup(mesh, comm, stat, errmsg[, tolerance][, max_iterations][, coarse][, isolated][, bc][, root_cells][, level_operator][, vcycle][, assembly])` | Validate the mesh, check the arguments, assemble the operators and the level hierarchy, size the root (a spectral solver of the class of `coarse`) |
| `solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy][, ghost])` | FCG to `tolerance`, warm-started while the mesh is unchanged; `phi` with every layer of its face ghosts (not the edge and corner ghosts); `ghost`: the value of each boundary face. At `max_iterations` above the stop tolerance: `ELISHA_ERR_NOT_CONVERGED` on every rank, `phi` the last iterate, `converged` `.false.` (ADAM printed a warning) |
| `symmetry_defect(x, y, stat)`, `apply_a(x, y, stat)` | Verification: `abs(x.Ay - y.Ax) / (abs(x) abs(Ay))`, `y = A x` |
| `vcycle_symmetry_defect(x, y, stat)` | Verification: `abs(x.My - y.Mx) / (abs(x) abs(My))`, `M` the V-cycle |
| `apply_b(g, y)`, `apply_bt(x, s)` | `y = B g`, `s = B^T x` over the boundary faces (`nbnd`, `bnd_row`, `bnd_w`, `bnd_kind`, `bnd_dir`, `bnd_x`) |

`MG_SEAM_GHOST_SCALE` (1.5) is the composite operator's face gradient across a 2:1 face: the fine-side ghost is
`phi_f - (mean_f - phi_C) / 1.5`.

## The method

| Part | Choice |
|------|--------|
| Operator | The 7-point stencil between leaves of one level; across a 2:1 face the mimetic mean flux, exact for linear `phi` face by face and symmetric in the V-weighted form (the exact energy exchange of self-gravity) |
| Krylov | FCG(1), relative max-norm residual: `tolerance` (default 1e-10) or the source's round-off floor, whichever is larger |
| V-cycle | Leaves plus one shadow block per refined node; red-black Gauss-Seidel, 2 sweeps down and 2 up; restriction by sum, piecewise-constant prolongation; across a 2:1 face, the coarser correction read on the way up and its adjoint applied on the way down, so the V-cycle is symmetric |
| Below the coarsest tree level | The blocks coarsen inside themselves while their cell counts are even and above `root_cells` |
| Root | The spectral solver's `solve_blocks`: on every rank (redundant) up to 32^3 cells, distributed above |
| Isolated | The James step: a zero-ghost solve, the screening charge, its free-space ghost values by the lattice Green's function, the solve with them |
| Distribution | CSR rows over global cell keys, one halo plan per operator |

The root follows Athena++ (Tomida and Stone 2023, section 3). ADAM measured, with its own V-cycle, what coarsening inside the blocks costs
(`test_poisson_mg --sub`, now the `sub` leg of `scripts/verification/poisson.sh`, `exe/elisha_sv_mg_mpi --sub`): 1 to
14 FCG iterations on a uniform box of 1:2:3 cells, from point smoothing on anisotropic cells, and 1 to 13 more on AMR
(36 to 47, 24 to 25, 39 to 41, 40 to 53 and 47 to 52 in its five AMR configurations). On cubic cells, where the exact
coarsest level takes 1 iteration, the same levels take 8 (periodic) and 11 to 15 (ghost-Dirichlet faces), and
trilinear prolongation gains nothing over piecewise constant: the `exact`, `pc` and `tl` columns of
[`mg_subblock_e5c.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/prototypes/mg_subblock_e5c.py)
(`python3 scripts/prototypes/mg_subblock_e5c.py`, about 1.5 minutes). In exchange, no V-cycle gathers the base grid on
rank 0.

**Boundary faces.** A bounded face has a kind. Its row stays homogeneous, so the operator stays symmetric, and its
value `g` goes to the right-hand side through `B`. Here `w` is the face area over the spacing, `h` the spacing and
`phi_C` the boundary cell's value:

| Kind | `g` | Row | `B` | Ghost |
|------|-----|-----|-----|-------|
| `ELISHA_BC_DIRICHLET_GHOST` (default on a bounded axis) | value at the ghost centre | `-w phi_C` | `w` | `g` |
| `ELISHA_BC_DIRICHLET` | value at the face | `-2 w phi_C` | `2 w` | `2 g - phi_C` |
| `ELISHA_BC_NEUMANN` | outward derivative at the face | none | `w h` | `phi_C + h g` |

Without a Dirichlet face `A` is singular, and the means of `factor V f - B g` and of the solution are removed. The
V-cycle's levels take the same kinds, homogeneous. Through the facade, `boundary_faces` lists the faces and `ghost`
gives their values ([The solver facade](/reference/solver#boundary-values)).

## The symmetric V-cycle

The default V-cycle (`ELISHA_VCYCLE_SYMMETRIC`) changes ADAM's numbers on every AMR mesh. ADAM's V-cycle was not
symmetric there, and FCG could stall on it.

Some notation: `E_l` couples a level-`l` cell to the coarser leaf across a 2:1 face. Its rows are the level's cells
next to that face.
- **ADAM's V-cycle.** On the way up, each level reads the coarser correction through `E_l`. On the way down, the
  restriction sums the residual into the shadow cells only. Nothing applies the adjoint `E_l^T` to the coarser
  leaves.
- **Why that stalls.** The preconditioner `M` is then not symmetric, and its symmetric part is indefinite on some
  modes at the seam. FCG steps by `(p, r) / (p, A p)` with `(p, r) = (M r, r)`, so the step vanishes where
  `(M r, r)` does.
- **The case that showed it.** The 2-D N/D P/P solve of E1c stalled at 200 iterations with the residual at 0.1157.
  ADAM's code compiled unchanged stalls identically. The step `alpha` fell to 1e-16 by iteration 40, and
  `(M r, r)` was 1.8e-16 with `abs(M r)` 10 and `abs(r)` 0.72.
- **Why only this case.** E1b solves on the same mesh, with the same operator and V-cycle, and converges in 21
  iterations: only the right-hand side differs.

Now the way down subtracts `E_l^T e_l` from the coarser level's right-hand side after the restriction. Here `e_l` is
the level's smoothed correction. The operator `etl` is assembled from the coarse side and holds `E_l`'s entries
transposed: each coarse leaf's faces to a refined neighbour, the fine cells on each such face, and the fine face
weight.

| Measure | ADAM's V-cycle | Symmetric V-cycle |
|---------|----------------|-------------------|
| `vcycle_symmetry_defect` on AMR (`elisha_test_hierarchical_solver_mpi`, np 1 to 3) | 7.9e-5 to 1.2e-3 | 1e-17 or less |
| The same, uniform meshes | 1e-17 or less | 1e-17 or less (no seam: the same V-cycle) |
| E1c, 2-D N/D P/P AMR | 200 iterations, residual 0.1157 | 15 iterations |
| AMR solves of `elisha_sv_mg_mpi`, every leg | 21 to 53 iterations | 15 to 38, 19 to 35% fewer |
| AMR cases of `elisha_test_hierarchical_solver_mpi` | 20 to 22 | 16 |
| Strong ladder, 3.9 M AMR cells, CPU, np 1, 16, 64: FCG iterations per solve, periodic; isolated final solve | 17.8; 16.0 | 12.0; 12.0 |
| The same: minimum solve time ([Performance](/guide/performance#scaling-eb), jobs 1369135 and 1369212) | 8.27, 1.05, 0.409 s periodic; 25.2, 3.41, 1.39 s isolated | 28 to 30% less periodic, 24% less isolated |

The extra work per V-cycle is one sparse product per level above `lmin`, over the seam rows only. Uniform meshes
have no seam, so their results are unchanged bit for bit. `setup(..., vcycle=ELISHA_VCYCLE_ADAM)` keeps ADAM's
V-cycle. The parity tests below use it to hold the rest of the port bitwise to ADAM's code. A client checking its
results against ADAM's can use it too, then switch to the default.

## What changed from ADAM, and what did not

The plumbing changed as in the spectral solver:
- the mesh descriptor replaces ADAM's `grid` and `field`, and the leaf list is the mesh's validated gather;
- every collective goes through `elisha_comm_object`, so the serial build runs on one rank;
- failures return a status;
- the root is the class of `elisha_spectral_solver_object`.

Five changes leave every result bit unchanged:
- **Persistent halo buffers.** The host `apply` and `smooth` reuse halo buffers allocated once; ADAM allocated them at every call.
- **Asserted colouring.** After each level operator is assembled, every off-diagonal entry must link the two red-black colours, which the device smoother relies on to update a colour in parallel.
- **Integer colour (M7, audit C3).** A level cell's red-black colour is an integer, 0 or 1, on the host and on the device. ADAM stored it as a real and compared `nint` of it. The colour only selects rows, so no arithmetic changes.
- **Neighbour halos (M7, audit B1).** Every halo exchange of `apply` and of the smoother (host and device) is `MPI_Neighbor_alltoallv` on a distributed-graph communicator. Each operator's plan builds that communicator once at setup from the ranks it sends to and receives from, and `destroy` frees it. ADAM called `MPI_ALLTOALLV` with P-sized count arrays at every call. The buffers, displacements and values received are unchanged. On the benchmark mesh with up to 16 ranks the solve time did not change measurably, because the count arrays there are small next to the messages. The cost per call now grows with the number of neighbours, not with P.
- **Mixed lookup keys.** The cell-key table mixes each key before its modulo. ADAM's `modulo(key, cap)` with linear probing clusters on the key lattice: about 51 000 probes per insert on a halo of 131 072 cells. The table maps a key to its value and ids are numbered in first-seen order, so no result depends on where a key sits.

`elisha_test_hierarchical_parity_mpi` runs ELISHA's solver with `vcycle=ELISHA_VCYCLE_ADAM` and ADAM's (compiled
unchanged in the tests) on the same source. It compares the bit patterns of:
- the potential with every layer of its face ghosts, and the energy;
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

## Stencil level operators

`setup(..., level_operator=ELISHA_OPERATOR_STENCIL)` is an M7 change (audit C2): the OpenACC backend's default; the
host solver's default stays `ELISHA_OPERATOR_CSR`, ADAM's. It changes how the V-cycle's level operators `A_l` are applied and smoothed. A row
whose cell lies inside its block on every active axis is the constant-coefficient 7-point stencil: its six
neighbours are in the same block and level, and no seam, halo or domain face touches it. Such a row is computed from
`x` at the seven cells and one weight per axis, without reading its 12 CSR entries and its row pointer (about 140 B a
row). Every other row keeps its CSR entries: a block face reads a halo, a 2:1 seam's ghost term, or a domain face's
kind. The CSR stays assembled, so both forms coexist on one level. The stencil takes rows in the CSR entries' order:
for each active axis, the diagonal's -w, the low neighbour's w, the diagonal's -w, the high neighbour's w; the
smoother takes the neighbours only.

| Compiler | Stencil against CSR |
|----------|---------------------|
| gfortran 16.2, ifx 2026.2 | bitwise: operators, smoother, potential, iterations |
| nvfortran 26.9 (`-fast`), host | operators and smoother within 4e-16, potential within 9e-16, the same iterations |
| nvfortran 26.9, A30 (jobs 1369032, 1369037) | bitwise CSR, and bitwise ADAM FNL |

The stencil takes 42% of the rows on blocks of 8^3 cells (6^3 of 8^3); levels with 4 cells per block or fewer have
few or no interior rows. `elisha_test_stencil_mpi` checks it on uniform, AMR, Dirichlet/Neumann with boundary values,
isolated and 2-D meshes.

Measured on the host (`elisha_bench_hierarchical_mpi ... stencil`; cases (a) AMR periodic and (d) uniform D/N, np 1,
2, 4, both orders, i9-13900KF under WSL), the mean E/A against ADAM was:

| Compiler | CSR | Stencil |
|----------|-----|---------|
| gfortran | 0.92 | 0.93 |
| nvfortran | 0.98 | 0.93 |

The run-to-run noise is about 5%, so the stencil shows no measurable gain on the host. On the A30, where memory
bandwidth is the limit (`spmv_sten_dev`, `sweep_sten_dev`), it gains 10 to 15% on the multigrid solve: on the
3.9 M-cell periodic case, E/A goes from 0.97 / 0.94 with CSR to 0.83 / 0.83 (job 1369037,
[Performance](/guide/performance#step-09-stencil-level-operators-audit-c2)). So it is the default of the OpenACC
backend, where it stays bitwise ADAM FNL's, and an option on the host, where CSR stays the default (no gain there,
and nvfortran `-fast` rounds differently).

## Performance

The M7 changes (integer colour, neighbour halos, the device root, the stencil option) are measured one by one in
[Performance](/guide/performance). The table below is the port (M4) against ADAM.

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
