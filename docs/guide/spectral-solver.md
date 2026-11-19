# The spectral solver

`elisha_spectral_solver_object` solves `L_h phi = factor (f - mean f)` (periodic) or `L_h phi = factor f` on a
uniform mesh, every block at one level, by FFT (FFTW 3 on the host). `L_h` is the compact second-order Laplacian
(7-point, 5-point in 2-D) or, with `dg_order`, PRISM's wide operator D(G). The solver is ADAM's `poisson_fft_object`
(ADAM 669d0bdd) moved onto ELISHA's mesh, communicator and status. Its arithmetic is ADAM's, held bitwise by a test
on every path (below).

```fortran
type(elisha_spectral_solver_object) :: fft
call fft%setup(mesh, comm, stat, errmsg, bc=bc)                        ! or isolated=.true., dg_order=p
call fft%solve_packed(mesh, buf, factor, phi, stat, errmsg, energy=e)  ! buf: interior source, block-major, i fastest
```

| Procedure | Role |
|-----------|------|
| `setup(mesh, comm, stat, errmsg[, isolated][, bc][, dg_order])` | Validate the mesh (one level), check the arguments, size the solver, make the plans (and, isolated, the kernel's spectrum) |
| `solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy])` | Solve from the packed interior source; `phi(1-ngc:n+ngc, ..., b)` returns with every ghost layer; `energy = sum f phi` |
| `initialize_blocks(nblk, nijk, ngc, dxyz, level, null_xyz, isolated, comm, stat, errmsg, ...)` | Size the solver for an explicit uniform block lattice without a mesh (the hierarchical solver's root); `redundant`: every rank solves |
| `solve_blocks(nbl, coords, buf, factor, gw, out, stat, errmsg[, energy])` | Solve on an explicit list of blocks, returning `gw` ghost layers |
| `global_potential(g, stat)`, `destroy()` | The transform array on rank 0 (diagnostics); cleanup |

The packing of one variable of a conservative array (ADAM's `solve(q, iv)`) belongs to the caller: ELISHA does not
assume a field layout.

## Problems and methods

| Problem | Method | Residual |
|---------|--------|----------|
| Periodic | Each Fourier mode divided by the eigenvalue of `L_h` | round-off |
| Dirichlet and Neumann faces | Odd and even extensions over 2n or 4n cells, the periodic transform of the extension | round-off |
| Ghost-Dirichlet | Odd about the ghost centres, 2 (n + 1) cells (type-I sine transform) | round-off |
| Isolated (Hockney and Eastwood) | Zero padding to a 13-smooth size at or above 2 (n + ngc), convolution with the lattice Green's function | about 1e-13 per offset |
| Wide D(G), order p | Eigenvalue `-s_p(theta)^2 / dx^2`; the checkerboard modes dropped | round-off |
| Zero-ghost faces with D(G) | CG on rank 0, preconditioned by the p = 2 operator inverted exactly (2^z FFT sub-solves) | 1e-12 |
| Free-space ghosts with D(G) | The zero-ghost solve with the ghosts' free-space potential on the right-hand side | 1e-12 |

The data path is ADAM's distributed slab transform: z-planes per rank, `MPI_ALLTOALLV` to z-lines, and every block
fetches its cells with their ghosts from the owners, so no ghost exchange follows. The potential is bitwise the same
on every rank count. The CG paths and the redundant root gather to rank 0 (or to every rank).

## What changed from ADAM, and what did not

Only the plumbing changed:
- `grid` and `field` became the mesh descriptor;
- `MPI_COMM_WORLD` became the caller's communicator, and every collective goes through `elisha_comm_object`, so the
  serial build runs on one rank;
- `error stop` became a status: argument refusals call the setup checks, and runtime refusals return
  `ELISHA_ERR_NOT_INITIALIZED`, `_SHAPE`, `_MESH_CHANGED`, `_ENERGY_REDUNDANT`, `_NO_FFT` or `_FFT_PLAN`.

Where a failure can reach some ranks only, one integer reduction agrees on the status, so no rank hangs in the next
collective.

Not changed: every floating-point expression, loop order, accumulation order and FFTW plan.
`elisha_test_spectral_parity_mpi` runs ELISHA's solver and ADAM's (compiled unchanged in the tests) on the same
source and compares bit patterns of the potential with every ghost layer, and of the energy, on 26 cases:
- periodic, in 3-D and 2-D;
- every face-kind family, and a mix across axes;
- isolated;
- D(G) for p = 2..8, with and without faces;
- the CG of zero and free ghosts;
- the redundant root;
- the ghost-Dirichlet root;
- the rank-0 FFT path.

All 26 are bitwise at np 1, 2, 3 on gfortran (debug and optimised), nvfortran and ifx. A one-ulp reassociation in
the library breaks 17 cases on the slab path and 7 on the rank-0 path, so the test detects one-ulp changes on both.

## Performance

`exe-bench/elisha_bench_spectral_mpi [reps [EA]]` (modes `bench-gnu`, `bench-nvf`) times both solvers on a Gaussian
on [-1, 1]^3. Every row checks that the two potentials are bitwise equal.

| Case | gfortran, E/A (median) | nvfortran CPU, E/A (median) |
|------|------------------------|-----------------------------|
| 128^3 periodic, np 1, 2, 4 | 0.97 to 1.02 | 0.95 to 1.01 |
| 128^3 isolated, np 1, 2, 4 | 0.97 to 1.01 | 0.95 to 0.99 |
| 64^3 periodic, np 1, 2, 4 | 0.99 to 1.02 | 0.99 to 1.02 |

Machine: an i9-13900KF under WSL, 15 and 61 repetitions, both orders of the pair. On this machine the first solve of
a pair is up to 10% faster, so a single order biases the minimum. Measure both orders, as `EA` allows. nvfortran 26.9
optimised the transpose's copy loops less when a component of the solver was passed to a procedure: the solver copies
its communication context into a local there (3 to 5% on the isolated solve at np 1).
