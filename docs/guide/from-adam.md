# Coming from ADAM's FLAIL

ELISHA is ADAM's FLAIL (ADAM 669d0bdd) extracted: the facade `adam_flail_object`, the FFT solver
`adam_poisson_fft_object`, the multigrid `adam_poisson_mg_object` and their FNL device twins. The arithmetic is
unchanged, and the tests hold every solver and the facade bitwise to ADAM's own code, compiled unchanged in
`src/tests`. What changed is the interface: a mesh descriptor instead of ADAM's grid and field, the caller's
communicator, a status instead of a stop, and private solvers behind the facade. This page maps a FLAIL client onto
ELISHA.

## The facade

| ADAM (`adam_flail_object`) | ELISHA (`elisha_solver_object`) |
|----------------------------|---------------------------------|
| `type(flail_object) :: flail` | `type(elisha_solver_object) :: solver` |
| `flail%setup(grid, field, isolated, backend[, option_name][, bc][, dg_order][, root_cells])` | `solver%setup(mesh, comm, stat, errmsg[, path][, isolated][, bc][, dg_order][, tolerance][, max_iterations][, root_cells][, spectral_mold][, hierarchical_mold])` |
| `FLAIL_BACKEND_AUTO`, `_FFT`, `_MULTIGRID` (0, 1, 2) | `ELISHA_PATH_AUTO`, `_SPECTRAL`, `_HIERARCHICAL` (0, 1, 2) |
| `FLAIL_BC_PERIODIC`, `_DIRICHLET`, `_NEUMANN`, `_DIRICHLET_GHOST`, `_ZERO_GHOSTS`, `_FREE_GHOSTS` | `ELISHA_BC_*` with the same names and values (0 to 5): an adapter passes them through |
| `flail%solve_packed(field, buf, factor, phi[, energy][, ghost])` | `solver%solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy][, ghost])` |
| `flail%solve(field, q, iv, factor, phi[, energy][, ghost])` | Not ported: pack `q(iv, 1:ni, 1:nj, 1:nk, b)` into `buf` (below), then `solve_packed` |
| `flail%is_multigrid()` | `solver%is_hierarchical()`, or `solver%path() == ELISHA_PATH_HIERARCHICAL` |
| `flail%iterations()` | `solver%diagnostics()%iterations`. On the FFT path ADAM returns 0; ELISHA returns the CG's iterations when the solve is the CG of D(G) |
| Allocating `flail%fft` before `setup` (the FNL backend allocates the cuFFT one) | `spectral_mold=` (`elisha_spectral_solver_cufft_object`) |
| Allocating `flail%mg_mold` before `setup` | `hierarchical_mold=` (`elisha_hierarchical_solver_oac_object`) |
| Reading `flail%fft%pcg`, `%iterations`, `%residual` | `solver%diagnostics()%pcg`, `%iterations`, `%residual` |
| Reading `flail%mg%iterations0`, `%residual`, `%tolerance_used` | `solver%diagnostics()%iterations0`, `%residual`, `%tolerance_used` |
| Reading `flail%mg%nbnd`, `%bnd_kind`, `%bnd_dir`, `%bnd_x` (to build `ghost`) | `solver%boundary_faces(kinds, dir, x, stat, errmsg)`: the same list in the same order |
| `poisson_fft_object%wall` | `solver%diagnostics()%wall`, `solver%reset_wall()` |
| `mpih%error_stop` (fatal) | `stat` and `errmsg`; the facade's refusals are listed in [The solver facade](./solver#refusals) |
| `option_name` (the client's key in the fatal message) | None: the caller words its own message from `stat` |
| `MG_SEAM_GHOST_SCALE` (from `adam_poisson_mg_object`) | `MG_SEAM_GHOST_SCALE` (from `elisha`) |

The energy has the same meaning, `sum_cells f phi dV`, on both paths. Both facades get the spectral solver's
`sum_cells f phi` and multiply it by the cell volume.

ELISHA's facade also takes the hierarchical solver's `tolerance` and `max_iterations`. ADAM's facade did not expose
them, so it always ran the multigrid with its defaults, 1e-10 and 200, and so does ELISHA's when they are not given.

## Grid and field: the mesh descriptor

The solvers read about ten members of ADAM's grid and field. `elisha_mesh_object%init` takes them directly:

| ADAM | `mesh%init` argument |
|------|----------------------|
| `grid%ni`, `grid%nj`, `grid%nk` | `nijk(3)` |
| `grid%ngc` | `ngc` |
| `nint((grid%domain_emax - grid%domain_emin) / grid%block_dxyz(:,0))` | `nblk0(3)` |
| `grid%cell_dxyz(:,0)` | `dx0(3)` |
| `grid%domain_emin` | `emin(3)` |
| `grid%is_ijk_periodic` | `periodic(3)` |
| `grid%null_xyz` | `null(3)` |
| `field%coordinates(:, 1:field%blocks_number)` | `coords(4, nb)`: (i, j, k) at the block's level, then the level |

`MPI_COMM_WORLD` becomes the caller's communicator, given to `elisha_comm_object%init` (which duplicates it). The
mesh is validated collectively by the first `setup` that sees it, or by `mesh%validate(comm, stat, errmsg)`.

## Packing a conservative variable

ADAM's `solve(q, iv)` copied one variable of `q(nv, 1-ngc:, 1-ngc:, 1-ngc:, nb)` into the packed source. An adapter
does the same:

```fortran
c = 0
do b=1, mesh%nb ; do k=1, nk ; do j=1, nj ; do i=1, ni
   c = c + 1 ; buf(c) = q(iv,i,j,k,b)
enddo ; enddo ; enddo ; enddo
call solver%solve_packed(mesh, buf, factor, phi, stat, errmsg, energy=energy)
```

## Clients

FLUME (self-gravity) uses `setup`, `solve`, `solve_packed`, `is_multigrid`, `iterations`, `mg_mold` and
`MG_SEAM_GHOST_SCALE`. Its input keys map onto the setup:

| FLUME key | Values | ELISHA |
|-----------|--------|--------|
| `[gravity] poisson_solver` | `auto` (default), `fft` (an AMR grid is fatal), `multigrid` | `path = ELISHA_PATH_AUTO`, `_SPECTRAL` (an AMR mesh returns `ELISHA_ERR_MESH_LEVELS`), `_HIERARCHICAL` |
| `[gravity] poisson_bc` | `periodic` (default), `isolated` | no `bc`, or `isolated=.true.` |

PRISM uses `setup`, `solve` and the `fft` component, and reads `fft%pcg`, `fft%iterations` and `fft%residual`, now
in the diagnostics. Its time-zero solves (`impose_pic_fields_time_zero`) solve `D(G) phi = -rho/eps` and `D(G) A_d =
-mu J_d` per component, with `dg_order = fdv_order` on the spectral path. A vector whose components take different
kinds uses one solver per set of kinds, and the instances coexist:

| PRISM face | Kind |
|------------|------|
| `periodic` | `ELISHA_BC_PERIODIC` |
| `Dirichlet` | `ELISHA_BC_ZERO_GHOSTS` |
| `PEC` | `ELISHA_BC_DIRICHLET` (the potential, tangential A), `ELISHA_BC_NEUMANN` (normal A) |
| `Neumann`, `Silver_Muller` (exact, open) | `ELISHA_BC_FREE_GHOSTS` |
| null axis | periodic |

| Limit (ADAM's, unchanged) | Detail |
|---------------------------|--------|
| Refined grids | Refused: D(G) is not defined across 2:1 faces (`ELISHA_ERR_DG_AMR`) |
| Memory of the CG solves | Zero-ghost and free-space solves gather the grid on rank 0 (and to its GPU on the device), about 24 B per transform cell. PEC solves are distributed (8 N cells with reflected faces) |
| Checkerboard | Centred D(G) annihilates `(-1)^(i+j+k)`; with PEC faces that component of `rho` stays in `div D - rho` |
| Free-space ghost sum | Direct, on rank 0, O(ghost-slab cells × cells) per component |

## ADAM's reference performance

ADAM's own documentation (`docs/flail.md`, ADAM 669d0bdd) times FLAIL on medooza (CPU nodes; A30 GPUs for FNL) with
`bench_poisson`: the cube `[-1, 1]^3`, a Gaussian source, uniform or with the central box refined, the minimum over
the repetitions and the maximum over the ranks. **These are ADAM's numbers, not ELISHA's measurements.** ELISHA's own
timings against ADAM, measured in pairs, are in the Performance sections of [the spectral
solver](./spectral-solver#performance), [the hierarchical solver](./hierarchical-solver#performance) and [the device
backends](./device#performance).

| Solver | Grid | Backend | Problem | 1 | 2 | 4 | 16 ranks / GPUs |
|--------|------|---------|---------|---|---|---|-----------------|
| FFT | 2.1 M uniform cells | CPU | periodic | 0.115 s | | | 0.026 s |
| FFT | 2.1 M uniform cells | CPU | isolated | 0.729 s | | | 0.108 s |
| FFT | 2.1 M uniform cells | FNL | periodic | 0.067 s | | 0.030 s | |
| FFT | 2.1 M uniform cells | FNL | isolated | 0.138 s | | 0.112 s | |
| Multigrid | 3.9 M AMR cells | CPU, 128^3 root | periodic | 8.6 s | | 2.7 s | 1.6 s |
| Multigrid | 3.9 M AMR cells | CPU, 64^3 root | isolated | 26.8 s | | 8.5 s | 5.2 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | periodic | 0.33 s | 0.19 s | 0.12 s | |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | isolated | 0.85 s | 0.52 s | 0.43 s | |

ADAM also reports weak scaling of the FFT solver, CPU, 8 ranks, 16.7 M uniform cells: periodic 0.24 s, isolated
1.5 s per solve.
