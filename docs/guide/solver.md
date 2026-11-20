# The solver facade

`elisha_solver_object` is the entry point of a client. It chooses the solver for the mesh, the spectral one on a mesh
at one level or the hierarchical one on a 2:1 AMR mesh, and forwards to it. Its results are bitwise those of the solver
it chose, called directly. It is ADAM's FLAIL facade (`adam_flail_object`, ADAM 669d0bdd) with the solvers made private:
the path is an argument, a backend is a mold, and what clients read from the solvers is a diagnostics type
(`elisha_diagnostics_object`). [Coming from ADAM's FLAIL](./from-adam) maps one API onto the other.

<<< @/examples/snippets/quickstart-solve.F90{fortran}

| Procedure | Role |
|-----------|------|
| `setup(mesh, comm, stat, errmsg[, path][, isolated][, bc][, dg_order][, tolerance][, max_iterations][, root_cells][, spectral_mold][, hierarchical_mold])` | Choose the path for the mesh and set up its solver; called on the initial mesh and after every regrid |
| `solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy][, ghost])` | Solve for `phi` from the packed interior source; every ghost layer of `phi` is filled |
| `diagnostics()` | An `elisha_diagnostics_object`: the path, iterations, residual, tolerance, CG flag, wall times |
| `path()`, `is_hierarchical()` | The path in use (`ELISHA_PATH_SPECTRAL`, `ELISHA_PATH_HIERARCHICAL`; 0 before setup) |
| `boundary_faces(kinds, dir, x, stat, errmsg)` | This rank's boundary faces, in the order `solve_packed(..., ghost)` takes their values (hierarchical path) |
| `reset_wall()` | Zero the accumulated wall times of the spectral solves |
| `destroy()` | Free the solver in use; the caller still owns the communicator and the mesh |

## Setup

`setup` is collective over `comm`, which the solver keeps as a copy. It validates the mesh (collectively) unless the
caller already did, so it knows the leaf levels over every rank, then chooses the path:

| `path` | Chooses |
|--------|---------|
| `ELISHA_PATH_AUTO` (0, the default) | The spectral solver when every leaf is at one level, the hierarchical one otherwise |
| `ELISHA_PATH_SPECTRAL` (1) | The spectral solver; a mesh at more than one level is refused |
| `ELISHA_PATH_HIERARCHICAL` (2) | The hierarchical solver on any 2:1 mesh. On a uniform mesh one V-cycle is the exact spectral solve, and boundary values are accepted |

| Argument | Meaning | Path |
|----------|---------|------|
| `isolated` | Free-space boundaries, `phi -> 0` at infinity: Hockney and Eastwood on the spectral path, the James step on the hierarchical one. Takes no `bc` | both |
| `bc(2,3)` | The kind of every face, `[lo/hi, axis]` (`ELISHA_BC_*`, [L0](./l0-math#boundary-kinds)); a periodic axis of the mesh must be periodic in `bc`, and the other way round. Without it: periodic, or isolated | both |
| `dg_order` | 0 (default) the compact Laplacian; 2, 4, 6, 8 PRISM's wide operator D(G) | spectral |
| `tolerance` | Relative max-norm residual of the FCG (default 1e-10; the source's round-off floor when larger) | hierarchical |
| `max_iterations` | FCG iterations at most (default 200) | hierarchical |
| `root_cells` | The root of the V-cycle at most this many cells (default: the root solver's) | hierarchical |
| `spectral_mold` | The spectral solver's type, and on the hierarchical path the type of its root (default: the host FFTW solver) | both |
| `hierarchical_mold` | The hierarchical solver's type (default: the host one) | hierarchical |

The hierarchical-only arguments are ignored on the spectral path. Every call rebuilds the solver: a hierarchical solve
warm-starts from the previous solution of the same setup, never across setups. A failed setup leaves the facade not set
up (`path()` is 0).

## Solve

`solve_packed(mesh, buf, factor, phi, stat, errmsg, energy, ghost)` solves `L_h phi = factor (f - mean f)` (periodic,
or no Dirichlet face) or `L_h phi = factor f`:

- `buf` holds the interior source of this rank's blocks, block-major with `i` fastest, `mesh%nb * ni * nj * nk` values;
- `phi(1-ngc:ni+ngc, 1-ngc:nj+ngc, 1-ngc:nk+ngc, nb)` returns with every ghost layer, filled by the boundary kinds or
  from the free-space solution;
- `energy` is `sum_cells f phi dV` on every rank, on both paths. The spectral solver's own energy is `sum_cells f phi`,
  and the facade multiplies it by the cell volume, as ADAM's facade did;
- `ghost` holds the value of each boundary face (`boundary_faces`), on the hierarchical path only. The spectral path
  solves the homogeneous problem.

The call is collective. The mesh must be the setup's: a regrid needs a new setup (`ELISHA_ERR_MESH_CHANGED`).

## Diagnostics

`diagnostics()` returns a copy of plain data:

| Field | Spectral path | Hierarchical path |
|-------|---------------|-------------------|
| `path` | `ELISHA_PATH_SPECTRAL` | `ELISHA_PATH_HIERARCHICAL` |
| `iterations` | CG iterations on the CG path (zero- or free-ghost faces with D(G)), 0 for a direct FFT | FCG iterations of the last solve; isolated, those of the second solve of the James step |
| `iterations0` | 0 | Isolated: FCG iterations of the zero-ghost solve of the James step; 0 otherwise |
| `residual` | Relative max-norm residual of the CG; 0 for a direct FFT | Relative max-norm residual of the FCG |
| `tolerance_used` | The CG's stop tolerance (1e-12); 0 for a direct FFT | The FCG's stop tolerance: `tolerance`, or the source's round-off floor when larger |
| `pcg` | The solve is the CG of the wide D(G) | `.false.` |
| `wall(3)` | Wall time (s) of the spectral solve on this rank: gather, solve, scatter, accumulated since the setup or `reset_wall` | The same for the root solver |

The spectral CG runs on rank 0: there `iterations` and `residual` are the CG's, while the other ranks report 0.

## Boundary values

On the hierarchical path a bounded face can take an inhomogeneous value. `boundary_faces` returns this rank's faces in
the order `ghost` takes their values: the kind of each face, its outward axis signed (-3 to 3), and where its value
lives (the face centre, or the centre of the first ghost cell for ghost-Dirichlet). A client evaluates its boundary
function there:

<<< @/examples/snippets/boundary_kinds-values.F90{fortran}

[Examples](./examples#boundary-kinds) runs it.

## Refusals

Besides the refusals of the mesh and the solvers (the codes in [Errors, communicator and
mesh](./errors-comm-mesh#status-a-code-and-a-message-never-a-stop)), the facade refuses:

| Code | Condition |
|------|-----------|
| `ELISHA_ERR_NOT_INITIALIZED` | The communication context is not initialised; `solve_packed` or `boundary_faces` before a setup, or after a failed one |
| `ELISHA_ERR_ARGUMENT` | `path` is not one of the three |
| `ELISHA_ERR_ISOLATED_BC` | `isolated` together with `bc` |
| `ELISHA_ERR_DG_ORDER` | `dg_order` is not 0, 2, 4, 6 or 8 |
| `ELISHA_ERR_MESH_LEVELS` | `ELISHA_PATH_SPECTRAL` on a mesh at more than one level |
| `ELISHA_ERR_DG_AMR` | `dg_order` other than 0 on the hierarchical path |
| `ELISHA_ERR_BC_FREE_GHOSTS` | Free-space ghost faces on the hierarchical path |
| `ELISHA_ERR_GHOST_VALUES` | `ghost` given, or `boundary_faces` called, on the spectral path |

The argument checks come first and are the same on every rank; the mesh's validation then agrees on one status over
the ranks.

## Device backends

A mold chooses a backend: the object is read for its type only. With the device library (mode `static-nvf-oac`,
[The device backends](./device)):

```fortran
use elisha, only : elisha_solver_object
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
type(elisha_spectral_solver_cufft_object)   :: cufft ! molds: only their types are read
type(elisha_hierarchical_solver_oac_object) :: oac
type(elisha_solver_object)                  :: solver
call solver%setup(mesh, comm, stat, errmsg, spectral_mold=cufft, hierarchical_mold=oac)
```

The spectral path then transforms on cuFFT, and the hierarchical path iterates on the device with its root on cuFFT.
Without `spectral_mold`, the OpenACC hierarchical solver already takes a cuFFT root.

## Regrid

Call `setup` again with the new mesh. It destroys the solver in use and builds the one the new mesh needs, so a run can
go from the spectral path to the hierarchical one and back:

```fortran
call solver%setup(mesh, comm, stat, errmsg)   ! uniform: spectral
! ... solves; the mesh is refined: a new descriptor, validated by setup
call solver%setup(mesh, comm, stat, errmsg)   ! 2:1 AMR: hierarchical
```

## Verification

- `elisha_test_solver_mpi` runs the facade against the solver its path names, called directly with the same arguments,
  on 9 cases: auto on uniform and AMR meshes, both paths forced, isolated on both, the CG of the wide D(G), boundary
  values, and the molds. It checks `phi` with every ghost layer and the energy bitwise, the diagnostics against the
  solver's fields, and each refusal above, on 1 to 3 ranks and in the serial build.
- `elisha_test_facade_parity_mpi` runs ELISHA's facade against ADAM's (`adam_flail_object`, compiled unchanged in the
  tests) on 20 steps of one setup and two solves each. The steps cover auto, fft and multigrid on uniform and AMR
  meshes, isolated on both paths, Dirichlet, Neumann and ghost-Dirichlet faces with boundary values, D(G) periodic,
  with zero ghosts and with free ghosts, and a regrid sequence from spectral to hierarchical and back. It compares the
  bit patterns of `phi` with every ghost layer, the energy, the iteration counts, the residuals and the boundary-face
  list. All 20 are bitwise at np 1, 2 and 3 on gfortran, nvfortran and ifx.

Four mutants of the facade each break the parity test:
- dropping the cell volume from the spectral energy breaks the 10 spectral steps;
- swapping auto's choice breaks the 14 auto steps;
- not forwarding `ghost` breaks the 3 steps with boundary values;
- not forwarding `root_cells` breaks the step that caps the root.
