# Status codes

Every procedure that can fail returns `stat` (mandatory, `intent(out)`) and, when asked, `errmsg`
(`character(:), allocatable`). The message is the code's name, a colon, and what failed with its values. The
library never executes `error stop`; the caller decides what a failure means.

```fortran
call mesh%validate(comm, stat, errmsg)
if (stat /= ELISHA_OK) then
   print '(A)', errmsg ! e.g. ELISHA_ERR_MESH_BALANCE: ... leaf (i, j, k, level) 7 6 0 2 faces a leaf at level 0 ...
   error stop          ! the caller's choice, not the library's
endif
```

`elisha_status_name(code)` and `elisha_status_text(code)` give a code's name and meaning. The codes, one per class
of ADAM's refusals:

| Code | Value | Meaning |
|------|-------|---------|
| `ELISHA_OK` | 0 | success |
| `ELISHA_ERR_ARGUMENT` | 1 | an argument is out of its range |
| `ELISHA_ERR_NO_FFT` | 2 | built without an FFT library (-D_FFTW on the host, -D_CUFFT on NVIDIA) |
| `ELISHA_ERR_FFT_PLAN` | 3 | the FFT library could not make or execute a plan |
| `ELISHA_ERR_MPI_NOT_INITIALIZED` | 4 | MPI is not initialised (the caller calls MPI_Init before ELISHA) |
| `ELISHA_ERR_MPI` | 5 | an MPI call failed |
| `ELISHA_ERR_INTERNAL` | 6 | an internal invariant failed (a bug in ELISHA) |
| `ELISHA_ERR_MESH_SHAPE` | 7 | the mesh shape is invalid (cells per block, ghosts, blocks at level 0, spacing, null axes) |
| `ELISHA_ERR_MESH_COORDS` | 8 | a block coordinate is outside the range of its level |
| `ELISHA_ERR_MESH_LEVELS` | 9 | the blocks sit at more than one level where one is needed (the spectral solver) |
| `ELISHA_ERR_MESH_COVERAGE` | 10 | the blocks do not cover the domain exactly (a gap, an overlap or a duplicate) |
| `ELISHA_ERR_MESH_BALANCE` | 11 | the mesh is not 2:1 balanced across faces |
| `ELISHA_ERR_MESH_KEY_LIMIT` | 12 | the finest level has 2^19 or more cells along an axis (the cell-key limit) |
| `ELISHA_ERR_MESH_CHANGED` | 13 | the mesh does not match the setup (a regrid needs a new setup) |
| `ELISHA_ERR_BC_UNKNOWN` | 14 | an unknown boundary kind |
| `ELISHA_ERR_BC_PERIODIC_PAIR` | 15 | an axis has one periodic face and one non-periodic face |
| `ELISHA_ERR_BC_GHOST_PAIR` | 16 | an axis mixes a ghost-Dirichlet face with another kind |
| `ELISHA_ERR_BC_ZERO_GHOST_PAIR` | 17 | an axis mixes a zero-ghost or free-ghost face with another kind |
| `ELISHA_ERR_BC_PERIODICITY` | 18 | the boundary kinds disagree with the periodic axes of the mesh |
| `ELISHA_ERR_BC_MISSING` | 19 | a non-periodic axis without boundary kinds |
| `ELISHA_ERR_BC_FREE_GHOSTS` | 20 | free-space ghosts need the wide operator D(G) on the spectral path |
| `ELISHA_ERR_ISOLATED_BC` | 21 | isolated boundaries take no boundary kinds |
| `ELISHA_ERR_ISOLATED_WIDE` | 22 | isolated boundaries take the compact operator only |
| `ELISHA_ERR_ISOLATED_DIMENSION` | 23 | isolated boundaries are 3-D only |
| `ELISHA_ERR_ISOLATED_PERIODIC` | 24 | isolated boundaries need every axis non-periodic |
| `ELISHA_ERR_ISOLATED_CELLS` | 25 | isolated boundaries need cubic cells |
| `ELISHA_ERR_ISOLATED_BOUNDARY_LEAF` | 26 | isolated boundaries need every boundary leaf at the coarsest level |
| `ELISHA_ERR_DG_ORDER` | 27 | the order of the wide operator is not 0, 2, 4, 6 or 8 |
| `ELISHA_ERR_DG_REACH` | 28 | an axis has fewer cells than the reach of the wide operator |
| `ELISHA_ERR_DG_AMR` | 29 | the wide operator needs a uniform mesh and the spectral path |
| `ELISHA_ERR_REDUNDANT` | 30 | the redundant spectral solve is for the compact, non-isolated operator |
| `ELISHA_ERR_GHOST_VALUES` | 31 | boundary values need the hierarchical path and non-isolated boundaries |
| `ELISHA_ERR_GHOST_COUNT` | 32 | fewer boundary values than boundary faces |
| `ELISHA_ERR_NOT_INITIALIZED` | 33 | the solver is used before its setup |
| `ELISHA_ERR_SHAPE` | 34 | an array argument has the wrong shape or size |
| `ELISHA_ERR_ENERGY_REDUNDANT` | 35 | no energy with the redundant spectral solve |
| `ELISHA_ERR_NOT_CONVERGED` | 36 | an iterative solve reached max_iterations above its tolerance (phi holds the last iterate, its ghosts filled) |

`ELISHA_ERR_NOT_CONVERGED` is not one of ADAM's refusals. It is returned when the hierarchical solver's FCG, or the
spectral solver's CG of the wide D(G), reaches `max_iterations` above its tolerance. Every rank returns it. ADAM
printed a warning on rank 0 and returned as if converged. Unlike the other codes, it comes with a result: `phi` is the
last iterate, with its ghosts filled, and so is the energy. A caller that kept ADAM's behaviour on purpose can treat
the code as a warning; `diagnostics()%converged` tells the same.


## A refusal, in a program

Two calls that fail, and what a program sees (`docs/examples/src/errors.F90`): `solve_packed` before any setup, and
the spectral path forced on a mesh at two levels. The solver stays not set up after a failed setup (`path()` is 0):

<<< @/examples/snippets/errors-refusal.F90{fortran}

<<< @/examples/snippets/errors-report.F90{fortran}

<<< @/examples/output/errors.txt{text}

Every code is the same on every rank: the argument checks come first and are the same everywhere, and a collective
check (the mesh's validation, a non-converged solve) agrees on one status over the ranks before returning.

<details><summary>The whole program</summary>

<<< @/examples/snippets/errors.F90{fortran}

</details>
