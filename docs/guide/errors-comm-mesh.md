# Errors, communicator and mesh

M2 adds the layer every solver stands on: how a failure is reported, which ranks take part, and what mesh they
solve on. ADAM's FLAIL had none of the three as objects of its own. It stopped the program on an error, called every
collective on `MPI_COMM_WORLD`, and read ADAM's `grid` and `field`.

## Status: a code and a message, never a stop

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

## The communication context

```fortran
type(elisha_comm_object) :: comm
call comm%init(MPI_COMM_WORLD, stat, errmsg) ! any communicator: ELISHA works on a duplicate (MPI_Comm_dup)
```

The duplicate keeps the library's messages apart from the caller's, and two solvers can live on two
sub-communicators. Every collective of ELISHA goes through this object. The caller initialises and finalises MPI;
`init` before `MPI_Init` returns `ELISHA_ERR_MPI_NOT_INITIALIZED`. Built with `-DELISHA_NO_MPI` (the `-serial`
modes, fpm) there is no MPI: one rank, the reductions are the identity.

## The mesh descriptor

`elisha_mesh_object` is plain data the caller gives:

| Field | Meaning |
|-------|---------|
| `nijk(3)` | Cells per block per axis |
| `ngc` | Ghost layers |
| `nblk0(3)` | Blocks per axis at level 0 |
| `dx0(3)` | Cell spacing at level 0 |
| `emin(3)` | Domain lower corner |
| `periodic(3)`, `null(3)` | Periodic axes; null axes (one cell, one block) |
| `coords(4,nb)` | This rank's blocks: (i, j, k) 0-based at the block's level l, and l (ADAM's `field%coordinates`) |

`init` checks what one rank can see: the shape (`ELISHA_ERR_MESH_SHAPE`) and the block coordinates
(`ELISHA_ERR_MESH_COORDS`). `validate(comm[, uniform])` checks the whole mesh collectively, and every rank gets the
same status:

| Check | Code |
|-------|------|
| One level, when `uniform` (the spectral solver) | `ELISHA_ERR_MESH_LEVELS` |
| Fewer than 2^19 cells along every axis at the finest level | `ELISHA_ERR_MESH_KEY_LIMIT` |
| Exact cover: no duplicate, no block inside another, the domain's volume | `ELISHA_ERR_MESH_COVERAGE` |
| 2:1 balance across faces, periodic axes wrapped | `ELISHA_ERR_MESH_BALANCE` |

A validated mesh keeps the gathered leaf list (`leaf_c`, `leaf_owner`, `leaf_loc`) and its table (`leaf_of`), which
the hierarchical solver uses.

## The setup checks

ADAM's refusals of a solver's arguments become pure checks (`elisha_setup_checks`), which the solvers compose:
`elisha_check_bc`, `elisha_check_dg_order`, `elisha_check_isolated`, `elisha_check_periodicity`,
`elisha_check_unbounded`, `elisha_check_reach`, `elisha_check_path`, `elisha_check_ghost_count` and the collective
`elisha_check_isolated_mesh`. Two are stricter than ADAM:
- an unknown kind is reported before the pair rules of its axis;
- a lone periodic face is refused on every path. ADAM's multigrid accepted `[periodic, Dirichlet]` on a non-periodic
  axis.

`elisha_test_setup_checks_mpi` feeds each check one case per ADAM refusal it replaces, plus a valid case.
