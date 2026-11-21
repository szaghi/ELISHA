# Mesh and communicator

Two objects every call takes: the communication context, which says which ranks take part, and the mesh descriptor,
which says what they solve on. ADAM's FLAIL had neither as an object of its own: it called every collective on
`MPI_COMM_WORLD` and read ADAM's `grid` and `field`. How a failure is reported is on [Status codes](/reference/status).

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
