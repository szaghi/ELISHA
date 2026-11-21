---
title: Reference
---

# Reference

Every public name of ELISHA, area by area. `use elisha` gives all of the host library: the facade, the solvers, the
mesh, the communicator, the status codes, the parameters, the L0 math, and the kinds `I4P`, `I8P`, `R8P` and `R_P`
(from PENF). `use elisha_device` adds the two device backends (the `*-nvf-oac` builds). Each page gives the calls,
their arguments and defaults, their refusals and how they are verified; the [API](/api/) pages, which formal
generates from the source comments, give every procedure's interface.

## What every call shares

- **A status, never a stop.** Every procedure that can fail returns `stat` (mandatory) and, when asked, `errmsg`; the
  library never executes `error stop` ([Status codes](./status)).
- **Collective where it must be.** `setup`, `solve_packed`, the mesh's `validate` and the communicator's reductions
  are collective over the communicator given to `elisha_comm_object%init`; the serial build (`-DELISHA_NO_MPI`) has
  one rank.
- **Your arrays, your layout.** The source goes in packed (this rank's blocks in `coords` order, interior cells, `i`
  fastest); the potential comes back as `phi(1-ngc:ni+ngc, ..., nb)` ([Concepts](/guide/concepts)).
- **One real kind**, `R_P`, double precision.

## Feature map

| Area | Names | Page |
|---|---|---|
| The facade: one solver for any mesh | `elisha_solver_object` (`setup`, `solve_packed`, `diagnostics`, `path`, `is_hierarchical`, `boundary_faces`, `reset_wall`, `destroy`), `elisha_diagnostics_object` | [The solver facade](./solver) |
| The mesh descriptor and the communication context | `elisha_mesh_object` (`init`, `validate`, `destroy`), `elisha_comm_object` (`init`, `destroy`, reductions, gathers), `elisha_halo_plan` | [Mesh and communicator](./mesh) |
| Status codes | `ELISHA_OK`, `ELISHA_ERR_*` (36 codes), `elisha_status_name`, `elisha_status_text`, `elisha_set_error` | [Status codes](./status) |
| Parameters | `ELISHA_BC_*`, `ELISHA_PATH_*`, `ELISHA_OPERATOR_*`, `ELISHA_VCYCLE_*`, `ELISHA_ASSEMBLY_*`, `MG_SEAM_GHOST_SCALE` | [Parameters](./parameters) |
| The spectral solver | `elisha_spectral_solver_object` | [The spectral solver](./spectral) |
| The hierarchical solver | `elisha_hierarchical_solver_object`, its vector and matrix types `rvec`, `ivec`, `dcsr` | [The hierarchical solver](./hierarchical) |
| The device backends | `elisha_spectral_solver_cufft_object`, `elisha_hierarchical_solver_oac_object` (`use elisha_device`) | [The device backends](./device) |
| L0: the math | `elisha_axis_eigenvalue`, `elisha_axis_transform_size`, `elisha_dg_stencil`, `elisha_smooth_size`, `ELISHA_D1_COEF`, `elisha_axis_map`, `elisha_set_plane`, `elisha_lattice_green` | [L0: the math](./math) |
| Setup checks | `elisha_check_bc`, `elisha_check_dg_order`, `elisha_check_isolated`, `elisha_check_isolated_mesh`, `elisha_check_periodicity`, `elisha_check_unbounded`, `elisha_check_reach`, `elisha_check_path`, `elisha_check_ghost_count` | [Mesh and communicator](./mesh#the-setup-checks) |
| Cell keys | `elisha_bkey`, `elisha_bkey4`, `elisha_ckey`, `elisha_kdecode`, `elisha_hash_slot`, `elisha_hash_table`, `ELISHA_KEY_*` | [API: elisha_keys](/api/src/lib/elisha_keys) |

## The facade at a glance

| Call | Collective | Returns |
|---|---|---|
| `comm%init(comm_in, stat, errmsg)` | yes | a duplicate of `comm_in` (ignored by the serial build) |
| `mesh%init(nijk, ngc, nblk0, dx0, emin, periodic, null, coords, stat, errmsg)` | no | this rank's blocks, checked for shape and coordinates |
| `solver%setup(mesh, comm, stat, errmsg[, path][, isolated][, bc][, dg_order][, tolerance][, max_iterations][, root_cells][, spectral_mold][, hierarchical_mold][, level_operator][, vcycle][, assembly])` | yes | the mesh validated, the path chosen, its solver built |
| `solver%solve_packed(mesh, buf, factor, phi, stat, errmsg[, energy][, ghost][, on_device])` | yes | `phi` with its ghosts, `energy = sum f phi dV` on every rank |
| `solver%diagnostics()` | no | path, iterations, residual, tolerance used, converged, wall times |
| `solver%boundary_faces(kinds, dir, x, stat, errmsg)` | no | this rank's boundary faces, in the order `ghost` takes their values |
