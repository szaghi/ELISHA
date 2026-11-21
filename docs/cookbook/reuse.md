---
title: One setup, many solves
---

# One setup, many solves

`setup` builds everything that depends on the mesh: it gathers and validates the leaves, builds the V-cycle's levels
and operators, plans the FFTs. `solve_packed` uses it. A time-dependent solver whose mesh does not change calls `setup`
once and `solve_packed` every step.

<<< @/examples/snippets/cookbook_reuse-reuse.F90{fortran}

A pulse moving by 0.01 per step (a sixth of its width), solved on one setup and on a fresh setup per step:

<<< @/examples/output/cookbook_reuse.txt{text}

- **The warm start saves one iteration in thirteen.** A solve on the same setup starts from the previous potential,
  which here is a sixth of a pulse width off. The multigrid reduces the error by about a decade per iteration
  whatever the start, so a start closer by less than a decade saves about one.
- **The setup is the saving.** On the scaling ladder's 3.9 M-cell AMR mesh, one rank, the hierarchical setup takes
  15.6 s and a solve 5.88 s ([Performance](/guide/performance#scaling-eb)); in a time loop the setup is paid once per
  mesh.
- **A regrid needs a new setup**: a mesh other than the setup's is refused (`ELISHA_ERR_MESH_CHANGED`). Call `setup`
  again with the new mesh ([tutorial 8](/tutorial/08-embers)).

<details><summary>The whole program</summary>

<<< @/examples/snippets/cookbook_reuse.F90{fortran}

</details>
