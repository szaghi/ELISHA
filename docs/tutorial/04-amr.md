---
title: 4. Refine where it matters
---

# 4. Refine where it matters

The star of chapter 3 lives in a small part of the box; resolving the whole box at the star's scale spends most of
the cells on empty space. An **AMR** mesh refines where the star is. In ELISHA's mesh a block at level $l$ can be
replaced by its 8 children at level $l+1$, each with the same number of cells and half the spacing, so the leaves
form an octree of blocks. The facade sees more than one level and chooses the **hierarchical** path: flexible
conjugate gradients preconditioned by a multigrid V-cycle, the spectral solver at its root.

## An octree of blocks

`coords` lists the leaves, each as `(i, j, k, level)` with `(i, j, k)` counted at its own level. The program builds
them recursively: a block near the star, and above level 2, is replaced by its children.

<<< @/examples/snippets/t04_amr-mesh.F90{fortran}

The two rules a mesh must keep, checked collectively at setup:

- **2:1 balance**: a leaf never faces a leaf more than one level coarser across a face (`ELISHA_ERR_MESH_BALANCE`);
- **exact cover**: no gap, no overlap, no duplicate (`ELISHA_ERR_MESH_COVERAGE`).

Isolated boundaries add one: every leaf touching the domain boundary is at the coarsest level
(`ELISHA_ERR_ISOLATED_BOUNDARY_LEAF`), which the James step needs for its boundary convolution.

The source is evaluated at each block's own cell centres, the spacing $h/2^l$:

<<< @/examples/snippets/t04_amr-source.F90{fortran}

## The solve

Nothing in the calls changes: `setup` sees the levels and chooses the path.

<<< @/examples/snippets/t04_amr-solve.F90{fortran}

## Running it

The program solves on a coarse uniform mesh, on the AMR mesh, and on the uniform mesh of the finest spacing:

<<< @/examples/output/t04_amr.txt{text}

- **The AMR mesh uses 4.3% of the fine mesh's cells** and has a quarter of the coarse mesh's error.
- **It does not reach the fine mesh's error**: 2.4 times larger. This run does not say which part of the mesh
  keeps it there; the candidates are the coarse far field, where the AMR mesh keeps level 0, and the 2:1 seams, where
  the composite operator is first order.
- **14 FCG iterations** to a relative residual of 1e-10, on the James step's final solve.

![the AMR mesh around the star on the plane z = 1/2, the block outlines darker for finer levels, the potential's rings
continuous across the seams](/tutorial/t04-amr.png)

## Two refusals

A refinement rule is easy to get wrong, and the mesh tells you which way. The program takes the two refinement reaches
on the command line. Reaching further at level 1 than level 0 can follow breaks the balance:

<<< @/examples/output/t04_amr-unbalanced.txt{text}

and reaching further at level 0 puts a refined leaf on the boundary, which the James step refuses:

<<< @/examples/output/t04_amr-boundary.txt{text}

The message names the leaf and the rule; `ERROR STOP 1` is the program's own `check`, not the library's: ELISHA
returned a status and left the decision to its caller.

A rule that always balances: a level-$l$ block is refined when the source is within $r_l$, with
$r_{l-1} \ge r_l + $ (the size of a level-$l$ block). Then every level-$(l+1)$ block's neighbours are at level $l$ or
finer.

::: tip What you learned
An AMR mesh is a list of leaves, each `(i, j, k, level)`; 2:1 balanced and covering the domain exactly. The facade
chooses the hierarchical path for it; the calls do not change. A bad mesh is refused, by name.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t04_amr.F90{fortran}

</details>

Next: [5. Boundary values](/tutorial/05-values).
