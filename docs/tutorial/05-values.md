---
title: 5. Boundary values
---

# 5. Boundary values

Chapter 2's faces held homogeneous values: zero at a Dirichlet face, zero derivative at a Neumann one. A box with a
heated lid holds something else on its faces. Here Laplace's equation, $L_h\phi = 0$, takes its values on every face
from the harmonic

$$ u = \sin\pi x\,\sin\pi y\,\frac{\sinh\sqrt2\pi z}{\sinh\sqrt2\pi}, $$

zero on five faces and $\sin\pi x \sin\pi y$ on the lid $z = 1$. Inhomogeneous values are a feature of the
**hierarchical** path; the spectral path solves the homogeneous problem only. The mesh is refined in its upper half,
where the lid's gradients are.

## Setup

Dirichlet on every face; two levels, so the hierarchical path:

<<< @/examples/snippets/t05_values-setup.F90{fortran}

## Ask where the values go

The solver knows its boundary faces; the caller knows the boundary function. `boundary_faces` lists this rank's
faces: the kind of each, its outward axis (-3 to 3, signed), and the point where its value lives (the face centre;
for `ELISHA_BC_DIRICHLET_GHOST`, the centre of the first ghost cell). The values go back in the same order:

<<< @/examples/snippets/t05_values-values.F90{fortran}

<<< @/examples/snippets/t05_values-solve.F90{fortran}

## Running it

<<< @/examples/output/t05_values.txt{text}

- **Second order, nearly**: 1.91 between the two meshes. The interior stencil is second order and the composite
  operator at the 2:1 seam first order; two resolutions do not separate their shares.
- **The iterations do not grow with the mesh**: 11 and 10. A multigrid preconditioner removes the error at every
  scale per V-cycle, so the count is set by the problem's shape, not its size.

![phi on the plane y = 1/2: hottest under the lid, cold at the other five faces, the upper half refined](/tutorial/t05-values.png)

::: tip What you learned
On the hierarchical path a face can hold a value: `boundary_faces` says where, `solve_packed(..., ghost=g)` takes the
values in the same order. The spectral path refuses `ghost` (`ELISHA_ERR_GHOST_VALUES`).
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t05_values.F90{fortran}

</details>

Next: [6. Watching the FCG](/tutorial/06-iterations).
