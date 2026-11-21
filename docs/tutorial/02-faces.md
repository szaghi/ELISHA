---
title: 2. Faces
---

# 2. Faces

A box that is not periodic needs a kind for each of its six faces. This chapter solves $\phi'' = f$ on a line for
four pairs of kinds, each against its continuum solution, and looks at what each kind writes into the ghost cell.

## A line is a box with two null axes

ELISHA's mesh is always three-dimensional. A **null axis** has one cell and one block, and the 7-point Laplacian
loses its two terms along it, so a box with null y and z axes is a line along x. The program declares the null axes
periodic, in the mesh and in `bc`, so that they hold no boundary at all; `bc` gives one pair `[low, high]` per axis:

<<< @/examples/snippets/t02_faces-bc.F90{fortran}

The four pairs, each with the mode that meets its faces with homogeneous values (zero at a Dirichlet face, zero
derivative at a Neumann one), and $f = u'' = -k^2 u$:

| Faces (x low / high) | $u$ | Singular |
|---|---|---|
| periodic / periodic | $\sin(2\pi x + 1)$ | yes: the mean of $f$ is removed, $\phi$ has zero mean |
| Dirichlet / Dirichlet | $\sin \pi x$ | no |
| Neumann / Neumann | $\cos \pi x$ | yes, as periodic |
| Dirichlet / Neumann | $\sin(\pi x / 2)$ | no |

Where no Dirichlet face pins the constant, $\phi$ and $u$ are compared with their means matched:

<<< @/examples/snippets/t02_faces-error.F90{fortran}

## What a kind writes into the ghost cell

The solver fills the ghost layer from the kind of its face, so a stencil applied to `phi` sees the boundary
condition. The program compares each end's ghost with the cell next to it and with the cell at the other end of the
line:

<<< @/examples/snippets/t02_faces-ghosts.F90{fortran}

## Running it

<<< @/examples/output/t02_faces.txt{text}

- **Every pair is second order**, the spectral path doing each by an odd or an even extension of the line
  (Dirichlet odd, Neumann even, a mixed pair a quarter-wave), so its residual is round-off as in chapter 1.
- **The ghosts carry the kinds.** At a Dirichlet face the ghost is the negated inner value: the face, halfway
  between them, holds zero. At a Neumann face the ghost equals the inner value: the derivative across the face is
  zero. A periodic ghost holds the cell at the far end.

![phi along the line for each pair of faces: the cells as dots on the continuum solution, the two ghost cells as
open squares beyond the dotted faces](/tutorial/t02-faces.png)

The open squares are the ghosts: mirrored through zero at a Dirichlet face, flat across a Neumann one, wrapped
around on the periodic line.

Two more kinds exist for PRISM's wide operator D(G) (zero ghosts, free-space ghosts), and one for a Dirichlet value
held at the first ghost centre instead of the face (`ELISHA_BC_DIRICHLET_GHOST`). Which kinds can share an axis, and
what is refused, is in [Which faces go together](/cookbook/faces).

::: tip What you learned
`bc(2,3)` gives the kind of every face, `[low, high]` per axis; a periodic axis of the mesh is periodic in `bc` too.
A null axis turns the box into a plane or a line. Each kind writes its own relation into the ghost layer.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t02_faces.F90{fortran}

</details>

Next: [3. A star in empty space](/tutorial/03-isolated).
