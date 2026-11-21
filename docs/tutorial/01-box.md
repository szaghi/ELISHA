---
title: 1. A periodic box
---

# 1. A periodic box

Poisson's equation on the periodic unit cube, discretised by the 7-point Laplacian,

$$ L_h\phi = \frac{\phi_{i+1} + \phi_{i-1} + \phi_{j+1} + \phi_{j-1} + \phi_{k+1} + \phi_{k-1} - 6\phi}{h^2} = f - \bar f, $$

with $f = \sin 2\pi x\, \sin 2\pi y\, \sin 2\pi z$. On a periodic box the problem is singular. A constant can be
added to any solution, and only a source of zero mean has one, so the solver solves for $f - \bar f$ and returns
the solution of zero mean. Here $\bar f = 0$ anyway, and the continuum solution is $-f/(12\pi^2)$.

## Three objects

Everything ELISHA does goes through three objects, built in this order:

1. **the communicator**, `elisha_comm_object`: the caller's MPI communicator, which ELISHA duplicates to keep its
   messages apart from yours. The serial build ignores it ([7. Many ranks](/tutorial/07-mpi) uses a real one);
2. **the mesh**, `elisha_mesh_object`: plain data that you give. The domain is a lattice of `nblk0` blocks at
   level 0, each of `nijk` cells, and `coords` lists this rank's blocks, each as `(i, j, k, level)`;
3. **the solver**, `elisha_solver_object`: the facade. Its `setup` chooses the path for the mesh and builds that
   solver; `solve_packed` solves.

<<< @/examples/snippets/t01_box-use.F90{fortran}

The mesh: 2³ blocks of `nc`³ cells on [0, 1]³, one ghost layer, every axis periodic.

<<< @/examples/snippets/t01_box-mesh.F90{fortran}

## The source goes in packed

`solve_packed` takes the source of this rank's blocks as one array, **packed**: block after block in the order of
`coords`, and in each block the interior cells with `i` fastest, then `j`, then `k`. That is Fortran's own order of
an array `f(nc, nc, nc, nb)`, so a field you already hold that way packs with one `reshape`
([Your fields in and out](/cookbook/fields)). The cell centres of block `b` are at
`(coords(1:3,b) * nc + [i, j, k] - 1/2) * h`.

<<< @/examples/snippets/t01_box-source.F90{fortran}

## The potential comes back with its ghosts

`phi` has the bounds of a block with its ghost layers, `(1-ngc:nc+ngc, ..., nb)`. The solver returns it with the
ghosts filled: here from the periodic neighbours, so a stencil can be applied to `phi` straight away.

<<< @/examples/snippets/t01_box-solve.F90{fortran}

The program checks both claims. It applies the 7-point Laplacian to `phi` through the ghosts the solver filled and
compares it with `f`, then compares `phi` with the continuum solution:

<<< @/examples/snippets/t01_box-check.F90{fortran}

## Running it

The same problem on 8³ to 64³ cells:

<<< @/examples/snippets/t01_box-sweep.F90{fortran}

<<< @/examples/output/t01_box.txt{text}

- **The residual is round-off on every mesh.** The spectral solver divides each Fourier mode by the eigenvalue of
  the *discrete* operator $L_h$, not by the continuum $-|k|^2$, so what it returns solves the discrete equation
  exactly, up to the rounding of the transforms.
- **The error is the discretisation's.** It is second order, 2.00 to 2.03: each halving of $h$ divides it by four.
  ELISHA solves $L_h\phi = f$ exactly; how far $L_h$ is from the Laplacian is the stencil's business.

![phi on the plane z = 1/2 with its contour rings, and the error against the cell spacing on a log-log plot with a
slope-2 reference](/tutorial/t01-box.png)

::: tip What you learned
A solve is three objects and two calls. The source goes in packed, block-major with `i` fastest, and the potential
comes back as blocks with their ghosts filled. On one level the facade chooses the spectral path, which solves the
discrete equation to round-off.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t01_box.F90{fortran}

</details>

Next: [2. Faces](/tutorial/02-faces).
