---
title: Your fields in and out
---

# Your fields in and out

A block-structured code keeps a field as `rho(1-ng:nc+ng, 1-ng:nc+ng, 1-ng:nc+ng, nb)`: the blocks of this rank, each
with its ghost layers. ELISHA takes the source **packed**, the interior only, block-major with `i` fastest, and returns
the potential in the layout of your fields, ghosts included.

The density, ghosts and all (the ghosts set to `huge` here, to show the solver never reads them):

<<< @/examples/snippets/cookbook_fields-fields.F90{fortran}

The interior of the array, in Fortran's order, is the packed order. One line:

<<< @/examples/snippets/cookbook_fields-pack.F90{fortran}

`reshape` makes one contiguous copy, which is what `solve_packed` reads. The order of the blocks is the order of the
mesh's `coords`: keep your field's fourth index in that order.

<<< @/examples/snippets/cookbook_fields-solve.F90{fortran}

`phi` comes back as `phi(1-ngc:nc+ngc, ..., nb)`: the solver fills every ghost layer it has, here two.

<<< @/examples/output/cookbook_fields.txt{text}

- **A source with a mean** on a periodic box (or with Neumann faces everywhere) has no solution as it stands. The
  solver solves for the source minus its mean, and returns the potential of zero mean; it does not tell you the mean
  it removed, so keep it if your physics needs it (cosmology's background density, say).
- **Which ghosts are filled**: on the spectral path every ghost of `phi`, edges and corners included. On the
  hierarchical path the face ghosts, every layer, but not the edge and corner ghosts, which keep what `phi` held
  before the call: a 7-point stencil never reads them, a 27-point one does ([the solver facade](/reference/solver#solve)).

<details><summary>The whole program</summary>

<<< @/examples/snippets/cookbook_fields.F90{fortran}

</details>
