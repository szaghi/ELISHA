# Examples

Four programs beyond the [Quick Start](./quickstart), each compiled against the serial library and run by
`scripts/docs_examples.sh`. The outputs below are theirs, regenerated and checked by CI. Every program builds its
mesh as the quick start does, and stops on a failed call by printing `errmsg`. The library itself never stops.

## AMR

A 2:1 mesh: the level-0 block (0,0,0) is replaced by its eight children at level 1, at coordinates
`(2i + o_x, 2j + o_y, 2k + o_z)` of that level:

<<< @/examples/snippets/amr-mesh.F90{fortran}

With two levels the automatic path is the hierarchical solver. `tolerance` is the FCG's relative max-norm residual,
and `energy` is `sum_cells f phi dV`:

<<< @/examples/snippets/amr-solve.F90{fortran}

<<< @/examples/output/amr.txt{text}

The potential is second order against the continuum solution, the 2:1 seam included: the composite operator is exact
for linear functions face by face, so only its face gradient across the seam drops to first order.

<details><summary>The whole program</summary>

<<< @/examples/snippets/amr.F90{fortran}

</details>

## Isolated boundaries

The free-space potential of a Gaussian density (`sigma = 0.08`, at the centre of the unit cube, 32^3 cells), `L_h phi
= 4 pi f` with `phi -> 0` at infinity. On a uniform mesh the spectral solver convolves the source with the lattice
Green's function (Hockney and Eastwood):

<<< @/examples/snippets/isolated-solve.F90{fortran}

The ghost layers hold the free-space solution too. Along a line from the centre to beyond the domain, the last cell is
a ghost of the outermost block. The analytic potential of the Gaussian is `-M erf(r / (sqrt(2) sigma)) / r`, `M` the
discrete mass:

<<< @/examples/snippets/isolated-compare.F90{fortran}

<<< @/examples/output/isolated.txt{text}

The relative error is the discretisation's, about 1e-3 at 2.6 cells per `sigma`. It decreases with the distance as
the potential tends to the point mass's.

<details><summary>The whole program</summary>

<<< @/examples/snippets/isolated.F90{fortran}

</details>

## Boundary kinds

On the spectral path, a kind per face: Dirichlet at the low x face and Neumann at the high one, the reverse on y,
Dirichlet on z. The ghost layer carries each kind: `phi_ghost = -phi_boundary` at a Dirichlet face (zero at the face),
`phi_ghost = phi_boundary` at a Neumann one (zero normal derivative):

<<< @/examples/snippets/boundary_kinds-spectral.F90{fortran}

On the hierarchical path the values can be inhomogeneous. After a regrid to a 2:1 mesh, Dirichlet everywhere, the
program gives each boundary face the value of the linear `u = 1 + x + 2 y - z` where the face's value lives
(`boundary_faces`), with no source. The compact operator is exact for linear functions, across the 2:1 seam too, so
the solution is `u` at every cell centre:

<<< @/examples/snippets/boundary_kinds-values.F90{fortran}

<<< @/examples/output/boundary_kinds.txt{text}

<details><summary>The whole program</summary>

<<< @/examples/snippets/boundary_kinds.F90{fortran}

</details>

## Errors

Every procedure returns `stat` and, when asked, `errmsg`, the code's name followed by what failed.
`elisha_status_name` and `elisha_status_text` give a code's name and its one-line meaning:

<<< @/examples/snippets/errors-report.F90{fortran}

A solve before the setup is refused. So is the spectral path on a mesh at two levels, which leaves the solver not set
up. The caller decides what follows, here a new setup on the automatic path:

<<< @/examples/snippets/errors-refusal.F90{fortran}

<<< @/examples/output/errors.txt{text}

The codes are listed in [Errors, communicator and mesh](./errors-comm-mesh), and the facade's own refusals in
[The solver facade](./solver#refusals).

<details><summary>The whole program</summary>

<<< @/examples/snippets/errors.F90{fortran}

</details>
