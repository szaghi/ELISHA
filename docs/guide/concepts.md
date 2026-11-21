# Concepts

Five ideas, and every call of ELISHA is made of them: the mesh, who holds which blocks, the packed source, the
potential with its ghosts, and the path. The [tutorial](/tutorial/) meets them one at a time; this page has them in
one place.

## The mesh: an octree of uniform blocks

The domain is a box, `emin` to `emin + nblk0 * nijk * dx0`, cut into a lattice of `nblk0(3)` blocks at **level 0**.
Every block has the same `nijk(3)` cells. A block at level $l$ can be replaced by its 8 children at level $l+1$
(4 on a quadtree, when an axis is null), each with the same number of cells at half the spacing. The leaves of that
octree are the mesh.

![an AMR mesh around a Gaussian star, three levels of blocks, the potential's rings across the seams](/tutorial/t04-amr.png)

A leaf is named by `(i, j, k, level)`: its position counted in blocks *of its own level*, 0-based. Its cells are at
`(coords(1:3) * nijk + [i, j, k] - 0.5) * dx0 / 2**level` from `emin`.

| Field of `elisha_mesh_object` | Meaning |
|---|---|
| `nijk(3)` | cells per block per axis |
| `ngc` | ghost layers around each block |
| `nblk0(3)` | blocks per axis at level 0 |
| `dx0(3)` | cell spacing at level 0 |
| `emin(3)` | the domain's lower corner |
| `periodic(3)` | periodic axes |
| `null(3)` | null axes: one cell, one block, no extent (a 2-D or 1-D problem) |
| `coords(4,nb)` | **this rank's** leaves |

A mesh is valid when its leaves cover the domain exactly (no gap, no overlap) and are **2:1 balanced**: two leaves that
share a face differ by one level at most, periodic axes wrapped. `setup` checks both, collectively, and refuses a mesh
that breaks them with the leaf and the rule named ([Mesh and communicator](/reference/mesh#the-mesh-descriptor)).

## Who holds which blocks

Each rank describes only its own leaves; how they are dealt is the caller's choice (contiguous chunks, round-robin, a
space-filling curve). `setup` gathers the leaf list over the communicator, so every rank knows the whole octree.
`setup` and `solve_packed` are collective: every rank calls them, with its own blocks. The serial build is one rank
holding every leaf ([tutorial 7](/tutorial/07-mpi)).

## The source goes in packed

`solve_packed` takes the source of this rank's leaves as one 1-D array: leaf after leaf in the order of `coords`, and
in each leaf the interior cells with `i` fastest, then `j`, then `k`. That is the order of an array
`f(ni, nj, nk, nb)`, so the interior of a field with ghosts packs in one line:

```fortran
buf = reshape(rho(1:ni,1:nj,1:nk,:), [ni * nj * nk * mesh%nb])
```

The solver solves $L_h\phi = \mathrm{factor}\,(f - \bar f)$ when no face pins the constant (periodic, or Neumann
everywhere), returning the $\phi$ of zero mean, and $L_h\phi = \mathrm{factor}\,f$ otherwise. `factor` is yours:
$4\pi G$ for gravity ([Self-gravity](/cookbook/gravity)), 1 for a plain Poisson problem. The optional `energy` is
$\sum f\phi\,\Delta V$ over the cells, on every rank.

## The potential comes back with its ghosts

`phi(1-ngc:ni+ngc, 1-ngc:nj+ngc, 1-ngc:nk+ngc, nb)` comes back with its ghost layers filled, so a stencil applies to
it straight away:

| Path | Ghosts filled |
|---|---|
| spectral | every ghost, edges and corners included |
| hierarchical | every layer of the face ghosts; the edge and corner ghosts keep what `phi` held |

## The path

`setup` chooses a solver for the mesh (`path=ELISHA_PATH_AUTO`, the default):

| The mesh | Path | How |
|---|---|---|
| every leaf at one level | spectral | direct, by FFT; the residual is round-off |
| leaves at two or more levels | hierarchical | iterative: FCG with a multigrid V-cycle, to `tolerance` |

Each path has things only it does: boundary values on the hierarchical path, the wide operator D(G) on the spectral
one ([Choose a path](/cookbook/choose)). A regrid is a new mesh and a new `setup`; the facade may change path.

## And every call returns a status

`stat` is `ELISHA_OK` or one of 36 error codes, the same on every rank, with a message in `errmsg`
([Status codes](/reference/status)). The library never stops your program; `ELISHA_ERR_NOT_CONVERGED` even comes with
a usable potential ([When it does not converge](/cookbook/not-converged)).
