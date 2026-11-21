---
title: Which faces go together
---

# Which faces go together

`setup(..., bc=bc)` takes the kind of every face, `bc(2,3)`, `[low, high]` per axis:

| Kind | Value | The face holds | Paths |
|---|---|---|---|
| `ELISHA_BC_PERIODIC` | 0 | nothing: the axis wraps around | both |
| `ELISHA_BC_DIRICHLET` | 1 | a value at the face | both (a value other than zero: hierarchical) |
| `ELISHA_BC_NEUMANN` | 2 | a normal derivative at the face | both (a derivative other than zero: hierarchical) |
| `ELISHA_BC_DIRICHLET_GHOST` | 3 | a value at the first ghost centre | both |
| `ELISHA_BC_ZERO_GHOSTS` | 4 | every ghost layer zero (PRISM's D(G), by CG) | both (hierarchical: as ghost-Dirichlet, zero) |
| `ELISHA_BC_FREE_GHOSTS` | 5 | the free-space potential in the ghosts (PRISM's D(G)) | spectral, with `dg_order` > 0 |

The rules, each with the status it earns:

- **a periodic axis of the mesh is periodic in `bc`, and only then** (`ELISHA_ERR_BC_PERIODICITY`), and an axis is
  periodic at both faces or at neither (`ELISHA_ERR_BC_PERIODIC_PAIR`);
- **ghost-Dirichlet pairs only with itself** (`ELISHA_ERR_BC_GHOST_PAIR`), and so do zero and free ghosts
  (`ELISHA_ERR_BC_ZERO_GHOST_PAIR`); Dirichlet and Neumann mix freely;
- **a bounded box needs kinds** (`ELISHA_ERR_BC_MISSING`), unless it is `isolated`;
- **isolated takes no kinds** (`ELISHA_ERR_ISOLATED_BC`), and needs every axis non-periodic
  (`ELISHA_ERR_ISOLATED_PERIODIC`), cubic cells and three dimensions;
- **with no Dirichlet face** (periodic, Neumann everywhere) the problem is singular: the solver removes the mean of
  the source and returns the potential of zero mean.

The program tries ten configurations on 2³ blocks:

<<< @/examples/snippets/cookbook_bc-cases.F90{fortran}

<<< @/examples/output/cookbook_bc.txt{text}

Every refusal is returned by `setup`, the same on every rank, with a message naming the face and the rule; the
solver is left not set up (`path()` is 0). The whole list of codes: [Status codes](/reference/status).

<details><summary>The whole program</summary>

<<< @/examples/snippets/cookbook_bc.F90{fortran}

</details>
