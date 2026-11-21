---
title: Parameters
---

# Parameters

The named constants of `elisha_parameters` (and one of the hierarchical solver), all `integer(I4P)` unless noted. Use
the names: the values are listed for reading logs, not for code.

## Boundary kinds

`bc(2,3)` at setup, `[low, high]` per axis; `boundary_faces` returns them per face.

| Name | Value | The face holds |
|---|---|---|
| `ELISHA_BC_PERIODIC` | 0 | nothing: the axis wraps around (both faces) |
| `ELISHA_BC_DIRICHLET` | 1 | a value at the face |
| `ELISHA_BC_NEUMANN` | 2 | a normal derivative at the face |
| `ELISHA_BC_DIRICHLET_GHOST` | 3 | a value at the first ghost centre |
| `ELISHA_BC_ZERO_GHOSTS` | 4 | every ghost layer zero (the wide D(G): by CG) |
| `ELISHA_BC_FREE_GHOSTS` | 5 | the free-space potential in the ghost layers (the wide D(G)) |
| `ELISHA_BC_KINDS(6)` | | all six, in this order |

Which pairs an axis accepts, and the refusals: [Which faces go together](/cookbook/faces).

## Paths

`path` at setup; `path()` and `diagnostics()%path` after it.

| Name | Value | Chooses |
|---|---|---|
| `ELISHA_PATH_AUTO` | 0 | (default) the spectral solver on a uniform mesh, the hierarchical one otherwise; `path()` before a setup |
| `ELISHA_PATH_SPECTRAL` | 1 | the spectral (FFT) solver; a mesh at more than one level is refused |
| `ELISHA_PATH_HIERARCHICAL` | 2 | the hierarchical (multigrid FCG) solver, on any 2:1 mesh |

## Options of the hierarchical solver

| Name | Value | Meaning |
|---|---|---|
| `ELISHA_OPERATOR_CSR` | 0 | `level_operator`: the V-cycle's level operators from their CSR entries (the host's default) |
| `ELISHA_OPERATOR_STENCIL` | 1 | `level_operator`: their block-interior rows as a matrix-free 7-point stencil (the OpenACC backend's default) |
| `ELISHA_VCYCLE_SYMMETRIC` | 0 | `vcycle`: the seam coupling both ways, a symmetric preconditioner (default) |
| `ELISHA_VCYCLE_ADAM` | 1 | `vcycle`: ADAM's, the seam coupling on the way up only; for bitwise parity with ADAM |
| `ELISHA_ASSEMBLY_HOST` | 0 | `assembly`: the right-hand side, ghost values and energy on the host (the host solver's only one) |
| `ELISHA_ASSEMBLY_DEVICE` | 1 | `assembly`: on the device (the OpenACC backend's default) |

## The 2:1 seam

`MG_SEAM_GHOST_SCALE` (`real(R_P)`, 1.5): the 2:1 fine-side ghost of the composite operator's face gradient, in the
source's own notation `phi_g = phi_f - (mean_f - phi_C) / 1.5`. A client that inverts that gradient (FLUME's seam
gravity does) uses this constant. See [the hierarchical solver](/reference/hierarchical).

## Kinds

`I4P`, `I8P`, `R8P` and `R_P` from PENF. Every solver computes in `R_P`, which is `R8P`: a build whose PENF `R_P` is
not double precision fails at compile time.
