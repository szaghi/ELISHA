---
title: Choose a path
---

# Choose a path

`setup(..., path=ELISHA_PATH_AUTO)`, the default, chooses for you: the **spectral** path when every leaf is at one
level, the **hierarchical** path otherwise. Force one only when you need what only it does.

| You need | Path | Why |
|---|---|---|
| A uniform mesh, periodic or with homogeneous faces | spectral (auto) | a direct solve, no iterations; the residual is round-off |
| A 2:1 AMR mesh | hierarchical (auto) | the spectral path refuses more than one level (`ELISHA_ERR_MESH_LEVELS`) |
| Inhomogeneous boundary values | hierarchical | `ghost` is the hierarchical path's (`ELISHA_ERR_GHOST_VALUES` on the spectral one); force it on a uniform mesh with `path=ELISHA_PATH_HIERARCHICAL` |
| Free space (`isolated`) | either | Hockney and Eastwood on the spectral path, the James step on the hierarchical one; they agree ([tutorial 3](/tutorial/03-isolated)) |
| PRISM's wide operator D(G) (`dg_order` 2 to 8) | spectral | uniform meshes only (`ELISHA_ERR_DG_AMR`) |
| Bitwise results across rank counts | spectral | the slab transform is rank-invariant; the hierarchical path's reductions are not ([tutorial 7](/tutorial/07-mpi)) |

## The options that matter

On the hierarchical path:

| Option | Default | When to change it |
|---|---|---|
| `tolerance` | 1e-10 | the relative max-norm residual of the FCG; raised to the source's round-off floor automatically when that is larger |
| `max_iterations` | 200 | lower it to bound the cost of a step, and handle [`ELISHA_ERR_NOT_CONVERGED`](/cookbook/not-converged) |
| `vcycle` | `ELISHA_VCYCLE_SYMMETRIC` | `ELISHA_VCYCLE_ADAM` only to reproduce ADAM's results bit for bit |
| `root_cells` | the root solver's | cap the root of the V-cycle (its spectral solve) at this many cells |
| `level_operator` | CSR on the host, stencil on OpenACC | `ELISHA_OPERATOR_STENCIL` on the host is an option, not faster there ([Performance](/guide/performance)) |
| `assembly` | host on the host, device on OpenACC | `ELISHA_ASSEMBLY_HOST` on OpenACC for ADAM's bitwise parity ([Host or device arrays](/cookbook/device-arrays)) |

On both: `spectral_mold` and `hierarchical_mold` choose the device backends ([tutorial 9](/tutorial/09-gpu)).
Every option and its refusal: [the solver facade](/reference/solver#setup).
