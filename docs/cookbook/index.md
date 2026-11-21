---
title: Cookbook
---

# Cookbook

Short recipes for specific tasks. Four are programs in `docs/examples/src/`, with the code and output on each page
taken from them when the documentation was built; the others are decisions, with the numbers taken from the
tutorial, the verification and the benchmarks.

| Recipe | Task |
|---|---|
| [Choose a path](/cookbook/choose) | spectral or hierarchical, and the options that matter on each |
| [Which faces go together](/cookbook/faces) | the face kinds, the pairs an axis accepts, and the refusals |
| [Your fields in and out](/cookbook/fields) | pack a field with ghost cells into the source, take the potential back in the same layout |
| [Self-gravity](/cookbook/gravity) | `factor = 4 pi G`, free space, and the binding energy from the solver's energy |
| [One setup, many solves](/cookbook/reuse) | what a setup costs, and what reusing it buys |
| [When it does not converge](/cookbook/not-converged) | `ELISHA_ERR_NOT_CONVERGED`: what comes back, and what to do with it |
| [Host or device arrays](/cookbook/device-arrays) | the device backends, `on_device` and `assembly` |

The recipes the tutorial already covers: isolated boundaries ([3](/tutorial/03-isolated)), an AMR mesh
([4](/tutorial/04-amr)), boundary values ([5](/tutorial/05-values)), MPI ([7](/tutorial/07-mpi)), a regrid
([8](/tutorial/08-embers)).
