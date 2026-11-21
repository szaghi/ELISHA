---
title: Host or device arrays
---

# Host or device arrays

With the device library (`fobis build --mode static-nvf-oac`) two molds choose the GPU backends:

```fortran
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
type(elisha_spectral_solver_cufft_object)   :: cufft ! molds: only their types are read
type(elisha_hierarchical_solver_oac_object) :: oac
call solver%setup(mesh, comm, stat, errmsg, spectral_mold=cufft, hierarchical_mold=oac)
```

Then a choice of where your arrays live ([tutorial 9](/tutorial/09-gpu)):

| Your fields | Call | What the solve copies |
|---|---|---|
| on the host | `solve_packed(mesh, f, factor, phi, ...)` | the source up, the potential down |
| on the GPU (`!$acc data`, `enter data`) | `solve_packed(..., on_device=.true.)` | nothing on the hierarchical path; the spectral path still gathers the source to rank 0 through the host |

The two give bitwise the same `phi`. On the medooza A30 ladder (job 1369224, minimum solve time):

| Problem, 1 GPU | Host arrays | `on_device` |
|---|---|---|
| multigrid, 3.9 M AMR cells, periodic | 0.191 s | 0.143 s |
| multigrid, isolated | 0.497 s | 0.408 s |
| FFT, 2.1 M cells, periodic | 0.0565 s | 0.0704 s |
| FFT, isolated | 0.114 s | 0.126 s |

So: `on_device` when your fields live on the GPU, which is when it pays off on the multigrid (15 to 25% faster than
host arrays at 1 and 2 GPUs). The spectral path takes 1 to 28% longer with it in that job, because its gather to
rank 0 still goes through the host.

## `assembly`

On the OpenACC hierarchical backend the right-hand side, the boundary terms, the mean removal, the ghost fill and the
energy are assembled on the GPU (`assembly=ELISHA_ASSEMBLY_DEVICE`, the default). `ELISHA_ASSEMBLY_HOST` assembles them
on the host as ADAM's GPU code did, for bitwise ADAM's results; with it, `on_device` has to stage the fields again.
Use it to compare with ADAM; the host solver takes only `ELISHA_ASSEMBLY_HOST`.
