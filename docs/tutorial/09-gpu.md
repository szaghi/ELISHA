---
title: 9. On the GPU
---

# 9. On the GPU

The device library (`fobis build --mode static-nvf-oac`: nvfortran, OpenACC, cuFFT) adds two backends: a spectral
solver on cuFFT and a hierarchical solver whose V-cycle, FCG and assembly run on the GPU with OpenACC, its root on
cuFFT. The facade picks them by **mold**: you pass an object of the backend's type, and only its type is read.

## The program

It builds with or without OpenACC: without it (the docs' serial build) the host solvers run and `on_device` is off.

<<< @/examples/snippets/t09_gpu-use.F90{fortran}

<<< @/examples/snippets/t09_gpu-molds.F90{fortran}

<<< @/examples/snippets/t09_gpu-setup.F90{fortran}

## Host arrays or device arrays

A device backend takes the source and returns the potential in **host arrays**, staging them itself:

<<< @/examples/snippets/t09_gpu-host_arrays.F90{fortran}

A solver whose fields already live on the GPU passes **device-resident arrays**: `f` and `phi` present on the device
(an `!$acc data` region, or `enter data`), and `on_device=.true.`. The solve then stages nothing:

<<< @/examples/snippets/t09_gpu-device_arrays.F90{fortran}

`copy`, not `copyout`: the hierarchical path writes the interior and the face ghosts of `phi`, not the edge and corner
ghosts, which keep what they held ([the solver facade](/reference/solver#solve)).

## Running it

On the host (the docs run):

<<< @/examples/output/t09_gpu.txt{text}

On an NVIDIA A30 of the medooza cluster, captured by hand (`docs/examples/device/`):

<<< @/examples/device/t09_gpu.txt{text}

- **The same numbers as the host**: errors, iterations.
- **Device-resident arrays are bitwise host arrays**: 0 values differ, on 1 rank and on 2 ranks sharing the GPU.
  The two modes run the same kernels and differ only in the copies at the ends.

How much faster: on an A30, ELISHA's device multigrid takes 17 to 37% less time than ADAM's GPU code on AMR meshes of
0.5 and 3.9 M cells (job 1369037). On the scaling ladder's 3.9 M-cell AMR problem one A30 with the arrays on the
device solves in 0.143 s, 64 ranks of a Xeon node in 0.288 s, with the same protocol (jobs 1369224 and 1369212,
[Performance](/guide/performance#scaling-eb)).

::: tip What you learned
A device backend is a mold at setup. Arrays already on the GPU go in with `on_device=.true.`, and the result is
bitwise the one from host arrays.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t09_gpu.F90{fortran}

</details>

That is the tutorial. Next: the [cookbook](/cookbook/), short recipes for specific tasks.
