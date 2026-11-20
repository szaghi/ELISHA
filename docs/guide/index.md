# About

ELISHA (**E**lliptic **L**aplacian **I**nversion: **S**pectral and **H**ierarchical **A**lgorithms) is a library of
elliptic solvers for block-structured grids: it inverts the discrete Laplacian, `L_h phi = factor f`, by a spectral
(FFT) solver on uniform block lattices and by a hierarchical one, multigrid-preconditioned flexible conjugate
gradients, on 2:1-balanced octrees of blocks (AMR), behind one facade.

It is extracted from the FLAIL solvers of ADAM (`adam_flail_object`, `adam_poisson_fft_object`,
`adam_poisson_mg_object` and their FNL backends), as SETH, SELAH and GIDEON were from ADAM's reconstruction,
time-integration and stencil layers. Unlike those, ELISHA is a distributed solver: it holds its plans and operators,
works over the MPI communicator the caller gives it (or serial, without MPI), and returns a status instead of
stopping the program.

The name comes from Elisha, who asked Elijah for a double portion of his spirit (2 Kings 2:9) and took up the mantle
that fell from him (2:13): two solvers behind one facade, and the work of the master carried on.

## Status

The port of ADAM's FLAIL is complete. Tests hold the math, the solvers and the facade bitwise to ADAM's own code, and
reproduce every ADAM refusal as a status code:

- the [L0 math](./l0-math): eigenvalues, extensions, the D(G) stencils, the lattice Green's function;
- [errors, communicator and mesh](./errors-comm-mesh): a status instead of a stop, the caller's communicator (or the
  serial build), the mesh descriptor and its collective validation;
- [the spectral solver](./spectral-solver), [the hierarchical solver](./hierarchical-solver) and [their device
  backends](./device) (cuFFT, OpenACC);
- [the solver facade](./solver), `elisha_solver_object`, which chooses the path for the mesh, and its diagnostics.

The [Quick Start](./quickstart) and the [Examples](./examples) are programs that CI compiles and runs.
[Coming from ADAM's FLAIL](./from-adam) maps a FLAIL client onto ELISHA. Next is performance (M7): neighbour
collectives, a matrix-free smoother, and a device-resident root and API, each change measured on its own.
