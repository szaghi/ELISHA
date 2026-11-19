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

The repository is being seeded: the build (MPI and serial, FFTW), the tests and the CI are in place, and the solvers
are being ported from ADAM's, bitwise first, faster second.
