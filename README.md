# ELISHA

**E**lliptic **L**aplacian **I**nversion: **S**pectral and **H**ierarchical **A**lgorithms: FFT and AMR multigrid
Poisson solvers for HPC, in modern Fortran.

[![CI](https://github.com/szaghi/ELISHA/actions/workflows/ci.yml/badge.svg)](https://github.com/szaghi/ELISHA/actions/workflows/ci.yml)
[![Matrix](https://github.com/szaghi/ELISHA/actions/workflows/matrix.yml/badge.svg)](https://github.com/szaghi/ELISHA/actions/workflows/matrix.yml)

ELISHA inverts the discrete Laplacian on block-structured grids: a spectral (FFT) solver on uniform block lattices,
periodic, with Dirichlet and Neumann faces or isolated, and a hierarchical one, multigrid-preconditioned flexible
conjugate gradients on 2:1-balanced octrees of blocks, behind one facade. It is extracted from the FLAIL solvers of
ADAM, as SETH, SELAH and GIDEON were from its other layers, and is distributed over a caller-given MPI communicator
(or serial); FFTW on the host, cuFFT and OpenACC on NVIDIA GPUs. Elisha asked for a double portion of his master's
spirit and took up his mantle (2 Kings 2:9-13): two solvers, one facade, the master's work carried on.

```bash
fobis fetch && fobis build --mode tests-gnu && bash scripts/run_tests.sh
```

Documentation: https://szaghi.github.io/ELISHA. License: your choice of GPL-3.0-or-later, BSD-2-Clause,
BSD-3-Clause or MIT.
