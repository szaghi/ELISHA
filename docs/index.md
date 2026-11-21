---
layout: home

hero:
  name: ELISHA
  text: Elliptic Laplacian Inversion, Spectral and Hierarchical Algorithms
  tagline: FFT and AMR multigrid Poisson solvers for HPC, in modern Fortran
  actions:
    - theme: brand
      text: Guide
      link: /guide/
    - theme: alt
      text: Verification
      link: /verification/
    - theme: alt
      text: API Reference
      link: /api/
    - theme: alt
      text: View on GitHub
      link: https://github.com/szaghi/ELISHA

features:
  - title: Spectral
    details: A direct FFT solver on uniform block lattices, periodic, with Dirichlet and Neumann faces, or isolated (Hockney–Eastwood with the lattice Green's function); the residual of the discrete operator at round-off.
  - title: Hierarchical
    details: Multigrid-preconditioned flexible conjugate gradients on 2:1-balanced octrees of blocks, with a symmetric, consistent composite Laplacian and the FFT solver at its root.
  - title: Distributed, CPU and GPU
    details: MPI over a caller-given communicator, or serial; FFTW on the host, cuFFT and OpenACC on NVIDIA GPUs; tested with gfortran, nvfortran and ifx.
---
