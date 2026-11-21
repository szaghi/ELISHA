<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/public/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/public/logo.svg">
  <img src="docs/public/logo.svg" alt="ELISHA, two embers in a hearth" width="320">
</picture>

</div>

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

```fortran
use elisha
type(elisha_solver_object) :: solver
call solver%setup(mesh, comm, stat, errmsg)                  ! spectral on one level, hierarchical on 2:1 AMR
call solver%solve_packed(mesh, f, factor, phi, stat, errmsg) ! L_h phi = factor (f - mean f), ghosts filled
```

Status: the whole of FLAIL is ported: both solvers, their device backends and the facade. Tests hold each one bitwise
to ADAM's code: the host solvers and the facade on 1 to 3 ranks with gfortran, nvfortran and ifx, the device backends
on an A30. The multigrid's default V-cycle is symmetric, a fix of ADAM's that changes its AMR results (fewer
iterations, no stall); `vcycle=ELISHA_VCYCLE_ADAM` keeps ADAM's, and the parity tests use it. The performance milestone (M7) is done: the device V-cycle root, neighbour halos, resident cuFFT
constants, integer colours, and stencil level operators (the OpenACC default), each measured. Next is M8, wiring ELISHA into ADAM (the owner's
call).

```bash
fobis fetch && fobis build --mode tests-gnu && bash scripts/run_tests.sh
```

Documentation: https://szaghi.github.io/ELISHA. License: your choice of GPL-3.0-or-later, BSD-2-Clause,
BSD-3-Clause or MIT.
