# Changelog

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/).
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-11-20
### Added
- Scaffold ELISHA, the elliptic solver library for HPC

- **l0**: Boundary kinds, spectral math and extension maps

- **l0**: Lattice Green's function of the 7-point Laplacian

- **status**: The stat/errmsg policy and its codes

- **comm**: The communication context on the caller's communicator

- **keys**: Cell keys and the hash table of the hierarchical solver

- **mesh**: The mesh descriptor and its collective checks

- **checks**: The setup checks of the solvers' arguments

- **spectral**: The spectral solver, ported from ADAM with the arithmetic unchanged

- **hierarchical**: The multigrid solver, ported from ADAM with the arithmetic unchanged

- **device**: The cuFFT spectral and OpenACC hierarchical backends

- **solver**: One facade over both solvers, with a diagnostics type


### Documentation
- **guide**: The L0 layer, the solvers' math

- **guide**: Errors, communicator and mesh

- **guide**: The spectral solver

- **guide**: The hierarchical solver

- **guide**: The device backends

- **examples**: Five compiled, run examples on SETH's scheme

- **guide**: The solver facade, the examples, coming from ADAM's FLAIL

- **guide**: The M7 performance page


### Performance
- **spectral**: Benchmark against ADAM; keep nvfortran's transpose loops vectorised

- **hierarchical**: Benchmark against ADAM's multigrid

- **keys**: Mix the cell keys before the table's modulo

- **device**: Device benchmark against ADAM's device solvers

- **hierarchical**: The red-black colour as an integer

- **hierarchical**: Neighbour collectives for the halo exchanges

- **device**: The cuFFT backend's constant arrays stay on the device

- **device**: The multigrid root on the device, and device-resident arrays

- **hierarchical**: An opt-in stencil for the block-interior rows of the level operators



