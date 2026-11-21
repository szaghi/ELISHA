# Changelog

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/).
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.2] — 2026-11-21
### Added
- **status**: ELISHA_ERR_NOT_CONVERGED, an iterative solve stopped above its tolerance


### Changed
- **hierarchical**: An overridable FCG solve, its device twin, and the `assembly` option

- **lib**: No argument temporaries in the library; ifx's arg_temp check on ELISHA's code


### Documentation
- **guide**: The scaling ladders, against FLAIL's table

- **prototypes**: FLAIL's design prototypes, each guide number linked to its producer

- **verification**: The solvers' verification pages, with ADAM's figures regenerated

- **verification**: CPU = device with the symmetric V-cycle (medooza job 1369211)

- **examples**: The AMR example's output with the symmetric V-cycle

- **performance**: The strong ladder re-timed with the symmetric V-cycle (medooza job 1369212)

- **performance**: The device ladder with the device assembly (medooza job 1369224)

- **install**: Known compiler issue, nvfortran's polymorphic-array deallocation (#2)

- **style**: New identity, Futatsu-bi mark in the Koyo ink

- Refactor on SETH's plan: tutorial, cookbook, reference, figures

- **landing**: The casts and figures under the hero, site and README


### Fixed
- **verification**: No stack temporaries in the verification tests' bit comparisons

- **verification**: No polymorphic solver arrays in the sub leg (nvfortran 26.9)

- **verification**: Interior_diff as a subroutine (nvfortran -Mbounds)

- **lib**: No argument temporaries in mesh validation and the spectral solver's sub-problems

- **verification**: No argument temporaries in the verification drivers

- **hierarchical**: A symmetric V-cycle, the adjoint of the seam coupling on the way down

- **tests,bench**: ADAM's V-cycle for the remaining bitwise comparisons with ADAM


### Performance
- **bench**: The scaling ladders of ADAM's bench.sh, on ELISHA alone

- **device**: The multigrid's assembly on the device, the OpenACC default

- **device**: The spectral solver's on_device does not stage phi down


## [0.1.1] — 2026-11-20
### Fixed
- **install**: The one-line installer builds the serial library


### Performance
- **device**: The stencil level operators by default on OpenACC


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



