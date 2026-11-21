# Installation

## One line

The installer downloads the latest release, builds the serial library with the build system you have (FoBiS, fpm,
CMake or make) and installs it under `~/.local`, one directory per compiler, with a `pkg-config` file:

```bash
curl -fsSL https://github.com/szaghi/ELISHA/releases/latest/download/install.sh | bash -s -- --gnu
```

`--help` lists the options (`--prefix`, `--tag`, `--intel`, `--nvidia`, `--dry-run`, `--uninstall`, ...).

## FoBiS

```bash
git clone https://github.com/szaghi/ELISHA && cd ELISHA
fobis fetch                      # PENF, into src/third_party/ (and GIDEON, for one test)
fobis build --mode static-gnu    # lib/gnu/static/libelisha.a (shared-gnu: libelisha.so)
```

`fobis build --lmodes` lists the modes: static and shared libraries and tests for gfortran (`-gnu`), nvfortran
(`-nvf`) and ifx (`-intel`), built with the compiler's MPI wrapper (the `mpif90` first on `PATH`); the serial twins
(`-serial`, `-DELISHA_NO_MPI`, no MPI library); `tests-nvf-oac`, the OpenACC build (`--varset cc80|cc89|cc90` picks
the compute capability; cc89 by default). Every mode compiles with `-D_FFTW`; a program that uses ELISHA links
`-lfftw3`. ELISHA computes in double precision only (`R_P = R8P`): a single-precision build stops at compile time.

The device library (cuFFT and OpenACC, nvfortran): `fobis build --mode static-nvf-oac --varset cc80` (A100/A30;
`cc89` RTX 40, `cc90` H100), then `use elisha_device` ([The device backends](/reference/device)).

## fpm

```toml
[dependencies]
ELISHA = { git = "https://github.com/szaghi/ELISHA", branch = "main" }
```

fpm builds the serial library (`-DELISHA_NO_MPI`) with FFTW: link `-lfftw3`.

## Known compiler issues

- **nvfortran 25.11 to 26.9: a polymorphic array of an extended solver crashes when it is deallocated**
  ([issue #2](https://github.com/szaghi/ELISHA/issues/2)). The pattern is
  `class(elisha_hierarchical_solver_object), allocatable :: s(:)` allocated, with two or more elements, as a type that
  extends the solver and adds a component, such as the OpenACC backend or the client's own extension. The runtime
  segfaults in `pgf90_dealloc_mbr03a_i8`, at an explicit `deallocate` or on leaving the scope. The bug needs a
  component default-initialised to a nonzero value next to an allocatable one, which ELISHA's solver types have; a
  20-line program without ELISHA reproduces it, and gfortran runs it correctly. ELISHA and ADAM hold their solvers as
  scalars, which work. To keep several solvers, use an array of a box type whose component is a scalar
  `class(...), allocatable :: s`, as `mg_box` in `src/tests/elisha_sv_mg_mpi.F90`.
