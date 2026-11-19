# Installation

## FoBiS

```bash
git clone https://github.com/szaghi/ELISHA && cd ELISHA
fobis fetch                      # PENF, into src/third_party/
fobis build --mode static-gnu    # lib/gnu/static/libelisha.a (shared-gnu: libelisha.so)
```

`fobis build --lmodes` lists the modes: static and shared libraries and tests for gfortran (`-gnu`), nvfortran
(`-nvf`) and ifx (`-intel`), built with the compiler's MPI wrapper (the `mpif90` first on `PATH`); the serial twins
(`-serial`, `-DELISHA_NO_MPI`, no MPI library); `tests-nvf-oac`, the OpenACC build (`--varset cc80|cc89|cc90` picks
the compute capability; cc89 by default). Every mode compiles with `-D_FFTW`; a program that uses ELISHA links
`-lfftw3`. ELISHA computes in double precision only (`R_P = R8P`): a single-precision build stops at compile time.

## fpm

```toml
[dependencies]
ELISHA = { git = "https://github.com/szaghi/ELISHA", branch = "main" }
```
