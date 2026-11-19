# Quick Start

```fortran
use elisha, only : I4P, R_P   ! the working kinds, re-exported from PENF (R_P is R8P)
```

The solvers are being ported from ADAM's; this page follows them.

## Tests

```bash
fobis fetch
fobis build --mode tests-gnu && bash scripts/run_tests.sh          # MPI (mpif90 first on PATH), FFTW
fobis build --mode tests-gnu-serial && bash scripts/run_tests.sh   # serial, no MPI
fobis rule --ex makecoverage-analysis   # what CI runs: coverage into docs/public/coverage.json
```
