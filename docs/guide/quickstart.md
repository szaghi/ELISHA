# Quick Start

One program, compiled against the library, that solves a Poisson problem on a uniform periodic mesh and checks the
result: the discrete Laplacian of the potential against the source, and the potential against the continuum solution.

## Build the library

```bash
git clone https://github.com/szaghi/ELISHA && cd ELISHA
fobis fetch                            # PENF, into src/third_party/
fobis build --mode static-gnu-serial   # lib/gnu/static-serial/libelisha_serial.a and its modules in .../mod
```

This is the serial library (`-DELISHA_NO_MPI`, no MPI at all). The MPI library (`static-gnu`), the other compilers and
fpm are in [Installation](./installation).

## The program

`use elisha` gives the whole library: the kinds (`R_P` is the working real kind), the mesh descriptor, the
communication context, the solver facade, the boundary kinds and the status codes:

<<< @/examples/snippets/quickstart-use.F90{fortran}

The mesh is a domain of uniform blocks: here 2^3 blocks of 8^3 cells at level 0, every axis periodic. Each rank
describes its own blocks by their integer coordinates and level. The communication context duplicates the caller's
communicator; the serial build ignores it:

<<< @/examples/snippets/quickstart-mesh.F90{fortran}

The source goes in packed: the interior cells of this rank's blocks, block-major, `i` fastest. The potential has
ghost layers, which the solver fills:

<<< @/examples/snippets/quickstart-source.F90{fortran}

`setup` chooses the path for the mesh, here the spectral one (every block at one level). `solve_packed` solves
`L_h phi = factor (f - mean f)`. Every call returns a status, which the program checks:

<<< @/examples/snippets/quickstart-solve.F90{fortran}

The ghost layers carry the periodic wrap, so the 7-point Laplacian of `phi` reads them directly:

<<< @/examples/snippets/quickstart-check.F90{fortran}

<details><summary>The whole program</summary>

<<< @/examples/snippets/quickstart.F90{fortran}

</details>

## Compile and run

```bash
gfortran -I lib/gnu/static-serial/mod quickstart.F90 lib/gnu/static-serial/libelisha_serial.a -lfftw3 -o quickstart
```

<<< @/examples/output/quickstart.txt{text}

The spectral solve is direct: no iterations, and the discrete residual is at round-off. The 1.3% error is the
discretisation's: the eigenvalue of the 7-point operator for this mode differs from the continuum one at second order
in the spacing, 16 cells per wavelength.

With MPI, build `static-gnu` (or `-nvf`, `-intel`), pass your communicator (`MPI_COMM_WORLD`) to `comm%init` after
`MPI_Init`, give each rank its own blocks, and run under `mpirun`. The spectral potential is bitwise the same on every number of
ranks; the hierarchical one agrees to round-off.

## Tests and examples

```bash
fobis fetch
fobis build --mode tests-gnu && bash scripts/run_tests.sh          # MPI (mpif90 first on PATH), FFTW
fobis build --mode tests-gnu-serial && bash scripts/run_tests.sh   # serial, no MPI
fobis rule --ex makecoverage-analysis   # what CI runs: coverage into docs/public/coverage.json
bash scripts/docs_examples.sh           # build and run docs/examples, regenerating the snippets and outputs
```

`scripts/docs_examples.sh` builds `static-gnu-serial` and compiles every program in `docs/examples/src` against it, as
above, with `-fcheck=bounds`. It writes the snippets and the outputs these pages include, and CI fails when the
committed ones differ.

## Where next

| To | Read |
|---|---|
| see the facade's arguments, diagnostics and refusals | [The solver facade](./solver) |
| run AMR, isolated boundaries, boundary kinds and error handling | [Examples](./examples) |
| port a FLAIL client | [Coming from ADAM's FLAIL](./from-adam) |
| see the solvers' methods and how they are checked | [The spectral solver](./spectral-solver), [the hierarchical solver](./hierarchical-solver) |
