---
title: 7. Many ranks
---

# 7. Many ranks

ELISHA is a distributed solver: every rank holds some of the blocks, and `setup` and `solve_packed` are collective
over the communicator. This chapter is one program that runs on one rank or many. Built against the serial library
it is a plain program; built against the MPI library it runs under `mpirun`, and nothing in the solver calls changes.

## One source, two builds

The MPI calls of the program are guarded by `ELISHA_NO_MPI`, the macro the serial library is built with:

<<< @/examples/snippets/t07_mpi-use.F90{fortran}

<<< @/examples/snippets/t07_mpi-init.F90{fortran}

`comm%init` duplicates the communicator: ELISHA's messages never mix with yours, and two solvers can live on two
sub-communicators. You initialise and finalise MPI; ELISHA refuses to start before `MPI_Init`
(`ELISHA_ERR_MPI_NOT_INITIALIZED`).

## Each rank its own blocks

The mesh descriptor lists *this rank's* blocks. How they are dealt is yours; here leaf $n$ goes to rank
$(n-1) \bmod P$:

<<< @/examples/snippets/t07_mpi-deal.F90{fortran}

`setup` gathers the leaf list over the ranks and validates it collectively: a gap, an overlap or an imbalance between
two ranks' blocks is found and reported to every rank alike. The packed source and `phi` hold this rank's blocks
only, in the order of its `coords`.

<<< @/examples/snippets/t07_mpi-solve.F90{fortran}

## Running it

The serial build, as the docs run it:

<<< @/examples/output/t07_mpi.txt{text}

The MPI build at 1, 2 and 4 ranks, with `bits`: the bit pattern of $\max\phi$, reduced over the ranks (a maximum is
exact in any order, so it tells whether $\phi$ itself changed). Captured by hand (`docs/examples/mpi/`):

<<< @/examples/mpi/t07_mpi.txt{text}

- **The answer does not depend on the rank count**, to the digits printed.
- **The spectral path is bitwise rank-invariant**: $\max\phi$ has the same 64 bits on 1, 2 and 4 ranks (and on 3,
  in a run not kept here). Its slab transform promises this, and the [verification](/verification/fft-poisson)
  checks it on every cell.
- **The hierarchical path is not**, in the last bit: its dot products are sums over the ranks, and a sum's rounding
  depends on its order. The difference is a unit or two in the last place, far below the tolerance.
- **The energy** is a sum over cells, reduced over the ranks: it differs in its last bits across rank counts on both
  paths, for the same reason.

::: tip What you learned
One program serves the serial and the MPI build. Each rank describes its own blocks; `setup` and `solve_packed` are
collective. The spectral path is bitwise across rank counts; the hierarchical path and every reduction are within
round-off.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t07_mpi.F90{fortran}

</details>

Next: [8. Two embers](/tutorial/08-embers).
