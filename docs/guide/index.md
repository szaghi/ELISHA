# About

ELISHA (**E**lliptic **L**aplacian **I**nversion, **S**pectral and **H**ierarchical **A**lgorithms) inverts the
discrete Laplacian on block-structured grids, $L_h\phi = \mathrm{factor}\cdot f$, for HPC codes in modern Fortran.
It has two solvers behind one facade:

- a **spectral** solver, direct, by FFT, on uniform block lattices: periodic, with Dirichlet and Neumann faces, or
  isolated (free space);
- a **hierarchical** solver on 2:1-balanced octrees of blocks (AMR): flexible conjugate gradients preconditioned by a
  multigrid V-cycle, the spectral solver at its root.

The facade (`elisha_solver_object`) chooses the solver for the mesh. Everything runs on the communicator you give it,
or serially, on the host or, with cuFFT and OpenACC, on NVIDIA GPUs. Failures come back as a status code, never as a
stopped program.

## Where it comes from

ELISHA is the elliptic solvers of [ADAM](https://github.com/szaghi/adam), a Fortran SDK for adaptive grids and its
solvers (FLUME, PRISM), where they were called FLAIL, pulled out into a library of their own. It is the fourth such extraction after SETH (reconstruction), SELAH (time integration) and
GIDEON (finite-difference operators), and the first that is a distributed solver: it holds plans and operators, and
talks over MPI. The port is bitwise: tests compile ADAM's code unchanged next to ELISHA's and compare the bits, on 1
to 3 ranks with gfortran, nvfortran and ifx, and on an A30. Then ELISHA departed from its master in one place on
purpose: ADAM's V-cycle was not symmetric on AMR meshes, and on one case FCG stalled on it. ELISHA's is symmetric,
converges where ADAM's stalled, and solves the AMR problems of the scaling ladders in 24 to 30% less time
([the hierarchical solver](/reference/hierarchical#the-symmetric-v-cycle)). ADAM's V-cycle stays available, for
parity.

## The name

Elisha asked Elijah for a double portion of his spirit (2 Kings 2:9) and took up the mantle that fell from him
(2:13): two solvers, one facade, the master's work carried on. The same chapter has Elisha, mocked for his bald head,
calling two she-bears out of the woods (2:23–24). ELISHA is gentler: give it a mesh that is not 2:1 balanced and it
returns `ELISHA_ERR_MESH_BALANCE`, naming the leaf. And as with the widow's oil (2 Kings 4:1–7), it fills every vessel
brought to it, provided, on the hierarchical path, the vessel shares a face with the interior: the edge and corner
ghosts it leaves to you ([the solver facade](/reference/solver#solve)).

## Status

- The port of FLAIL is complete: both solvers, their device backends and the facade, held bitwise to ADAM's code by
  the tests (the multigrid with `vcycle=ELISHA_VCYCLE_ADAM`).
- The physics verification moved from ADAM to ELISHA: the solvers against analytic and discrete solutions
  ([Verification](/verification/)), and the scaling ladders on up to 64 CPU ranks and 4 A30s
  ([Performance](/guide/performance#scaling-eb)).
- The performance milestone is done: five changes, each measured on its own ([Performance](/guide/performance)).
- Next: ADAM uses ELISHA in place of FLAIL (ADAM issue #136).

## How these pages are organised

| To | Read |
|---|---|
| build it and solve one problem | [Installation](/guide/installation), [Quick start](/guide/quickstart) |
| know the mesh, the packed source and the ghosts before anything else | [Concepts](/guide/concepts) |
| learn it, from a box to two embers on the GPU | [the Tutorial](/tutorial/) |
| do one specific thing | [the Cookbook](/cookbook/) |
| look up a call, an argument, a status code | [the Reference](/reference/), the [API](/api/) |
| see how it is checked | [Verification](/verification/), [Performance](/guide/performance) |
| move a FLAIL client over | [Coming from ADAM's FLAIL](/guide/from-adam) |
