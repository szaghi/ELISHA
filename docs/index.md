---
layout: home

hero:
  name: ELISHA
  text: Elliptic Laplacian Inversion, Spectral and Hierarchical Algorithms
  tagline: FFT and AMR multigrid Poisson solvers for HPC, in modern Fortran. It took up the mantle of ADAM's FLAIL, asked for a double portion, and got two solvers and twice the tests.
  image:
    light: /logo.svg
    dark: /logo-dark.svg
    alt: ELISHA, two embers in a hearth
  actions:
    - theme: brand
      text: Quick start
      link: /guide/quickstart
    - theme: alt
      text: Tutorial
      link: /tutorial/
    - theme: alt
      text: Reference
      link: /reference/
    - theme: alt
      text: Verification
      link: /verification/
    - theme: alt
      text: View on GitHub
      link: https://github.com/szaghi/ELISHA

features:
  - icon:
      light: /icons/light/fft-spectral.svg
      dark: /icons/dark/fft-spectral.svg
      width: 44
      height: 44
    title: Spectral, to round-off
    details: A direct FFT solver on uniform block lattices. It divides by the eigenvalues of the discrete Laplacian, not the continuum's, so the residual of what it returns is round-off; periodic, Dirichlet and Neumann faces by odd and even extensions, and PRISM's wide operator.
    link: /tutorial/01-box
    linkText: A periodic box
  - icon:
      light: /icons/light/amr-multigrid.svg
      dark: /icons/dark/amr-multigrid.svg
      width: 44
      height: 44
    title: Hierarchical, on 2:1 octrees
    details: Flexible conjugate gradients preconditioned by a multigrid V-cycle on 2:1-balanced octrees of blocks, the FFT solver at its root. The V-cycle is symmetric, which ADAM's was not; it now converges where ADAM's stalled, in 24 to 30% less time.
    link: /tutorial/04-amr
    linkText: Refine where it matters
  - icon:
      light: /icons/light/isolated.svg
      dark: /icons/dark/isolated.svg
      width: 44
      height: 44
    title: Stars in empty space
    details: Isolated boundaries, the potential going to zero at infinity. Hockney and Eastwood with the lattice Green's function on the spectral path, the James step on the hierarchical one; both paths agree to 1e-10 and match the analytic potential of a Gaussian star.
    link: /tutorial/03-isolated
    linkText: A star in empty space
  - icon:
      light: /icons/light/dirichlet-neumann.svg
      dark: /icons/dark/dirichlet-neumann.svg
      width: 44
      height: 44
    title: Refuses politely
    details: Six face kinds and inhomogeneous boundary values. A mesh that is not 2:1, faces that do not pair, a solve that does not converge come back as one of 36 named status codes with a message; the library never stops your program.
    link: /cookbook/faces
    linkText: Which faces go together
  - icon:
      light: /icons/light/gpu.svg
      dark: /icons/dark/gpu.svg
      width: 44
      height: 44
    title: Many ranks, many GPUs
    details: Any MPI communicator, or none (the serial build). FFTW on the host, cuFFT and OpenACC on NVIDIA GPUs, with the arrays on the device if you like. On an A30 the multigrid takes 17 to 37% less time than ADAM's GPU code, and solves a 3.9 M-cell AMR problem in half the time of 64 CPU cores.
    link: /tutorial/09-gpu
    linkText: On the GPU
  - icon:
      light: /icons/light/round-off.svg
      dark: /icons/dark/round-off.svg
      width: 44
      height: 44
    title: Verification you can rerun
    details: Bitwise to ADAM's code on 1 to 3 ranks with gfortran, nvfortran and ifx, and on an A30; the physics against analytic solutions; every number in these pages printed by a program that CI compiles and runs.
    link: /verification/
    linkText: Verification
---

<script setup>
import { withBase } from 'vitepress'
import { VPTeamMembers } from 'vitepress/theme'

const authors = [
  {
    avatar: 'https://github.com/szaghi.png',
    name: 'Stefano Zaghi',
    title: 'Chief Yak Shaver & Inheritor of Mantles',
    desc: "Wrote FLAIL inside ADAM, then pulled it out again as its own library, because the fourth extraction of a monorepo is when you finally get the naming right.",
    links: [{ icon: 'github', link: 'https://github.com/szaghi' }],
  },
  {
    avatar: withBase('/avatars/claude.svg'),
    name: 'Claude',
    org: 'Anthropic',
    orgLink: 'https://claude.ai',
    title: 'Tireless Rubber Duck & Professional Ghost Filler',
    desc: "AI pair programmer; fills the ghost layers so humans don't have to, found the V-cycle that was not symmetric, and has mentioned it ever since.",
  },
]
</script>

<div class="elisha-home">

## Quick start

<p class="elisha-lead">Build the library and run its tests, or pull ELISHA into your own project.</p>

::: code-group

```bash [Build and test]
git clone https://github.com/szaghi/ELISHA && cd ELISHA
fobis fetch                                               # PENF (and GIDEON, for one test)
fobis build --mode static-gnu                             # lib/gnu/static/libelisha.a, MPI
fobis build --mode tests-gnu && bash scripts/run_tests.sh # the test programs, MPI ones on 2 ranks
```

```ini [FoBiS dependency]
# fobos of your project; then: fobis fetch && fobis build
[dependencies]
deps_dir = src/third_party
PENF     = https://github.com/szaghi/PENF :: branch=master   # ELISHA's own dependency: fobis fetch is not transitive
ELISHA   = https://github.com/szaghi/ELISHA :: branch=main
```

```toml [fpm dependency]
# fpm.toml of your project (the serial build: no MPI)
[dependencies]
ELISHA = { git = "https://github.com/szaghi/ELISHA", branch = "main" }
```

:::

<p class="elisha-lead">Describe your blocks, give the solver your communicator, solve. The path (spectral or
hierarchical) is chosen for the mesh:</p>

```fortran
use elisha
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver

call comm%init(MPI_COMM_WORLD, stat, errmsg)                 ! ELISHA works on its own duplicate
call mesh%init(nijk, ngc, nblk0, dx0, emin, periodic, null, coords, stat, errmsg) ! this rank's blocks
call solver%setup(mesh, comm, stat, errmsg)                  ! one level: spectral; 2:1 AMR: hierarchical
call solver%solve_packed(mesh, f, factor, phi, stat, errmsg) ! L_h phi = factor (f - mean f), ghosts filled
```

<p class="elisha-lead">The <a href="./tutorial/">tutorial</a> goes from one periodic box to two embers orbiting on an
adaptive mesh and onto the GPU; the <a href="./reference/">reference</a> lists every argument, kind and status code.</p>

## Authors

</div>

<VPTeamMembers size="medium" :members="authors" />

<div class="elisha-home">

## License

<p class="elisha-lead">ELISHA is available under your choice of license.</p>

| License | SPDX | File |
|---|---|---|
| GNU General Public License v3 | `GPL-3.0-or-later` | [LICENSE.gpl3.md](https://github.com/szaghi/ELISHA/blob/main/LICENSE.gpl3.md) |
| BSD 2-Clause | `BSD-2-Clause` | [LICENSE.bsd-2.md](https://github.com/szaghi/ELISHA/blob/main/LICENSE.bsd-2.md) |
| BSD 3-Clause | `BSD-3-Clause` | [LICENSE.bsd-3.md](https://github.com/szaghi/ELISHA/blob/main/LICENSE.bsd-3.md) |
| MIT | `MIT` | [LICENSE.mit.md](https://github.com/szaghi/ELISHA/blob/main/LICENSE.mit.md) |

<p class="elisha-lead">© <a href="https://github.com/szaghi">Stefano Zaghi</a></p>

</div>
