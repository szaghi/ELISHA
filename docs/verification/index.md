# Verification

ELISHA's solvers are verified against analytic and discrete solutions, not against the code they came from. The
bitwise parity tests hold every solver to ADAM's FLAIL ([the spectral solver](/guide/spectral-solver#what-changed-from-adam-and-what-did-not),
[the hierarchical solver](/guide/hierarchical-solver#what-changed-from-adam-and-what-did-not)). The physics
verification, which ADAM ran with its FLAIL drivers, moved here with ADAM issue #136 (decision D2): ELISHA is the
source of truth for these solvers. The cases, sources, exact solutions and bounds are ADAM's. The drivers call ELISHA,
build their meshes with `elisha_mesh_object`, and report a refusal as the status code ELISHA returns (decision D3).

| Page | Case | Solver | Checked against |
|---|---|---|---|
| [SV-2: periodic](./fft-poisson) | two oblique modes on the box $[0,1]\times[0,2]\times[0,3]$ | spectral | the discrete solution (round-off), the continuum one (second order) |
| [SV-6: isolated](./isolated-solver) | the $n = 1$ polytrope and the uniform sphere on $[-1,1]^3$ | spectral, zero padding | the exact potentials ($-GM/r$ outside) |
| [SV-9, SV-12: AMR](./multigrid-amr) | SV-2's source, the central box refined; periodic and Dirichlet | hierarchical | the spectral solver (uniform), the exact potential, the operator's symmetry and consistency |
| [SV-13: the James step](./james-step) | SV-6's stars, the star's box refined | hierarchical, isolated | the spectral solver's isolated mode (uniform), the exact potentials, the symmetry of the solution map |
| [Face kinds and D(G)](./face-kinds) | per-axis eigenfunctions of every face kind; manufactured solutions | spectral, compact and wide D(G) | the discrete solutions (round-off), the continuum (D/D order), PRISM's own operators, a direct Green's sum |

The multigrid page also covers the hierarchical solver's face kinds (E1b), the levels inside the blocks (E5c) and
the facade against the solvers it chooses (E1c).

## The drivers

Three test programs, each also a CI test (`scripts/run_tests.sh` runs them at np 2 with the rest of the suite;
without flags they run every leg):

| Program | Legs | Wall time, np 2 (gfortran, release) |
|---|---|---|
| [`elisha_sv_fft_mpi`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_fft_mpi.F90) | `--periodic` (SV-2), `--isolated` (SV-6), the rank invariance, `--dump DIR` | 1.4 s |
| [`elisha_sv_mg_mpi`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_mg_mpi.F90) | `--sv9` (SV-9, SV-12), `--isolated` (SV-13), `--bc` (E1b), `--sub` (E5c), `--flail` (E1c), `--dump DIR` | 20 s |
| [`elisha_sv_bc_mpi`](https://github.com/szaghi/ELISHA/blob/main/src/tests/elisha_sv_bc_mpi.F90) | `--compact` (E1a), `--wide` (E2a), `--zero` (E2b), `--prism` (E3a), `--free` (E3c), `--order` | 0.6 s |

Each prints ADAM's per-level lines, then a summary table with the measured numbers and their bounds, then
`<program>: PASSED` (exit 0) or `FAILED` (exit 1). A refusal flag (`--no-periodic`, `--refuse-*`, `--mixed`) runs that
refusal only and exits 0 when ELISHA returns the expected status, printed by name. The test programs build with every
test mode (`fobis build --mode tests-gnu`, `tests-nvf`, `tests-intel`, the serial `tests-gnu-serial`) and with fpm.
In the device build (`tests-nvf-oac`, into `exe-oac/`) `--device` runs them on the cuFFT and OpenACC backends.

## The harness

[`scripts/verification/poisson.sh`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/poisson.sh) is
ADAM's `poisson.sh` on these drivers: each leg on every rank count, its summary rows and its refusals.

```bash
fobis build --mode tests-gnu
scripts/verification/poisson.sh                            # every leg at np 1, 2, 3 (about 100 s)
scripts/verification/poisson.sh --legs "sv2 slab" --np-list "2 4"
scripts/verification/poisson.sh --dump DIR                 # also dump the potentials (np 2)
scripts/verification/poisson.sh --device --dump DIR_DEV    # the device build, exe-oac/
scripts/verification/poisson.sh --compare DIR DIR_DEV     # host against device
```

| Leg | Program and flags | Refusals (by ELISHA status) |
|---|---|---|
| `sv2` | `elisha_sv_fft_mpi --periodic` | `ELISHA_ERR_BC_MISSING` |
| `sv6` | `elisha_sv_fft_mpi --isolated` | `_ISOLATED_PERIODIC`, `_ISOLATED_CELLS`, `_ISOLATED_DIMENSION` |
| `slab` | `elisha_sv_fft_mpi --dump` at every rank count, the dumps compared byte for byte | |
| `sv9` | `elisha_sv_mg_mpi --sv9` | `_BC_MISSING`, `_BC_PERIODICITY`, `_BC_GHOST_PAIR` |
| `sv13` | `elisha_sv_mg_mpi --isolated` | `_ISOLATED_BOUNDARY_LEAF` |
| `bc` | `elisha_sv_bc_mpi` | `_BC_PERIODIC_PAIR`, `_BC_GHOST_PAIR`, `_ISOLATED_WIDE`, `_BC_ZERO_GHOST_PAIR`, `_BC_FREE_GHOSTS` |
| `bcmg` | `elisha_sv_mg_mpi --bc` | |
| `flail` | `elisha_sv_mg_mpi --flail` | `_GHOST_VALUES`, `_ISOLATED_BC`, `_DG_AMR` |
| `sub` | `elisha_sv_mg_mpi --sub` | |

The variables `ELISHA_EXE_DIR`, `POISSON_EXE`, `POISSON_MG_EXE`, `POISSON_BC_EXE` choose the executables, and
`LAUNCH` the launcher (`mpirun --oversubscribe -np` by default, `srun --mpi=pmix --overlap -n` under Slurm).

**Host and device.** [`scripts/medooza/elisha-sv.slurm`](https://github.com/szaghi/ELISHA/blob/main/scripts/medooza/elisha-sv.slurm)
builds the drivers in `tests-nvf` and `tests-nvf-oac` on a medooza node, runs every leg at np 1, 2, 3 on the host and
with `--device` (one A30 shared by the ranks), and dumps both builds' potentials. `poisson.sh --compare` compares
them (`poisson_oracle.py compare --pattern`, numpy) on the node when its Python has numpy, on the workstation otherwise
(medooza's nodes have none). The spectral dumps (`phi-r*.bin`) are direct solves, compared at $10^{-13}$; the
hierarchical ones (`phi-mg-*.bin`) are FCG solutions stopped at a relative residual of $10^{-10}$: two builds sum their
dots in different orders and stop at different iterates within that tolerance, so they are compared at $10^{-8}$, the
bound of the device oracle test.

CPU = device, job 1369178 (medooza, A30): every host leg passes; every device leg passes except `sub` (a crash in the
device library, under repair). The 13 spectral dumps agree within $8.93\cdot10^{-16}$, the 18 hierarchical ones within
$7.85\cdot10^{-12}$.

## Figures and prototypes

[`scripts/verification/make_verification_figures.py`](https://github.com/szaghi/ELISHA/blob/main/scripts/verification/make_verification_figures.py)
draws the figures of these pages from a run of the legs (ADAM's figure functions, ported; matplotlib and numpy):

```bash
scripts/verification/poisson.sh --legs "sv2 sv6 sv9 sv13" --np-list 2 --work WORK --keep --dump DUMP
python3 scripts/verification/make_verification_figures.py --work WORK --dump DUMP   # docs/public/verification/
```

The figures carry no time stamp: a rerun on the same data writes the same bytes. The operator study behind the 2:1
coupling (`seam_operators.py`) and the prototype of the James step (`james_proto.py`) are in `scripts/verification/`,
the prototypes of the wide D(G) and of the levels inside the blocks in `scripts/prototypes/`
([the spectral solver](/guide/spectral-solver#problems-and-methods), [the hierarchical solver](/guide/hierarchical-solver#the-method)).
