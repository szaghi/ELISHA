# The device backends

ELISHA runs both solvers on NVIDIA GPUs: the spectral one with cuFFT, the hierarchical one with OpenACC. They are
ADAM's device twins (`poisson_fft_fnl_object`, `poisson_mg_fnl_object`, ADAM 669d0bdd) on ELISHA's host solvers, with
the device code unchanged: the same kernels, loop mappings, clauses, data directives, reductions and cuFFT plans
(`-Minfo=accel` reports the same 9 + 10 kernels, line for line). One M7 change is in the smoother's kernel: the colour
it tests is an integer instead of ADAM's real (audit C3), so it selects the same rows. A second M7 change (audit C5) is in
the cuFFT backend: it places the eigenvalue parts and the transpose's row maps on the device once, with the plans,
where ADAM copied them in at every solve. The arithmetic is unchanged.

```fortran
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
type(elisha_hierarchical_solver_oac_object) :: mg
call mg%setup(mesh, comm, stat, errmsg)                      ! the root on cuFFT by default
call mg%solve_packed(mesh, buf, factor, phi, stat, errmsg)
```

| Type | Extends | Where it computes |
|------|---------|-------------------|
| `elisha_spectral_solver_cufft_object` | `elisha_spectral_solver_object` | The slab transform on each rank's GPU (cuFFT), the transpose on device buffers (GPU-aware MPI) |
| `elisha_hierarchical_solver_oac_object` | `elisha_hierarchical_solver_object` | The FCG iteration and the V-cycle on the device: SpMV and red-black sweeps one row per thread, device dots and max-norms; the operators assembled on the host and placed once |

The right-hand side, the root (through the host to the cuFFT solver), the James step and the ghost fill's placement
stay on the host, as in ADAM.

## The root on the device and device-resident arrays (M7, audit C4)

**The V-cycle's root.** ADAM's device multigrid takes its root through the host at every V-cycle: the level residual
goes down, the root is gathered, placed and extended on the host, cuFFT transforms it, and the correction goes back
up. With the cuFFT root on its redundant path (at most 32^3 cells, the default), ELISHA keeps the whole root on the
device:

1. the division `r / v0` runs as a device kernel;
2. the source is gathered on device buffers, `MPI_ALLGATHERV` through `host_data` (GPU-aware), or copied on the device
   at one rank;
3. it is placed into the global array and extended (the plane reflections);
4. cuFFT runs with the same plans and the same call as before;
5. this rank's blocks are extracted.

Steps 2, 3 and 5 are copies and sign products, and the division is IEEE `div.rn.f64` on the device, so the potential
stays bitwise ADAM FNL's. The root's layout and its global array are placed on the device once, at the first V-cycle
after a setup. A root above the cap, on the slab path, and any non-cuFFT root keep the host trip.

**Device-resident arrays.** `solve_packed(..., on_device=.true.)` takes `buf` and `phi` already present on the device
(OpenACC, `phi` contiguous). It works on the device backends and through the facade with the device molds:

```fortran
!$acc enter data copyin(buf) create(phi)
call solver%solve_packed(mesh, buf, factor, phi, stat, errmsg, on_device=.true.)
! phi is current on the device (and on the host)
```

The right-hand side (its sums) and the ghost fill stay on the host, in the host's order, which keeps the results
bitwise. The backend therefore copies the source to a host copy of its own (the client's `buf` is not touched) and
brings `phi` to the host and back around the solve. The option saves the client its host mirror and its own copies,
not the per-solve transfers: the gain is the root's, at every V-cycle. A host solver given `on_device=.true.` returns
`ELISHA_ERR_ARGUMENT`.

`src/tests/device/elisha_test_device_api_mpi` solves each case three ways: ADAM FNL, ELISHA with host arrays, and
ELISHA with `on_device`. The cases are hierarchical (periodic, ghost-Dirichlet and D/N roots, isolated), spectral, and
the facade with the device molds. It compares the bit patterns of the potential, the energy, the iterations and the
residual.

## Build

| Mode | Content |
|------|---------|
| `static-nvf-oac` | The library with the device backends (target `src/device/elisha_device.F90`): `-acc -gpu=$GPU_CC -cudalib=cufft -D_CUFFT` |
| `tests-nvf-oac` | Every test, plus the device tests in `src/tests/device/`, into `exe-oac/` |
| `bench-nvf-oac` | The device benchmark against ADAM's device solvers, into `exe-bench-oac/` |

The device sources live in `src/device/`, a sibling of `src/lib/`, so that the host modes and fpm never compile
them. The compute capability comes from `--varset`: `cc80`, `cc89`, `cc90`, or `iac` (cc80 with CUDA 12.9, for
drivers that cannot run CUDA 13). One GPU per node-local rank. More than one rank needs a GPU-aware MPI, because the
halos and the transpose are exchanged on device buffers.

## Verification

Every device test ran on an A30 (driver 575.51, nvfortran 26.9, CUDA 12.9) through `scripts/medooza/elisha-tests.slurm`.

| Test | Result |
|------|--------|
| `elisha_test_device_parity_mpi`: ELISHA's device solvers against ADAM's device solvers on the same source (phi with ghosts, energy; multigrid iterations, residuals, `A x`, symmetry defect, cold and warm solves); 11 spectral and 12 hierarchical cases | Bitwise on all 23 at np 1, 2, 3 (ranks sharing the GPU) |
| The same, device against host | FFT 6.5e-16, multigrid `A x` bitwise, potential 1.8e-15, energy 1.3e-15 (relative) |
| `elisha_test_adam_device_oracle_mpi`: ADAM's own device against its host | FFT 5.2e-16, `A x` bitwise, potential 5.9e-16 |

The device and the host differ in the last bits only: the device sums its dots in another order and contracts
products into FMAs.

## Performance

`exe-bench-oac/elisha_bench_device_mpi [reps [AE|EA [cases]]]` and `scripts/medooza/elisha-bench.slurm`. These are
one A30, 15 repetitions, both orders, every row bitwise.

| Case | Cells | Solve, ELISHA (median) | E/A (median), AE / EA | Setup ADAM / ELISHA |
|------|-------|------------------------|-----------------------|---------------------|
| FFT periodic | 128^3 | 0.029 s | 0.99 / 0.99 | 0.04 / 0.04 s |
| FFT isolated | 128^3 | 0.10 s | 0.96 / 0.93 | 0.7 / 0.7 s |
| FFT periodic | 64^3 | 0.0057 s | 0.98 / 0.98 | 0.02 / 0.006 s |
| Multigrid AMR periodic | 491 520 | 0.065 s | 1.00 / 0.98 | 6.5 / 2.5 s |
| Multigrid AMR isolated | 491 520 | 0.13 s | 1.03 / 0.94 | 6.5 / 2.6 s |
| Multigrid AMR periodic | 3 932 160 | 0.34 s | 0.99 / 0.99 | 22.7 / 21.7 s |
| Multigrid AMR isolated | 3 932 160 | 0.60 s | 0.99 / 0.98 | 22.2 / 22.2 s |

The 3.9 M-cell periodic solve (0.34 s) is ADAM's documented FNL time on one A30 (0.33 s).
