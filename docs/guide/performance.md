# Performance

ELISHA is ADAM's FLAIL, bitwise first, faster second. Milestones M3 to M5 ported the solvers and held their time to
ADAM's. M7 (2026-11-20) then made five performance changes, one at a time, each measured against ADAM in the same
binary. This page collects those measurements. The solver pages keep their own tables of the port:
[spectral](./spectral-solver#performance), [hierarchical](./hierarchical-solver#performance),
[device](./device#performance).

## How it is measured

- **The ratio E/A.** Every benchmark runs ADAM's solver and ELISHA's in one program, from the same source, alternating
  the two solves. E/A is the median of ELISHA's solve times over the median of ADAM's: below 1, ELISHA is faster.
  Every timed pair is first checked bitwise, and the check is printed with each row.
- **Both orders.** On these machines the first solve of a pair can be up to 10% faster, so every case runs with ADAM
  first (AE) and with ELISHA first (EA). Both are reported.
- **Before and after.** For a change measured on the host, the "before" binary is built from a copy of the tree taken
  just before the change. The two run interleaved in one session on one machine.
- **Machines.**

  | Label | Machine | Used for |
  |-------|---------|----------|
  | WSL | Intel i9-13900KF, WSL 2, 8 cores (16 threads) given to Linux; gfortran 16.2 + Open MPI 5.0.11, nvfortran 26.9 + HPC-X | host benches |
  | A30 | medooza (CNR-IAC), one NVIDIA A30, driver 575.51, nvfortran 26.9, CUDA 12.9 | device benches, jobs `scripts/medooza/elisha-bench.slurm` |

- **Benches.**
  - Host: `exe-bench/elisha_bench_hierarchical_mpi reps AE|EA cases [stencil]` (modes `bench-hierarchical-gnu` and
    `-nvf`). Case (a) is 2:1 AMR, periodic, 491 520 cells; case (d) is uniform 64^3 with D/N faces and a root capped
    at 32^3 cells.
  - Device: `exe-bench-oac/elisha_bench_device_mpi reps AE|EA cases [stencil]`. The cases are 128^3 periodic (a)
    and isolated (b), 64^3 periodic (c), and AMR with 491 520 cells, periodic (d) and isolated (e), or with 3 932 160
    cells, periodic (f) and isolated (g).
- **Noise.** On WSL, ADAM's own median, whose code did not change, moves by up to 10% between runs (up to 50% at 16
  ranks on hardware threads). A difference of a few percent between "before" and "after" is below what these runs
  can resolve.
- **Raw outputs.** The host runs are kept in the working tree under `notes/bench-m7/` (one file per compiler, ranks,
  order and tree; local, not tracked). The device runs are the medooza jobs named below.

## M7 at a glance

| Step | Change (audit) | Bitwise ADAM | Measured | Verdict |
|------|----------------|--------------|----------|---------|
| 05 | Integer red-black colour (C3) | yes | WSL: mean E/A 0.95 to 0.93 (gfortran), 0.97 to 0.93 (nvfortran); A30: no change | kept: exact, small, reads 4 B instead of 8 |
| 06 | Neighbour halo exchange (B1) | yes | WSL up to 16 ranks: no change; A30 np 1, 2: no change | kept: the cost per call now grows with neighbours, not with P; the gain needs large P |
| 07 | Resident cuFFT constants (C5) | yes | A30 np 1: FFT isolated 128^3 0.86 / 0.94, others within noise | kept |
| 08 | Device V-cycle root, `on_device` API (C4) | yes | A30 np 1: AMR 491 520 cells periodic 0.84 / 0.81, isolated 0.65 / 0.68; 3.9 M cells 0.90 to 0.96 | kept: the largest gain of M7 |
| 09 | Stencil level operators, opt-in (C2) | A30, gfortran, ifx: yes; nvfortran host: round-off | WSL: no change; A30 np 1: multigrid E/A 10 to 15% lower than CSR (job 1369037) | opt-in; the default (CSR) is the owner's choice |

The ratios are AE / EA unless a range is given.

## Step 05: integer colour (audit C3)

The red-black colour of a level cell is stored as an integer, 0 or 1, on the host and on the device. ADAM stored it
as a real and compared `nint` of it. The colour only selects rows, so every result is bitwise unchanged.

| Where | Before | After |
|-------|--------|-------|
| WSL, mean E/A over cases (a) and (d), np 1, 2, 4, both orders, gfortran | 0.951 | 0.931 |
| WSL, the same, nvfortran | 0.973 | 0.931 |
| A30 np 1, AMR periodic (d), AE / EA (job 1368962) | 1.00 / 0.98 (M5) | 1.008 / 1.037 |
| A30 np 1, AMR isolated (e), AE / EA (job 1368962) | 1.03 / 0.94 (M5) | 1.000 / 0.948 |

On WSL, 19 of 24 rows were faster after the change, by 2 to 7%: consistent in direction, at the edge of the noise.
On the A30 there was no measurable change: a device kernel reads the colour once per row, next to 12 CSR entries.

## Step 06: neighbour halo exchange (audit B1)

Every halo exchange of the hierarchical solver's `apply` and smoother, host and device, is an
`MPI_Neighbor_alltoallv` on a distributed-graph communicator. Each operator's halo plan builds that communicator once
at setup. ADAM called `MPI_ALLTOALLV` with P-sized count arrays at every call. The buffers and values are the same, so
the results are bitwise unchanged: np 1 to 8 on Open MPI and Intel MPI, np 1 to 4 on HPC-X.

| Where | Before | After |
|-------|--------|-------|
| WSL, mean E/A over (a) and (d), np 1 to 16, both orders, gfortran | 0.943 | 0.944 |
| WSL, the same, nvfortran (one disturbed np 16 pair rerun) | 0.956 | 0.943 |
| A30 np 1, (d) / (e), AE / EA (job 1368965) | | 0.989 / 1.019, 1.054 / 0.978 |
| A30 np 2, (d) / (e), AE / EA (job 1368965) | | 0.989 / 0.996, 0.990 / 0.996 |

There is no measurable change up to 16 ranks. At this size the P-sized count arrays cost nothing next to the
messages. The bench deals the AMR leaves round-robin, so the composite operator of case (a) talks to 13 to 15 of the
15 other ranks at np 16; its level operators talk to 6. The change matters at large P with a block distribution that
keeps neighbours local. That was not measured here.

## Step 07: resident cuFFT constants (audit C5)

The cuFFT backend places the eigenvalue parts (`lam1..3`) and the transpose's row maps (`yr`, `yown`) on the device
once, with the plans. ADAM copied them in at every solve. The arithmetic is unchanged.

| A30, job 1368967 | AE / EA |
|------------------|---------|
| np 1, FFT 128^3 periodic (a) | 1.004 / 0.921 |
| np 1, FFT 128^3 isolated (b) | 0.858 / 0.937 |
| np 1, FFT 64^3 periodic (c) | 0.959 / 0.984 |
| np 1, AMR periodic (d) | 0.982 / 0.962 |
| np 1, AMR isolated (e) | 0.970 / 1.053 |
| np 2, every case | 0.95 to 1.14, no consistent direction |

The isolated transform gains a few percent at np 1. Elsewhere the copies were too small to see.

## Step 08: the root on the device and the device-resident API (audit C4)

The hierarchical solver's V-cycle root used to go through the host at every V-cycle: the residual down, the root
gathered, placed and extended on the host, cuFFT, the correction back up. With the cuFFT root on its redundant path,
the whole root now runs on the device. The gather is GPU-aware; placement, extension and extraction are device
kernels; the division by the cell volume is IEEE `div.rn.f64`. The potential is bitwise ADAM FNL's (job 1368968: the
device parity, 23 cases, and `elisha_test_device_api_mpi`, 8 cases, at np 1 to 3). `solve_packed(..., on_device=.true.)`
takes `buf` and `phi` already on the device.

| A30 np 1, job 1368969 | AE / EA |
|-----------------------|---------|
| AMR periodic, 491 520 cells (d) | 0.835 / 0.806 |
| AMR isolated, 491 520 cells (e) | 0.649 / 0.680 |
| AMR periodic, 3 932 160 cells (f) | 0.959 / 0.952 |
| AMR isolated, 3 932 160 cells (g) | 0.902 / 0.939 |

The 3.9 M-cell periodic solve takes 0.33 s against ADAM's 0.35 s in the same job. The gain is largest where the root
is a large share of the V-cycle: the 491 520-cell cases, and the isolated ones, whose James step runs two solves.
`on_device` saves the client its own copies, not the per-solve transfers: the right-hand side and the ghost fill stay
on the host to keep their summation order.

## Step 09: stencil level operators (audit C2, opt-in)

`setup(..., level_operator=ELISHA_OPERATOR_STENCIL)` applies and smooths the block-interior rows of the V-cycle's level
operators as the 7-point stencil they are, without reading their CSR entries. That is 42% of the rows on blocks of
8^3 cells. The other rows keep CSR ([the hierarchical solver](./hierarchical-solver#stencil-level-operators)). The
default is CSR.

| | CSR | Stencil |
|-|-----|---------|
| WSL, mean E/A over (a) and (d), np 1, 2, 4, both orders, gfortran | 0.919 | 0.925 |
| WSL, the same, nvfortran | 0.980 | 0.939 |
| A30 np 1, AMR periodic, 491 520 cells (d), AE / EA (job 1369037) | 0.851 / 0.859 | 0.768 / 0.766 |
| A30 np 1, AMR isolated, 491 520 cells (e) | 0.657 / 0.654 | 0.631 / 0.634 |
| A30 np 1, AMR periodic, 3 932 160 cells (f) | 0.974 / 0.944 | 0.826 / 0.828 |
| A30 np 1, AMR isolated, 3 932 160 cells (g) | 0.909 / 0.917 | 0.811 / 0.816 |

Job 1369037 ran CSR and the stencil interleaved in one job, 15 repetitions, both orders.

| A30 np 1, job 1369037 | ADAM | ELISHA, CSR | ELISHA, stencil |
|-----------------------|------|-------------|-----------------|
| AMR periodic, 3.9 M cells (f), median, AE / EA | 0.343 / 0.342 s | 0.334 / 0.323 s | 0.284 / 0.289 s |
| AMR isolated, 3.9 M cells (g), median, AE / EA | 0.614 / 0.605 s | 0.558 / 0.555 s | 0.505 / 0.501 s |

- **Results.** Against CSR, bitwise with gfortran and ifx on the host and with nvfortran on the A30. On the host,
  nvfortran `-fast` differs by at most 8e-16 in the potential, with the same iterations.
- **Against ADAM FNL.** Every row of job 1369037, stencil included, is bitwise ADAM FNL's: the device stencil keeps
  the CSR entries' order.
- **On the A30**, the device is limited by memory bandwidth, and the stencil gains 10 to 15% on the multigrid solve.
  It is 3 to 4% on the isolated 491 520-cell case, where the James step's two solves and the root weigh more.
- **On the host** it shows no gain beyond the noise.
- **Verdict.** It stays opt-in. Making it the default is the owner's decision.

## Where the time goes now

| A30, np 1 | ELISHA solve | E/A |
|-----------|--------------|-----|
| FFT 128^3 periodic | 0.029 s (M5) | 1.00 / 0.92 (step 07) |
| FFT 128^3 isolated | 0.10 s (M5) | 0.86 / 0.94 (step 07) |
| AMR periodic, 491 520 cells | | 0.84 / 0.81 (step 08) |
| AMR isolated, 491 520 cells | | 0.65 / 0.68 (step 08) |
| AMR periodic, 3 932 160 cells | 0.33 s (step 08; ADAM 0.35 s) | 0.96 / 0.95 (step 08) |
| AMR isolated, 3 932 160 cells | | 0.90 / 0.94 (step 08) |
| AMR periodic, 3 932 160 cells, stencil (opt-in) | 0.28 / 0.29 s (step 09; ADAM 0.34 s) | 0.83 / 0.83 (step 09) |
| AMR isolated, 3 932 160 cells, stencil (opt-in) | 0.51 / 0.50 s (step 09; ADAM 0.61 s) | 0.81 / 0.82 (step 09) |
| AMR periodic, 491 520 cells, stencil (opt-in) | | 0.77 / 0.77 (step 09) |
| AMR isolated, 491 520 cells, stencil (opt-in) | | 0.63 / 0.63 (step 09) |

On the host, after M7, ELISHA's solve stays within the noise of ADAM's. The tables of the
[spectral](./spectral-solver#performance) and [hierarchical](./hierarchical-solver#performance) solvers hold. ELISHA's
setup is faster than ADAM's because of the key mixing of M4.
