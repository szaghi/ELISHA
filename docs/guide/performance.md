# Performance

ELISHA is ADAM's FLAIL, bitwise first, faster second. Milestones M3 to M5 ported the solvers and held their time to
ADAM's. M7 (2026-11-20) then made five performance changes, one at a time, each measured against ADAM in the same
binary. This page collects those measurements. The solver pages keep their own tables of the port:
[spectral](./spectral-solver#performance), [hierarchical](./hierarchical-solver#performance),
[device](./device#performance). The scaling ladders, which time ELISHA alone in absolute seconds from 1 to 64 CPU
ranks and 1 to 4 GPUs, are in [Scaling (EB)](#scaling-eb).

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
  | medooza CPU | one medooza node (gnode19): Intel Xeon Gold 6338 (Ice Lake, 2.0 GHz), 64 ranks one per core, 1 TB; nvfortran 26.9 + HPC-X | scaling ladders, jobs `scripts/medooza/elisha-ladder.slurm` |
  | 4 × A30 | one medooza node (gnode12), four NVIDIA A30, driver 545.23.08, nvfortran 26.9, CUDA 12.9, one GPU per rank | device scaling ladder |

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
| 09 | Stencil level operators (C2): OpenACC default, host option | A30, gfortran, ifx: yes; nvfortran host: round-off | WSL: no change; A30 np 1: multigrid E/A 10 to 15% lower than CSR (job 1369037) | default on OpenACC, option on the host (owner, 2026-11-20) |

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
on the host to keep their summation order. The scaling ladder measured what that staging costs: `on_device` is 9 to 24%
slower than host arrays on the multigrid and up to 51% on the FFT ([Scaling](#device-a30-job-1369163)).

## Step 09: stencil level operators (audit C2)

`setup(..., level_operator=ELISHA_OPERATOR_STENCIL)` applies and smooths the block-interior rows of the V-cycle's level
operators as the 7-point stencil they are, without reading their CSR entries. That is 42% of the rows on blocks of
8^3 cells. The other rows keep CSR ([the hierarchical solver](./hierarchical-solver#stencil-level-operators)). It is
the OpenACC backend's default; on the host the default is CSR.

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
- **Verdict.** The OpenACC backend's default (owner, 2026-11-20): faster there and still bitwise ADAM FNL's. On the
  host CSR stays the default; the stencil is an option.

## Where the time goes now

| A30, np 1 | ELISHA solve | E/A |
|-----------|--------------|-----|
| FFT 128^3 periodic | 0.029 s (M5) | 1.00 / 0.92 (step 07) |
| FFT 128^3 isolated | 0.10 s (M5) | 0.86 / 0.94 (step 07) |
| AMR periodic, 491 520 cells | | 0.84 / 0.81 (step 08) |
| AMR isolated, 491 520 cells | | 0.65 / 0.68 (step 08) |
| AMR periodic, 3 932 160 cells | 0.33 s (step 08; ADAM 0.35 s) | 0.96 / 0.95 (step 08) |
| AMR isolated, 3 932 160 cells | | 0.90 / 0.94 (step 08) |
| AMR periodic, 3 932 160 cells, stencil (OpenACC default) | 0.28 / 0.29 s (step 09; ADAM 0.34 s) | 0.83 / 0.83 (step 09) |
| AMR isolated, 3 932 160 cells, stencil (OpenACC default) | 0.51 / 0.50 s (step 09; ADAM 0.61 s) | 0.81 / 0.82 (step 09) |
| AMR periodic, 491 520 cells, stencil (OpenACC default) | | 0.77 / 0.77 (step 09) |
| AMR isolated, 491 520 cells, stencil (OpenACC default) | | 0.63 / 0.63 (step 09) |

On the host, after M7, ELISHA's solve stays within the noise of ADAM's. The tables of the
[spectral](./spectral-solver#performance) and [hierarchical](./hierarchical-solver#performance) solvers hold. ELISHA's
setup is faster than ADAM's because of the key mixing of M4.

## Scaling (EB)

The scaling ladders moved here from ADAM (ELISHA issue #1, EB; ADAM issue #136, decision D7). ADAM's FLAIL ladders
were the only multi-node scaling evidence for these solvers. They now run on ELISHA's own program, in absolute
seconds, with no ADAM oracle in the binary. FLAIL's published numbers are quoted next to them
[below](#against-flail-adam-s-numbers).

The multigrid rows below were measured on 2026-11-20 with ADAM's V-cycle, before the symmetric V-cycle became the
default ([the hierarchical solver](./hierarchical-solver#the-symmetric-v-cycle)). On the verification's AMR meshes the
symmetric V-cycle takes 19 to 35% fewer FCG iterations, at one extra sparse product per level and V-cycle over the
seam rows only; the ladders' meshes have not been re-timed with it.

### How the ladders are measured

- **Program.** `src/bench/ladder/elisha_bench_ladder_mpi.F90`, in the modes `bench-ladder-gnu` and `bench-ladder-nvf`
  (host, into `exe-bench/`) and `bench-ladder-nvf-oac` (the cuFFT and OpenACC backends, into `exe-bench-oac/`):

  ```sh
  mpirun -np N elisha_bench_ladder_mpi [--path fft|mg] [--bc periodic|isolated] [--level L] [--ni N] [--reps R]
    [--root-cells C] [--ngc G] [--deal morton|rr] [--cold] [--operator csr|stencil] [--on-device] [--tag LABEL]
  ```

  It prints one `BENCH key=value ...` line per run. `scripts/medooza/elisha-ladder.slurm` builds the program inside
  the job, runs one ladder set (`ELISHA_LADDER=strong`, `weak` or `fnl`) and writes `bench.txt`, `summary.txt` and
  `env.txt` to the run directory.
- **Problems.** These are the problems of ADAM's `bench_poisson`. The domain is the cube `[-1,1]^3` (an octree whose
  root block has `ni`^3 cells) with the source `exp(-|x - (s,0,0)|^2 / 0.05)` and 3 ghost layers.
  - **FFT** is the spectral solver on the uniform mesh at `--level`. At level 4 with `ni` 8 that is 128^3 =
    2 097 152 cells.
  - **MG** is the hierarchical solver on the same mesh with the blocks inside `[-1/2,1/2]^3` refined once. At
    level 4 that is 7 680 blocks and 3 932 160 cells.
  - **Boundaries.** Periodic on every axis, or isolated: the lattice Green's function on the spectral path, the
    James step on the hierarchical one.
  - **Roots.** The hierarchical root is the backend's default: 128^3 cells periodic and 64^3 isolated on the host,
    32^3 on cuFFT.
- **Distribution.** The leaves are cut into contiguous chunks along their Morton order, one chunk per rank.
- **Protocol** (ADAM's). The setup is timed once. One untimed warm-up solve follows. Then come 5 timed solves, with
  the source moved by 0.01 along x each time. The multigrid warm-starts from its last solution, as it would in a time
  step. Each solve is bracketed by a barrier, `MPI_Wtime` and, on the device, `!$acc wait`. A solve's time is the
  maximum over the ranks.
- **Reported quantities.**
  - The tables give the minimum ("min") and the median over the 5 solves.
  - Speed-up is t(1)/t(np) on the minima. Strong efficiency is speed-up / np; weak efficiency is t(1)/t(np).
  - The phases (gather or redistribution, solve, scatter) are rank 0's spectral timers per solve. On the FFT path
    they cover the whole solve. On the MG path they cover the V-cycle root, summed over the V-cycles of one solve,
    and `james` is the James step's ghost-to-ghost solve.
- **Bitwise check.** `phi_xor` hashes the warm-up potential bit for bit. Each cell is keyed by its global index, so
  the hash depends neither on the ranks nor on the distribution. Equal hashes across np mean bitwise equal
  potentials.
- **Jobs** (2026-11-20, nvfortran 26.9, release profile):
  - 1369135: strong, CPU, np 1 to 64;
  - 1369145: weak, CPU, np 1, 8, 64;
  - 1369163: FNL, np 1, 2, 4, with host arrays and with `--on-device`.

  Every run passed. The CPU ladders ran on a whole node (`--exclusive`), so all ranks share one node in every row.

### Strong scaling, CPU (job 1369135)

FFT, 2 097 152 uniform cells. The phases are in milliseconds per solve.

| Problem | np | min [s] | median [s] | Speed-up | Efficiency | Gather | Solve | Scatter |
|---------|----|---------|------------|----------|------------|--------|-------|---------|
| periodic | 1 | 0.124 | 0.125 | 1.00 | 1.00 | 10.1 | 76.4 | 26.1 |
| periodic | 2 | 0.0675 | 0.0689 | 1.83 | 0.92 | 5.2 | 43.4 | 14.1 |
| periodic | 4 | 0.0437 | 0.0486 | 2.83 | 0.71 | 8.0 | 23.5 | 13.4 |
| periodic | 8 | 0.0275 | 0.0332 | 4.50 | 0.56 | 4.9 | 13.1 | 11.2 |
| periodic | 16 | 0.0179 | 0.0310 | 6.90 | 0.43 | 1.6 | 14.7 | 7.1 |
| periodic | 32 | 0.0141 | 0.0151 | 8.80 | 0.27 | 1.2 | 9.6 | 5.7 |
| periodic | 64 | 0.0102 | 0.0104 | 12.1 | 0.19 | 1.0 | 2.9 | 5.7 |
| isolated | 1 | 0.823 | 0.823 | 1.00 | 1.00 | 32.7 | 752.6 | 26.0 |
| isolated | 2 | 0.503 | 0.504 | 1.64 | 0.82 | 25.0 | 430.3 | 42.4 |
| isolated | 4 | 0.277 | 0.278 | 2.97 | 0.74 | 12.7 | 236.1 | 27.1 |
| isolated | 8 | 0.144 | 0.147 | 5.70 | 0.71 | 8.2 | 123.6 | 13.2 |
| isolated | 16 | 0.0872 | 0.0882 | 9.44 | 0.59 | 5.2 | 69.5 | 9.7 |
| isolated | 32 | 0.0607 | 0.0653 | 13.6 | 0.42 | 2.8 | 53.4 | 7.2 |
| isolated | 64 | 0.0459 | 0.0470 | 17.9 | 0.28 | 2.6 | 35.9 | 9.1 |

Multigrid, 3 932 160 AMR cells. Periodic takes 17.8 FCG iterations per solve. Isolated takes 19.0 + 16.0: the
zero-ghost solve, then the final solve of the James step. The counts are the same at every np. The root phases and
`james` are in seconds per solve.

| Problem | np | min [s] | median [s] | Speed-up | Efficiency | Root gather | Root solve | Root scatter | James |
|---------|----|---------|------------|----------|------------|-------------|------------|--------------|-------|
| periodic | 1 | 8.27 | 8.74 | 1.00 | 1.00 | 0.173 | 1.354 | 0.080 | |
| periodic | 2 | 4.16 | 4.39 | 1.99 | 0.99 | 0.091 | 0.711 | 0.039 | |
| periodic | 4 | 2.23 | 2.37 | 3.72 | 0.93 | 0.057 | 0.382 | 0.032 | |
| periodic | 8 | 1.31 | 1.38 | 6.31 | 0.79 | 0.035 | 0.201 | 0.022 | |
| periodic | 16 | 1.05 | 1.12 | 7.87 | 0.49 | 0.085 | 0.107 | 0.027 | |
| periodic | 32 | 0.776 | 0.821 | 10.7 | 0.33 | 0.271 | 0.067 | 0.021 | |
| periodic | 64 | 0.409 | 0.437 | 20.2 | 0.32 | 0.141 | 0.053 | 0.019 | |
| isolated | 1 | 25.2 | 25.2 | 1.00 | 1.00 | 0.216 | 2.749 | 0.034 | 0.677 |
| isolated | 2 | 12.9 | 12.9 | 1.95 | 0.98 | 0.144 | 1.441 | 0.038 | 0.385 |
| isolated | 4 | 6.89 | 6.90 | 3.66 | 0.91 | 0.073 | 0.776 | 0.037 | 0.211 |
| isolated | 8 | 4.13 | 4.15 | 6.10 | 0.76 | 0.045 | 0.415 | 0.023 | 0.109 |
| isolated | 16 | 3.41 | 3.42 | 7.38 | 0.46 | 0.024 | 0.231 | 0.018 | 0.060 |
| isolated | 32 | 2.55 | 2.56 | 9.89 | 0.31 | 0.018 | 0.145 | 0.016 | 0.040 |
| isolated | 64 | 1.39 | 1.40 | 18.2 | 0.28 | 0.029 | 0.122 | 0.017 | 0.026 |

The multigrid setup (host assembly) takes 14.6 s periodic and 18.3 s isolated at np 1, and 0.41 s and 0.58 s at
np 64.

### Weak scaling, CPU (job 1369145)

The cells per rank are fixed, with (np, level) = (1, 3), (8, 4), (64, 5) as in ADAM's ladder:

- `ni` 8 gives 262 144 uniform cells (FFT) or 491 520 AMR cells (MG) per rank;
- `ni` 16 gives 2 097 152 uniform cells per rank, FFT only, as in ADAM's ladder.

The phases are rank 0's, in seconds per solve, at the largest np.

| Problem | ni | np 1 min [s] | np 8 min [s] | np 64 min [s] | Efficiency np 8 / np 64 | Cells at np 64 | Phases at np 64: gather / solve / scatter |
|---------|----|--------------|--------------|---------------|-------------------------|----------------|-------------------------------------------|
| FFT periodic | 8 | 0.0109 | 0.0267 | 0.0836 | 0.41 / 0.13 | 16 777 216 | 0.009 / 0.030 / 0.050 |
| FFT isolated | 8 | 0.100 | 0.147 | 0.363 | 0.68 / 0.28 | 16 777 216 | 0.027 / 0.267 / 0.063 |
| MG periodic | 8 | 0.878 | 1.30 | 5.59 | 0.67 / 0.16 | 31 457 280 | root 0.045 / 0.054 / 0.020 |
| MG isolated | 8 | 4.04 | 4.12 | 15.2 | 0.98 / 0.27 | 31 457 280 | root 0.064 / 0.158 / 0.026, James 0.241 |
| FFT periodic | 16 | 0.109 | 0.163 | 0.596 | 0.67 / 0.18 | 134 217 728 | 0.073 / 0.250 / 0.184 |
| FFT isolated | 16 | 0.805 | 1.25 | 3.53 | 0.64 / 0.23 | 134 217 728 | 0.190 / 2.965 / 0.343 |

The medians are within 20% of the minima. MG periodic takes 16.0, 17.8 and 18.0 iterations per solve at np 1, 8
and 64. MG isolated takes 16 + 14, 19 + 16 and 25 + 20 (zero-ghost + final solve).

### Device, A30 (job 1369163)

The strong problems at level 4 on np 1, 2, 4, one A30 per rank. Each cell is the minimum / median in seconds. "Host
arrays" passes `buf` and `phi` on the host (ADAM's FNL protocol). `--on-device` passes them present on the device
(`solve_packed(..., on_device=.true.)`).

| Problem | np | Host arrays | `--on-device` | `on_device` cost |
|---------|----|-------------|---------------|------------------|
| FFT periodic | 1 | 0.0563 / 0.0567 | 0.0836 / 0.0859 | +48% |
| FFT periodic | 2 | 0.0300 / 0.0302 | 0.0453 / 0.0481 | +51% |
| FFT periodic | 4 | 0.0260 / 0.0339 | 0.0347 / 0.0422 | +33% |
| FFT isolated | 1 | 0.114 / 0.115 | 0.140 / 0.143 | +23% |
| FFT isolated | 2 | 0.0891 / 0.0910 | 0.105 / 0.109 | +18% |
| FFT isolated | 4 | 0.110 / 0.119 | 0.112 / 0.124 | +2% |
| MG periodic | 1 | 0.258 / 0.258 | 0.320 / 0.335 | +24% |
| MG periodic | 2 | 0.152 / 0.152 | 0.177 / 0.177 | +16% |
| MG periodic | 4 | 0.103 / 0.107 | 0.116 / 0.123 | +13% |
| MG isolated | 1 | 0.649 / 0.649 | 0.707 / 0.729 | +9% |
| MG isolated | 2 | 0.391 / 0.392 | 0.438 / 0.438 | +12% |
| MG isolated | 4 | 0.342 / 0.343 | 0.357 / 0.358 | +5% |

The multigrid runs 17.0 FCG iterations per solve periodic and 22.0 + 19.0 isolated, with the 32^3 cuFFT root on its
redundant path. Its setup is host assembly: 20.2 s at np 1, 9.7 s at np 2 and 5.0 s at np 4. One setup costs as
much as about 80 solves at np 1.

These rows are not comparable with the device bench rows above, which time cold solves with 2 ghost layers on
another node. The ladder uses ADAM's ladder protocol (warm-started, the source moving), 3 ghost layers, and node
gnode12 with driver 545.

### Against FLAIL (ADAM's numbers)

The ADAM column is FLAIL's table from ADAM's `docs/flail.md` ("Performance", ADAM 669d0bdd), measured with
`bench_poisson` and `bench.sh` on medooza. **These are ADAM's numbers, from another date and build, not measured
here.** Its protocol is the one above: the same problems, 5 warm-started solves, the minimum over the repetitions,
the maximum over the ranks. ADAM's table does not record the nodes.

| Solver | Grid | Backend | Problem | np | ADAM (FLAIL) | ELISHA (min) |
|--------|------|---------|---------|----|--------------|--------------|
| FFT | 2.1 M uniform cells | CPU | periodic | 1 | 0.115 s | 0.124 s |
| FFT | 2.1 M uniform cells | CPU | periodic | 16 | 0.026 s | 0.0179 s |
| FFT | 2.1 M uniform cells | CPU | isolated | 1 | 0.729 s | 0.823 s |
| FFT | 2.1 M uniform cells | CPU | isolated | 16 | 0.108 s | 0.0872 s |
| FFT | 2.1 M uniform cells | FNL | periodic | 1 | 0.067 s | 0.0563 s |
| FFT | 2.1 M uniform cells | FNL | periodic | 4 | 0.030 s | 0.0260 s |
| FFT | 2.1 M uniform cells | FNL | isolated | 1 | 0.138 s | 0.114 s |
| FFT | 2.1 M uniform cells | FNL | isolated | 4 | 0.112 s | 0.110 s |
| Multigrid | 3.9 M AMR cells | CPU, 128^3 root | periodic | 1 | 8.6 s | 8.27 s |
| Multigrid | 3.9 M AMR cells | CPU, 128^3 root | periodic | 4 | 2.7 s | 2.23 s |
| Multigrid | 3.9 M AMR cells | CPU, 128^3 root | periodic | 16 | 1.6 s | 1.05 s |
| Multigrid | 3.9 M AMR cells | CPU, 64^3 root | isolated | 1 | 26.8 s | 25.2 s |
| Multigrid | 3.9 M AMR cells | CPU, 64^3 root | isolated | 4 | 8.5 s | 6.89 s |
| Multigrid | 3.9 M AMR cells | CPU, 64^3 root | isolated | 16 | 5.2 s | 3.41 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | periodic | 1 | 0.33 s | 0.258 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | periodic | 2 | 0.19 s | 0.152 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | periodic | 4 | 0.12 s | 0.103 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | isolated | 1 | 0.85 s | 0.649 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | isolated | 2 | 0.52 s | 0.391 s |
| Multigrid | 3.9 M AMR cells | FNL, 32^3 cuFFT root | isolated | 4 | 0.43 s | 0.342 s |
| FFT, weak | 16.7 M uniform cells | CPU | periodic | 8 | 0.24 s | 0.163 s |
| FFT, weak | 16.7 M uniform cells | CPU | isolated | 8 | 1.5 s | 1.25 s |

### Readings

- **Bitwise across np.**
  - On the slab path, the FFT potential is bitwise equal from 1 to 64 CPU ranks, periodic and isolated. It is also
    bitwise equal from 1 to 4 GPUs, with host arrays and with `on_device`.
  - The multigrid potential is not bitwise across np. Its FCG dot products and residual norms reduce over the ranks,
    and the summation order changes with the partition. The iteration counts are nevertheless the same at every np
    in every ladder.
- **FFT, strong.** The transform scales; the block-to-slab redistribution does not.
  - Periodic: efficiency 0.71 at 4 ranks, 0.43 at 16, 0.19 at 64. At np 1 the redistribution (gather + scatter) is
    29% of the solve; at np 64 it is 66% (6.7 of 10.2 ms).
  - Isolated holds better (0.59 at 16, 0.28 at 64) because its transform (264^3 cells, 8.6 times the grid) stays
    78% of the solve at np 64. At 64 ranks its 264 planes no longer divide evenly (4 or 5 per rank).
- **Multigrid, strong.** Efficiency is 0.91 to 0.99 up to 4 ranks, 0.76 to 0.79 at 8, 0.46 to 0.49 at 16, and about
  0.3 at 32 and 64.
  - Periodic: the root is the whole 128^3 base level, transformed at every V-cycle. Its share of the solve grows
    from 19% at np 1 to 52% at np 64. Above 16 ranks its gather alone takes about a third of the solve.
  - Isolated: the root and the James step take at most 16% of the solve at any np (9 to 10% at 16 and 32 ranks).
    The drop past 8 ranks is in the levels (smoothing, halo exchanges, FCG reductions), which the ladder does not
    time apart.
  - Up to 16 ranks both problems are faster than FLAIL's: 1.05 s against 1.6 s periodic and 3.41 s against 5.2 s
    isolated at np 16.
- **Weak.**
  - Efficiency at np 64 is 0.13 to 0.28 on every problem. MG periodic at np 64 takes 5.6 s, of which the root
    phases are 0.12 s, so the time is in the levels.
  - Every weak row ran on one node, so np 64 fills all 64 cores. These solvers are memory-bound, so this ladder
    mixes communication with the sharing of one node's memory bandwidth. It does not separate the two. A run with 8
    ranks per node on 8 nodes would.
  - At np 8, the weak problems are the ones FLAIL quoted: 16.7 M cells take 0.163 s periodic and 1.25 s isolated,
    against FLAIL's 0.24 s and 1.5 s.
- **FFT at np 1 on the CPU** is 8% (periodic) and 13% (isolated) slower than FLAIL's table. This comparison is
  across dates and builds. In the paired benches of the same binary, ELISHA's spectral solve stays within the noise
  of ADAM's ([spectral solver](./spectral-solver#performance)).
- **FNL.**
  - The multigrid is 1.2 to 1.3 times faster than FLAIL's FNL at every np: 0.103 s against 0.12 s periodic and
    0.342 s against 0.43 s isolated at np 4. That fits the M7 device root and stencil operators (E/A 0.81 to 0.83 in
    step 09).
  - From 1 to 4 GPUs it gains 2.5× (periodic) and 1.9× (isolated); FLAIL gained 2.75× and 2.0× from slower
    starting points.
  - The isolated FFT does not scale past 2 GPUs: 0.089 s at np 2, 0.110 s at np 4. Its solve phase, which includes
    the GPU-to-GPU slab transposes, rises from 24 to 67 ms. FLAIL measured the same at np 4 (0.112 s).
- **`on_device` is slower than host arrays**, by 9 to 24% for the multigrid and 18 to 51% for the FFT at np 1
  and 2. Concretely: 0.320 against 0.258 s for MG periodic at np 1, and 0.0836 against 0.0563 s for FFT periodic.
  - The cause: the right-hand side and the ghost fill run on the host, in the host's order, to stay bitwise. So
    `on_device` copies the source from the device to a host copy and moves `phi` (every ghost layer) to the host and
    back, around a solve that is otherwise the host-array path.
  - For the 3.9 M-cell multigrid at np 1 that is about 370 MB per solve (`phi` with 3 ghost layers is 169 MB),
    which matches the 0.062 s measured at about 6 GB/s. The FFT case matches too: 197 MB and 0.027 s, about 7 GB/s.
  - The cost shrinks with np because the data per rank shrinks.
  - **Today, pass host arrays.** `on_device` pays off only once the right-hand side and the ghost fill move to the
    device.
