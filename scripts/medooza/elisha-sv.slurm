#!/bin/bash
# ELISHA on medooza (CNR-IAC): the solvers' verification legs (scripts/verification/poisson.sh, ADAM issue #136, D2) on
# the host build (tests-nvf, exe/) and on the device build (tests-nvf-oac, exe-oac/, the drivers run with --device:
# cuFFT and OpenACC on one A30 shared by the ranks), every leg at np 1, 2, 3; each build's potentials dumped
# (poisson.sh --dump) for the host = device comparison.
#
# The source tree is built where it lies (ELISHA_SRC, a sandbox under /home/szaghi; only the three drivers, by target)
# and everything runs from the run directory on /ifs (ELISHA_RUN, default /ifs/hpc/home/szaghi/elisha-tests/sv-<job>):
#   build-<mode>.log, legs-host.log, legs-device.log, work-host/ and work-device/ (every driver log), dump-host/ and
#   dump-device/ (the potentials), compare.txt, summary.txt.
# The host = device comparison, scripts/verification/poisson_oracle.py compare dump-host dump-device (numpy), runs on
# the node when ELISHA_PY (default python3) imports numpy; otherwise summary.txt says so, and the comparison is made on
# the workstation after copying the two dump directories back (rsync -a medooza:$RUN/dump-{host,device} .).
# Submit from the login node (only sbatch runs there), e.g.
#
#   sbatch --export=ALL,ELISHA_SRC=/home/szaghi/elisha-ev scripts/medooza/elisha-sv.slurm
#   sbatch --export=ALL,ELISHA_SRC=...,ELISHA_SV_LEGS='sv2 sv6 slab',ELISHA_SV_SIDES='device' ...
#
# The dev-gn nodes (driver 575) run CUDA 12.9: varset iac. medooza has no FFTW: the user-space FFTW 3.3.10 of the ADAM
# tests is linked (LIBRARY_PATH, LD_LIBRARY_PATH). HPC-X runs need UCC off on the older prod-gn drivers (harmless here).
#SBATCH -A iacroma
#SBATCH -p dev-gn
#SBATCH -J elisha-sv
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:55:00
#SBATCH -o /ifs/hpc/home/szaghi/elisha-tests/elisha-sv-%j.out
#SBATCH -e /ifs/hpc/home/szaghi/elisha-tests/elisha-sv-%j.out
set -uo pipefail

SRC=${ELISHA_SRC:?set ELISHA_SRC to the sandbox}
LEGS=${ELISHA_SV_LEGS:-all}
SIDES=${ELISHA_SV_SIDES:-host device}
NPS=${ELISHA_SV_NP:-1 2 3}
PY=${ELISHA_PY:-python3}
RUN=${ELISHA_RUN:-/ifs/hpc/home/szaghi/elisha-tests/sv-$SLURM_JOB_ID}
FFTW=/ifs/hpc/home/szaghi/adam-tests/opt/fftw-3.3.10/lib
DRIVERS="src/tests/elisha_sv_fft_mpi.F90 src/tests/elisha_sv_mg_mpi.F90 src/tests/elisha_sv_bc_mpi.F90"

source /ifs/hpc/home/szaghi/nvidia/nvhpc.sh || exit 1
export PATH=$HOME/.local/bin:$PATH
export LIBRARY_PATH=$FFTW:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$FFTW:${LD_LIBRARY_PATH:-}
export OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_coll_hcoll_enable=0
export LAUNCH="srun --mpi=pmix --overlap -n" # poisson.sh's launcher: the job's own allocation
mkdir -p "$RUN"
{
   echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"
   echo "source $SRC, legs $LEGS, sides $SIDES, np $NPS"
   nvfortran --version | sed -n 2p
   nvidia-smi --query-gpu=name,driver_version,memory.free,memory.total --format=csv,noheader
} | tee "$RUN/env.txt"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free:-0}" -lt 20000 ]; then echo "GPU busy on $(hostname): ${free} MiB free" | tee "$RUN/summary.txt"; exit 4; fi

status=0
for side in $SIDES; do
   case "$side" in
      host)   mode=tests-nvf     ; exe_dir=exe     ; dev="" ;;
      device) mode=tests-nvf-oac ; exe_dir=exe-oac ; dev="--device" ;;
      *) echo "unknown side $side (host, device)" | tee -a "$RUN/summary.txt"; status=1; continue ;;
   esac
   cd "$SRC" || exit 1
   built=1
   for d in $DRIVERS; do
      if ! fobis build --mode "$mode" --varset iac --target "$d" >> "$RUN/build-$mode.log" 2>&1; then built=0; fi
   done
   if [ $built -eq 0 ]; then
      echo "BUILD FAILED $mode: see $RUN/build-$mode.log" | tee -a "$RUN/summary.txt"; status=1; continue
   fi
   # shellcheck disable=SC2086
   if ELISHA_EXE_DIR="$SRC/$exe_dir" bash "$SRC/scripts/verification/poisson.sh" --np-list "$NPS" --legs "$LEGS" $dev \
         --dump "$RUN/dump-$side" --work "$RUN/work-$side" --keep > "$RUN/legs-$side.log" 2>&1; then
      echo "PASS  $side legs ($LEGS), np $NPS" | tee -a "$RUN/summary.txt"
   else
      echo "FAIL  $side legs ($LEGS), np $NPS: see $RUN/legs-$side.log" | tee -a "$RUN/summary.txt"; status=1
   fi
   grep -E '^>>|refusal|slab:|dump:|FAIL|PASSED|FAILED' "$RUN/legs-$side.log" | sed "s/^/   $side: /" >> "$RUN/summary.txt"
done
if [ -d "$RUN/dump-host" ] && [ -d "$RUN/dump-device" ]; then
   if "$PY" -I -c 'import numpy' 2> /dev/null; then
      if "$PY" -I "$SRC/scripts/verification/poisson_oracle.py" compare "$RUN/dump-host" "$RUN/dump-device" \
            > "$RUN/compare.txt" 2>&1; then
         echo "PASS  host = device ($(tail -1 "$RUN/compare.txt"))" | tee -a "$RUN/summary.txt"
      else
         echo "FAIL  host = device: see $RUN/compare.txt" | tee -a "$RUN/summary.txt"; status=1
      fi
   else
      echo "NOTE  no numpy for $PY on $(hostname): compare dump-host and dump-device on the workstation" \
           "(scripts/verification/poisson_oracle.py compare)" | tee -a "$RUN/summary.txt"
   fi
fi
if [ $status -eq 0 ] && grep -q '^PASS' "$RUN/summary.txt"; then echo "all passed" | tee -a "$RUN/summary.txt"; else
   echo "some leg failed, or none ran" | tee -a "$RUN/summary.txt"; exit 1; fi
