#!/bin/bash
# ELISHA on medooza (CNR-IAC): build test modes on a compute node and run every test program there, the MPI ones at
# np 1, 2, 3, the device (tests-nvf-oac) ones on one A30 (shared by the ranks).
#
# The source tree is built where it lies (ELISHA_SRC, a sandbox under /home/szaghi) and the programs run from their own
# run directory on /ifs (ELISHA_RUN, default /ifs/hpc/home/szaghi/elisha-tests/<job id>): nothing but build products is
# written next to the sources. Submit from the login node (only sbatch runs there), e.g.
#
#   sbatch --export=ALL,ELISHA_SRC=/home/szaghi/elisha-m4m5 scripts/medooza/elisha-tests.slurm
#   sbatch --export=ALL,ELISHA_SRC=...,ELISHA_MODES='tests-nvf-oac',ELISHA_GLOB='elisha_test_*device*' ...
#
# The dev-gn nodes (driver 575) run CUDA 12.9: varset iac. medooza has no FFTW: the user-space FFTW 3.3.10 of the ADAM
# tests is linked (LIBRARY_PATH, LD_LIBRARY_PATH). HPC-X runs need UCC off on the older prod-gn drivers (harmless here).
#SBATCH -A iacroma
#SBATCH -p dev-gn
#SBATCH -J elisha-tests
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 00:55:00
#SBATCH -o /ifs/hpc/home/szaghi/elisha-tests/elisha-%j.out
#SBATCH -e /ifs/hpc/home/szaghi/elisha-tests/elisha-%j.out
set -uo pipefail

SRC=${ELISHA_SRC:?set ELISHA_SRC to the sandbox}
MODES=${ELISHA_MODES:-tests-nvf tests-nvf-oac}
GLOB=${ELISHA_GLOB:-elisha_test_*}
RUN=${ELISHA_RUN:-/ifs/hpc/home/szaghi/elisha-tests/$SLURM_JOB_ID}
FFTW=/ifs/hpc/home/szaghi/adam-tests/opt/fftw-3.3.10/lib

source /ifs/hpc/home/szaghi/nvidia/nvhpc.sh || exit 1
export PATH=$HOME/.local/bin:$PATH
export LIBRARY_PATH=$FFTW:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$FFTW:${LD_LIBRARY_PATH:-}
export OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_coll_hcoll_enable=0
mkdir -p "$RUN"
{
   echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"
   echo "source $SRC, modes $MODES"
   nvfortran --version | sed -n 2p
   nvidia-smi --query-gpu=name,driver_version,memory.free,memory.total --format=csv,noheader
} | tee "$RUN/env.txt"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free:-0}" -lt 20000 ]; then echo "GPU busy on $(hostname): ${free} MiB free" | tee "$RUN/summary.txt"; exit 4; fi

status=0
for mode in $MODES; do
   cd "$SRC" || exit 1
   if ! fobis build --mode "$mode" --varset iac > "$RUN/build-$mode.log" 2>&1; then
      echo "BUILD FAILED $mode: see $RUN/build-$mode.log" | tee -a "$RUN/summary.txt"; status=1; continue
   fi
   mkdir -p "$RUN/$mode" && cd "$RUN/$mode" || exit 1
   exe_dir=exe ; [[ $mode == *oac* ]] && exe_dir=exe-oac # fobos: the device mode builds into exe-oac/
   for t in "$SRC"/$exe_dir/$GLOB; do
      [ -x "$t" ] && [ -f "$t" ] || continue
      name=$(basename "$t")
      # MPI programs at np 1, 2, 3, the device ones too: their ranks share the A30, which exercises the GPU-aware
      # (device-buffer) exchanges of the halos and the slab transpose
      if [[ $name == *mpi* ]]; then nps="1 2 3"; else nps="1"; fi
      for np in $nps; do
         if srun --mpi=pmix -n "$np" --overlap "$t" > "$name.np$np.log" 2>&1; then
            echo "PASS  $mode $name np $np"
         else
            echo "FAIL  $mode $name np $np (exit $?)"; status=1
         fi
      done
   done | tee -a "$RUN/summary.txt"
   grep -q "^FAIL  $mode" "$RUN/summary.txt" && status=1
done
if [ $status -eq 0 ] && grep -q '^PASS' "$RUN/summary.txt"; then echo "all passed" | tee -a "$RUN/summary.txt"; else
   echo "some program failed, or none ran" | tee -a "$RUN/summary.txt"; exit 1; fi
