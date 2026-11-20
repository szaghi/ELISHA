#!/bin/bash
# ELISHA on medooza (CNR-IAC): build the device benchmark on a compute node and run it on one A30, both orders of the
# ADAM/ELISHA pair (the first slot of a pair can be faster: always measure both).
#
#   sbatch --export=ALL,ELISHA_SRC=/home/szaghi/elisha-m4m5 scripts/medooza/elisha-bench.slurm
#   sbatch --export=ALL,ELISHA_SRC=...,ELISHA_BENCH_ARGS='21',ELISHA_BENCH_CASES='abcde',ELISHA_BENCH_NP='1 2' ...
#SBATCH -A iacroma
#SBATCH -p dev-gn
#SBATCH -J elisha-bench
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 00:55:00
#SBATCH -o /ifs/hpc/home/szaghi/elisha-tests/elisha-bench-%j.out
#SBATCH -e /ifs/hpc/home/szaghi/elisha-tests/elisha-bench-%j.out
set -uo pipefail

SRC=${ELISHA_SRC:?set ELISHA_SRC to the sandbox}
REPS=${ELISHA_BENCH_ARGS:-15}
CASES=${ELISHA_BENCH_CASES:-abcde}
NPS=${ELISHA_BENCH_NP:-1}
RUN=${ELISHA_RUN:-/ifs/hpc/home/szaghi/elisha-tests/bench-$SLURM_JOB_ID}
FFTW=/ifs/hpc/home/szaghi/adam-tests/opt/fftw-3.3.10/lib

source /ifs/hpc/home/szaghi/nvidia/nvhpc.sh || exit 1
export PATH=$HOME/.local/bin:$PATH
export LIBRARY_PATH=$FFTW:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$FFTW:${LD_LIBRARY_PATH:-}
export OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_coll_hcoll_enable=0
mkdir -p "$RUN"
{
   echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), reps $REPS, cases $CASES, np $NPS"
   nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.free --format=csv,noheader
} | tee "$RUN/env.txt"
cd "$SRC" || exit 1
fobis build --mode bench-nvf-oac --varset iac > "$RUN/build.log" 2>&1 || { echo "BUILD FAILED" | tee "$RUN/summary.txt"; exit 1; }
cd "$RUN" || exit 1
for np in $NPS; do
   for order in AE EA; do
      echo "== np $np order $order" | tee -a "$RUN/summary.txt"
      srun --mpi=pmix -n "$np" --overlap "$SRC/exe-bench-oac/elisha_bench_device_mpi" "$REPS" "$order" "$CASES" \
         2>&1 | tee -a "$RUN/summary.txt"
   done
done
echo "bench done" | tee -a "$RUN/summary.txt"
