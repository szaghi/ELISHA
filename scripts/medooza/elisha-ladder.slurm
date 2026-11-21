#!/bin/bash
# ELISHA on medooza (CNR-IAC): the scaling ladders of the solvers (ELISHA issue #1, EB; ADAM issue #136, D7), ADAM's
# `bench.sh` ladders on ELISHA's own program (src/bench/ladder/elisha_bench_ladder_mpi.F90): absolute times, no ADAM
# oracle. Builds inside the job, runs one ladder set, writes every BENCH line (with the job id) to $RUN/bench.txt and
# prints the summary table at the end ($RUN/summary.txt).
#
# Ladders (ELISHA_LADDER, a space-separated list; the problems are ADAM's, see the program's header):
#   strong  CPU, level 4, ni 8: FFT 2 097 152 uniform cells, multigrid 3 932 160 AMR cells (the central box refined
#           once), periodic and isolated; np 1 2 4 8 16 32 64 (ELISHA_STRONG_NP)
#   weak    CPU, the cells per rank fixed: (np, level) = (1, 3) (8, 4) (64, 5) (ELISHA_WEAK_PAIRS); ni 8 on the four
#           problems (262 144 uniform cells per rank), ni 16 on the two FFT ones (2 097 152 per rank: 16.7 M at np 8)
#   fnl     the device backends (cuFFT + OpenACC), the strong problems at np 1 2 4 (ELISHA_FNL_NP), one A30 per rank
#           (the program takes the GPU of its node-local rank), source and potential on the host (ADAM's protocol)
#           and on the device (`--on-device`, ELISHA's M7 API)
# One job per ladder set, in this order (one job at a time; the CPU ladders take the whole node, --exclusive):
#
#   sbatch --export=ALL,ELISHA_SRC=/home/szaghi/elisha-eb,ELISHA_LADDER=strong scripts/medooza/elisha-ladder.slurm
#   sbatch --export=ALL,ELISHA_SRC=/home/szaghi/elisha-eb,ELISHA_LADDER=weak   scripts/medooza/elisha-ladder.slurm
#   sbatch --ntasks=4 --gres=gpu:4 --mem=120G -t 01:00:00 \
#          --export=ALL,ELISHA_SRC=/home/szaghi/elisha-eb,ELISHA_LADDER=fnl     scripts/medooza/elisha-ladder.slurm
#
# Knobs: ELISHA_LADDER_TC (CPU toolchain: nvf, the default, or gnu), ELISHA_REPS (timed solves per run, default 5),
# ELISHA_RUN (run directory, default /ifs/hpc/home/szaghi/elisha-tests/ladder-<job id>), ELISHA_BUILD=0 (use the
# built program), ELISHA_RUN_TIMEOUT (seconds per run, default 1800).
# Local dry run (no Slurm: mpirun, no toolchain sourcing, small problems), e.g. on a workstation:
#   ELISHA_LOCAL=1 ELISHA_SRC=$PWD ELISHA_LADDER='strong weak' ELISHA_STRONG_LEVEL=3 ELISHA_STRONG_NP='1 2' \
#   ELISHA_WEAK_PAIRS='1 2,8 3' ELISHA_BUILD=0 ELISHA_RUN=/tmp/ladder bash scripts/medooza/elisha-ladder.slurm
#
# medooza has no FFTW: the user-space FFTW 3.3.10 of the ADAM tests is linked (LIBRARY_PATH, LD_LIBRARY_PATH). HPC-X
# needs UCC and HCOLL off on the older prod-gn drivers.
#SBATCH -A iacroma
#SBATCH -p prod-gn
#SBATCH -J elisha-ladder
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o /ifs/hpc/home/szaghi/elisha-tests/elisha-ladder-%j.out
#SBATCH -e /ifs/hpc/home/szaghi/elisha-tests/elisha-ladder-%j.out
set -uo pipefail

SRC=${ELISHA_SRC:?set ELISHA_SRC to the sandbox}
LADDERS=${ELISHA_LADDER:-strong}
TC=${ELISHA_LADDER_TC:-nvf}
REPS=${ELISHA_REPS:-5}
LOCAL=${ELISHA_LOCAL:-0}
BUILD=${ELISHA_BUILD:-1}
TMO=${ELISHA_RUN_TIMEOUT:-1800}
STRONG_LEVEL=${ELISHA_STRONG_LEVEL:-4}
STRONG_NP=${ELISHA_STRONG_NP:-1 2 4 8 16 32 64}
WEAK_PAIRS=${ELISHA_WEAK_PAIRS:-1 3,8 4,64 5}
FNL_NP=${ELISHA_FNL_NP:-1 2 4}
JOB=${SLURM_JOB_ID:-local}
RUN=${ELISHA_RUN:-/ifs/hpc/home/szaghi/elisha-tests/ladder-$JOB}
FFTW=/ifs/hpc/home/szaghi/adam-tests/opt/fftw-3.3.10/lib

for l in $LADDERS; do
   case $l in strong|weak|fnl) ;; *) echo "elisha-ladder: unknown ladder '$l' (strong, weak, fnl)" >&2; exit 2 ;; esac
done
case $TC in nvf|gnu) ;; *) echo "elisha-ladder: ELISHA_LADDER_TC is nvf or gnu, not '$TC'" >&2; exit 2 ;; esac
cpu=0 ; fnl=0
for l in $LADDERS; do [ "$l" = fnl ] && fnl=1 || cpu=1; done
# the device build is nvfortran's; a CPU ladder in the same job then runs nvfortran too
if [ $fnl -eq 1 ] && [ $cpu -eq 1 ] && [ "$TC" = gnu ]; then
   echo "elisha-ladder: fnl needs nvfortran; run the gnu CPU ladders in their own job" >&2; exit 2
fi
[ $fnl -eq 1 ] && TC=nvf

if [ "$LOCAL" = 1 ]; then
   LAUNCH=(mpirun --oversubscribe -np)
else
   if [ "$TC" = nvf ]; then
      source /ifs/hpc/home/szaghi/nvidia/nvhpc.sh || exit 1
   else
      source /ifs/hpc/home/szaghi/gnu/gnu.sh || exit 1
   fi
   export PATH=$HOME/.local/bin:$PATH
   export LIBRARY_PATH=$FFTW:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$FFTW:${LD_LIBRARY_PATH:-}
   export OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_coll_hcoll_enable=0
   LAUNCH=(srun --mpi=pmix --cpu-bind=cores -n)
fi
export OMP_NUM_THREADS=1
mkdir -p "$RUN" || exit 1
OUT=$RUN/bench.txt
{
   echo "job $JOB on $(hostname), $(date -Is), ladders: $LADDERS, toolchain $TC, reps $REPS"
   echo "source $SRC ($(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo 'no git')), run $RUN"
   echo "cores ${SLURM_CPUS_ON_NODE:-$(nproc)}, memory $(free -g | awk '/^Mem:/ {print $2}') GiB, $(lscpu | sed -n 's/^Model name: *//p')"
   mpif90 --version 2>/dev/null | grep -m1 -E "nvfortran|GNU Fortran"
   mpirun --version 2>/dev/null | head -1
   [ $fnl -eq 1 ] && nvidia-smi --query-gpu=index,name,driver_version,memory.free --format=csv,noheader
} | tee "$RUN/env.txt"
if [ "$TC" = gnu ] && [ "$LOCAL" != 1 ] && ldd "$FFTW/libfftw3.so" 2>/dev/null | grep -qiE 'nvhpc|libnvc|libpgc'; then
   echo "elisha-ladder: $FFTW/libfftw3.so needs the NVIDIA runtime: run with ELISHA_LADDER_TC=nvf" | tee -a "$RUN/env.txt"
   exit 3
fi
if [ $fnl -eq 1 ] && [ "$LOCAL" != 1 ]; then
   free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
   if [ "${free:-0}" -lt 20000 ]; then echo "GPU busy on $(hostname): ${free} MiB free" | tee -a "$RUN/env.txt"; exit 4; fi
fi

# the builds (one mode per toolchain; the host modes share exe-bench/, the device one has exe-bench-oac/)
build() { # build <mode>
   [ "$BUILD" = 1 ] || return 0
   (cd "$SRC" && fobis build --mode "$1" --varset iac) > "$RUN/build-$1.log" 2>&1 \
      || { echo "BUILD FAILED $1: see $RUN/build-$1.log" | tee -a "$RUN/summary.txt"; exit 1; }
}
EXE_CPU=$SRC/exe-bench/elisha_bench_ladder_mpi
EXE_FNL=$SRC/exe-bench-oac/elisha_bench_ladder_mpi
[ $cpu -eq 1 ] && build "bench-ladder-$TC"
[ $fnl -eq 1 ] && build bench-ladder-nvf-oac

status=0
run() { # run <ladder tag> <exe> <np> <program arguments...>: one run, its BENCH line (with the job id) into $OUT
   local tag=$1 exe=$2 np=$3 log; shift 3
   log="$RUN/log-$tag-np$np$(echo "$*" | tr -d ' -').txt"
   if (cd "$RUN" && timeout "$TMO" "${LAUNCH[@]}" "$np" "$exe" "$@" --reps "$REPS" --tag "$tag" < /dev/null > "$log" 2>&1) \
      && grep -aq '^BENCH ' "$log"; then
      grep -a '^BENCH ' "$log" | sed "s/^BENCH /BENCH job=$JOB /" >> "$OUT"
      echo "  ok   $tag np $np $*"
   else
      echo "  FAIL $tag np $np $* (see $log)"
      status=1
   fi
}
PROBLEMS=("--path fft --bc periodic" "--path fft --bc isolated" "--path mg --bc periodic" "--path mg --bc isolated")

for l in $LADDERS; do
   case $l in
   strong)
      echo "strong ladder (CPU, $TC), level $STRONG_LEVEL, ni 8, np $STRONG_NP" | tee -a "$RUN/summary.txt"
      for p in "${PROBLEMS[@]}"; do
         for np in $STRONG_NP; do run strong "$EXE_CPU" "$np" $p --level "$STRONG_LEVEL" --ni 8; done
      done ;;
   weak)
      echo "weak ladder (CPU, $TC), (np level): $WEAK_PAIRS" | tee -a "$RUN/summary.txt"
      IFS=',' read -r -a pairs <<< "$WEAK_PAIRS"
      for ni in 8 16; do
         for p in "${PROBLEMS[@]}"; do
            [[ $ni -eq 16 && $p == *mg* ]] && continue # ADAM's ladder: the multigrid at ni 16 would take hours
            for pair in "${pairs[@]}"; do
               set -- $pair
               run weak "$EXE_CPU" "$1" $p --level "$2" --ni "$ni"
            done
         done
      done ;;
   fnl)
      echo "FNL ladder (cuFFT + OpenACC), level $STRONG_LEVEL, ni 8, np $FNL_NP" | tee -a "$RUN/summary.txt"
      for mode in host dev; do
         extra="" ; [ $mode = dev ] && extra="--on-device"
         for p in "${PROBLEMS[@]}"; do
            for np in $FNL_NP; do run "fnl-$mode" "$EXE_FNL" "$np" $p --level "$STRONG_LEVEL" --ni 8 $extra; done
         done
      done ;;
   esac
done 2>&1 | tee -a "$RUN/summary.txt"
grep -q '^  FAIL' "$RUN/summary.txt" && status=1

# the summary: solve time (min over the repetitions, max over the ranks), speed-up and efficiency against the ladder's
# smallest np (strong: t1 / (np t), weak: t1 / t), rank 0's spectral phases per solve, and whether the potential is
# bitwise equal across np (phi_xor) per problem
{
   echo
   printf '%-9s %-4s %-9s %-3s %-3s %-3s %-10s %-5s %-5s %-10s %-10s %-10s %-7s %-6s %-10s %-10s %-10s %-10s\n' \
      ladder path bc np lev ni cells lay its setup solve_min solve_med speedup eff gather solve scatter james_solve
   [ -s "$OUT" ] && awk '
   function kv(s,   i, m, a) { delete v; m = split(s, f, " "); for (i = 2; i <= m; i++) { split(f[i], a, "="); v[a[1]] = a[2] } }
   { kv($0); key = v["tag"] "|" v["path"] "|" v["bc"] "|" v["ni"] (v["tag"] == "weak" ? "" : "|" v["level"])
     if (!(key in np0) || v["np"] + 0 < np0[key]) { np0[key] = v["np"] + 0; t0[key] = v["solve_min"] + 0 }
     row[NR] = $0; k[NR] = key
     hkey = v["tag"] "|" v["path"] "|" v["bc"] "|" v["level"] "|" v["ni"]
     if (!(hkey in h)) { h[hkey] = v["phi_xor"]; hn[hkey] = 1; order[++no] = hkey } else { hn[hkey]++; if (h[hkey] != v["phi_xor"]) hd[hkey] = 1 } }
   END {
     for (r = 1; r <= NR; r++) {
        kv(row[r]); t = v["solve_min"] + 0; s = (t > 0 ? t0[k[r]] / t : 0)
        e = (v["tag"] == "weak" ? s : s * np0[k[r]] / v["np"])
        printf "%-9s %-4s %-9s %-3s %-3s %-3s %-10s %-5s %-5s %-10.3e %-10.3e %-10.3e %-7.2f %-6.2f %-10.3e %-10.3e %-10.3e %-10.3e\n", \
           v["tag"], v["path"], v["bc"], v["np"], v["level"], v["ni"], v["cells"], v["layout"], v["iterations"], \
           v["setup"], t, v["solve_med"], s, e, v["gather"], v["solve"], v["scatter"], v["james_solve"]
     }
     print ""
     for (i = 1; i <= no; i++) if (hn[order[i]] > 1) \
        printf "phi bitwise across np  %-40s %s\n", order[i], (order[i] in hd ? "NO" : "yes")
   }' "$OUT"
} | tee -a "$RUN/summary.txt"
echo "ladder done (status $status): $RUN" | tee -a "$RUN/summary.txt"
exit $status
