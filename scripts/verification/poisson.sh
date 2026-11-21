#!/usr/bin/env bash
# ELISHA verification of the elliptic solvers: ADAM's poisson.sh (src/tests/flume/verification/selfgravity, ADAM
# 669d0bdd) ported onto ELISHA's three drivers (ADAM issue #136, D2). Each leg runs a driver on every rank count of
# --np-list and passes when every run prints `<driver>: PASSED`; each refusal runs at np 2 and passes when the driver
# reports the expected ELISHA status by name (owner decision D3: ELISHA returns a status where FLAIL stopped the run).
#
#   leg    driver and flags                       what (the drivers' headers have the cases and the bounds)
#   sv2    elisha_sv_fft_mpi --periodic           SV-2: the spectral solver, periodic, against the discrete and the
#                                                 continuum solutions; refusal: a non-periodic grid
#                                                 (ELISHA_ERR_BC_MISSING)
#   sv6    elisha_sv_fft_mpi --isolated           SV-6: isolated, the polytrope and the uniform sphere; refusals: a
#                                                 periodic grid (ELISHA_ERR_ISOLATED_PERIODIC), anisotropic cells
#                                                 (_ISOLATED_CELLS), 2-D (_ISOLATED_DIMENSION)
#   slab   elisha_sv_fft_mpi --dump               the slab transform's results do not depend on the rank count: the
#                                                 potentials of SV-2 and SV-6 (13 transform arrays, images and padding
#                                                 included) dumped at every rank count and compared byte for byte
#   sv9    elisha_sv_mg_mpi --sv9                 SV-9a/b, SV-12a/b: the hierarchical solver, periodic and Dirichlet,
#                                                 uniform and AMR; refusals: mixed axes without bc (_BC_MISSING), bc
#                                                 against the grid (_BC_PERIODICITY), a ghost-Dirichlet face against a
#                                                 face kind (_BC_GHOST_PAIR)
#   sv13   elisha_sv_mg_mpi --isolated            SV-13a/b: isolated on AMR (the James step); refusal: a refined leaf
#                                                 on the boundary (_ISOLATED_BOUNDARY_LEAF)
#   bc     elisha_sv_bc_mpi                       E1a, E2a, E2b, E3a, E3c: the spectral solver's face kinds and the
#                                                 wide D(G); refusals: _BC_PERIODIC_PAIR, _BC_GHOST_PAIR,
#                                                 _ISOLATED_WIDE, _BC_ZERO_GHOST_PAIR, _BC_FREE_GHOSTS
#   bcmg   elisha_sv_mg_mpi --bc                  E1b: the hierarchical solver's face kinds
#   flail  elisha_sv_mg_mpi --flail               E1c: the facade against the solvers it chooses, bit for bit;
#                                                 refusals: _GHOST_VALUES, _ISOLATED_BC, _DG_AMR
#   sub    elisha_sv_mg_mpi --sub                 E5c: the levels inside the blocks
#
# `--dump DIR` also runs elisha_sv_fft_mpi (every SV-2 and SV-6 level) and elisha_sv_mg_mpi --sv9 --isolated (every
# SV-9, SV-12 and SV-13 grid) at np 2 with --dump DIR: run it once per build (the host one and, with --device, the
# device one), then compare the two directories with
#   scripts/verification/poisson_oracle.py compare DIR_HOST DIR_DEVICE      (max |phi_A - phi_B| / max |phi_A|, 1e-13)
# `--device` runs every driver with --device (the cuFFT and OpenACC backends, the device build tests-nvf-oac) and makes
# exe-oac/ the default executable directory.
#
# Usage: scripts/verification/poisson.sh [--np-list "1 2 3"] [--legs LIST] [--dump DIR] [--device] [--work DIR] [--keep]
#        LIST: sv2, sv6, slab, sv9, sv13, bc, bcmg, flail, sub or all (default all), space or comma separated
# Environment: ELISHA_EXE_DIR (default exe/, exe-oac/ with --device), and per driver POISSON_EXE (elisha_sv_fft_mpi),
# POISSON_MG_EXE (elisha_sv_mg_mpi), POISSON_BC_EXE (elisha_sv_bc_mpi); LAUNCH, the launcher before the rank count
# (default "mpirun --oversubscribe -np"; under Slurm e.g. "srun --mpi=pmix --overlap -n"). The caller owns the
# toolchain environment matching the build. The work directory (default under ${TMPDIR:-/tmp}) is removed on success
# unless --keep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ALL_LEGS="sv2 sv6 slab sv9 sv13 bc bcmg flail sub"
NP_LIST="1 2 3"
LEGS="$ALL_LEGS"
DUMP=""
DEVICE=0
WORK=""
KEEP=0
while [[ $# -gt 0 ]]; do
   case "$1" in
      --np-list) NP_LIST="$2" ; shift 2 ;;
      --legs)    LEGS="${2//,/ }" ; shift 2 ;;
      --dump)    DUMP="$2" ; shift 2 ;;
      --device)  DEVICE=1 ; shift ;;
      --work)    WORK="$2" ; shift 2 ;;
      --keep)    KEEP=1 ; shift ;;
      *) echo "poisson.sh: unknown argument '$1' (accepted: --np-list LIST, --legs LIST, --dump DIR, --device," \
              "--work DIR, --keep)" >&2 ; exit 2 ;;
   esac
done
[[ "$LEGS" == "all" ]] && LEGS="$ALL_LEGS"
for leg in $LEGS; do
   [[ " $ALL_LEGS " == *" $leg "* ]] || { echo "poisson.sh: unknown leg '$leg' (legs: $ALL_LEGS, all)" >&2 ; exit 2 ; }
done

DEV_FLAG=""
EXE_DIR="${ELISHA_EXE_DIR:-$REPO_ROOT/exe}"
if [[ $DEVICE -eq 1 ]]; then
   DEV_FLAG="--device"
   EXE_DIR="${ELISHA_EXE_DIR:-$REPO_ROOT/exe-oac}"
fi
FFT_EXE="${POISSON_EXE:-$EXE_DIR/elisha_sv_fft_mpi}"
MG_EXE="${POISSON_MG_EXE:-$EXE_DIR/elisha_sv_mg_mpi}"
BC_EXE="${POISSON_BC_EXE:-$EXE_DIR/elisha_sv_bc_mpi}"
read -r -a LAUNCH_CMD <<< "${LAUNCH:-mpirun --oversubscribe -np}"

needs_fft=0 ; needs_mg=0 ; needs_bc=0
for leg in $LEGS; do
   case "$leg" in sv2|sv6|slab) needs_fft=1 ;; sv9|sv13|bcmg|flail|sub) needs_mg=1 ;; bc) needs_bc=1 ;; esac
done
[[ -n "$DUMP" ]] && { needs_fft=1 ; needs_mg=1 ; }
check_exe() { # check_exe <path> <needed> <variable>: the executable exists, made absolute
   if [[ $2 -eq 1 && ! -x "$1" ]]; then
      echo "poisson.sh: executable '$1' not found (build it, or set $3 / ELISHA_EXE_DIR)" >&2 ; exit 2
   fi
}
check_exe "$FFT_EXE" $needs_fft POISSON_EXE
check_exe "$MG_EXE" $needs_mg POISSON_MG_EXE
check_exe "$BC_EXE" $needs_bc POISSON_BC_EXE
[[ $needs_fft -eq 1 ]] && FFT_EXE="$(realpath "$FFT_EXE")"
[[ $needs_mg -eq 1 ]] && MG_EXE="$(realpath "$MG_EXE")"
[[ $needs_bc -eq 1 ]] && BC_EXE="$(realpath "$BC_EXE")"

TAG="$(basename "$EXE_DIR")${DEV_FLAG:+-device}"
WORK="${WORK:-${TMPDIR:-/tmp}/elisha-sv-$TAG-$$}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
FAILED=0

run() { # run <np> <log> <exe> [flags...]: the driver on np ranks, from the work directory
   local np="$1" log="$2" exe="$3"
   shift 3
   # shellcheck disable=SC2086
   (cd "$WORK" && "${LAUNCH_CMD[@]}" "$np" "$exe" "$@" $DEV_FLAG > "$log" 2>&1)
}
suite() { # suite <label> <exe> [flags...]: the driver on every rank count of the list, its summary rows and verdict
   local label="$1" exe="$2" np log name
   shift 2
   name="$(basename "$exe")"
   for np in $NP_LIST; do
      log="$WORK/log-$label-np$np.txt"
      if ! run "$np" "$log" "$exe" "$@"; then
         echo "   np $np: the run failed, see $log  FAIL"
         FAILED=1
         continue
      fi
      sed -n '/ summary (np /,/: PASSED$\|: FAILED$/p' "$log" | { grep '^  ' || true ; } | sed "s/^/   np $np:/"
      if ! grep -q "^$name: PASSED" "$log"; then
         echo "   np $np: $name did not pass, see $log  FAIL"
         FAILED=1
      fi
   done
}
refusal() { # refusal <exe> <flag> <status name>: the driver must report that status (and exit 0)
   local exe="$1" flag="$2" code="$3" log="$WORK/log-refusal${2}.txt"
   if run 2 "$log" "$exe" "$flag" && grep -q "$code (expected $code)  PASS" "$log"; then
      echo "   refusal $flag: $code  PASS"
   else
      echo "   refusal $flag: not $code, see $log  FAIL"
      FAILED=1
   fi
}

echo "ELISHA Poisson legs: $LEGS; ranks $NP_LIST; executables $EXE_DIR${DEV_FLAG:+ (--device)}; work $WORK"
for leg in $LEGS; do
   t0=$SECONDS
   case "$leg" in
      sv2)
         echo ">> SV-2: the spectral solver, periodic ($(basename "$FFT_EXE"))"
         suite sv2 "$FFT_EXE" --periodic
         refusal "$FFT_EXE" --no-periodic ELISHA_ERR_BC_MISSING ;;
      sv6)
         echo ">> SV-6: the spectral solver, isolated ($(basename "$FFT_EXE"))"
         suite sv6 "$FFT_EXE" --isolated
         refusal "$FFT_EXE" --refuse-isolated-periodic ELISHA_ERR_ISOLATED_PERIODIC
         refusal "$FFT_EXE" --refuse-isolated-anisotropic ELISHA_ERR_ISOLATED_CELLS
         refusal "$FFT_EXE" --refuse-isolated-2d ELISHA_ERR_ISOLATED_DIMENSION ;;
      slab)
         echo ">> The slab transform, bitwise across the rank counts ($(basename "$FFT_EXE"))"
         first="" ; nsame=0 ; ndiff=0
         for np in $NP_LIST; do
            dir="$WORK/slab-np$np"
            mkdir -p "$dir"
            if ! run "$np" "$dir.log" "$FFT_EXE" --no-rank-check --dump "$dir"; then
               echo "   slab, np $np: the run failed, see $dir.log  FAIL" ; FAILED=1 ; continue
            fi
            if [[ -z "$first" ]]; then first="$np" ; continue ; fi
            for f in "$WORK/slab-np$first"/*.bin; do
               g="$dir/$(basename "$f")"
               if cmp -s "$f" "$g"; then nsame=$((nsame + 1)) ; else ndiff=$((ndiff + 1)) ; echo "   slab: $g differs" ; fi
            done
         done
         if [[ $ndiff -eq 0 && $nsame -gt 0 ]]; then
            echo "   slab: $nsame potentials identical to np $first  PASS"
         else
            echo "   slab: $ndiff of $((nsame + ndiff)) potentials differ from np $first (one rank count: nothing to" \
                 "compare)  FAIL"
            FAILED=1
         fi ;;
      sv9)
         echo ">> SV-9, SV-12: the hierarchical solver, periodic and Dirichlet ($(basename "$MG_EXE"))"
         suite sv9 "$MG_EXE" --sv9
         refusal "$MG_EXE" --mixed ELISHA_ERR_BC_MISSING
         refusal "$MG_EXE" --refuse-bc-mismatch ELISHA_ERR_BC_PERIODICITY
         refusal "$MG_EXE" --refuse-bc-mixed-ghost ELISHA_ERR_BC_GHOST_PAIR ;;
      sv13)
         echo ">> SV-13: the hierarchical solver, isolated on AMR ($(basename "$MG_EXE"))"
         suite sv13 "$MG_EXE" --isolated
         refusal "$MG_EXE" --refuse-boundary-refined ELISHA_ERR_ISOLATED_BOUNDARY_LEAF ;;
      bc)
         echo ">> E1a, E2a, E2b, E3a, E3c: the spectral solver's face kinds and the wide D(G) ($(basename "$BC_EXE"))"
         suite bc "$BC_EXE"
         refusal "$BC_EXE" --refuse-mixed-periodic ELISHA_ERR_BC_PERIODIC_PAIR
         refusal "$BC_EXE" --refuse-mixed-ghost ELISHA_ERR_BC_GHOST_PAIR
         refusal "$BC_EXE" --refuse-wide-isolated ELISHA_ERR_ISOLATED_WIDE
         refusal "$BC_EXE" --refuse-zero-mixed ELISHA_ERR_BC_ZERO_GHOST_PAIR
         refusal "$BC_EXE" --refuse-free-compact ELISHA_ERR_BC_FREE_GHOSTS ;;
      bcmg)
         echo ">> E1b: the hierarchical solver's face kinds ($(basename "$MG_EXE"))"
         suite bcmg "$MG_EXE" --bc ;;
      flail)
         echo ">> E1c: the facade ($(basename "$MG_EXE"))"
         suite flail "$MG_EXE" --flail
         refusal "$MG_EXE" --refuse-flail-ghost-fft ELISHA_ERR_GHOST_VALUES
         refusal "$MG_EXE" --refuse-flail-isolated-bc ELISHA_ERR_ISOLATED_BC
         refusal "$MG_EXE" --refuse-flail-wide-amr ELISHA_ERR_DG_AMR ;;
      sub)
         echo ">> E5c: the levels inside the blocks ($(basename "$MG_EXE"))"
         suite sub "$MG_EXE" --sub ;;
   esac
   echo "   leg $leg: $((SECONDS - t0)) s"
done
if [[ -n "$DUMP" ]]; then
   mkdir -p "$DUMP"
   DUMP="$(cd "$DUMP" && pwd)"
   if ! run 2 "$WORK/log-dump-fft.txt" "$FFT_EXE" --no-rank-check --dump "$DUMP"; then
      echo "   dump, $(basename "$FFT_EXE"): the run failed, see $WORK/log-dump-fft.txt  FAIL" ; FAILED=1
   fi
   if ! run 2 "$WORK/log-dump-mg.txt" "$MG_EXE" --sv9 --isolated --dump "$DUMP"; then
      echo "   dump, $(basename "$MG_EXE"): the run failed, see $WORK/log-dump-mg.txt  FAIL" ; FAILED=1
   fi
   echo "   dump: $(find "$DUMP" -maxdepth 1 -name 'phi-*.bin' | wc -l) potentials in $DUMP"
fi
if [[ $KEEP -eq 0 && $FAILED -eq 0 ]]; then rm -rf "$WORK" ; fi
if [[ $FAILED -ne 0 ]]; then
   echo "Poisson legs ($LEGS) FAILED ($TAG; logs in $WORK)" ; exit 1
fi
echo "Poisson legs ($LEGS) PASSED ($TAG)"
