#!/usr/bin/env bash
# Run the tests of the fobos mode tests-intel-argtemp (ifx, every check on, arg_temp_created included) and fail when
# the library builds an argument temporary: a "forrtl: warning (406)" whose first stack frame is in a library source
# (src/lib/, src/device/). The tests' own one-off temporaries (array constructors passed to a test's helpers) are
# reported, not failed. Each run's output is capped (OUT_CAP bytes): a temporary in a per-cell loop prints a traceback
# per call, and one such loop flooded a log to 82 GB (2026-11-20).
#
# Usage: fobis build --mode tests-intel-argtemp && bash scripts/check_argtemp.sh [--np N] [--exe-dir DIR]
NP=2
EXE=exe
OUT_CAP=${OUT_CAP:-20000000}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --np | -n ) NP="$2"; shift 2 ;;
    --exe-dir | -e ) EXE="${2%/}"; shift 2 ;;
    * ) printf "Unknown argument: %s\n" "$1" >&2; exit 2 ;;
  esac
done
LIB_SOURCES=$(cd "$(dirname "$0")/.." && ls src/lib/*.F90 src/device/*.F90 2>/dev/null | xargs -n1 basename)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
lib_sites=0
test_sites=0
for t in "$EXE"/*; do
  [[ -f "$t" && -x "$t" ]] || continue
  name=$(basename "$t")
  if [[ "$name" == *mpi* ]]; then cmd=(mpirun -np "$NP" "$t"); else cmd=("$t"); fi
  "${cmd[@]}" 2>&1 | head -c "$OUT_CAP" > "$tmp/out"
  # the first frame after each warning: "<image> <pc> <routine> <line> <source>"
  awk '/forrtl: warning \(406\)/ {w = $0
         if ((getline) <= 0 || (getline) <= 0 || (getline) <= 0) next # truncated by the cap
         if (NF >= 5) print $NF ":" $(NF-1) " " w; else print "unknown:0 " w}' "$tmp/out" \
    | sort | uniq -c > "$tmp/sites"
  while read -r count where rest; do
    src=${where%%:*}
    if grep -qx "$src" <<< "$LIB_SOURCES"; then
      printf 'LIBRARY  %-20s %8d  %s\n' "$where" "$count" "${rest#*In call to }"
      lib_sites=$((lib_sites + 1))
    else
      test_sites=$((test_sites + 1))
    fi
  done < "$tmp/sites"
done
printf 'check_argtemp: %d library site(s), %d test site(s) (tests: reported, not failed)\n' "$lib_sites" "$test_sites"
[[ $lib_sites -eq 0 ]]
