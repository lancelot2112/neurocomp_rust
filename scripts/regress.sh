#!/usr/bin/env bash
# Experiment regression suite: reruns the key experiments of research/experiments at
# their recorded settings (seed 0, full length) and compares the answer accuracy with the
# figures in scripts/regress.tsv. Runs are deterministic (neurocomp::det), so any
# difference means the code's behaviour changed.
#
#   scripts/regress.sh            run all, fail if any result drops more than MARGIN points
#   scripts/regress.sh --record   run all and write the results into the table
#   ONLY=saccades scripts/regress.sh      run only the named entries (comma-separated)
#   MARGIN=3 JOBS=4                       allowed drop in points; parallel runs (default: cores)
set -euo pipefail
cd "$(dirname "$0")/.."
TABLE=scripts/regress.tsv
MARGIN=${MARGIN:-3}
JOBS=${JOBS:-$(nproc)}
RECORD=0
[[ "${1:-}" == "--record" ]] && RECORD=1
OUT=target/regress
mkdir -p "$OUT"

cargo build --release --example episodic >/dev/null 2>&1 || { echo "build failed"; exit 1; }
BIN=target/release/examples/episodic

entries=()
while IFS=$'\t' read -r name page seen held settings; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ -n "${ONLY:-}" && ",$ONLY," != *",$name,"* ]]; then continue; fi
  entries+=("$name")
  (
    # a sub-shell per run, at most JOBS at a time
    env $settings SEEDS=1 "$BIN" >"$OUT/$name.log" 2>&1 || echo "run failed" >>"$OUT/$name.log"
  ) &
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
done <"$TABLE"
wait

# the policy summary line: "NoMemory  seen pairs  63.1%   held-out pairs  22.3% ..."
result() { grep -m1 "seen pairs" "$OUT/$1.log" | sed -E 's/.*seen pairs +([0-9.]+)%.*held-out pairs +([0-9.]+)%.*/\1 \2/'; }

fail=0
tmp=$(mktemp)
printf '%-22s %5s %16s %16s  %s\n' entry page "seen (expected)" "held (expected)" status
while IFS= read -r line; do
  IFS=$'\t' read -r name page seen held settings <<<"$line"
  if [[ -z "$name" || "$name" == \#* ]] || [[ ! " ${entries[*]} " == *" $name "* ]]; then
    printf '%s\n' "$line" >>"$tmp"; continue
  fi
  read -r s h <<<"$(result "$name" || true)"
  if [[ -z "${s:-}" ]]; then
    status="NO RESULT (see $OUT/$name.log)"; fail=1
  elif (( RECORD )); then
    status="recorded"; seen=$s; held=$h
  elif [[ "$seen" == "-" ]]; then
    status="no expected figure (run --record)"
  else
    status=$(awk -v s="$s" -v h="$h" -v es="$seen" -v eh="$held" -v m="$MARGIN" 'BEGIN {
      if (s < es - m || h < eh - m) print "FAIL"; else if (s != es || h != eh) print "changed (within margin)"; else print "same" }')
    [[ "$status" == FAIL ]] && fail=1
  fi
  printf '%-22s %5s %7s (%6s) %7s (%6s)  %s\n' "$name" "$page" "${s:--}" "$seen" "${h:--}" "$held" "$status"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$page" "$seen" "$held" "$settings" >>"$tmp"
done <"$TABLE"
if (( RECORD )); then mv "$tmp" "$TABLE"; else rm -f "$tmp"; fi
exit $fail
