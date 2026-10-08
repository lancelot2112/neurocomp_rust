#!/usr/bin/env bash
# Experiment regression suite: reruns the key experiments of research/experiments at
# their recorded settings (full length) on several seeds and compares the mean answer
# accuracy with the figures in scripts/regress.tsv. Runs are deterministic
# (neurocomp::det), so any difference means the code's behaviour changed.
#
# One seed is one draw from a wide spread (a seed-0 figure moved by 20+ points when the
# random streams changed, the three-seed mean did not), so the table holds the mean over
# SEEDS seeds and the margin applies to the mean. Five seeds by default: kernel growth is
# path-dependent (a change at 0.2% of training steps moved one seed by 22 points,
# experiment 85), and three-seed means could not separate effects of 5–10 points. Every (entry, seed) is its own job, so
# seeds run in parallel.
#
#   scripts/regress.sh            run all, fail if any mean drops more than MARGIN points
#   scripts/regress.sh --record   run all and write the means into the table
#   ONLY=saccades scripts/regress.sh      run only the named entries (comma-separated)
#   SEEDS=5 MARGIN=3 JOBS=4               seeds per entry; allowed drop of the mean in
#                                         points; parallel jobs (default: cores)
#   QUICK=1 scripts/regress.sh            seed 0 only, reported without pass or fail (one
#                                         seed is not comparable with the recorded mean)
set -euo pipefail
cd "$(dirname "$0")/.."
TABLE=scripts/regress.tsv
MARGIN=${MARGIN:-3}
JOBS=${JOBS:-$(nproc)}
SEEDS=${SEEDS:-5}
QUICK=${QUICK:-0}
(( QUICK )) && SEEDS=1
RECORD=0
[[ "${1:-}" == "--record" ]] && RECORD=1
(( QUICK && RECORD )) && { echo "QUICK runs cannot be recorded"; exit 1; }
OUT=target/regress
mkdir -p "$OUT"

cargo build --release --example episodic >/dev/null 2>&1 || { echo "build failed"; exit 1; }
BIN=target/release/examples/episodic

entries=()
while IFS=$'\t' read -r name page seen held settings; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ -n "${ONLY:-}" && ",$ONLY," != *",$name,"* ]]; then continue; fi
  entries+=("$name")
  for ((k = 0; k < SEEDS; k++)); do
    (
      # a sub-shell per (entry, seed), at most JOBS at a time
      env $settings SEEDS=1 SEED_START=$k "$BIN" >"$OUT/$name.s$k.log" 2>&1 || echo "run failed" >>"$OUT/$name.s$k.log"
    ) &
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  done
done <"$TABLE"
wait

# the policy summary line: "NoMemory  seen pairs  63.1%   held-out pairs  22.3% ..."
result() { grep -m1 "seen pairs" "$OUT/$1.log" | sed -E 's/.*seen pairs +([0-9.]+)%.*held-out pairs +([0-9.]+)%.*/\1 \2/'; }

fail=0
tmp=$(mktemp)
printf '%-26s %5s %16s %16s  %-22s %s\n' entry page "seen (expected)" "held (expected)" "held per seed" status
while IFS= read -r line; do
  IFS=$'\t' read -r name page seen held settings <<<"$line"
  if [[ -z "$name" || "$name" == \#* ]] || [[ ! " ${entries[*]} " == *" $name "* ]]; then
    printf '%s\n' "$line" >>"$tmp"; continue
  fi
  # the mean over seeds, and each seed's held-out figure
  ss=(); hs=(); missing=0
  for ((k = 0; k < SEEDS; k++)); do
    read -r a b <<<"$(result "$name.s$k" || true)"
    if [[ -z "${a:-}" ]]; then missing=1; else ss+=("$a"); hs+=("$b"); fi
  done
  if (( missing )); then
    s=""; h=""; per="-"
  else
    s=$(printf '%s\n' "${ss[@]}" | awk '{t += $1} END {printf "%.1f", t / NR}')
    h=$(printf '%s\n' "${hs[@]}" | awk '{t += $1} END {printf "%.1f", t / NR}')
    per=$(IFS=/; echo "${hs[*]}")
  fi
  if [[ -z "$s" ]]; then
    status="NO RESULT (see $OUT/$name.s*.log)"; fail=1
  elif (( QUICK )); then
    status="quick (seed 0 only)"
  elif (( RECORD )); then
    status="recorded"; seen=$s; held=$h
  elif [[ "$seen" == "-" ]]; then
    status="no expected figure (run --record)"
  else
    status=$(awk -v s="$s" -v h="$h" -v es="$seen" -v eh="$held" -v m="$MARGIN" 'BEGIN {
      if (s < es - m || h < eh - m) print "FAIL"; else if (s != es || h != eh) print "changed (within margin)"; else print "same" }')
    [[ "$status" == FAIL ]] && fail=1
  fi
  printf '%-26s %5s %7s (%6s) %7s (%6s)  %-22s %s\n' "$name" "$page" "${s:--}" "$seen" "${h:--}" "$held" "$per" "$status"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$page" "$seen" "$held" "$settings" >>"$tmp"
done <"$TABLE"
if (( RECORD )); then mv "$tmp" "$TABLE"; else rm -f "$tmp"; fi
exit $fail
