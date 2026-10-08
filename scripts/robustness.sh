#!/usr/bin/env bash
# Mutational robustness of a genome: how much do single mutations change what it does?
# Runs the base genome and single mutants of it, each over SEEDS seeds on TASK, in two
# forms: the gene list (MUTATE: nudge, rewire, add, duplicate, toggle) and the stack text
# (MUTATE_STACK: number, delete, insert, swap). Prints each mutant's mean held-out accuracy
# and its change from its form's base.
#
#   GENOME=genomes/hierarchy.gen TASK=habit scripts/robustness.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GENOME=${GENOME:-genomes/hierarchy.gen}
TASK=${TASK:-habit}
SEEDS=${SEEDS:-3}
JOBS=${JOBS:-$(nproc)}
BIN=${BIN:-target/release/examples/episodic}
OUT=${OUT:-target/robustness}
GENE_OPS=${GENE_OPS:-"nudge:10 rewire:6 add:3 duplicate:3 toggle:4"}
STACK_OPS=${STACK_OPS:-"number:10 delete:6 insert:5 swap:5"}
mkdir -p "$OUT"

# jobs: "<name>|<env>"
jobs=("genes-base|GENES=1" "stack-base|")
for spec in $GENE_OPS; do op=${spec%:*}; n=${spec#*:}; for ((k = 1; k <= n; k++)); do jobs+=("genes-$op-$k|MUTATE=$op:$k"); done; done
for spec in $STACK_OPS; do op=${spec%:*}; n=${spec#*:}; for ((k = 1; k <= n; k++)); do jobs+=("stack-$op-$k|MUTATE_STACK=$op:$k"); done; done

run() {
  local name=${1%%|*} env=${1#*|} s=$2
  env $env GENOME="$GENOME" TASK="$TASK" POLICIES=nomemory SEEDS=1 SEED_START=$s "$BIN" >"$OUT/$name.s$s.log" 2>&1 || echo "run failed" >>"$OUT/$name.s$s.log"
}
export -f run; export GENOME TASK BIN OUT
for j in "${jobs[@]}"; do for ((s = 0; s < SEEDS; s++)); do echo "$j"$'\t'"$s"; done; done | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r j s <<< "{}"; run "$j" "$s"'

held() { grep -h "seen pairs" "$1" 2>/dev/null | sed -E 's/.*held-out pairs +([0-9.]+)%.*/\1/' | head -1; }
mean() { local name=$1 sum=0 n=0 v; for ((s = 0; s < SEEDS; s++)); do v=$(held "$OUT/$name.s$s.log"); v=${v:-0}; sum=$(echo "$sum + $v" | bc -l); n=$((n + 1)); done; echo "scale=1; $sum / $n" | bc -l; }
gb=$(mean genes-base); sb=$(mean stack-base)
printf '%-22s %7s %7s  %s\n' mutant held change what
printf '%-22s %7s %7s\n' genes-base "$gb" "" ; printf '%-22s %7s %7s\n' stack-base "$sb" ""
for j in "${jobs[@]:2}"; do
  name=${j%%|*}; m=$(mean "$name"); base=$gb; [[ $name == stack-* ]] && base=$sb
  d=$(echo "scale=1; $m - $base" | bc -l)
  what=$(grep -h "MUTANT seed 0" "$OUT/$name.s0.log" 2>/dev/null | sed 's/.*seed 0: //' | head -1)
  printf '%-22s %7s %7s  %s\n' "$name" "$m" "$d" "$what"
done
