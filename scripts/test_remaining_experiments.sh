#!/usr/bin/env bash
#
# Run the experiments that don't yet have full model coverage on the website:
#
#   public_goods       (0/21 models tested)
#   centipede_game     (1/21)
#   travellers_dilemma (1/21)
#   trust_game         (1/21)
#   stag_hunt          (3/21)
#
# Skips any (experiment, model) pair whose dashboard JSON already exists, so
# you can re-run safely after partial completion. Continues past per-run
# failures and writes a summary at the end.
#
# Env overrides:
#   ONLY_MODEL=gpt-4o            run for just one model
#   ONLY_EXPERIMENT=public_goods run just one experiment
#   FORCE=1                      re-run even if the output JSON exists
#   DRY_RUN=1                    print the commands without executing
#
# Note: transitivity.py and risk.py are "coming soon" stubs and are skipped.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENTS=(public_goods centipede_game travellers_dilemma trust_game stag_hunt)

# Pull the registered model list directly from the website's source of truth.
mapfile -t MODELS < <(python3 -c "
import json
for m in json.load(open('web/data/models.json')):
    print(m)
")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Could not load model list from web/data/models.json" >&2
  exit 1
fi

LOG_DIR="data/results/runs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/test_remaining_$(date +%Y%m%d_%H%M%S).log"

ONLY_MODEL="${ONLY_MODEL:-}"
ONLY_EXPERIMENT="${ONLY_EXPERIMENT:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

total=0; ran=0; skipped=0; failed=0
failed_list=()

log() { echo "$@" | tee -a "$RUN_LOG"; }

log "Run log: $RUN_LOG"
log "Models    : ${#MODELS[@]}"
log "Experiments: ${EXPERIMENTS[*]}"
[[ -n "$ONLY_MODEL"      ]] && log "Filter ONLY_MODEL=$ONLY_MODEL"
[[ -n "$ONLY_EXPERIMENT" ]] && log "Filter ONLY_EXPERIMENT=$ONLY_EXPERIMENT"
[[ "$FORCE"   == "1"     ]] && log "FORCE=1 (re-running existing results)"
[[ "$DRY_RUN" == "1"     ]] && log "DRY_RUN=1 (no commands will execute)"
log ""

for exp in "${EXPERIMENTS[@]}"; do
  [[ -n "$ONLY_EXPERIMENT" && "$exp" != "$ONLY_EXPERIMENT" ]] && continue
  for model in "${MODELS[@]}"; do
    [[ -n "$ONLY_MODEL" && "$model" != "$ONLY_MODEL" ]] && continue

    total=$((total + 1))
    label="$exp / $model"
    out_file="web/data/${exp}_experiment_${model}.json"

    if [[ "$FORCE" != "1" && -f "$out_file" ]]; then
      log "[SKIP] $label  (exists: $out_file)"
      skipped=$((skipped + 1))
      continue
    fi

    log "[RUN ] $label"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "       would run: python src/tasks/${exp}.py --model $model"
      ran=$((ran + 1))
      continue
    fi

    if ECONBENCH_EXPERIMENT="$exp" \
       python src/tasks/"${exp}".py --model "$model" >> "$RUN_LOG" 2>&1; then
      log "[OK  ] $label"
      ran=$((ran + 1))
    else
      rc=$?
      log "[FAIL] $label  (exit=$rc, see $RUN_LOG)"
      failed=$((failed + 1))
      failed_list+=("$label")
    fi
  done
done

log ""
log "----- Summary -----"
log "Total : $total"
log "Ran   : $ran"
log "Skip  : $skipped"
log "Fail  : $failed"

if (( ${#failed_list[@]} > 0 )); then
  log ""
  log "Failed runs:"
  for f in "${failed_list[@]}"; do
    log "  - $f"
  done
fi

# Refresh aggregate stats whenever we produced new data.
if (( ran > 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  log ""
  log "Recalculating rationality stats..."
  if python src/tools/calculate_rationality_stats.py >> "$RUN_LOG" 2>&1; then
    log "[OK  ] calculate_rationality_stats.py"
  else
    log "[WARN] calculate_rationality_stats.py failed (see $RUN_LOG)"
  fi
fi

(( failed == 0 ))
