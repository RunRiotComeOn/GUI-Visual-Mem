#!/bin/bash
#SBATCH --job-name=guimem_m2w_v3
#SBATCH --account=bgek-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-4%1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

cd /u/yhuang48/GUIMem

mkdir -p outputs/slurm outputs/offline

set -a
source /u/yhuang48/GUIMem/.env
set +a

CONFIGS=(
  ""
  "configs/experiments/exp1_baseline_v3.yaml"
  "configs/experiments/exp2_in_task_v3.yaml"
  "configs/experiments/exp3_visual_skill_v3.yaml"
  "configs/experiments/exp4_both_v3.yaml"
)

PRED_FILES=(
  ""
  "outputs/offline/exp1_baseline_v3_predictions_100.jsonl"
  "outputs/offline/exp2_in_task_v3_predictions_100.jsonl"
  "outputs/offline/exp3_visual_skill_v3_predictions_100.jsonl"
  "outputs/offline/exp4_both_v3_predictions_100.jsonl"
)

LABELS=(
  ""
  "exp1_baseline_v3"
  "exp2_in_task_v3"
  "exp3_visual_skill_v3"
  "exp4_both_v3"
)

run_one() {
  local task_id="$1"
  if [[ "${task_id}" -lt 1 || "${task_id}" -gt 4 ]]; then
    echo "Invalid task id: ${task_id}. Expected 1..4." >&2
    return 2
  fi

  local config="${CONFIGS[${task_id}]}"
  local pred_file="${PRED_FILES[${task_id}]}"
  local label="${LABELS[${task_id}]}"

  echo "============================================================"
  echo "Task ${task_id}: ${label}"
  echo "Config: ${config}"
  echo "Predictions: ${pred_file}"
  echo "============================================================"

  python evaluate.py --config "${config}"

  echo ""
  echo "Evaluating ${label}"
  python evaluation/offline_evaluation/Multimodal-Mind2Web/eval_guimem.py \
    --pred_file "${pred_file}" --label "${label}"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_one "${SLURM_ARRAY_TASK_ID}"
else
  # Local mode: run all four by default, or set RUN_TASK_IDS="3" / "3 4".
  for task_id in ${RUN_TASK_IDS:-1 2 3 4}; do
    run_one "${task_id}"
  done
fi
