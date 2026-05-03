#!/bin/bash
#SBATCH --job-name=guimem_m2w_v4
#SBATCH --account=bgek-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /u/yhuang48/GUIMem

mkdir -p outputs/slurm outputs/offline outputs/visual_skills/mind2web_v4

set -a
source /u/yhuang48/GUIMem/.env
set +a

TRAIN_DATA_DIR="evaluation/offline_evaluation/Multimodal-Mind2Web/data/samples"
VISUAL_SKILL_STORE="outputs/visual_skills/mind2web_v4"

CONFIGS=(
  ""
  "configs/experiments/exp1_baseline_v4.yaml"
  "configs/experiments/exp2_in_task_v4.yaml"
  "configs/experiments/exp3_visual_skill_v4.yaml"
  "configs/experiments/exp4_both_v4.yaml"
)

PRED_FILES=(
  ""
  "outputs/offline/exp1_baseline_v4_predictions_100.jsonl"
  "outputs/offline/exp2_in_task_v4_predictions_100.jsonl"
  "outputs/offline/exp3_visual_skill_v4_predictions_100.jsonl"
  "outputs/offline/exp4_both_v4_predictions_100.jsonl"
)

LABELS=(
  ""
  "exp1_baseline_v4"
  "exp2_in_task_v4"
  "exp3_visual_skill_v4"
  "exp4_both_v4"
)

echo "============================================================"
echo "Step 0: Mining v4 visual skills from training trajectories"
echo "============================================================"

python -m guimemorysystem.visual_skill_memory \
  --version v4 \
  --dataset mind2web \
  --input \
    "${TRAIN_DATA_DIR}/cross_domain_blocks_natural.jsonl" \
    "${TRAIN_DATA_DIR}/cross_task_blocks_natural.jsonl" \
    "${TRAIN_DATA_DIR}/cross_website_blocks_natural.jsonl" \
  --output-dir "${VISUAL_SKILL_STORE}" \
  --min-support 3 \
  --min-tasks 2

echo "Mining complete. Visual skill store: ${VISUAL_SKILL_STORE}"
echo ""

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

# Run all four experiments sequentially.
# To run a subset locally: RUN_TASK_IDS="3 4" bash scripts/run_offline_mind2web_experiments_v4.sbatch
for task_id in ${RUN_TASK_IDS:-1 2 3 4}; do
  run_one "${task_id}"
done
