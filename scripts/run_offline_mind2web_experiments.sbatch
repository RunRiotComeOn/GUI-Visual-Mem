#!/bin/bash
#SBATCH --job-name=guimem_m2w_exp
#SBATCH --account=bgek-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-4
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
  "configs/experiments/exp1_baseline.yaml"
  "configs/experiments/exp2_in_task.yaml"
  "configs/experiments/exp3_visual_skill.yaml"
  "configs/experiments/exp4_both.yaml"
)

PRED_FILES=(
  ""
  "outputs/offline/exp1_baseline_predictions_100.jsonl"
  "outputs/offline/exp2_in_task_predictions_100.jsonl"
  "outputs/offline/exp3_visual_skill_predictions_100.jsonl"
  "outputs/offline/exp4_both_predictions_100.jsonl"
)

LABELS=(
  ""
  "exp1_baseline"
  "exp2_in_task"
  "exp3_visual_skill"
  "exp4_both"
)

CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
PRED_FILE="${PRED_FILES[${SLURM_ARRAY_TASK_ID}]}"
LABEL="${LABELS[${SLURM_ARRAY_TASK_ID}]}"

echo "Running ${CONFIG}"
python evaluate.py --config "${CONFIG}"

echo ""
echo "Evaluating ${LABEL}"
python evaluation/offline_evaluation/Multimodal-Mind2Web/eval_guimem.py \
  --pred_file "${PRED_FILE}" --label "${LABEL}"
