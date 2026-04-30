"""Evaluate GUIMem offline Mind2Web predictions (block-selection format)."""
import argparse
import collections
import json
import re
import string
import unicodedata

import numpy as np


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(s):
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def calculate_f1(pred, label):
    pred_tokens = set(normalize_text(pred).split()) - set(string.punctuation)
    label_tokens = set(normalize_text(label).split()) - set(string.punctuation)
    if not pred_tokens and not label_tokens:
        return 1.0
    if not pred_tokens or not label_tokens:
        return 0.0
    tp = len(pred_tokens & label_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(label_tokens)
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(records):
    all_element_acc = []
    all_operation_f1 = []
    all_step_acc = []
    sample_to_website = {}
    seen_annotation_ids = set()

    for rec in records:
        annotation_id = rec["annotation_id"]
        sample_to_website[annotation_id] = rec.get("website", "")
        seen_annotation_ids.add(annotation_id)

        target_blocks = rec.get("target_blocks", {})
        ans_block = rec.get("ans_block", -1)
        element_correct = str(ans_block) in target_blocks

        gpt_action = rec.get("gpt_action", "").lower().strip()
        gpt_value = rec.get("gpt_value", "").lower().strip()
        gt_operation = rec.get("operation", "").lower().strip()
        gt_value = rec.get("value", "").lower().strip()

        action_map = {"write": "type"}
        gpt_action = action_map.get(gpt_action, gpt_action)

        if gpt_action in ("click", "longpress") or gpt_value in ("none", ""):
            pred_text = gpt_action
        else:
            pred_text = f"{gpt_action} {gpt_value}"

        gold_text = f"{gt_operation} {gt_value}".strip()
        f1 = calculate_f1(pred_text, gold_text)

        all_element_acc.append([int(element_correct), annotation_id])
        all_operation_f1.append([f1, annotation_id])
        all_step_acc.append([int(element_correct and f1 == 1.0), annotation_id])

    # Pad missing steps to match total_steps per annotation
    total_steps = {rec["annotation_id"]: rec["total_steps"] for rec in records}
    current_steps = collections.defaultdict(int)
    for _, aid in all_element_acc:
        current_steps[aid] += 1
    for aid, steps in total_steps.items():
        while current_steps[aid] < steps:
            all_element_acc.append([0, aid])
            all_operation_f1.append([0.0, aid])
            all_step_acc.append([0, aid])
            current_steps[aid] += 1

    return {
        "element_acc": np.mean([x[0] for x in all_element_acc]),
        "operation_f1": np.mean([x[0] for x in all_operation_f1]),
        "step_acc": np.mean([x[0] for x in all_step_acc]),
        "n_steps": len(all_element_acc),
        "n_tasks": len(seen_annotation_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GUIMem offline Mind2Web predictions")
    parser.add_argument("--pred_file", required=True, help="Predictions JSONL from run_memory.py")
    parser.add_argument("--label", default="", help="Optional label for display")
    args = parser.parse_args()

    records = load_jsonl(args.pred_file)
    metrics = evaluate(records)

    label = args.label or args.pred_file
    print(f"\n=== {label} ===")
    print(f"  element_acc  : {metrics['element_acc']*100:.2f}%")
    print(f"  operation_f1 : {metrics['operation_f1']*100:.2f}%")
    print(f"  step_acc     : {metrics['step_acc']*100:.2f}%")
    print(f"  steps        : {metrics['n_steps']}  tasks: {metrics['n_tasks']}")


if __name__ == "__main__":
    main()
