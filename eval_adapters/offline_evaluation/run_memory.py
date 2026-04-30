"""Static offline-evaluation runner for AndroidControl and Multimodal-Mind2Web."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


def _setup_paths() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_setup_paths()

from eval_adapters.common import MemoryAdapterConfig, build_policy  # noqa: E402


ANDROID_ACTION_HINT = """Android action space additions:
- OpenApp: value is the app name.
- NavigateBack: press the Android back button.
- NavigateHome: go to the home screen.
- LongPress: press and hold at point_2d.
- Terminate: finish the task when complete."""


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_action_fields(data: dict) -> tuple[str, str, str]:
    description = _field_text(data.get("action_description") or data.get("action_target"))
    action_type = _field_text(data.get("action_type"))
    value = _field_text(data.get("value"))
    return description, action_type, "" if value == "None" else value


def _policy_for_row(config: MemoryAdapterConfig, instruction: str, previous_actions: list[str]):
    policy = build_policy(config)
    policy.reset(instruction)
    policy.previous_actions = [str(item) for item in previous_actions]
    return policy


def run_android_control(args, config: MemoryAdapterConfig) -> None:
    rows = _load_json_or_jsonl(args.input_file)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="android_control"):
            instruction = row["goal"] if args.level == "high" else row["step_instruction"]
            screenshot_path = os.path.join(args.screenshot_dir, row["screenshot"])
            image = Image.open(screenshot_path).convert("RGB")
            policy = _policy_for_row(
                config,
                f"{instruction}\n\n{ANDROID_ACTION_HINT}",
                row.get("previous_actions", []) or [],
            )
            result = policy.step(image)
            description, action_type, value = _extract_action_fields(result.action_data)
            record = {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "instruction": instruction,
                "action": description,
                "action_type": action_type,
                "element_description": _field_text(result.action_data.get("action_target")),
                "value": value,
                "reason": result.thought,
                "response": result.raw_response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_multimodal_mind2web(args, config: MemoryAdapterConfig) -> None:
    rows = _load_json_or_jsonl(args.input_file)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Group rows by task (annotation_id) and sort steps sequentially so that
    # a single policy instance can accumulate in-task memory across steps.
    from collections import defaultdict
    tasks_map: dict[str, list] = defaultdict(list)
    task_order: list[str] = []
    for row in rows:
        aid = row["annotation_id"]
        if aid not in tasks_map:
            task_order.append(aid)
        tasks_map[aid].append(row)
    for aid in tasks_map:
        tasks_map[aid].sort(key=lambda r: r.get("step", 0))

    with open(args.output_file, "w", encoding="utf-8") as out:
        for aid in tqdm(task_order, desc="multimodal_mind2web"):
            task_rows = tasks_map[aid]
            task_desc = task_rows[0]["task"]

            # One policy per task: reset once, then step+commit for each step.
            policy = build_policy(config)
            policy.reset(task_desc)

            for row in task_rows:
                # Seed previous_actions from dataset ground truth so the
                # action-history context is consistent with the baseline.
                # In-task memory accumulates separately via commit().
                previous_actions = row.get("previous_actions_descriptions", row.get("previous_actions", [])) or []
                policy.previous_actions = [str(a) for a in previous_actions]

                target_blocks = row.get("target_blocks", []) or []
                max_target = max([int(item) for item in target_blocks], default=-1)
                block_num = 0
                result = None
                image = None
                while True:
                    image_path = os.path.join(args.blocks, row["blocks_path"], f"{block_num}.png")
                    if not os.path.exists(image_path):
                        break
                    image = Image.open(image_path).convert("RGB")
                    result = policy.step(image)
                    _, action_type, value = _extract_action_fields(result.action_data)
                    next_block = os.path.join(args.blocks, row["blocks_path"], f"{block_num + 1}.png")
                    if action_type.lower() != "scroll" or value.lower() != "down" or block_num > max_target or not os.path.exists(next_block):
                        break
                    block_num += 1

                if result is None or image is None:
                    continue
                description, action_type, value = _extract_action_fields(result.action_data)
                width, height = image.size

                # Commit the final action so in-task memory updates before the
                # next step's prepare_step() call.
                action_repr = _field_text(result.action_data.get("action_target") or description) or f"{action_type} {value}".strip()
                policy.commit(action_repr)

                record = dict(row)
                record.update(
                    {
                        "ans_block": block_num,
                        "gpt_action": action_type,
                        "gpt_value": value,
                        "description": _field_text(result.action_data.get("action_target") or description),
                        "response": result.raw_response,
                        "reason": result.thought,
                        "image_width": width,
                        "image_height": height,
                    }
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory policy on GUI-Libra offline benches.")
    parser.add_argument("--benchmark", choices=["android_control", "multimodal_mind2web"], required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--screenshot_dir", default="")
    parser.add_argument("--blocks", default="")
    parser.add_argument("--level", choices=["high", "low"], default="high")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--api_base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MemoryAdapterConfig.from_mapping(
        {
            "policy_model": args.model,
            "policy_api_base": args.api_base,
            "policy_api_key": args.api_key,
            "policy_temperature": args.temperature,
        }
    )
    if args.benchmark == "android_control":
        if not args.screenshot_dir:
            raise SystemExit("--screenshot_dir is required for android_control")
        run_android_control(args, config)
    else:
        if not args.blocks:
            raise SystemExit("--blocks is required for multimodal_mind2web")
        run_multimodal_mind2web(args, config)


if __name__ == "__main__":
    main()
