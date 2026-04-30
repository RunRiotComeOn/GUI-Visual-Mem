"""Docker-gateway Android World runner for the memory-augmented agent."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _setup_paths() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    android_root = repo_root / "evaluation" / "android_world_seeact_v"
    if not android_root.exists():
        raise SystemExit(f"android_world_seeact_v not found at {android_root}")
    for path in (str(android_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root


_setup_paths()

import run_suite_on_docker as _docker_run  # noqa: E402

from eval_adapters.android_world.agent import MemoryAugmentedAndroidWorldAgent  # noqa: E402
from eval_adapters.common import MemoryAdapterConfig  # noqa: E402


def _build_config(args) -> MemoryAdapterConfig:
    return MemoryAdapterConfig.from_mapping(
        {
            "policy_model": args.model,
            "policy_api_base": args.base_url,
            "policy_api_key": os.getenv("OPENAI_API_KEY", "token-abc123"),
            "policy_temperature": args.temperature,
        }
    )


def main() -> None:
    args = _docker_run.get_args()
    os.makedirs(args.output_path, exist_ok=True)

    client = _docker_run.AndroidEnvClient(base_url=args.env_url)
    while not client.health():
        print("Environment is not healthy, waiting for 1 second...")
        time.sleep(1)

    print(f"reset response: {client.reset(go_home=True)}")
    task_list = client.get_suite_task_list(max_index=-1)
    if args.task_index >= 0:
        task_list = task_list[args.task_index:]
    print(task_list)

    print(f"reinitialize_suite response: {client.reinitialize_suite()}")
    agent = MemoryAugmentedAndroidWorldAgent(client, _build_config(args))

    for task_name in task_list:
        num_tasks = client.get_suite_task_length(task_type=task_name)
        print(f"num_tasks: {num_tasks}")
        for cur_idx in range(num_tasks):
            agent.reset(task_id=cur_idx, repeat_id=cur_idx)
            task_template = client.get_task_template(task_type=task_name, task_idx=cur_idx)
            if not ("{" in task_template and "}" in task_template) and cur_idx > 0:
                break

            task_goal = client.get_task_goal(task_type=task_name, task_idx=cur_idx)
            try:
                print(f"initialize_task response: {client.initialize_task(task_type=task_name, task_idx=cur_idx)}")
                print(f"Goal: {task_goal}")
                is_done = False
                for _ in range(args.max_steps):
                    response = agent.step(task_goal)
                    if response.done:
                        is_done = True
                        break
                task_score = client.get_task_score(task_type=task_name, task_idx=cur_idx)
                print(f"task_score: {task_score}")
                print(f"Task {'Successful' if is_done and task_score == 1 else 'Failed'}; {task_goal}")
                client.tear_down_task(task_type=task_name, task_idx=cur_idx)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"Error processing task {task_name} {cur_idx}: {exc}")
            client.reset(go_home=True)

    client.close()


if __name__ == "__main__":
    main()
