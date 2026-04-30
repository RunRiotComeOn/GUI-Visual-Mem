"""Entrypoint for WebArenaLiteV2 with the memory-augmented agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_paths() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    webarena_root = repo_root / "evaluation" / "WebArenaLiteV2"
    if not webarena_root.exists():
        raise SystemExit(f"WebArenaLiteV2 not found at {webarena_root}")
    for path in (str(webarena_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root


_setup_paths()

import yaml  # noqa: E402
import agent_run as _agent_run  # noqa: E402
from envs.web.web_env import WebEnv  # noqa: E402

from eval_adapters.webarena_lite.agent import MemoryAugmentedWebArenaLiteAgent  # noqa: E402


def init_env_and_memory_agent(args):
    with open(args.env_config_path, "r", encoding="utf-8") as f:
        env_config = yaml.safe_load(f)
    env = WebEnv(**env_config)

    with open(args.agent_config_path, "r", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)

    model_config = dict(agent_config.get("model_config", {}))
    screen_width, screen_height = env.screen_size
    agent = MemoryAugmentedWebArenaLiteAgent(
        model_config,
        width=screen_width,
        height=screen_height,
    )
    return env, agent


def main() -> None:
    _agent_run.init_env_and_agent = init_env_and_memory_agent
    _agent_run.main()


if __name__ == "__main__":
    main()
