"""Entrypoint for Android World with the memory-augmented agent."""
from __future__ import annotations

import os
import sys
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

from absl import app  # noqa: E402
import run as _run_module  # noqa: E402

from eval_adapters.android_world.agent import MemoryAugmentedAndroidWorldAgent  # noqa: E402
from eval_adapters.common import MemoryAdapterConfig  # noqa: E402


def _get_memory_agent(env, family=None):  # noqa: ARG001
    config = MemoryAdapterConfig.from_mapping(
        {
            "policy_model": os.getenv("OPENAI_MODEL", ""),
            "policy_api_base": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "policy_api_key": os.getenv("OPENAI_API_KEY", ""),
            "policy_temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
        }
    )
    return MemoryAugmentedAndroidWorldAgent(env, config)


def main() -> None:
    _run_module._get_agent = _get_memory_agent
    app.run(_run_module._main)


if __name__ == "__main__":
    main()
