"""Online-Mind2Web entrypoint for the memory-augmented agent.

This is a thin shim that:
  1. Adds evaluation/online-mind2web-eval/ to sys.path so we can
     import its `agent`, `syn`, and `run` modules.
  2. Adds the GUIMem root to sys.path so local packages (`policy`,
     `guimemorysystem`, `eval_adapters`)
     are importable.
  3. Monkey-patches the GUI-Libra ``run`` module so that
     - ``run.AgentConfig`` is replaced by our ``MemoryAgentConfig`` (lets
       ``MultiAgent.__init__`` reconstruct per-process configs with our extra
       memory/selector fields), and
     - ``run.run_single_agent`` is replaced by a worker that constructs
       ``MemoryAugmentedAgent`` instead of the base ``Agent``.
  4. Delegates everything else to GUI-Libra's existing ``MultiAgent`` orchestrator.
"""
import copy
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _setup_paths() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    omm2w_root = repo_root / "evaluation" / "online-mind2web-eval"
    if not omm2w_root.exists():
        raise SystemExit(f"online-mind2web-eval not found at {omm2w_root}")
    for path in (str(omm2w_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root


_setup_paths()


from loguru import logger  # noqa: E402

import run as _run_module  # noqa: E402  (GUI-Libra/online-mind2web-eval/run.py)
from simpleArgParser import parse_args  # noqa: E402

from eval_adapters.online_mind2web.agent import (  # noqa: E402
    MemoryAgentConfig,
    MemoryAugmentedAgent,
)


@dataclass
class MemoryMultiAgentConfig(MemoryAgentConfig):
    num_processes: int = field(default=4, kw_only=True)

    def pre_process(self):
        super().pre_process()
        self.num_processes = max(1, self.num_processes)

    def __repr__(self) -> str:
        return MemoryAgentConfig.__repr__(self)


def run_single_agent_memory(idx: int, shared_configs: list[MemoryAgentConfig]) -> None:
    config = shared_configs[idx]
    if idx != 0:
        logger.remove()
    log_file = f"{config.output}/run.log"
    logger.add(
        log_file,
        format=(
            "<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>"
        ),
        level="DEBUG", colorize=False, rotation=None,
    )
    logger.info(f"Running MemoryAgent={idx}/{len(shared_configs)-1} with config=\n{config}")
    agent = MemoryAugmentedAgent(config)
    agent.run_episode()
    logger.info(f"MemoryAgent={idx}/{len(shared_configs)-1} finished with output={config.output}")


def run_monitoring_process_memory(
    multi_config: MemoryMultiAgentConfig,
    shared_configs: list[MemoryAgentConfig],
    interval_minutes: int = 10,
) -> None:
    # The upstream monitor reconstructs AgentConfig inside a spawned process,
    # which drops our memory-only fields. Final result gathering still runs in
    # the parent, so a passive monitor is enough for the adapter path.
    logger.remove()
    log_file = f"{multi_config.output}/run_monitor.log"
    logger.add(
        log_file,
        format=(
            "<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>"
        ),
        level="DEBUG", colorize=False, rotation=None,
    )
    logger.info("Memory adapter monitor active; parent process gathers final results")
    while True:
        time.sleep(max(1, interval_minutes) * 60)


def main() -> None:
    mp.set_start_method("spawn", force=True)

    # Patch the GUI-Libra run module so its MultiAgent uses our types/workers.
    _run_module.AgentConfig = MemoryAgentConfig
    _run_module.run_single_agent = run_single_agent_memory
    _run_module.run_monitoring_process = run_monitoring_process_memory

    args: MemoryMultiAgentConfig = parse_args(MemoryMultiAgentConfig)
    multiagent = _run_module.MultiAgent(args)

    retry_cnt = 3
    error_code = 0
    while retry_cnt > 0:
        retry_cnt -= 1
        error_code = multiagent.run_episode()
        if error_code == 0:
            break
        logger.error(f"run_episode failed with error_code={error_code}, {retry_cnt} retries left")
    logger.info(f"MultiAgent finished with error_code={error_code}")


if __name__ == "__main__":
    main()
