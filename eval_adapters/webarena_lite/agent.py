"""WebArenaLiteV2 adapter for the memory-augmented policy."""
from __future__ import annotations

import logging
from typing import Any, Mapping

from eval_adapters.common import (
    MemoryAdapterConfig,
    build_policy,
    exec_actions_from_policy_result,
    history_repr_for_exec_actions,
    policy_debug_info,
)

logger = logging.getLogger(__name__)


class MemoryAugmentedWebArenaLiteAgent:
    """Drop-in agent for ``WebArenaLiteV2/agent_run.py``.

    It exposes the same ``reset`` and ``predict(instruction, observation)``
    surface as GUI-Libra's native agents. The observation screenshot is raw PNG
    bytes from ``WebEnv.get_obs()``.
    """

    def __init__(
        self,
        engine_params: Mapping[str, Any],
        platform: str = "web",
        width: int = 1600,
        height: int = 2560,
    ) -> None:
        self.config = MemoryAdapterConfig.from_mapping(engine_params)
        self.platform = platform
        self.width = width
        self.height = height
        self.policy = build_policy(self.config)
        self.previous_operations_list: list[str] = []
        self._task: str | None = None

    def reset(self) -> None:
        self.previous_operations_list = []
        self._task = None
        self.policy.reset("")

    def _ensure_task(self, instruction: str) -> None:
        if self._task != instruction:
            self.policy.reset(instruction)
            self.previous_operations_list = []
            self._task = instruction

    def predict(self, instruction: str, observation: dict) -> tuple[dict, list[dict]]:
        self._ensure_task(instruction)
        screenshot = observation["screenshot"]
        current_url = str(observation.get("url", "") or observation.get("current_url", ""))
        title = str(observation.get("title", "") or observation.get("page_title", ""))

        result = self.policy.step(
            screenshot,
            current_url=current_url,
            current_page_title=title,
        )
        exec_actions = exec_actions_from_policy_result(result, screenshot)

        action_repr = result.operation or history_repr_for_exec_actions(exec_actions)
        self.previous_operations_list.append(action_repr)
        self.policy.commit(action_repr)

        info = policy_debug_info(result)
        info["operation"] = action_repr
        info["exec_actions"] = exec_actions
        return info, exec_actions
