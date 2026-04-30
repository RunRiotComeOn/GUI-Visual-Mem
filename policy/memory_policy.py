"""Memory-augmented GUI policy.

Composes:
  - per-task `MemoryState` (rolling older_summary + recent_buffer + keyframes)
  - cross-task experience selector (Stage-C)
  - GUIPivot policy LLM call

Per-task contract:
    policy.reset(task)
    while not done:
        screenshot = env.observe()
        result = policy.step(screenshot)              # uses screenshot as `current`
        action = result.action_data
        env.execute(action)                           # adapter converts via parsers.convert_to_exec_actions
        policy.commit(action_repr=...)                # tells the policy what was actually done

`commit` and `step` are split so that the adapter retains the freedom to alter
the action between policy proposal and execution (e.g., downgrade to wait when
required coordinates are missing). The next call to `step` will then pass the
*post-execution* screenshot into the change-memory call together with the
previous screenshot and the committed action_repr.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from guimemorysystem import GUIMemorySystem
from policy.engines import OpenAICompatibleEngine
from policy.images import image_size as _image_size
from policy.parsers import (
    parse_guipivot_json,
    parse_plan_guipivot,
)
from policy.prompts import build_policy_messages

logger = logging.getLogger(__name__)


@dataclass
class PolicyStepResult:
    action_data: dict
    thought: str
    operation: str
    raw_response: str
    raw_messages: list[dict]
    history_text: str
    history_image_count: int
    experience_id: str | None
    experience_reason: str
    active_experience_text: str


class MemoryAugmentedPolicy:
    def __init__(
        self,
        *,
        policy_engine: OpenAICompatibleEngine,
        memory_engine: OpenAICompatibleEngine | None = None,
        selector_engine: OpenAICompatibleEngine | None = None,
        experience_catalog: list[dict] | None = None,
        experience_library: dict[str, dict] | None = None,
        recent_k: int = 3,
        last_k_actions: int = 15,
        history_text_char_budget: int = 24000,
        max_policy_tokens: int = 4096,
        use_visual_skill: bool = False,
    ) -> None:
        self.policy_engine = policy_engine
        self.memory_engine = memory_engine
        self.selector_engine = selector_engine
        self.experience_catalog = experience_catalog or []
        self.experience_library = experience_library or {}
        self.recent_k = recent_k
        self.last_k_actions = last_k_actions
        self.history_text_char_budget = history_text_char_budget
        self.max_policy_tokens = max_policy_tokens

        self._task: str | None = None
        self.memory_system = GUIMemorySystem(
            memory_engine=memory_engine,
            selector_engine=selector_engine,
            experience_catalog=experience_catalog,
            experience_library=experience_library,
            recent_k=recent_k,
            use_visual_skill=use_visual_skill,
        )
        # Kept as a public compatibility attribute for existing GUI-Libra
        # adapters that seed static previous action history directly.
        self.previous_actions: list[str] = []

    # ------------------------------------------------------------------ task

    def reset(self, task: str) -> None:
        self._task = task
        self.memory_system.reset(task)
        self.previous_actions = []

    # ------------------------------------------------------------------ step

    def step(
        self,
        screenshot: Any,
        *,
        current_url: str = "",
        current_page_title: str = "",
    ) -> PolicyStepResult:
        if self._task is None:
            raise RuntimeError("MemoryAugmentedPolicy.step called before reset(task)")
        task = self._task
        # External runners may assign policy.previous_actions for static rows.
        # Synchronize that compatibility attribute into GUIMemorySystem before
        # rendering memory or selecting cross-task experience.
        self.memory_system.previous_actions = list(self.previous_actions)
        context = self.memory_system.prepare_step(
            screenshot,
            current_url=current_url,
            current_page_title=current_page_title,
        )
        self.previous_actions = self.memory_system.previous_actions

        img_size = _image_size(screenshot)
        messages = build_policy_messages(
            task=task,
            screenshot=screenshot,
            image_size=img_size,
            previous_actions=self.previous_actions,
            history_text=context.task_memory_text,
            active_experience_text=context.active_experience_text,
            history_images=context.task_memory_images,
            last_k=self.last_k_actions,
            history_text_char_budget=self.history_text_char_budget,
        )

        raw_response = self.policy_engine.chat(messages, max_tokens=self.max_policy_tokens)
        thought, operation, action_json_str = parse_plan_guipivot(raw_response)
        if not action_json_str:
            raise ValueError(
                f"Policy did not return a parseable <answer> JSON. raw={raw_response[:300]!r}"
            )
        action_data = parse_guipivot_json(action_json_str)
        if action_data.get("action_description"):
            operation = action_data["action_description"]
        return PolicyStepResult(
            action_data=action_data,
            thought=thought,
            operation=operation,
            raw_response=raw_response,
            raw_messages=messages,
            history_text=context.task_memory_text,
            history_image_count=context.history_image_count,
            experience_id=context.experience_id,
            experience_reason=context.experience_reason,
            active_experience_text=context.active_experience_text,
        )

    def commit(self, action_repr: str) -> None:
        """Tell the policy what action_repr was actually executed for *this* step.

        Must be called after `step()` and before the next `step()`. The
        recorded `action_repr` is what the change-memory agent will see as
        the cause of the next observed transition, and what the next selector
        call will see as `previous_actions[-1]`.
        """
        repr_str = action_repr or "(no explicit action repr)"
        self.memory_system.previous_actions = list(self.previous_actions)
        self.memory_system.commit_action(repr_str)
        self.previous_actions = self.memory_system.previous_actions
