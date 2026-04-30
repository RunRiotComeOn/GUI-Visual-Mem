"""Android World adapter for the memory-augmented policy."""
from __future__ import annotations

import os
import time
from typing import Any, Mapping

from eval_adapters.common import (
    MemoryAdapterConfig,
    build_policy,
    exec_actions_from_policy_result,
    history_repr_for_exec_actions,
    policy_debug_info,
)
from policy.parsers import normalize_action_type

# Imported from GUI-Libra/android_world once the run shim places that benchmark
# directory on sys.path.
from android_world.agents import base_agent
from android_world.env import json_action


ANDROID_ACTION_HINT = """Android action space additions:
- OpenApp: value is the app name.
- NavigateBack: press the Android back button.
- NavigateHome: go to the home screen.
- LongPress: press and hold at point_2d.
- Terminate: finish the task when complete."""


def _android_action_from_policy(data: dict, exec_actions: list[dict]) -> json_action.JSONAction:
    action_type = normalize_action_type(data.get("action_type", ""))
    value = data.get("value", "")
    if value == "None" or value is None:
        value = ""

    primary = exec_actions[0] if exec_actions else {"name": "wait", "parameters": {"seconds": 1}}
    name = primary.get("name", "")
    params = primary.get("parameters", {})

    if action_type in ("Click", "Select") or name == "click":
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            x=params.get("x"),
            y=params.get("y"),
        )
    if action_type == "LongPress" or name == "long_press":
        return json_action.JSONAction(
            action_type=json_action.LONG_PRESS,
            x=params.get("x"),
            y=params.get("y"),
        )
    if action_type == "Write" or name == "write":
        click = next((item for item in exec_actions if item.get("name") == "click"), None)
        click_params = click.get("parameters", {}) if click else {}
        return json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            text=str(value),
            x=click_params.get("x"),
            y=click_params.get("y"),
        )
    if action_type == "KeyboardPress" or name == "press":
        keys = str(value or params.get("keys", "")).lower()
        if keys in ("enter", "return"):
            return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)
        return json_action.JSONAction(action_type=json_action.UNKNOWN)
    if action_type == "Scroll" or name == "swipe":
        direction = str(value or params.get("direction", "down")).lower()
        if direction not in ("up", "down", "left", "right"):
            direction = "down"
        return json_action.JSONAction(action_type=json_action.SCROLL, direction=direction)
    if action_type in ("Back", "NavigateBack") or name == "back":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
    if action_type == "NavigateHome":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
    if action_type == "OpenApp" or name == "open_app":
        return json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=str(value or params.get("app_name", "")))
    if action_type == "Answer" or name == "response":
        answer = str(value or params.get("answer", ""))
        return json_action.JSONAction(action_type=json_action.ANSWER, text=answer)
    if action_type == "terminate" or name == "terminate":
        return json_action.JSONAction(action_type=json_action.STATUS, goal_status="complete")
    if action_type == "wait" or name == "wait":
        return json_action.JSONAction(action_type=json_action.WAIT)
    return json_action.JSONAction(action_type=json_action.UNKNOWN)


class MemoryAugmentedAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
    """Drop-in Android World agent that executes ``MemoryAugmentedPolicy``."""

    def __init__(
        self,
        env,
        config: Mapping[str, Any] | MemoryAdapterConfig,
        name: str = "memory_augmented",
        wait_after_action_seconds: float = 2.0,
    ) -> None:
        super().__init__(env, name)
        self.config = config if isinstance(config, MemoryAdapterConfig) else MemoryAdapterConfig.from_mapping(config)
        self.policy = build_policy(self.config)
        self.history: list[dict[str, Any]] = []
        self.wait_after_action_seconds = wait_after_action_seconds
        self.task_idx: int | None = None

    @classmethod
    def from_env(
        cls,
        env,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> "MemoryAugmentedAndroidWorldAgent":
        data = {
            "policy_model": model or os.getenv("OPENAI_MODEL", ""),
            "policy_api_base": api_base or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "policy_api_key": api_key or os.getenv("OPENAI_API_KEY", ""),
            "policy_temperature": temperature,
        }
        return cls(env, data)

    def reset(self, task_id: int | None = None, repeat_id: int | None = None, go_home_on_reset: bool = False):
        if task_id is None:
            super().reset(go_home_on_reset)
        else:
            super().reset(go_home_on_reset)
        self.task_idx = task_id
        self.history = []
        self.policy.reset("")

    def _ensure_task(self, goal: str) -> None:
        policy_goal = f"{goal}\n\n{ANDROID_ACTION_HINT}"
        if not self.policy._task or self.policy._task != policy_goal:  # noqa: SLF001 - adapter owns policy lifecycle.
            self.policy.reset(policy_goal)
            self.history = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._ensure_task(goal)
        step_data: dict[str, Any] = {
            "raw_screenshot": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "summary": None,
        }

        before_screenshot = self.env.get_screenshot()
        step_data["raw_screenshot"] = before_screenshot.copy()

        result = self.policy.step(before_screenshot)
        exec_actions = exec_actions_from_policy_result(result, before_screenshot)
        converted_action = _android_action_from_policy(result.action_data, exec_actions)

        action_repr = result.operation or history_repr_for_exec_actions(exec_actions)
        self.policy.commit(action_repr)

        step_data.update(policy_debug_info(result))
        step_data["action"] = action_repr
        step_data["action_reason"] = result.thought
        step_data["action_output"] = result.raw_response
        step_data["action_output_json"] = converted_action

        if converted_action.action_type == json_action.STATUS:
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == json_action.ANSWER:
            self.env.execute_action(converted_action)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        try:
            self.env.execute_action(converted_action)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            step_data["summary"] = f"Failed to execute action: {exc}"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        time.sleep(self.wait_after_action_seconds)
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
