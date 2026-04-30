"""Online-Mind2Web adapter for the memory-augmented policy.

Subclasses ``GUI-Libra/evaluation/online-mind2web-eval/agent.py:Agent`` and
overrides only ``_cot_step`` to route the per-step decision through our
``MemoryAugmentedPolicy``. Every other piece — playwright env init, mind2web
result writer, browser recovery, multi-process orchestration — is reused
unchanged from the base class.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, fields, is_dataclass

from loguru import logger

# `agent` and `syn` come from the GUI-Libra repo. Caller is expected to set
# PYTHONPATH so this directory is importable, e.g. via the run.py shim below.
from agent import Agent as _BaseAgent
from syn.args import AgentConfig
from syn.data import (
    Action,
    ActionType,
    HighLevelTask,
    LowLevelTask,
    LowTaskStatus,
    StateInfo,
)

from policy import MemoryAugmentedPolicy, OpenAICompatibleEngine
from guimemorysystem import load_catalog, load_library_by_id
from guimemorysystem.online_experience import (
    OnlineExperienceStoreConfig,
    update_online_experience_store,
)
from eval_adapters.online_mind2web.stealth_env import (
    DEFAULT_USER_AGENT,
    StealthScriptBrowserEnv,
)

_BLOCKED_FINAL_RE = re.compile(
    r"\b("
    r"access denied|permission denied|blocked|inaccessible|captcha|cloudflare|"
    r"verify you are human|verification failed|login wall|paywall"
    r")\b",
    re.IGNORECASE,
)


def _safe_dataclass_repr(obj) -> str:
    """Dataclass repr that masks API keys before configs hit logs."""
    parts = []
    for item in fields(obj):
        value = getattr(obj, item.name)
        if "api_key" in item.name:
            value_repr = repr("[REDACTED]" if value else "")
        elif is_dataclass(value):
            value_repr = _safe_dataclass_repr(value)
        else:
            value_repr = repr(value)
        parts.append(f"{item.name}={value_repr}")
    return f"{obj.__class__.__name__}({', '.join(parts)})"


@dataclass
class MemoryAgentConfig(AgentConfig):
    """Extends the base ``AgentConfig`` with memory-side options.

    All extra knobs are keyword-only so the base parser keeps working.
    """

    # Memory engine (change-memory + summarizer). Empty model disables it.
    memory_model: str = field(default="", kw_only=True)
    memory_api_base: str = field(default="https://api.openai.com/v1", kw_only=True)
    memory_api_key: str = field(default="", kw_only=True)
    memory_rate_limit: int = field(default=-1, kw_only=True)
    memory_recent_k: int = field(default=3, kw_only=True)

    # Selector engine (Stage-C). Empty paths disable it.
    selector_model: str = field(default="", kw_only=True)
    selector_api_base: str = field(default="https://api.openai.com/v1", kw_only=True)
    selector_api_key: str = field(default="", kw_only=True)
    selector_rate_limit: int = field(default=-1, kw_only=True)
    experience_catalog_path: str = field(default="", kw_only=True)
    experience_library_path: str = field(default="", kw_only=True)

    # Online Stage-A/B writer. Enabled only when online_experience_enabled is
    # true; successful trajectories are summarized and upserted into these
    # files. Online records intentionally omit supporting_trajectories.
    online_experience_enabled: bool = field(default=False, kw_only=True)
    online_experience_model: str = field(default="", kw_only=True)
    online_experience_api_base: str = field(default="https://api.openai.com/v1", kw_only=True)
    online_experience_api_key: str = field(default="", kw_only=True)
    online_experience_rate_limit: int = field(default=12, kw_only=True)
    online_experience_summary_buffer_path: str = field(
        default="outputs/cross_task_experience/summary_buffer_online.jsonl",
        kw_only=True,
    )
    online_experience_library_path: str = field(
        default="outputs/cross_task_experience/experience_library_online.jsonl",
        kw_only=True,
    )
    online_experience_catalog_path: str = field(
        default="outputs/cross_task_experience/catalog_online.json",
        kw_only=True,
    )
    online_experience_catalog_cap: int = field(default=100, kw_only=True)
    online_experience_stage_a_max_tokens: int = field(default=1800, kw_only=True)
    online_experience_stage_b_max_tokens: int = field(default=1800, kw_only=True)

    # Browser hardening for public websites that are sensitive to bare
    # Playwright contexts. This does not solve interactive CAPTCHA puzzles; it
    # only uses normal browser headers/profile signals and clicks simple
    # consent/verification controls when visible.
    browser_stealth: bool = field(default=True, kw_only=True)
    browser_user_agent: str = field(default=DEFAULT_USER_AGENT, kw_only=True)
    browser_locale: str = field(default="en-US", kw_only=True)
    browser_timezone_id: str = field(default="America/New_York", kw_only=True)
    browser_accept_language: str = field(default="en-US,en;q=0.9", kw_only=True)
    browser_storage_state_path: str = field(default="", kw_only=True)
    browser_auto_handle_blockers: bool = field(default=True, kw_only=True)

    def __repr__(self) -> str:
        return _safe_dataclass_repr(self)


class MemoryAugmentedAgent(_BaseAgent):
    """Online-Mind2Web Agent that consults our memory + experience layers."""

    def __init__(self, config: MemoryAgentConfig):
        super().__init__(config)
        self._memory_config: MemoryAgentConfig = config

        policy_engine = OpenAICompatibleEngine(
            model=config.gpt.model,
            api_base=config.gpt.openai_api_base,
            api_key=config.gpt.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
            temperature=float(getattr(config.gpt, "temperature", 0.0)),
        )

        memory_engine = None
        if config.memory_model:
            memory_engine = OpenAICompatibleEngine(
                model=config.memory_model,
                api_base=config.memory_api_base,
                api_key=config.memory_api_key or os.getenv("OPENAI_API_KEY", ""),
                rate_limit=config.memory_rate_limit,
            )

        selector_engine = None
        experience_catalog = None
        experience_library = None
        if (
            config.selector_model
            and config.experience_catalog_path
            and config.experience_library_path
        ):
            selector_engine = OpenAICompatibleEngine(
                model=config.selector_model,
                api_base=config.selector_api_base,
                api_key=config.selector_api_key or os.getenv("OPENAI_API_KEY", ""),
                rate_limit=config.selector_rate_limit,
            )
            experience_catalog = load_catalog(config.experience_catalog_path)
            experience_library = load_library_by_id(config.experience_library_path)
            logger.info(
                "loaded experience catalog (n=%d) and library (n=%d)",
                len(experience_catalog), len(experience_library),
            )

        self.policy = MemoryAugmentedPolicy(
            policy_engine=policy_engine,
            memory_engine=memory_engine,
            selector_engine=selector_engine,
            experience_catalog=experience_catalog,
            experience_library=experience_library,
            recent_k=config.memory_recent_k,
            last_k_actions=config.history_last_k or 15,
        )
        self._policy_task: str | None = None

        self._online_experience_engine = None
        self._online_experience_store_config = None
        if config.online_experience_enabled:
            online_model = (
                config.online_experience_model
                or config.selector_model
                or config.memory_model
                or config.gpt.model
            )
            if online_model:
                self._online_experience_engine = OpenAICompatibleEngine(
                    model=online_model,
                    api_base=config.online_experience_api_base,
                    api_key=config.online_experience_api_key or os.getenv("OPENAI_API_KEY", ""),
                    rate_limit=config.online_experience_rate_limit,
                    temperature=0.0,
                )
                self._online_experience_store_config = OnlineExperienceStoreConfig(
                    summary_buffer_path=config.online_experience_summary_buffer_path,
                    experience_library_path=config.online_experience_library_path,
                    catalog_path=config.online_experience_catalog_path,
                    catalog_cap=config.online_experience_catalog_cap,
                    max_stage_a_tokens=config.online_experience_stage_a_max_tokens,
                    max_stage_b_tokens=config.online_experience_stage_b_max_tokens,
                )
                logger.info(
                    "online experience writer enabled: summary=%s library=%s catalog=%s",
                    config.online_experience_summary_buffer_path,
                    config.online_experience_library_path,
                    config.online_experience_catalog_path,
                )
            else:
                logger.warning("online_experience_enabled=True but no model was provided; writer disabled")

    # ------------------------------------------------------------------ utils

    def _init_env_for_episode(self, start_url: str):
        if not self._memory_config.browser_stealth:
            return super()._init_env_for_episode(start_url)

        env = StealthScriptBrowserEnv(
            headless=self.config.browser.headless,
            slow_mo=self.config.browser.slow_mo,
            observation_type=self.config.browser.observation_type,
            current_viewport_only=self.config.browser.current_viewport_only,
            viewport_size=self.config.browser.viewport_size,
            sleep_after_execution=self.config.sleep_after_action,
            user_agent=self._memory_config.browser_user_agent,
            locale=self._memory_config.browser_locale,
            timezone_id=self._memory_config.browser_timezone_id,
            accept_language=self._memory_config.browser_accept_language,
        )
        self._env = env
        observation, info = self._reset_env(env, start_url=start_url)
        observation_metadata = info["observation_metadata"]
        current_state = self._get_env_state(env, obs=observation, observation_metadata=observation_metadata)
        return env, current_state

    def _reset_env(self, env, start_url: str):
        storage_state = self._memory_config.browser_storage_state_path or None
        if storage_state and not os.path.exists(storage_state):
            logger.warning("browser_storage_state_path does not exist: %s", storage_state)
            storage_state = None

        with open(f"{self.config.output}/init_env.json", "w") as f:
            state = {"start_url": start_url, "storage_state": storage_state}
            json.dump(state, f)
            logger.info(f"Resetting environment with state: {state}")
        observation, info = env.reset(options={"config_file": f"{self.config.output}/init_env.json"})
        env.context.set_default_timeout(30000)
        env.context.set_default_navigation_timeout(60000)

        if self._memory_config.browser_auto_handle_blockers:
            if self._maybe_handle_common_blockers(env):
                observation = env._get_obs()
                info["observation_metadata"] = env._get_obs_metadata()
        return observation, info

    def _maybe_handle_common_blockers(self, env) -> bool:
        changed = False
        page = env.page

        # Consent banners are not part of the task and often cover useful UI.
        consent_patterns = [
            r"^(accept|accept all|allow all|agree|i agree|got it|ok)$",
            r"(accept|allow|agree).*(cookie|cookies|privacy|consent)",
            r"(cookie|cookies|privacy|consent).*(accept|allow|agree)",
        ]
        for pattern in consent_patterns:
            try:
                button = page.get_by_role("button", name=re.compile(pattern, re.I)).first
                if button.count() > 0 and button.is_visible(timeout=1000):
                    button.click(timeout=1500)
                    time.sleep(1.0)
                    changed = True
                    break
            except Exception:
                continue

        # Cloudflare Turnstile is usually inside an iframe that is invisible to
        # the accessibility tree. Click the iframe center once; if it is an
        # automatic check, the later prompt can wait.
        try:
            iframe = page.locator(
                "iframe[title*='Cloudflare'], "
                "iframe[title*='challenge'], "
                "iframe[src*='challenges.cloudflare.com']"
            ).first
            if iframe.count() > 0 and iframe.is_visible(timeout=1000):
                box = iframe.bounding_box(timeout=1500)
                if box:
                    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    time.sleep(5.0)
                    changed = True
        except Exception:
            pass

        return changed

    def _sync_committed_actions(self, previous_traj: list[LowLevelTask]) -> None:
        """Replay any uncommitted entries from ``previous_traj`` into the policy.

        Uses each ``LowLevelTask.task`` field — that is the (possibly
        downgraded) action repr the bench actually executed.
        """
        already = len(self.policy.previous_actions)
        for low_task in previous_traj[already:]:
            repr_str = low_task.task or "(no explicit action repr)"
            self.policy.commit(action_repr=repr_str)

    # --------------------------------------------------------------- override

    def _cot_step(
        self,
        task: str,
        current_state: StateInfo,
        previous_traj: list[LowLevelTask],
    ) -> LowLevelTask:
        from syn.prompts import (
            convert_guipivot_to_exec_actions,
            guipivot_action_to_mind2web_str,
        )
        from syn.tools import tools_ndarray_to_base64_image_raw
        from syn.args import is_qwen25_model

        failed_low_level_task = LowLevelTask(
            task="failed during cot_step",
            action=Action(element=None, action_type=ActionType.STOP, value="error during cot_step"),
            curr_state=current_state,
            task_status=LowTaskStatus.IN_PROGRESS,
        )

        screenshot = current_state.raw_state.screenshot
        if screenshot is None:
            error_msg = "policy requires screenshot but none available"
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        if self._policy_task != task:
            self.policy.reset(task)
            self._policy_task = task
        self._sync_committed_actions(previous_traj)

        try:
            result = self.policy.step(screenshot)
        except Exception as exc:
            error_msg = f"policy.step failed: {exc}"
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        # Coordinate scaling matches the base class. Online-Mind2Web treats
        # ``is_qwen25_model`` as the smart-resize signal; for our gpt-4o style
        # policy that's False, so coords are interpreted as 0..1000.
        _, (image_width, image_height) = tools_ndarray_to_base64_image_raw(screenshot)
        use_smart_resize = is_qwen25_model(self._memory_config.gpt.model)

        try:
            exec_actions = convert_guipivot_to_exec_actions(
                data=result.action_data,
                screen_width=image_width,
                screen_height=image_height,
                use_smart_resize=use_smart_resize,
            )
        except Exception as exc:
            error_msg = f"convert_guipivot_to_exec_actions failed: {exc}"
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        if not exec_actions:
            error_msg = f"No executable actions from response: {result.action_data}"
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        primary_action = exec_actions[0]
        action_name = primary_action.get("name", "")
        params = primary_action.get("parameters", {})

        action_type_mapping = {
            "click": ActionType.CLICK,
            "write": ActionType.TYPE,
            "press": ActionType.PRESS,
            "swipe": ActionType.SCROLL,
            "back": ActionType.GO_BACK,
            "response": ActionType.NONE,
            "terminate": ActionType.NONE,
            "wait": ActionType.PRESS,
        }
        action_type = action_type_mapping.get(action_name, ActionType.PRESS)

        coordinates = None
        value = ""
        if action_name == "click":
            x, y = params.get("x"), params.get("y")
            if x is not None and y is not None:
                coordinates = (x, y)
        elif action_name == "write":
            value = params.get("message", "")
            if len(exec_actions) > 1 and exec_actions[0].get("name") == "click":
                cp = exec_actions[0].get("parameters", {})
                x, y = cp.get("x"), cp.get("y")
                if x is not None and y is not None:
                    coordinates = (x, y)
        elif action_name == "press":
            value = params.get("keys", "")
        elif action_name == "swipe":
            swipe_dir = params.get("direction", "up")
            direction_mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
            value = direction_mapping.get(swipe_dir, swipe_dir)
        elif action_name in ("response", "terminate"):
            value = params.get("answer", "") or params.get("info", "") or "Task Completed"
        elif action_name == "wait":
            value = f"wait {params.get('seconds', 1)}s"
        else:
            value = f"noop ({action_name})"

        if Action._is_required_element(action_type) and coordinates is None:
            logger.warning(
                f"action_type={action_type} requires coordinates but none available. Downgrading to PRESS."
            )
            action_type = ActionType.PRESS
            if not value or not value.strip():
                value = f"noop (missing coords for {action_name})"

        current_state.summary = result.thought

        action = Action(
            action_type=action_type,
            element=None,
            value=value,
            coordinates=coordinates,
        )
        action.exec_actions = exec_actions
        action.raw_response = result.raw_response
        action.raw_input_messages = result.raw_messages
        # stash for offline inspection
        action.memory_history_text = result.history_text
        action.memory_history_image_count = result.history_image_count
        action.experience_id = result.experience_id
        action.experience_reason = result.experience_reason

        low_level_instruction = result.operation or guipivot_action_to_mind2web_str(primary_action)

        return LowLevelTask(
            task=low_level_instruction,
            curr_state=current_state,
            action=action,
            task_status=LowTaskStatus.IN_PROGRESS,
            reasoning=result.thought,
        )

    # -------------------------------------------------------- online Stage A/B

    def _looks_like_blocked_completion(self, high_level_task: HighLevelTask) -> bool:
        if not high_level_task.trajectories:
            return False
        last = high_level_task.trajectories[-1]
        fields_to_check = [
            last.task or "",
            last.reasoning or "",
            last.action.value if last.action is not None else "",
        ]
        state = last.curr_state
        if state is not None and state.raw_state is not None:
            fields_to_check.append(state.raw_state.accessibility_tree or "")
            fields_to_check.append(state.raw_state.url or "")
        return any(_BLOCKED_FINAL_RE.search(str(value)) for value in fields_to_check if value)

    def _trajectory_to_online_experience_steps(
        self,
        previous_traj: list[LowLevelTask],
    ) -> list[dict]:
        steps: list[dict] = []
        for idx, low_task in enumerate(previous_traj, start=1):
            action = low_task.action
            before_url = ""
            after_url = ""
            if low_task.curr_state is not None and low_task.curr_state.raw_state is not None:
                before_url = low_task.curr_state.raw_state.url
            if low_task.state_after is not None and low_task.state_after.raw_state is not None:
                after_url = low_task.state_after.raw_state.url
            steps.append(
                {
                    "step": idx,
                    "action_repr": low_task.task or (action.get_action_str() if action else ""),
                    "action_type": action.action_type.value if action and action.action_type else "",
                    "value": action.value if action else "",
                    "coordinates": list(action.coordinates) if action and action.coordinates else None,
                    "reasoning": low_task.reasoning or "",
                    "before_url": before_url,
                    "after_url": after_url,
                    "state_after_summary": (
                        low_task.state_after.summary
                        if low_task.state_after is not None
                        else ""
                    ),
                }
            )
        return steps

    def _maybe_update_online_experience(
        self,
        task_status: dict,
        high_level_task: HighLevelTask,
        task_id: str | None,
    ) -> None:
        if task_status.get("end_reason") != "completed":
            return
        if self._online_experience_engine is None or self._online_experience_store_config is None:
            return
        steps = self._trajectory_to_online_experience_steps(high_level_task.trajectories)
        if not steps:
            return
        try:
            result = update_online_experience_store(
                engine=self._online_experience_engine,
                config=self._online_experience_store_config,
                task_id=str(task_id or high_level_task.task),
                task=high_level_task.task,
                start_url=high_level_task.start_url,
                steps=steps,
            )
            logger.info(
                "online experience updated for task_id=%s: n=%s ids=%s catalog=%s",
                task_id,
                result["num_experiences"],
                result["experience_ids"],
                result["catalog_path"],
            )
        except Exception as exc:
            logger.warning("online experience update failed for task_id=%s: %s", task_id, exc)

    def _finalize_mind2web_task(self, task_status: dict, high_level_task: HighLevelTask):
        task_id = self._current_mind2web_task_id
        if (
            task_status.get("end_reason") == "completed"
            and self._looks_like_blocked_completion(high_level_task)
        ):
            task_status["end_reason"] = "blocked"
            logger.info("task_id=%s ended on blocked/inaccessible page; marking as blocked", task_id)
        super()._finalize_mind2web_task(task_status, high_level_task)
        self._maybe_update_online_experience(task_status, high_level_task, task_id)
