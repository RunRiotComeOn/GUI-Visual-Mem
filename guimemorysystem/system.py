"""Unified GUI memory system used by evaluation adapters."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guimemorysystem.cross_task_memory import (
    ExperienceSelection,
    build_selector_context,
    load_catalog,
    load_library_by_id,
    select_experience,
)
from guimemorysystem.engine import EngineProtocol
from guimemorysystem.task_memory import MemoryState
from guimemorysystem.visual_skill_selector import (
    RecentFrameContext,
    select_visual_skills,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryStepContext:
    """Memory payload produced before one policy step."""

    task_memory_text: str = ""
    task_memory_images: list[Any] = field(default_factory=list)
    active_experience_text: str = ""
    experience_id: str | None = None
    experience_reason: str = ""
    previous_actions: list[str] = field(default_factory=list)
    current_observation: str = ""
    memory_update: dict | None = None

    @property
    def history_image_count(self) -> int:
        return len(self.task_memory_images)

    @property
    def context_text(self) -> str:
        blocks = [x for x in (self.task_memory_text, self.active_experience_text) if x]
        return "\n\n".join(blocks)

    def as_debug_dict(self) -> dict:
        return {
            "task_memory_text": self.task_memory_text,
            "task_memory_image_count": self.history_image_count,
            "active_experience_text": self.active_experience_text,
            "experience_id": self.experience_id,
            "experience_reason": self.experience_reason,
            "previous_actions": list(self.previous_actions),
            "current_observation": self.current_observation,
            "memory_update": self.memory_update,
        }


class GUIMemorySystem:
    """Composes task-specific memory and cross-task experience selection.

    Eval-loop usage:

    1. ``reset(task)`` once per task.
    2. ``prepare_step(screenshot, ...)`` before the policy call.
    3. Execute the policy action.
    4. ``commit_action(action_repr)`` with the action that actually ran.

    The next ``prepare_step`` call updates task-specific memory from the
    previous screenshot, current screenshot, and committed action string.
    """

    def __init__(
        self,
        *,
        memory_engine: EngineProtocol | None = None,
        selector_engine: EngineProtocol | None = None,
        experience_catalog: list[dict] | None = None,
        experience_library: dict[str, dict] | None = None,
        recent_k: int = 3,
        selector_recent_k: int = 3,
        use_visual_skill: bool = False,
        visual_skill_max_selected: int = 3,
        visual_skill_max_candidates: int = 20,
    ) -> None:
        self.memory_engine = memory_engine
        self.selector_engine = selector_engine
        self.experience_catalog = experience_catalog or []
        self.experience_library = experience_library or {}
        self.recent_k = recent_k
        self.selector_recent_k = selector_recent_k
        self.use_visual_skill = use_visual_skill
        self.visual_skill_max_selected = visual_skill_max_selected
        self.visual_skill_max_candidates = visual_skill_max_candidates

        self._task: str | None = None
        self._task_id: str = ""
        self.task_memory = MemoryState(keep_recent_items=recent_k)
        self.previous_actions: list[str] = []
        self._prev_screenshot: Any | None = None
        self._pending_action_repr: str | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        memory_engine: EngineProtocol | None = None,
        selector_engine: EngineProtocol | None = None,
        experience_catalog_path: str | Path | None = None,
        experience_library_path: str | Path | None = None,
        recent_k: int = 3,
        selector_recent_k: int = 3,
    ) -> "GUIMemorySystem":
        catalog: list[dict] | None = None
        library: dict[str, dict] | None = None
        if experience_catalog_path and experience_library_path:
            catalog = load_catalog(experience_catalog_path)
            library = load_library_by_id(experience_library_path)
        return cls(
            memory_engine=memory_engine,
            selector_engine=selector_engine,
            experience_catalog=catalog,
            experience_library=library,
            recent_k=recent_k,
            selector_recent_k=selector_recent_k,
        )

    def reset(self, task: str, *, task_id: str = "") -> None:
        self._task = task
        self._task_id = task_id
        self.task_memory = MemoryState(keep_recent_items=self.recent_k)
        self.previous_actions = []
        self._prev_screenshot = None
        self._pending_action_repr = None

    def prepare_step(
        self,
        screenshot: Any,
        *,
        current_url: str = "",
        current_page_title: str = "",
        observation_summary: str = "",
    ) -> MemoryStepContext:
        """Update memory for the last transition and render this step's context."""
        task = self._require_task()
        memory_update = self._update_pending_transition(task=task, screenshot=screenshot)

        history_text = ""
        history_images: list[Any] = []
        if self.memory_engine is not None:
            history_text, history_images = self.task_memory.render(task)

        if self.use_visual_skill:
            selection = self._select_visual_skill(
                task=task,
                current_screenshot=screenshot,
                current_url=current_url,
                current_page_title=current_page_title,
                observation_summary=observation_summary,
            )
        else:
            selection = self._select_experience(
                task=task,
                current_url=current_url,
                current_page_title=current_page_title,
                observation_summary=observation_summary,
            )

        self._prev_screenshot = screenshot
        return MemoryStepContext(
            task_memory_text=history_text,
            task_memory_images=history_images,
            active_experience_text=selection.injection,
            experience_id=selection.experience_id,
            experience_reason=selection.reason,
            previous_actions=list(self.previous_actions),
            current_observation=build_selector_context(
                task=task,
                previous_actions=self.previous_actions,
                current_url=current_url,
                current_page_title=current_page_title,
                observation_summary=observation_summary,
                recent_k=self.selector_recent_k,
            )[2],
            memory_update=memory_update,
        )

    def commit_action(self, action_repr: str) -> None:
        """Record the executed action for the next transition update."""
        self._require_task()
        repr_str = action_repr or "(no explicit action repr)"
        self.previous_actions.append(repr_str)
        self._pending_action_repr = repr_str

    def update_transition(
        self,
        *,
        prev_screenshot: Any,
        curr_screenshot: Any,
        action_repr: str,
    ) -> dict | None:
        """Directly update task memory for callers that own their own loop state."""
        task = self._require_task()
        if self.memory_engine is None:
            return None
        return self.task_memory.update(
            engine=self.memory_engine,
            task=task,
            prev_screenshot=prev_screenshot,
            curr_screenshot=curr_screenshot,
            action_repr=action_repr,
        )

    def render_task_memory(self) -> tuple[str, list[Any]]:
        return self.task_memory.render(self._require_task())

    def snapshot(self) -> dict:
        return {
            "task": self._task,
            "task_id": self._task_id,
            "task_memory": self.task_memory.snapshot(),
            "previous_actions": list(self.previous_actions),
            "has_pending_action": self._pending_action_repr is not None,
            "experience_catalog_size": len(self.experience_catalog),
            "experience_library_size": len(self.experience_library),
        }

    def _update_pending_transition(self, *, task: str, screenshot: Any) -> dict | None:
        if self.memory_engine is None:
            self._pending_action_repr = None
            return None
        if self._prev_screenshot is None:
            self._pending_action_repr = None
            return None
        if self._pending_action_repr is None:
            return None
        action_repr = self._pending_action_repr
        self._pending_action_repr = None
        try:
            return self.task_memory.update(
                engine=self.memory_engine,
                task=task,
                prev_screenshot=self._prev_screenshot,
                curr_screenshot=screenshot,
                action_repr=action_repr,
            )
        except Exception as exc:
            logger.warning("Task-specific memory update failed; continuing without update: %s", exc)
            return {"error": str(exc), "action_repr": action_repr}

    def _select_experience(
        self,
        *,
        task: str,
        current_url: str,
        current_page_title: str,
        observation_summary: str,
    ) -> ExperienceSelection:
        if not (self.selector_engine and self.experience_catalog and self.experience_library):
            return ExperienceSelection(None, "", "")
        sel_task, recent_steps, current_obs = build_selector_context(
            task=task,
            previous_actions=self.previous_actions,
            current_url=current_url,
            current_page_title=current_page_title,
            observation_summary=observation_summary,
            recent_k=self.selector_recent_k,
        )
        try:
            return select_experience(
                engine=self.selector_engine,
                task=sel_task,
                recent_steps=recent_steps,
                current_obs=current_obs,
                catalog=self.experience_catalog,
                library=self.experience_library,
            )
        except Exception as exc:
            logger.warning("Cross-task experience selector failed; continuing without experience: %s", exc)
            return ExperienceSelection(None, f"selector failed: {exc}", "")

    def _select_visual_skill(
        self,
        *,
        task: str,
        current_screenshot: Any,
        current_url: str,
        current_page_title: str,
        observation_summary: str,
    ) -> ExperienceSelection:
        if not (self.selector_engine and self.experience_catalog and self.experience_library):
            return ExperienceSelection(None, "", "")
        recent_frames: list[RecentFrameContext] = []
        for action in self.previous_actions[-self.selector_recent_k :]:
            recent_frames.append(RecentFrameContext(action=str(action)))
        obs_summary = observation_summary
        if not obs_summary and (current_url or current_page_title):
            obs_summary = " ".join(
                part for part in (current_page_title, current_url) if part
            ).strip()
        try:
            selection = select_visual_skills(
                engine=self.selector_engine,
                task=task,
                current_frame=current_screenshot,
                catalog=self.experience_catalog,
                library=self.experience_library,
                current_observation=obs_summary,
                recent_frames=recent_frames,
                max_catalog_candidates=self.visual_skill_max_candidates,
                max_selected_skills=self.visual_skill_max_selected,
            )
        except Exception as exc:
            logger.warning("Visual skill selector failed; continuing without skill: %s", exc)
            return ExperienceSelection(None, f"visual selector failed: {exc}", "")
        if not selection.selected_skills:
            return ExperienceSelection(None, selection.selector_notes or "no visual skill selected", "")
        chosen_ids = ",".join(item.skill_id for item in selection.selected_skills)
        reason = selection.selected_skills[0].reason or selection.selector_notes
        return ExperienceSelection(chosen_ids, reason, selection.injection)

    def _require_task(self) -> str:
        if self._task is None:
            raise RuntimeError("GUIMemorySystem.reset(task) must be called before use.")
        return self._task
