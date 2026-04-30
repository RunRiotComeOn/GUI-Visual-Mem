"""Online task-specific memory: older summary + recent buffer + keyframes."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from guimemorysystem.engine import EngineProtocol

logger = logging.getLogger(__name__)


MEMORY_AGENT_SYSTEM = """You are the memory agent in a GUI agentic system.
Your job is to update rolling interaction memory online.

Rules:
- Compare only the previous screenshot and the current screenshot for the newest step.
- Use the provided action string only as auxiliary context.
- First emit one newest item as either `recent_change` or `recent_change_keyframe`.
- A `recent_change` is for a small local update on the same working screen.
- A `recent_change_keyframe` is for a major state transition such as page jump, modal/dialog open, checkout/payment transition, or a clear interaction-focus shift.
- The memory must preserve the exact interaction primitive, not only a high-level workflow story.
- Always ground the update in the acted-on element and the action type.
- If a value was typed or selected, copy that value in a short normalized form.
- Always state what interaction focus the UI is in after this step, and what the next likely local goal is.
- Interaction focus means the current sub-flow or active workspace, such as `search form`, `guest modal`, `results list`, `truck options`, `location continuation`, `add-ons`, `checkout form`, or `payment`.
- Keep the change description extremely short and grounded in visible UI changes.
- Do not narrate the whole task.
- Do not speculate about hidden state.
- Do not replace precise controls with vague summaries like `continued checkout` or `moved forward` if a more specific element or sub-flow is visible.
- If the UI focus has shifted away from one branch to another, say that explicitly in `focus_after` or `next_goal`.

Output strict JSON only with this schema:
{
  "item_type": "recent_change" | "recent_change_keyframe",
  "change": "<very short visible state change>",
  "element": "<very short acted-on element description>",
  "action_type": "CLICK" | "SELECT" | "TYPE" | "DUAL_POINT" | "GO_BACK" | "GO_HOME" | "ENTER" | "TASK_COMPLETE" | "TASK_IMPOSSIBLE" | "",
  "action_value": "<typed/selected value or empty string>",
  "focus_after": "<current active UI sub-flow after this step>",
  "next_goal": "<the next immediate local target implied by the UI, not the whole task>",
  "keep_image": true | false,
  "action": "<copy the provided action string only if item_type is recent_change_keyframe, else empty string>"
}

Do not wrap JSON in markdown fences.
"""

MEMORY_AGENT_USER_TEMPLATE = """Task:
{task}

Current memory before this update:
[older_summary]
{older_summary}

[recent_buffer]
{recent_buffer}

Newest action:
{action_repr}

Compare the two screenshots and update memory for only this newest transition."""


SUMMARIZER_SYSTEM = """You are the summarizer for old GUI interaction memory.
You compress older recent-memory items into a single older_summary.

Rules:
- Summarize only the older prefix, not the newest tail.
- Keep it concise and high-level.
- Preserve major page or workflow transitions.
- Merge repeated small local changes aggressively.
- Mention keyframe transitions at a coarse level.
- Do not narrate every step.
- Do not drop the interaction anchors that are necessary for future action grounding.
- Keep explicit references to:
  - key acted-on elements when they define the current branch,
  - exact action types when they matter,
  - important typed/selected values,
  - interaction-focus shifts such as `guest modal -> results`, `truck options -> location continuation`, or `payment options -> personal info form`.
- Prefer short structured prose over vague storytelling.
- If an older prefix establishes that one branch is active and another branch is no longer active, retain that distinction.

Output strict JSON only:
{
  "older_summary": "<compressed older summary>"
}
Do not wrap JSON in markdown fences.
"""

SUMMARIZER_USER_TEMPLATE = """Task:
{task}

Existing older summary:
{older_summary}

Older recent-memory prefix to compress:
{prefix_text}
"""


def _safe_json_load(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def render_recent_item(item: dict) -> str:
    parts = [f"[{item['item_type']}] {item.get('change', '')}"]
    if item.get("element"):
        parts.append(f"element: {item['element']}")
    if item.get("action_type"):
        parts.append(f"action_type: {item['action_type']}")
    if item.get("action_value"):
        parts.append(f"action_value: {item['action_value']}")
    if item.get("focus_after"):
        parts.append(f"focus_after: {item['focus_after']}")
    if item.get("next_goal"):
        parts.append(f"next_goal: {item['next_goal']}")
    if item.get("item_type") == "recent_change_keyframe" and item.get("action"):
        parts.append(f"action: {item['action']}")
    return " | ".join(parts)


def _fallback_memory_item(action_repr: str) -> dict:
    return {
        "item_type": "recent_change",
        "change": "the interface updates locally on the same screen.",
        "element": "",
        "action_type": "",
        "action_value": "",
        "focus_after": "same local screen",
        "next_goal": "continue the current local interaction",
        "keep_image": False,
        "action": "",
        "_fallback": True,
        "_action_repr": action_repr,
    }


def _fallback_older_summary(older_summary: str, prefix: list[dict]) -> str:
    prefix_lines = [item.get("change", "").strip() for item in prefix if item.get("change")]
    merged = " ".join(prefix_lines[:3]).strip()
    if older_summary and merged:
        return f"{older_summary} {merged}".strip()
    if older_summary:
        return older_summary
    return merged or "Earlier interaction history contains local state updates in the same task flow."


@dataclass
class MemoryState:
    """Rolling task-specific memory for one task lifecycle."""

    keep_recent_items: int = 3
    older_summary: str = ""
    recent_buffer: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.older_summary = ""
        self.recent_buffer = []

    def update(
        self,
        *,
        engine: EngineProtocol,
        task: str,
        prev_screenshot: Any,
        curr_screenshot: Any,
        action_repr: str,
    ) -> dict:
        """Add the newest observed transition to task-specific memory."""
        user_text = MEMORY_AGENT_USER_TEMPLATE.format(
            task=task,
            older_summary=self.older_summary or "(empty)",
            recent_buffer="\n".join(f"- {render_recent_item(x)}" for x in self.recent_buffer) or "(empty)",
            action_repr=action_repr or "(no explicit action repr)",
        )
        item: dict | None = None
        for _ in range(2):
            response = engine.chat_with_images(
                system_prompt=MEMORY_AGENT_SYSTEM,
                user_text=user_text,
                previous_image=prev_screenshot,
                current_image=curr_screenshot,
                max_tokens=320,
            )
            try:
                item = _safe_json_load(response)
                break
            except json.JSONDecodeError:
                item = None

        if item is None:
            logger.warning("Memory agent returned non-JSON output for action %r. Using fallback.", action_repr)
            item = _fallback_memory_item(action_repr)

        item_type = item.get("item_type", "recent_change")
        if item_type not in {"recent_change", "recent_change_keyframe"}:
            item_type = "recent_change"
        keep_image = bool(item.get("keep_image", False))
        memory_item = {
            "item_type": item_type,
            "change": (item.get("change") or "").strip(),
            "element": (item.get("element") or "").strip(),
            "action_type": (item.get("action_type") or "").strip(),
            "action_value": (item.get("action_value") or "").strip(),
            "focus_after": (item.get("focus_after") or "").strip(),
            "next_goal": (item.get("next_goal") or "").strip(),
            "keep_image": keep_image,
            "action": (item.get("action") or "").strip() if item_type == "recent_change_keyframe" else "",
            "image": curr_screenshot if keep_image else None,
        }
        if item.get("_fallback"):
            memory_item["_fallback"] = True
        self.recent_buffer.append(memory_item)
        summary_trace = self._summarize_prefix_if_needed(engine=engine, task=task)
        return {
            "memory_item": _json_safe_item(memory_item),
            "older_summary": self.older_summary,
            "recent_buffer": [_json_safe_item(x) for x in self.recent_buffer],
            "summary_trace": summary_trace,
        }

    def _summarize_prefix_if_needed(self, *, engine: EngineProtocol, task: str) -> dict | None:
        if len(self.recent_buffer) <= self.keep_recent_items:
            return None
        prefix = self.recent_buffer[: -self.keep_recent_items]
        tail = self.recent_buffer[-self.keep_recent_items :]
        prefix_text = "\n".join(f"- {render_recent_item(item)}" for item in prefix)
        user_text = SUMMARIZER_USER_TEMPLATE.format(
            task=task,
            older_summary=self.older_summary or "(empty)",
            prefix_text=prefix_text,
        )

        data: dict | None = None
        for _ in range(2):
            response = engine.chat(
                [
                    {"role": "system", "content": SUMMARIZER_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=320,
            )
            try:
                data = _safe_json_load(response)
                break
            except json.JSONDecodeError:
                data = None

        trace = {
            "compressed_items": [_json_safe_item(x) for x in prefix],
            "older_summary_before": self.older_summary,
            "fallback": data is None,
        }
        if data is None:
            logger.warning("Summarizer returned non-JSON output. Falling back to heuristic merge.")
            self.older_summary = _fallback_older_summary(self.older_summary, prefix)
        else:
            self.older_summary = data.get("older_summary", self.older_summary) or self.older_summary
        self.recent_buffer = tail
        trace["older_summary_after"] = self.older_summary
        return trace

    def render(self, task: str) -> tuple[str, list[Any]]:
        """Render memory text and keyframe images for policy prompt injection."""
        if not self.recent_buffer and not self.older_summary:
            return "", []
        lines = [
            "Hybrid trajectory context for the same task:",
            f"Task: {task}",
        ]
        if self.older_summary:
            lines.append("[older_summary]")
            lines.append(self.older_summary)
        if self.recent_buffer:
            lines.append("[recent_buffer]")
            for idx, item in enumerate(self.recent_buffer, start=1):
                lines.append(f"{idx}. {render_recent_item(item)}")
        history_images = [
            item["image"]
            for item in self.recent_buffer
            if item.get("keep_image") and item.get("image") is not None
        ]
        return "\n".join(lines), history_images

    def snapshot(self) -> dict:
        return {
            "older_summary": self.older_summary,
            "recent_buffer": [_json_safe_item(item) for item in self.recent_buffer],
        }


def _json_safe_item(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "image"}
