"""Cross-task experience memory: Stage-C selector and prompt rendering."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from guimemorysystem.engine import EngineProtocol

logger = logging.getLogger(__name__)


SELECTOR_SYSTEM_PROMPT = """You are the experience selector for a GUI agent.

You are given a list of cross-task experiences. Each one is a generalizable
rule learned from prior successful trajectories. Your job is to decide if
exactly one of them applies to the current step.

Hard rules:
- Return at most one experience.
- "null" is always a valid answer. Prefer null over a weak match.
- The reason must reference a concrete signal from the current step or the
  recent history, not the experience description.
- Output strict JSON only.
"""

SELECTOR_USER_TEMPLATE = """Task instruction:
{task}

Recent steps (most recent last):
{recent_block}

Current observation (short summary):
{current_obs}

Available experiences (catalog):
{catalog_block}

Choose the single most useful experience for this step, or null. Return JSON:
{{"experience_id": "<id or null>", "reason": "<one sentence grounded in current step>"}}
"""


@dataclass(frozen=True)
class ExperienceSelection:
    experience_id: str | None
    reason: str = ""
    injection: str = ""

    @property
    def matched(self) -> bool:
        return self.experience_id is not None

    def as_dict(self) -> dict:
        return {
            "experience_id": self.experience_id,
            "reason": self.reason,
            "injection": self.injection or None,
        }


def load_library_by_id(path: str | Path) -> dict[str, dict]:
    library: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            exp_id = record.get("experience_id")
            if exp_id:
                library[exp_id] = record
    return library


def load_catalog(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"experience catalog must be a list: {path}")
    return data


def select_experience(
    *,
    engine: EngineProtocol,
    task: str,
    recent_steps: list[dict],
    current_obs: str,
    catalog: list[dict],
    library: dict[str, dict],
    max_tokens: int = 200,
) -> ExperienceSelection:
    """Pick at most one cross-task experience for the current step."""
    if not catalog:
        return ExperienceSelection(None, "empty catalog", "")

    user_prompt = SELECTOR_USER_TEMPLATE.format(
        task=task,
        recent_block=_format_recent_steps(recent_steps[-3:]),
        current_obs=current_obs or "(no current observation summary)",
        catalog_block=_format_catalog(catalog),
    )
    raw = engine.chat(
        [
            {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    parsed = _extract_json_object(raw)
    exp_id = parsed.get("experience_id")
    reason = parsed.get("reason", "") or ""
    if exp_id is None or exp_id == "null":
        return ExperienceSelection(None, reason, "")

    record = library.get(str(exp_id))
    if record is None:
        logger.warning("Selector returned unknown experience_id=%s; treating as null.", exp_id)
        return ExperienceSelection(None, f"unknown id {exp_id}", "")
    return ExperienceSelection(str(exp_id), reason, render_experience_slot(record))


def render_experience_slot(record: dict) -> str:
    """Render one full experience into the fixed policy prompt slot."""
    context = record.get("applicable_context") or {}
    lines = [
        "[active_experience]",
        f"Title: {record.get('title', '')}",
        f"When it applies: {context.get('when', '')}",
        f"Guidance: {record.get('action_guidance', '')}",
    ]
    if record.get("trigger_ui_state"):
        lines.append(f"Trigger UI state: {record['trigger_ui_state']}")
    if record.get("forbidden_alternative"):
        lines.append(f"Avoid: {record['forbidden_alternative']}")
    if record.get("expected_postcondition"):
        lines.append(f"Expected after action: {record['expected_postcondition']}")
    templates = record.get("action_templates") or []
    if templates:
        lines.append("Suggested action shapes:")
        for template in templates[:2]:
            lines.append(f"- {template}")
    return "\n".join(lines)


def build_selector_context(
    *,
    task: str,
    previous_actions: list[str],
    current_url: str = "",
    current_page_title: str = "",
    observation_summary: str = "",
    recent_k: int = 3,
) -> tuple[str, list[dict], str]:
    """Build the Stage-C ``(task, recent_steps, current_obs)`` tuple."""
    recent_steps = [{"action": action} for action in previous_actions[-recent_k:]]
    obs_parts: list[str] = []
    if observation_summary:
        obs_parts.append(observation_summary)
    if current_url:
        obs_parts.append(f"url={current_url}")
    if current_page_title:
        obs_parts.append(f"title={current_page_title}")
    return task, recent_steps, " ".join(obs_parts)


def _format_catalog(catalog: list[dict]) -> str:
    lines: list[str] = []
    for entry in catalog:
        exp_id = entry.get("experience_id") or entry.get("id")
        if not exp_id:
            continue
        lines.append(
            "- {id}: {title} | when: {when}".format(
                id=exp_id,
                title=entry.get("title", ""),
                when=entry.get("when", entry.get("trigger", "")),
            )
        )
    return "\n".join(lines) if lines else "(empty)"


def _format_recent_steps(recent_steps: list[dict]) -> str:
    if not recent_steps:
        return "(no recent steps)"
    lines: list[str] = []
    for idx, step in enumerate(recent_steps, start=1):
        action = step.get("action", "")
        obs = step.get("observation_summary") or ""
        if obs:
            lines.append(f"{idx}. action={action} | obs={obs}")
        else:
            lines.append(f"{idx}. action={action}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in selector output")
    return json.loads(text[start : end + 1])
