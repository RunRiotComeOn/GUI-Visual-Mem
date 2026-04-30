"""Online selector for v2/v3 visual skills.

The selector is intentionally benchmark-agnostic.  It does not assume golden
targets, DOM nodes, accessibility trees, or precomputed candidate elements.
Instead, it gives the selector the current frame, a short recent-frame/action
history, and a small catalog shortlist.  The selector is instructed to inspect
the frame first, infer likely target affordances, and then choose applicable
visual skills from the shortlist.

For v3 stores, historical skill images are used as retrieval/state-matching
evidence, while the injected content is procedural planning guidance.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from guimemorysystem.cross_task_memory import load_catalog, load_library_by_id
from guimemorysystem.engine import EngineProtocol
from guimemorysystem.images import image_to_chat_content


VISUAL_SKILL_SELECTOR_SYSTEM_PROMPT = """You are the visual skill selector for a GUI agent.

Your job is to choose a few reusable visual skills that may help the next GUI
action.  You are not executing the action.

Important assumptions:
- The benchmark does not provide target candidates or golden element labels.
- You must inspect the current frame yourself and infer likely actionable
  targets such as search bars, text fields, dropdowns, menus, tabs, date cells,
  buttons, and scrollable areas.
- Recent frames/actions may indicate that a multi-step skill is already in
  progress, for example a dropdown is open after the previous click.
- v3 skills are UI planning skills: historical images are matching evidence,
  but the useful payload is the procedure/preconditions/failure recovery.
- The catalog shortlist is not exhaustive. Prefer no skill over a weak match.

Selection rules:
- Return at most the requested number of skills.
- A selected skill must be grounded in visible evidence from the current frame
  or in a concrete recent transition, and the task goal must be compatible with
  the skill intent.
- Do not select a skill only because the task text is similar; also match the
  current page/state against the historical evidence or preconditions.
- For scroll or key-only skills, no bounding box is needed. For localized
  skills, describe the expected target visually, but do not invent coordinates.
- Output strict JSON only.
- Set confidence >= 0.7 only when you are highly certain the skill applies to
  the CURRENT step. Lower confidence means the skill should not be used.
- If the next action is most likely a simple button/link click with no text
  input or dropdown involved, set no_skill_needed=true and return an empty list.
- When in doubt, set no_skill_needed=true. A missing skill is less harmful than
  a wrong skill that changes the action type.
"""

VISUAL_SKILL_SELECTOR_USER_TEMPLATE = """Task goal:
{task}

Current observation summary:
{current_observation}

Recent frames/actions, oldest to newest:
{recent_block}

Catalog shortlist:
{catalog_block}

Historical evidence images, when provided, are labeled by skill_id. First
inspect the current frame and recent frames. Then compare the current
page/task against both the historical page evidence and the skill text. Select
up to {max_selected} skills from the catalog shortlist.

Return JSON:
{{
  "selected_skills": [
    {{
      "skill_id": "<catalog skill_id>",
      "confidence": 0.0,
      "matched_visual_evidence": "<what you saw in current/recent frames>",
      "matched_skill_evidence": "<which historical evidence/precondition matched>",
      "expected_target_description": "<visual target to look for; no coordinates>",
      "suggested_plan": "<short procedural plan for the current state>",
      "slot_values": {{"slot_name": "value if clear"}},
      "reason": "<one concise reason grounded in the current step>"
    }}
  ],
  "no_skill_needed": false,
  "selector_notes": "<brief note if helpful>"
}}
"""


@dataclass(frozen=True)
class RecentFrameContext:
    """One recent online frame/action item for selector context."""

    image: Any | None = None
    action: str = ""
    observation_summary: str = ""
    result_summary: str = ""


@dataclass(frozen=True)
class SelectedVisualSkill:
    """One selector decision tied to a library record when available."""

    skill_id: str
    confidence: float = 0.0
    reason: str = ""
    matched_visual_evidence: str = ""
    matched_skill_evidence: str = ""
    expected_target_description: str = ""
    suggested_plan: str = ""
    slot_values: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] | None = None


@dataclass(frozen=True)
class VisualSkillSelection:
    """Selector output plus rendered injection text."""

    selected_skills: list[SelectedVisualSkill]
    injection: str = ""
    raw_response: str = ""
    selector_notes: str = ""
    catalog_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.selected_skills)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_skills": [
                {
                    "skill_id": item.skill_id,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "matched_visual_evidence": item.matched_visual_evidence,
                    "matched_skill_evidence": item.matched_skill_evidence,
                    "expected_target_description": item.expected_target_description,
                    "suggested_plan": item.suggested_plan,
                    "slot_values": item.slot_values,
                }
                for item in self.selected_skills
            ],
            "injection": self.injection or None,
            "selector_notes": self.selector_notes,
            "catalog_candidate_ids": [
                item.get("skill_id") or item.get("experience_id") or item.get("id")
                for item in self.catalog_candidates
            ],
        }


def load_visual_skill_store(store_dir: str | Path) -> tuple[list[dict], dict[str, dict]]:
    """Load ``catalog.json`` and ``skill_library.jsonl`` from a v2 store dir."""
    root = Path(store_dir)
    return load_catalog(root / "catalog.json"), load_library_by_id(root / "skill_library.jsonl")


def retrieve_visual_skill_catalog_candidates(
    catalog: Sequence[dict[str, Any]],
    *,
    task: str,
    current_observation: str = "",
    recent_frames: Sequence[RecentFrameContext | dict[str, Any]] | None = None,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Cheap, non-oracle shortlist before the VLM selector sees the catalog.

    This retrieval stage uses only task/observation/history text and structured
    catalog metadata.  It never uses target annotations from the benchmark.
    """
    if max_candidates <= 0:
        return []
    recent_frames = recent_frames or []
    query_text = _selector_query_text(task, current_observation, recent_frames)
    query_tokens = _tokens(query_text)
    history_text = " ".join(_recent_text(item) for item in recent_frames).lower()

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, entry in enumerate(catalog):
        skill_id = _entry_id(entry)
        if not skill_id:
            continue
        score = _catalog_entry_score(entry, query_text=query_text, query_tokens=query_tokens, history_text=history_text)
        copied = dict(entry)
        copied["_retrieval_score"] = round(score, 4)
        scored.append((score, -idx, copied))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    positives = [entry for score, _, entry in scored if score > 0]
    if len(positives) >= max_candidates:
        return positives[:max_candidates]
    remainder = [entry for score, _, entry in scored if score <= 0]
    return (positives + remainder)[:max_candidates]


def select_visual_skills(
    *,
    engine: EngineProtocol,
    task: str,
    current_frame: Any | None,
    catalog: Sequence[dict[str, Any]],
    library: dict[str, dict],
    current_observation: str = "",
    recent_frames: Sequence[RecentFrameContext | dict[str, Any]] | None = None,
    max_catalog_candidates: int = 20,
    max_selected_skills: int = 3,
    max_tokens: int = 600,
    lossy_images: bool = True,
    min_confidence: float = 0.7,
    include_candidate_images: bool = True,
    max_candidate_images: int = 3,
) -> VisualSkillSelection:
    """Select top visual skills for the current online step."""
    if not catalog or not library:
        return VisualSkillSelection([], selector_notes="empty visual skill store")
    recent_frames = list(recent_frames or [])
    candidates = retrieve_visual_skill_catalog_candidates(
        catalog,
        task=task,
        current_observation=current_observation,
        recent_frames=recent_frames,
        max_candidates=max_catalog_candidates,
    )
    if not candidates:
        return VisualSkillSelection([], selector_notes="empty catalog shortlist")

    user_text = VISUAL_SKILL_SELECTOR_USER_TEMPLATE.format(
        task=task or "(empty task)",
        current_observation=current_observation or "(no text summary)",
        recent_block=_format_recent_frames(recent_frames),
        catalog_block=_format_visual_skill_catalog(candidates),
        max_selected=max_selected_skills,
    )
    content = _selector_multimodal_content(
        user_text=user_text,
        current_frame=current_frame,
        recent_frames=recent_frames,
        candidate_images=_candidate_evidence_images(
            candidates,
            library,
            max_images=max_candidate_images if include_candidate_images else 0,
        ),
        lossy=lossy_images,
    )
    raw = engine.chat(
        [
            {"role": "system", "content": VISUAL_SKILL_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
    )
    parsed = _extract_json_object(raw)
    if parsed.get("no_skill_needed"):
        return VisualSkillSelection(
            [],
            raw_response=raw,
            selector_notes=str(parsed.get("selector_notes") or "no_skill_needed"),
            catalog_candidates=candidates,
        )
    selected = _parse_selected_skills(
        parsed,
        candidates=candidates,
        library=library,
        max_selected_skills=max_selected_skills,
        min_confidence=min_confidence,
    )
    return VisualSkillSelection(
        selected_skills=selected,
        injection=render_selected_visual_skills(selected),
        raw_response=raw,
        selector_notes=str(parsed.get("selector_notes") or ""),
        catalog_candidates=candidates,
    )


def render_selected_visual_skills(selected: Sequence[SelectedVisualSkill]) -> str:
    """Render selected full skill records for policy prompt injection."""
    selected = [item for item in selected if item.record]
    if not selected:
        return ""
    lines = [
        "[active_visual_skills]",
        "Use these only when the current frame visually matches the stated target. Ignore weak matches.",
    ]
    for idx, item in enumerate(selected, start=1):
        record = item.record or {}
        if record.get("version") == "visual_skill_v3":
            lines.extend(_render_selected_v3_skill(idx, item, record))
            continue
        context = record.get("applicable_context") or {}
        example = record.get("example") or {}
        lines.extend(
            [
                f"{idx}. {record.get('title') or item.skill_id}",
                f"   skill_id: {item.skill_id}",
                f"   selector_reason: {item.reason}",
                f"   visual_evidence: {item.matched_visual_evidence}",
                f"   expected_target: {item.expected_target_description}",
                f"   when: {context.get('when', '')}",
                f"   target_instruction: {record.get('target_instruction', '')}",
                f"   action_guidance: {record.get('action_guidance', '')}",
            ]
        )
        templates = record.get("action_templates") or []
        if templates:
            lines.append("   action_templates: " + "; ".join(str(x) for x in templates[:3]))
        value_slots = record.get("value_slots") or []
        if value_slots:
            lines.append("   value_slots: " + json.dumps(value_slots, ensure_ascii=False))
        if item.slot_values:
            lines.append("   filled_slots: " + json.dumps(item.slot_values, ensure_ascii=False))
        if example.get("target_role"):
            lines.append(f"   example_target_role: {example.get('target_role')}")
    return "\n".join(lines)


def _render_selected_v3_skill(idx: int, item: SelectedVisualSkill, record: dict[str, Any]) -> list[str]:
    context = record.get("applicable_context") or {}
    planning = record.get("planning") or {}
    lines = [
        f"{idx}. {record.get('title') or item.skill_id}",
        f"   skill_id: {item.skill_id}",
        f"   selector_reason: {item.reason}",
        f"   current_state_match: {item.matched_visual_evidence}",
    ]
    if item.matched_skill_evidence:
        lines.append(f"   historical_evidence_match: {item.matched_skill_evidence}")
    lines.append(f"   intent: {record.get('intent', '')}")
    preconditions = context.get("preconditions") or []
    if preconditions:
        lines.append("   preconditions: " + " | ".join(str(x) for x in preconditions[:5]))
    procedure = planning.get("procedure") or []
    if procedure:
        lines.append("   planning_procedure:")
        lines.extend(f"     - {step}" for step in procedure[:6])
    if item.suggested_plan:
        lines.append(f"   selector_adapted_plan: {item.suggested_plan}")
    postconditions = planning.get("postcondition_checks") or []
    if postconditions:
        lines.append("   postcondition_checks: " + " | ".join(str(x) for x in postconditions[:4]))
    failures = planning.get("failure_modes") or []
    if failures:
        lines.append("   watch_for_failures: " + " | ".join(str(x) for x in failures[:4]))
    recovery = planning.get("recovery_steps") or []
    if recovery:
        lines.append("   recovery: " + " | ".join(str(x) for x in recovery[:4]))
    if item.slot_values:
        lines.append("   filled_slots: " + json.dumps(item.slot_values, ensure_ascii=False))
    return lines


def _selector_multimodal_content(
    *,
    user_text: str,
    current_frame: Any | None,
    recent_frames: Sequence[RecentFrameContext | dict[str, Any]],
    candidate_images: Sequence[tuple[str, Any]],
    lossy: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for idx, item in enumerate(recent_frames, start=1):
        image = _recent_value(item, "image")
        if image is None:
            continue
        content.append({"type": "text", "text": f"Recent frame {idx} image:"})
        content.append(image_to_chat_content(image, lossy=lossy))
    if current_frame is not None:
        content.append({"type": "text", "text": "Current frame image:"})
        content.append(image_to_chat_content(current_frame, lossy=lossy))
    for label, image in candidate_images:
        content.append({"type": "text", "text": f"Historical skill evidence image: {label}"})
        content.append(image_to_chat_content(image, lossy=lossy))
    content.append({"type": "text", "text": user_text})
    return content


def _parse_selected_skills(
    parsed: dict[str, Any],
    *,
    candidates: Sequence[dict[str, Any]],
    library: dict[str, dict],
    max_selected_skills: int,
    min_confidence: float = 0.0,
) -> list[SelectedVisualSkill]:
    candidate_ids = {_entry_id(item) for item in candidates}
    selected_rows = parsed.get("selected_skills") or []
    if not isinstance(selected_rows, list):
        return []
    selected: list[SelectedVisualSkill] = []
    seen: set[str] = set()
    for row in selected_rows:
        if not isinstance(row, dict):
            continue
        skill_id = str(row.get("skill_id") or row.get("experience_id") or row.get("id") or "")
        if not skill_id or skill_id in seen or skill_id not in candidate_ids:
            continue
        confidence = _clamp_float(row.get("confidence"), 0.0, 1.0)
        if confidence < min_confidence:
            continue
        record = library.get(skill_id)
        if record is None:
            continue
        selected.append(
            SelectedVisualSkill(
                skill_id=skill_id,
                confidence=confidence,
                reason=str(row.get("reason") or ""),
                matched_visual_evidence=str(row.get("matched_visual_evidence") or ""),
                matched_skill_evidence=str(row.get("matched_skill_evidence") or ""),
                expected_target_description=str(row.get("expected_target_description") or ""),
                suggested_plan=str(row.get("suggested_plan") or ""),
                slot_values=row.get("slot_values") if isinstance(row.get("slot_values"), dict) else {},
                record=record,
            )
        )
        seen.add(skill_id)
        if len(selected) >= max_selected_skills:
            break
    return selected


def _catalog_entry_score(
    entry: dict[str, Any],
    *,
    query_text: str,
    query_tokens: set[str],
    history_text: str,
) -> float:
    signature = str(entry.get("signature") or "")
    roles, actions, value_kinds = _parse_signature(signature)
    entry_text = " ".join(
        [
            str(entry.get("title") or ""),
            str(entry.get("intent") or ""),
            str(entry.get("when") or ""),
            " ".join(str(x) for x in entry.get("preconditions") or []),
            " ".join(str(x) for x in entry.get("page_state_cues") or []),
            " ".join(str(x) for x in entry.get("task_goal_cues") or []),
            " ".join(str(x) for x in entry.get("procedure") or []),
            _retrieval_entry_text(entry.get("retrieval") or {}),
            " ".join(str(x) for x in entry.get("visual_cues") or []),
            signature,
        ]
    ).lower()
    entry_tokens = _tokens(entry_text)
    overlap = len(query_tokens & entry_tokens) / max(len(entry_tokens), 1)

    score = 2.0 * overlap
    score += _action_need_score(actions, value_kinds, query_text)
    score += _history_continuation_score(roles, actions, history_text)
    score += _support_prior(entry)
    score -= _generic_penalty(roles, actions, value_kinds)
    return score


def _action_need_score(actions: Sequence[str], value_kinds: Sequence[str], query_text: str) -> float:
    text = query_text.lower()
    score = 0.0
    if "input_text" in actions and _has_any(text, ["type", "enter", "input", "search for", "look up", "find "]):
        score += 0.8
    if "select" in actions and _has_any(text, ["select", "choose", "pick"]):
        score += 0.5
    if "hover" in actions and _has_any(text, ["menu", "nav", "hover"]):
        score += 0.4
    if "date_or_time" in value_kinds and _has_any(text, ["date", "time", "calendar", "pickup", "return"]):
        score += 0.5
    if "query" in value_kinds and _has_any(text, ["search", "find", "look up", "query"]):
        score += 0.4
    return score


def _history_continuation_score(roles: Sequence[str], actions: Sequence[str], history_text: str) -> float:
    if not history_text:
        return 0.0
    score = 0.0
    if "text_option" in roles and _has_any(history_text, ["dropdown", "select", "menu", "suggestion"]):
        score += 0.7
    if "date_cell" in roles and _has_any(history_text, ["date", "calendar"]):
        score += 0.7
    if "search_bar" in roles and _has_any(history_text, ["opened search", "clicked search", "search icon"]):
        score += 0.4
    if "input_text" in actions and _has_any(history_text, ["clicked field", "focused field"]):
        score += 0.3
    return score


def _support_prior(entry: dict[str, Any]) -> float:
    support = entry.get("support") or {}
    num_tasks = float(support.get("num_tasks") or 0)
    num_domains = float(support.get("num_domains") or 0)
    num_segments = float(support.get("num_trajectory_segments") or support.get("num_occurrences") or 0)
    return min(
        0.4,
        math.log1p(num_tasks) / 20.0
        + math.log1p(num_domains) / 30.0
        + math.log1p(num_segments) / 60.0,
    )


def _generic_penalty(roles: Sequence[str], actions: Sequence[str], value_kinds: Sequence[str]) -> float:
    if len(roles) == 1 and actions == ["click"] and value_kinds == ["none"]:
        return 0.5
    if roles and all(role in {"text_option", "button", "confirm_button", "target_element"} for role in roles):
        return 0.25
    return 0.0


def _format_visual_skill_catalog(catalog: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, entry in enumerate(catalog, start=1):
        skill_id = _entry_id(entry)
        if not skill_id:
            continue
        support = entry.get("support") or {}
        retrieval = entry.get("retrieval") or {}
        procedure = entry.get("procedure") or []
        preconditions = entry.get("preconditions") or []
        lines.append(
            (
                "{idx}. skill_id={skill_id}\n"
                "   version={version}\n"
                "   title={title}\n"
                "   intent={intent}\n"
                "   signature={signature}\n"
                "   when={when}\n"
                "   preconditions={preconditions}\n"
                "   procedure={procedure}\n"
                "   retrieval_summary={retrieval_summary}\n"
                "   visual_cues={visual_cues}\n"
                "   support=tasks:{tasks}, domains:{domains}, segments:{segments}, occurrences:{occurrences}\n"
                "   retrieval_score={score}"
            ).format(
                idx=idx,
                skill_id=skill_id,
                version=entry.get("version", ""),
                title=entry.get("title", ""),
                intent=entry.get("intent", ""),
                signature=entry.get("signature", ""),
                when=entry.get("when", ""),
                preconditions=" | ".join(str(x) for x in preconditions[:4]),
                procedure=" | ".join(str(x) for x in procedure[:4]),
                retrieval_summary=retrieval.get("page_state_summary", ""),
                visual_cues=", ".join(str(x) for x in (entry.get("visual_cues") or entry.get("page_state_cues") or [])[:5]),
                tasks=support.get("num_tasks", ""),
                domains=support.get("num_domains", ""),
                segments=support.get("num_trajectory_segments", ""),
                occurrences=support.get("num_occurrences", ""),
                score=entry.get("_retrieval_score", ""),
            )
        )
    return "\n".join(lines) if lines else "(empty)"


def _format_recent_frames(recent_frames: Sequence[RecentFrameContext | dict[str, Any]]) -> str:
    if not recent_frames:
        return "(no recent frames/actions)"
    lines: list[str] = []
    for idx, item in enumerate(recent_frames, start=1):
        pieces = []
        action = _recent_value(item, "action")
        obs = _recent_value(item, "observation_summary")
        result = _recent_value(item, "result_summary")
        if action:
            pieces.append(f"action={action}")
        if obs:
            pieces.append(f"observation={obs}")
        if result:
            pieces.append(f"result={result}")
        if _recent_value(item, "image") is not None:
            pieces.append(f"image=Recent frame {idx}")
        lines.append(f"{idx}. " + (" | ".join(pieces) if pieces else "(image only)"))
    return "\n".join(lines)


def _candidate_evidence_images(
    candidates: Sequence[dict[str, Any]],
    library: dict[str, dict],
    *,
    max_images: int,
) -> list[tuple[str, Any]]:
    if max_images <= 0:
        return []
    images: list[tuple[str, Any]] = []
    seen_paths: set[str] = set()
    for entry in candidates:
        skill_id = _entry_id(entry)
        record = library.get(skill_id) or {}
        for path in _record_evidence_image_paths(record):
            if path in seen_paths:
                continue
            image = _load_image_if_available(path)
            if image is None:
                continue
            seen_paths.add(path)
            images.append((f"skill_id={skill_id} path={path}", image))
            if len(images) >= max_images:
                return images
    return images


def _record_evidence_image_paths(record: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    retrieval = record.get("retrieval") or {}
    for item in retrieval.get("visual_evidence") or []:
        if isinstance(item, dict) and item.get("image_path"):
            paths.append(str(item["image_path"]))
    example = record.get("example") or {}
    if example.get("image_path"):
        paths.append(str(example["image_path"]))
    return paths


def _load_image_if_available(path: str) -> Any | None:
    try:
        image_path = Path(path).expanduser()
        if not image_path.exists():
            return None
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _retrieval_entry_text(retrieval: dict[str, Any]) -> str:
    return " ".join(
        [
            str(retrieval.get("page_state_summary") or ""),
            " ".join(str(x) for x in retrieval.get("query_terms") or []),
            " ".join(str(x) for x in retrieval.get("text_evidence") or []),
        ]
    )


def _selector_query_text(
    task: str,
    current_observation: str,
    recent_frames: Sequence[RecentFrameContext | dict[str, Any]],
) -> str:
    return " ".join([task or "", current_observation or "", " ".join(_recent_text(item) for item in recent_frames)])


def _recent_text(item: RecentFrameContext | dict[str, Any]) -> str:
    return " ".join(
        str(_recent_value(item, key) or "")
        for key in ("action", "observation_summary", "result_summary")
    )


def _recent_value(item: RecentFrameContext | dict[str, Any], key: str) -> Any:
    if isinstance(item, RecentFrameContext):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key)
    return None


def _parse_signature(signature: str) -> tuple[list[str], list[str], list[str]]:
    roles: list[str] = []
    actions: list[str] = []
    value_kinds: list[str] = []
    for motif in str(signature or "").split(" -> "):
        role, action, value = (motif.split(":") + ["", "", ""])[:3]
        if role:
            roles.append(role)
        if action:
            actions.append(action)
        if value:
            value_kinds.append(value)
    return roles, actions, value_kinds


def _entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("skill_id") or entry.get("experience_id") or entry.get("id") or "")


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def _has_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        return low
    return min(high, max(low, number))


def _extract_json_object(text: str) -> dict[str, Any]:
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
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("selector JSON must be an object")
    return parsed
