"""Online Stage-A/B experience writer for successful live trajectories.

Unlike the offline Stage-B miner, online updates are allowed to create or
refresh an experience from a single successful trajectory. Therefore new
records intentionally do not require or emit ``supporting_trajectories``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from guimemorysystem.engine import EngineProtocol

logger = logging.getLogger(__name__)


ONLINE_STAGE_A_SYSTEM = """You are an online trajectory summarizer for GUI agents.

You receive one successful live GUI trajectory. Extract only reusable GUI
interaction lessons. Do not summarize the whole task.

Output strict JSON only:
{
  "goal": "<abstract task intent without one-off values>",
  "task_shape": "search|filter|form|checkout|navigation|booking|account|content|other",
  "turning_points": [
    {
      "step": 1,
      "pre_state": "<short UI state before the action>",
      "action": "<short action description>",
      "post_state": "<short UI state after the action>",
      "commit_signal": ["visible cue that made this action correct"],
      "failure_if_skipped": "<concrete mistake this prevents>",
      "generalizable_pattern": "<short snake_case pattern>"
    }
  ],
  "rejected_branches": ["tempting wrong alternatives avoided"],
  "outcome": "<one sentence explaining why the trajectory succeeded>"
}

Rules:
- Keep only transferable state-transition rules.
- Prefer confirmation, modal discipline, branch control, autocomplete
  commitment, required-field completion, and semantic-click targeting.
- If no reusable rule exists, use an empty turning_points list.
- Do not include coordinates unless they are essential to the rule.
- Do not output markdown fences.
"""


ONLINE_STAGE_A_USER = """Task:
{task}

Task id: {task_id}
Start URL: {start_url}

Successful trajectory:
{steps_block}

Emit the online Stage-A summary JSON now."""


ONLINE_STAGE_B_SYSTEM = """You convert one successful trajectory summary into
zero or more reusable GUI-agent experiences.

Output strict JSON only:
{
  "experiences": [
    {
      "proposed_id": "short_snake_case_slug",
      "title": "<= 10 word GUI-rule title",
      "applicable_context": {
        "when": "triggering UI situation",
        "ui_signals": ["concrete visible cues"],
        "domain_hint": "generic|travel|shopping|info|entertainment|service"
      },
      "action_guidance": "what to do and why",
      "action_templates": ["<= 2 short templates"],
      "prevents_mistake": "concrete mistake this rule prevents",
      "trigger_ui_state": "short phrase for the pre-state",
      "forbidden_alternative": "tempting wrong action to avoid",
      "expected_postcondition": "what should become committed after the action",
      "confidence": 0.0
    }
  ]
}

Important:
- Do NOT emit supporting_trajectories.
- Do NOT require multiple supporting tasks; this is an online single-success update.
- Drop task skeletons like "search then select". Keep interaction rules that
  prevent concrete mistakes.
- Do not output markdown fences.
"""


ONLINE_STAGE_B_USER = """Online Stage-A summary:
{summary_json}

Emit the online Stage-B experiences JSON now."""


@dataclass
class OnlineExperienceStoreConfig:
    summary_buffer_path: str = "outputs/cross_task_experience/summary_buffer_online.jsonl"
    experience_library_path: str = "outputs/cross_task_experience/experience_library_online.jsonl"
    catalog_path: str = "outputs/cross_task_experience/catalog_online.json"
    catalog_cap: int = 100
    max_stage_a_tokens: int = 1800
    max_stage_b_tokens: int = 1800


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def _json_default(value: Any) -> str:
    return str(value)


def _format_steps_block(steps: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in steps:
        parts = [
            f"{step.get('step', '?')}: {step.get('action_repr', '')}",
            f"type={step.get('action_type', '')}",
        ]
        if step.get("value"):
            parts.append(f"value={step['value']!r}")
        if step.get("before_url"):
            parts.append(f"before_url={step['before_url']}")
        if step.get("after_url"):
            parts.append(f"after_url={step['after_url']}")
        if step.get("reasoning"):
            parts.append(f"reason={str(step['reasoning'])[:240]}")
        if step.get("state_after_summary"):
            parts.append(f"after={str(step['state_after_summary'])[:240]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def summarize_successful_trajectory(
    *,
    engine: EngineProtocol,
    task_id: str,
    task: str,
    start_url: str,
    steps: list[dict[str, Any]],
    max_tokens: int,
) -> dict:
    messages = [
        {"role": "system", "content": ONLINE_STAGE_A_SYSTEM},
        {
            "role": "user",
            "content": ONLINE_STAGE_A_USER.format(
                task=task,
                task_id=task_id,
                start_url=start_url,
                steps_block=_format_steps_block(steps),
            ),
        },
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    summary = _extract_json_object(raw)
    if not isinstance(summary.get("turning_points", []), list):
        raise ValueError("Stage-A summary field turning_points must be a list")
    return {
        "annotation_id": task_id,
        "source": "online",
        "task": task,
        "start_url": start_url,
        "num_steps": len(steps),
        "summary": summary,
        "created_at": _today(),
    }


def extract_experiences_from_summary(
    *,
    engine: EngineProtocol,
    summary_record: dict,
    max_tokens: int,
) -> list[dict]:
    messages = [
        {"role": "system", "content": ONLINE_STAGE_B_SYSTEM},
        {
            "role": "user",
            "content": ONLINE_STAGE_B_USER.format(
                summary_json=json.dumps(summary_record["summary"], ensure_ascii=False)
            ),
        },
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    parsed = _extract_json_object(raw)
    experiences = parsed.get("experiences") or []
    if not isinstance(experiences, list):
        raise ValueError("Stage-B output field experiences must be a list")
    return [_clean_experience(exp) for exp in experiences if _is_valid_experience(exp)]


def _is_valid_experience(exp: dict) -> bool:
    required = ("proposed_id", "title", "applicable_context", "action_guidance", "prevents_mistake")
    return isinstance(exp, dict) and all(exp.get(key) for key in required)


def _clean_experience(exp: dict) -> dict:
    context = exp.get("applicable_context") or {}
    templates = exp.get("action_templates") or []
    if not isinstance(templates, list):
        templates = []
    cleaned = {
        "proposed_id": str(exp["proposed_id"]),
        "title": str(exp["title"]),
        "applicable_context": {
            "when": str(context.get("when", "")),
            "ui_signals": list(context.get("ui_signals") or []),
            "domain_hint": str(context.get("domain_hint", "generic") or "generic"),
        },
        "action_guidance": str(exp["action_guidance"]),
        "action_templates": [str(item) for item in templates[:2]],
        "prevents_mistake": str(exp.get("prevents_mistake", "")),
        "trigger_ui_state": str(exp.get("trigger_ui_state", "")),
        "forbidden_alternative": str(exp.get("forbidden_alternative", "")),
        "expected_postcondition": str(exp.get("expected_postcondition", "")),
        "confidence": float(exp.get("confidence") or 0.0),
    }
    # Explicitly do not preserve supporting_trajectories for online experiences.
    return cleaned


def update_online_experience_store(
    *,
    engine: EngineProtocol,
    config: OnlineExperienceStoreConfig,
    task_id: str,
    task: str,
    start_url: str,
    steps: list[dict[str, Any]],
) -> dict:
    summary_record = summarize_successful_trajectory(
        engine=engine,
        task_id=task_id,
        task=task,
        start_url=start_url,
        steps=steps,
        max_tokens=config.max_stage_a_tokens,
    )
    experiences = extract_experiences_from_summary(
        engine=engine,
        summary_record=summary_record,
        max_tokens=config.max_stage_b_tokens,
    )

    summary_path = Path(config.summary_buffer_path)
    library_path = Path(config.experience_library_path)
    catalog_path = Path(config.catalog_path)
    for path in (summary_path, library_path, catalog_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with _file_lock(catalog_path):
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_record, ensure_ascii=False, default=_json_default) + "\n")

        library = _load_library(library_path)
        changed_ids = _upsert_experiences(library, experiences)
        _write_library(library_path, library)
        catalog = _build_catalog(library, cap=config.catalog_cap)
        _write_json_atomic(catalog_path, catalog)

    return {
        "summary_path": str(summary_path),
        "library_path": str(library_path),
        "catalog_path": str(catalog_path),
        "num_experiences": len(experiences),
        "experience_ids": changed_ids,
    }


def _upsert_experiences(library: dict[str, dict], experiences: list[dict]) -> list[str]:
    changed: list[str] = []
    timestamp = _today()
    existing_by_title = {_normalize_title(v.get("title", "")): k for k, v in library.items()}
    for exp in experiences:
        title_key = _normalize_title(exp["title"])
        exp_id = existing_by_title.get(title_key)
        if not exp_id:
            exp_id = _new_experience_id(exp, set(library))
        existing = library.get(exp_id, {})
        record = {
            "experience_id": exp_id,
            "title": exp["title"],
            "applicable_context": exp["applicable_context"],
            "action_guidance": exp["action_guidance"],
            "action_templates": exp.get("action_templates") or [],
            "prevents_mistake": exp.get("prevents_mistake", ""),
            "trigger_ui_state": exp.get("trigger_ui_state", ""),
            "forbidden_alternative": exp.get("forbidden_alternative", ""),
            "expected_postcondition": exp.get("expected_postcondition", ""),
            "confidence": exp.get("confidence") or existing.get("confidence", 0.0),
            "source": "online",
            "last_updated": timestamp,
            "created_at": existing.get("created_at", timestamp),
        }
        library[exp_id] = record
        existing_by_title[title_key] = exp_id
        changed.append(exp_id)
    return changed


def _new_experience_id(exp: dict, used_ids: set[str]) -> str:
    slug_source = exp.get("proposed_id") or exp.get("title") or "online_pattern"
    slug = re.sub(r"[^a-z0-9]+", "_", slug_source.lower()).strip("_")
    base = f"exp_{slug}" if slug and not slug.startswith("exp_") else slug or "exp_online_pattern"
    exp_id = base
    suffix = 2
    while exp_id in used_ids:
        exp_id = f"{base}_{suffix}"
        suffix += 1
    return exp_id


def _load_library(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    library: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed experience record in %s", path)
                continue
            exp_id = record.get("experience_id")
            if exp_id:
                library[exp_id] = record
    return library


def _write_library(path: Path, library: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in library.values():
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    tmp.replace(path)


def _build_catalog(library: dict[str, dict], cap: int) -> list[dict]:
    entries: list[dict] = []
    for record in library.values():
        when = (record.get("applicable_context") or {}).get("when", "")
        entries.append(
            {
                "id": record["experience_id"],
                "experience_id": record["experience_id"],
                "title": record.get("title", ""),
                "trigger": when,
                "when": when,
                "_last_updated": record.get("last_updated", ""),
            }
        )
    entries.sort(key=lambda entry: entry.get("_last_updated", ""), reverse=True)
    for entry in entries:
        entry.pop("_last_updated", None)
    return entries[:cap]


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title: str) -> str:
    return " ".join(_NORM_RE.sub(" ", title.lower()).split())


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
