"""Visual-skill cross-task memory mining (v2/v3).

This module keeps the old text-only cross-task memory path intact and adds a
separate offline miner for reusable visual GUI skills.  The miner consumes
expert trajectory steps, abstracts each target into an ``image + bbox + action``
event, counts recurring step/segment motifs, and writes a visual skill store:

``catalog.json``
    Lightweight selector entries.
``skill_library.jsonl``
    Full skill records with action templates, support stats, and example image.
``support/<skill_id>.jsonl``
    The source occurrences used to create each skill.
``images/<skill_id>.png``
    An annotated representative screenshot when the source image is available.

The first implementation is deliberately deterministic and dataset tolerant.
It supports the standard JSONL shape we want going forward, AndroidControl
JSON-list rows, and Multimodal-Mind2Web JSONL rows.

The v3 store keeps the same deterministic support filtering, but changes the
skill payload from visual-grounding demonstrations to UI planning skills:
historical screenshots are evidence for retrieval/state matching, while the
runtime injection is primarily procedural guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_NON_INTERACTIVE_ACTIONS = {"", "unknown", "noop", "none", "status", "wait"}
_LOCALIZATION_FREE_ACTIONS = {"scroll", "press_key"}
_GENERIC_SINGLE_CLICK_ROLES = {
    "button",
    "confirm_button",
    "dismiss_or_back_control",
    "dropdown_or_select",
    "menu_control",
    "search_bar",
    "search_button",
    "tab",
    "target_element",
    "text_field",
    "text_option",
}


@dataclass(frozen=True)
class BBox:
    """Pixel-space bounding box in x1/y1/x2/y2 form."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clamp(self, width: int | None = None, height: int | None = None) -> "BBox":
        x1 = max(0.0, self.x1)
        y1 = max(0.0, self.y1)
        x2 = self.x2
        y2 = self.y2
        if width is not None:
            x1 = min(float(width), x1)
            x2 = min(float(width), max(x1, x2))
        if height is not None:
            y1 = min(float(height), y1)
            y2 = min(float(height), max(y1, y2))
        return BBox(x1, y1, x2, y2)

    def to_list(self) -> list[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]


@dataclass
class VisualSkillStep:
    """One normalized expert action target."""

    source_id: str
    task_id: str
    step_index: int
    task: str
    action_type: str
    action_value: str = ""
    bbox: BBox | None = None
    screenshot_path: str = ""
    screenshot_width: int | None = None
    screenshot_height: int | None = None
    target_role: str = "target_element"
    target_text: str = ""
    visual_cues: list[str] = field(default_factory=list)
    domain: str = "generic"
    app: str = "generic"
    dataset: str = "standard"
    postcondition_hint: str = ""

    @property
    def value_kind(self) -> str:
        return infer_value_kind(self.action_value, self.task)

    @property
    def motif(self) -> str:
        return f"{self.target_role}:{self.action_type}:{self.value_kind}"

    def as_support_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "task": self.task,
            "action_type": self.action_type,
            "action_value": self.action_value,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "screenshot_path": self.screenshot_path or None,
            "target_role": self.target_role,
            "target_text": self.target_text,
            "visual_cues": self.visual_cues,
            "domain": self.domain,
            "app": self.app,
            "dataset": self.dataset,
        }


@dataclass
class VisualSkillOccurrence:
    """A contiguous sequence of normalized steps from one trajectory."""

    steps: list[VisualSkillStep]

    @property
    def signature(self) -> str:
        return " -> ".join(step.motif for step in self.steps)

    @property
    def task_id(self) -> str:
        return self.steps[0].task_id

    @property
    def domain(self) -> str:
        return self.steps[0].domain

    @property
    def app(self) -> str:
        return self.steps[0].app

    def as_support_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "task_id": self.task_id,
            "domain": self.domain,
            "app": self.app,
            "steps": [step.as_support_dict() for step in self.steps],
        }


@dataclass
class VisualSkillCandidate:
    """A mined visual skill before it is written as a library record."""

    skill_id: str
    signature: str
    title: str
    occurrences: list[VisualSkillOccurrence]
    support_steps: int
    support_tasks: int
    support_domains: int
    support_apps: int
    dominant_roles: list[str]
    dominant_actions: list[str]
    representative: VisualSkillOccurrence
    confidence: float
    family: str = ""
    variant: str = ""


@dataclass
class VisualSkillMiningConfig:
    """Controls deterministic visual skill mining."""

    min_support: int = 5
    min_tasks: int = 3
    min_domains: int = 1
    max_segment_len: int = 3
    max_examples_per_skill: int = 40
    catalog_cap: int = 200
    include_single_step_skills: bool = True
    filter_low_information_skills: bool = True
    compact_v3_skills: bool = True
    v3_keep_single_step_skills: bool = False
    v3_subsequence_task_coverage_threshold: float = 0.65


def load_offline_steps(
    path: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
) -> list[VisualSkillStep]:
    """Load and normalize expert trajectory rows from JSON/JSONL input."""
    records = list(_read_records(Path(path)))
    if dataset == "auto":
        dataset = _detect_dataset(records)
    return normalize_records(records, dataset=dataset, image_root=image_root)


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    dataset: str = "standard",
    image_root: str | Path | None = None,
) -> list[VisualSkillStep]:
    """Normalize supported offline row schemas into ``VisualSkillStep`` rows."""
    steps: list[VisualSkillStep] = []
    root = Path(image_root) if image_root else None
    for idx, record in enumerate(records):
        try:
            if dataset == "android_control":
                step = _normalize_android_control(record, idx, root)
            elif dataset == "mind2web":
                step = _normalize_mind2web(record, idx, root)
            else:
                step = _normalize_standard(record, idx, root)
        except Exception as exc:
            logger.warning("Skipping visual skill row %s: %s", idx, exc)
            continue
        if step.action_type in _LOCALIZATION_FREE_ACTIONS:
            step.bbox = None
        if step.action_type not in _NON_INTERACTIVE_ACTIONS and step.target_role != "noise":
            steps.append(step)
    return steps


def mine_visual_skill_candidates(
    steps: Sequence[VisualSkillStep],
    config: VisualSkillMiningConfig | None = None,
) -> list[VisualSkillCandidate]:
    """Mine recurring visual step/segment motifs from normalized steps."""
    config = config or VisualSkillMiningConfig()
    grouped = _group_steps_by_task(steps)
    occurrences_by_signature: dict[str, list[VisualSkillOccurrence]] = defaultdict(list)

    for task_steps in grouped.values():
        task_steps = sorted(task_steps, key=lambda item: item.step_index)
        for start in range(len(task_steps)):
            max_len = min(config.max_segment_len, len(task_steps) - start)
            for seg_len in range(1, max_len + 1):
                if seg_len == 1 and not config.include_single_step_skills:
                    continue
                occurrence = VisualSkillOccurrence(task_steps[start : start + seg_len])
                if config.filter_low_information_skills and _is_low_information_occurrence(occurrence):
                    continue
                occurrences_by_signature[occurrence.signature].append(occurrence)

    candidates: list[VisualSkillCandidate] = []
    for signature, occurrences in occurrences_by_signature.items():
        support_tasks = len({item.task_id for item in occurrences})
        support_domains = len({item.domain for item in occurrences})
        if (
            len(occurrences) < config.min_support
            or support_tasks < config.min_tasks
            or support_domains < config.min_domains
        ):
            continue
        trimmed = _trim_occurrences(occurrences, config.max_examples_per_skill)
        representative = _choose_representative(trimmed)
        roles = _counter_top(step.target_role for occ in trimmed for step in occ.steps)
        actions = _counter_top(step.action_type for occ in trimmed for step in occ.steps)
        skill_id = _skill_id_for_signature(signature)
        confidence = _confidence(
            occurrences=occurrences,
            support_tasks=support_tasks,
            support_domains=support_domains,
        )
        candidates.append(
            VisualSkillCandidate(
                skill_id=skill_id,
                signature=signature,
                title=_title_for_signature(signature),
                occurrences=trimmed,
                support_steps=len(occurrences),
                support_tasks=support_tasks,
                support_domains=support_domains,
                support_apps=len({item.app for item in occurrences}),
                dominant_roles=roles,
                dominant_actions=actions,
                representative=representative,
                confidence=confidence,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.support_tasks,
            item.support_domains,
            item.support_steps,
            item.confidence,
        ),
        reverse=True,
    )
    return candidates


def compact_visual_skill_v3_candidates(
    candidates: Sequence[VisualSkillCandidate],
    *,
    keep_single_step: bool = False,
    subsequence_task_coverage_threshold: float = 0.65,
) -> tuple[list[VisualSkillCandidate], dict[str, int]]:
    """Collapse sliding-window fragments into larger v3 planning skills.

    Mining still counts every contiguous segment so support is measured fairly.
    This pass decides which segments deserve to be standalone planning skills:
    single-step affordances are dropped by default, and shorter n-grams are
    removed when their task support is mostly covered by longer n-grams that
    contain them as contiguous subsequences.
    """
    kept: list[VisualSkillCandidate] = []
    dropped_single = 0
    dropped_subsequence = 0

    indexed = [
        (
            candidate,
            _signature_motifs(candidate.signature),
            _candidate_task_ids(candidate),
        )
        for candidate in candidates
    ]

    for candidate, motifs, task_ids in indexed:
        if len(motifs) <= 1 and not keep_single_step:
            dropped_single += 1
            continue

        longer_matches: list[set[str]] = []
        for other, other_motifs, other_task_ids in indexed:
            if other is candidate or len(other_motifs) <= len(motifs):
                continue
            if _contains_contiguous_subsequence(other_motifs, motifs):
                longer_matches.append(other_task_ids)

        if longer_matches and task_ids:
            covered_tasks = set().union(*longer_matches)
            coverage = len(task_ids & covered_tasks) / len(task_ids)
            if coverage >= subsequence_task_coverage_threshold:
                dropped_subsequence += 1
                continue

        kept.append(candidate)

    return kept, {
        "num_candidates_before_compaction": len(candidates),
        "num_candidates_after_compaction": len(kept),
        "num_dropped_single_step": dropped_single,
        "num_dropped_subsequence": dropped_subsequence,
    }


def mine_visual_skill_v3_family_candidates(
    steps: Sequence[VisualSkillStep],
    config: VisualSkillMiningConfig | None = None,
) -> tuple[list[VisualSkillCandidate], dict[str, int]]:
    """Mine v3 skills as semantic interaction families, not action n-grams."""
    config = config or VisualSkillMiningConfig()
    grouped = _group_steps_by_task(steps)
    occurrences_by_key: dict[tuple[str, str], list[VisualSkillOccurrence]] = defaultdict(list)
    raw_episode_count = 0

    for task_steps in grouped.values():
        task_steps = sorted(task_steps, key=lambda item: item.step_index)
        idx = 0
        while idx < len(task_steps):
            episode = _extract_v3_episode_at(task_steps, idx)
            if episode is None:
                idx += 1
                continue
            family, variant, occurrence, next_idx = episode
            if len(occurrence.steps) < 2 or _is_low_information_v3_episode(family, occurrence):
                idx = max(idx + 1, next_idx)
                continue
            occurrences_by_key[(family, variant)].append(occurrence)
            raw_episode_count += 1
            idx = max(idx + 1, next_idx)

    candidates: list[VisualSkillCandidate] = []
    for (family, variant), occurrences in occurrences_by_key.items():
        support_tasks = len({item.task_id for item in occurrences})
        support_domains = len({item.domain for item in occurrences})
        if (
            len(occurrences) < config.min_support
            or support_tasks < config.min_tasks
            or support_domains < config.min_domains
        ):
            continue

        trimmed = _trim_occurrences(occurrences, config.max_examples_per_skill)
        representative = _choose_representative(trimmed)
        dominant_roles = _counter_top(step.target_role for occ in trimmed for step in occ.steps)
        dominant_actions = _counter_top(step.action_type for occ in trimmed for step in occ.steps)
        signature = _family_signature_for(family, variant, trimmed)
        candidates.append(
            VisualSkillCandidate(
                skill_id=_v3_skill_id_for_family(family, variant),
                signature=signature,
                title=_v3_family_title(family, variant),
                occurrences=trimmed,
                support_steps=len(occurrences),
                support_tasks=support_tasks,
                support_domains=support_domains,
                support_apps=len({item.app for item in occurrences}),
                dominant_roles=dominant_roles,
                dominant_actions=dominant_actions,
                representative=representative,
                confidence=_confidence(
                    occurrences=occurrences,
                    support_tasks=support_tasks,
                    support_domains=support_domains,
                ),
                family=family,
                variant=variant,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.support_tasks,
            item.support_domains,
            item.support_steps,
            item.confidence,
        ),
        reverse=True,
    )
    return candidates, {
        "num_v3_raw_episodes": raw_episode_count,
        "num_v3_family_keys": len(occurrences_by_key),
        "num_v3_family_candidates": len(candidates),
    }


def build_visual_skill_record(
    candidate: VisualSkillCandidate,
    *,
    output_dir: str | Path | None = None,
    draw_example: bool = True,
) -> dict[str, Any]:
    """Convert one candidate to the v2 skill library schema."""
    example_image = ""
    if output_dir and draw_example:
        example_image = _write_annotated_example(candidate, Path(output_dir))

    first = candidate.representative.steps[0]
    action_templates = _action_templates_for(candidate)
    record = {
        "version": "visual_skill_v2",
        "skill_id": candidate.skill_id,
        "experience_id": candidate.skill_id,
        "title": candidate.title,
        "signature": candidate.signature,
        "applicable_context": {
            "when": _when_for_candidate(candidate),
            "visual_cues": _visual_cues_for(candidate),
            "domain_hint": _domain_hint_for(candidate),
        },
        "target_instruction": _target_instruction_for(candidate),
        "action_guidance": _action_guidance_for(candidate),
        "action_templates": action_templates,
        "value_slots": _value_slots_for(candidate),
        "expected_postcondition": _postcondition_for(candidate),
        "forbidden_alternative": _avoid_for(candidate),
        "example": {
            "image_path": example_image or None,
            "bbox": first.bbox.to_list() if first.bbox else None,
            "target_role": first.target_role,
            "target_text": first.target_text,
        },
        "support": {
            "num_occurrences": candidate.support_steps,
            "num_tasks": candidate.support_tasks,
            "num_domains": candidate.support_domains,
            "num_apps": candidate.support_apps,
            "dominant_roles": candidate.dominant_roles,
            "dominant_actions": candidate.dominant_actions,
        },
        "confidence": candidate.confidence,
        "source": "offline_visual_skill_mining_v2",
    }
    return record


def build_visual_skill_v3_record(
    candidate: VisualSkillCandidate,
    *,
    output_dir: str | Path | None = None,
    draw_example: bool = True,
) -> dict[str, Any]:
    """Convert one candidate to the v3 procedural skill schema."""
    example_image = ""
    if output_dir and draw_example:
        example_image = _write_annotated_example(candidate, Path(output_dir))

    family = candidate.family or _v3_family_from_signature(candidate.signature)
    variant = candidate.variant or "generic"
    skill_id = candidate.skill_id or _v3_skill_id_for_family(family, variant)
    action_templates = _action_templates_for(candidate)
    record = {
        "version": "visual_skill_v3",
        "skill_type": "ui_planning_skill",
        "family": family,
        "variant": variant,
        "skill_id": skill_id,
        "experience_id": skill_id,
        "title": _v3_family_title(family, variant),
        "signature": candidate.signature,
        "intent": _v3_family_intent(family, variant),
        "applicable_context": {
            "when": _v3_family_when(family, variant),
            "preconditions": _v3_family_preconditions(family, variant),
            "page_state_cues": _visual_cues_for(candidate),
            "task_goal_cues": _v3_task_goal_cues_for(candidate),
            "negative_conditions": _v3_family_negative_conditions(family),
            "domain_hint": _domain_hint_for(candidate),
        },
        "planning": {
            "procedure": _v3_family_procedure(family, variant),
            "action_templates": action_templates,
            "postcondition_checks": _v3_family_postcondition_checks(family),
            "failure_modes": _v3_family_failure_modes(family),
            "recovery_steps": _v3_family_recovery_steps(family),
        },
        "retrieval": {
            "query_terms": _v3_query_terms_for(candidate),
            "page_state_summary": _v3_page_state_summary_for(candidate),
            "action_pattern": candidate.signature,
            "visual_evidence": _v3_visual_evidence_for(candidate, example_image=example_image),
            "text_evidence": _v3_text_evidence_for(candidate),
        },
        # Compatibility fields let the existing selector and prompt renderer
        # consume v3 stores without a separate policy path.
        "target_instruction": "Match the current page state to the preconditions, then use the procedural plan.",
        "action_guidance": "Use the planning procedure only when the current task and page state match the mined evidence.",
        "action_templates": action_templates,
        "value_slots": _value_slots_for(candidate),
        "expected_postcondition": "; ".join(_v3_postcondition_checks_for(candidate)),
        "forbidden_alternative": _avoid_for(candidate),
        "example": {
            "image_path": example_image or None,
            "bbox": candidate.representative.steps[0].bbox.to_list()
            if candidate.representative.steps[0].bbox
            else None,
            "target_role": candidate.representative.steps[0].target_role,
            "target_text": candidate.representative.steps[0].target_text,
        },
        "support": {
            "num_occurrences": candidate.support_steps,
            "num_trajectory_segments": candidate.support_steps,
            "num_tasks": candidate.support_tasks,
            "num_domains": candidate.support_domains,
            "num_apps": candidate.support_apps,
            "dominant_roles": candidate.dominant_roles,
            "dominant_actions": candidate.dominant_actions,
            "action_pattern_agreement": _v3_action_pattern_agreement(candidate),
            "qualification": (
                "Mined only after multiple semantic episodes mapped to the same "
                "interaction family and passed support/task/domain filters."
            ),
        },
        "confidence": candidate.confidence,
        "source": "offline_visual_skill_mining_v3",
    }
    return record


def build_visual_skill_v4_record(
    candidate: VisualSkillCandidate,
    *,
    output_dir: str | Path | None = None,
    draw_example: bool = True,
) -> dict[str, Any]:
    """Convert one candidate to the v4 visual episode memory schema.

    v4 stores the concrete action plan from real past trajectories keyed to a
    specific trigger visual state, not an abstract interaction family.  The VLM
    selector matches the current screenshot against the trigger state and injects
    the grounded plan as a planning template.
    """
    example_image = ""
    if output_dir and draw_example:
        example_image = _write_annotated_example(candidate, Path(output_dir))

    rep = candidate.representative
    trigger_step = rep.steps[0]
    action_plan_steps = _v4_action_plan_steps(rep)
    slot_analysis = _v4_slot_analysis(candidate)

    record = {
        "version": "visual_skill_v4",
        "skill_type": "visual_episode_memory",
        "skill_id": candidate.skill_id,
        "experience_id": candidate.skill_id,
        "title": _v4_title(candidate),
        "signature": candidate.signature,
        "trigger_visual_state": {
            "screenshot_path": trigger_step.screenshot_path or None,
            "visual_cues": trigger_step.visual_cues[:6],
            "target_role": trigger_step.target_role,
            "domain": trigger_step.domain,
            "app": trigger_step.app,
        },
        "action_plan": {
            "steps": action_plan_steps,
            "plan_description": _v4_plan_description(rep),
            "num_steps": len(rep.steps),
            "slot_analysis": slot_analysis,
        },
        "applicable_context": {
            "when": _v4_when(candidate),
            "visual_triggers": _v4_visual_triggers(candidate),
            "task_keywords": _v4_task_keywords(candidate),
            "negative_conditions": _v4_negative_conditions(candidate),
        },
        "retrieval": {
            "query_terms": _v4_query_terms(candidate),
            "trigger_state_summary": _v4_trigger_state_summary(candidate),
            "example_tasks": _v4_example_tasks(candidate),
            "visual_evidence": _v4_visual_evidence(candidate, example_image=example_image),
        },
        # Compat fields so existing selector and prompt renderer work unchanged.
        "target_instruction": _v4_plan_description(rep),
        "action_guidance": _v4_when(candidate),
        "action_templates": _action_templates_for(candidate),
        "value_slots": _value_slots_for(candidate),
        "expected_postcondition": _postcondition_for(candidate),
        "forbidden_alternative": _avoid_for(candidate),
        "example": {
            "image_path": example_image or None,
            "bbox": trigger_step.bbox.to_list() if trigger_step.bbox else None,
            "target_role": trigger_step.target_role,
            "target_text": trigger_step.target_text,
        },
        "support": {
            "num_occurrences": candidate.support_steps,
            "num_tasks": candidate.support_tasks,
            "num_domains": candidate.support_domains,
            "num_apps": candidate.support_apps,
            "dominant_roles": candidate.dominant_roles,
            "dominant_actions": candidate.dominant_actions,
        },
        "confidence": candidate.confidence,
        "source": "offline_visual_skill_mining_v4",
    }
    return record


def write_visual_skill_store(
    candidates: Sequence[VisualSkillCandidate],
    output_dir: str | Path,
    *,
    catalog_cap: int = 200,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """Write catalog, library, support rows, and annotated images."""
    out = Path(output_dir)
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "support", ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "support").mkdir(parents=True, exist_ok=True)

    library_path = out / "skill_library.jsonl"
    catalog_path = out / "catalog.json"
    records: list[dict[str, Any]] = []

    with library_path.open("w", encoding="utf-8") as library_handle:
        for candidate in candidates:
            record = build_visual_skill_record(candidate, output_dir=out, draw_example=draw_examples)
            records.append(record)
            library_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            support_path = out / "support" / f"{candidate.skill_id}.jsonl"
            with support_path.open("w", encoding="utf-8") as support_handle:
                for occurrence in candidate.occurrences:
                    support_handle.write(json.dumps(occurrence.as_support_dict(), ensure_ascii=False) + "\n")

    catalog = [_catalog_entry(record) for record in records[:catalog_cap]]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(out),
        "library_path": str(library_path),
        "catalog_path": str(catalog_path),
        "num_skills": len(records),
    }


def write_visual_skill_v3_store(
    candidates: Sequence[VisualSkillCandidate],
    output_dir: str | Path,
    *,
    catalog_cap: int = 200,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """Write catalog, library, support rows, and evidence images for v3."""
    out = Path(output_dir)
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "support", ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "support").mkdir(parents=True, exist_ok=True)

    library_path = out / "skill_library.jsonl"
    catalog_path = out / "catalog.json"
    records: list[dict[str, Any]] = []

    with library_path.open("w", encoding="utf-8") as library_handle:
        for candidate in candidates:
            record = build_visual_skill_v3_record(candidate, output_dir=out, draw_example=draw_examples)
            records.append(record)
            library_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            support_path = out / "support" / f"{record['skill_id']}.jsonl"
            with support_path.open("w", encoding="utf-8") as support_handle:
                for occurrence in candidate.occurrences:
                    support_handle.write(json.dumps(occurrence.as_support_dict(), ensure_ascii=False) + "\n")

    catalog = [_catalog_v3_entry(record) for record in records[:catalog_cap]]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(out),
        "library_path": str(library_path),
        "catalog_path": str(catalog_path),
        "num_skills": len(records),
        "version": "visual_skill_v3",
    }


def write_visual_skill_v4_store(
    candidates: Sequence[VisualSkillCandidate],
    output_dir: str | Path,
    *,
    catalog_cap: int = 200,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """Write catalog, library, support rows, and trigger images for v4."""
    out = Path(output_dir)
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "support", ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "support").mkdir(parents=True, exist_ok=True)

    library_path = out / "skill_library.jsonl"
    catalog_path = out / "catalog.json"
    records: list[dict[str, Any]] = []

    with library_path.open("w", encoding="utf-8") as library_handle:
        for candidate in candidates:
            record = build_visual_skill_v4_record(candidate, output_dir=out, draw_example=draw_examples)
            records.append(record)
            library_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            support_path = out / "support" / f"{record['skill_id']}.jsonl"
            with support_path.open("w", encoding="utf-8") as support_handle:
                for occurrence in candidate.occurrences:
                    support_handle.write(json.dumps(occurrence.as_support_dict(), ensure_ascii=False) + "\n")

    catalog = [_catalog_v4_entry(record) for record in records[:catalog_cap]]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(out),
        "library_path": str(library_path),
        "catalog_path": str(catalog_path),
        "num_skills": len(records),
        "version": "visual_skill_v4",
    }


def mine_visual_skills_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v2 mining helper."""
    config = config or VisualSkillMiningConfig()
    steps = load_offline_steps(input_path, dataset=dataset, image_root=image_root)
    candidates = mine_visual_skill_candidates(steps, config=config)
    result = write_visual_skill_store(
        candidates,
        output_dir,
        catalog_cap=config.catalog_cap,
        draw_examples=draw_examples,
    )
    result["num_steps"] = len(steps)
    return result


def mine_visual_skill_v3_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v3 mining helper."""
    config = config or VisualSkillMiningConfig()
    steps = load_offline_steps(input_path, dataset=dataset, image_root=image_root)
    candidates, mining_stats = mine_visual_skill_v3_family_candidates(steps, config=config)
    result = write_visual_skill_v3_store(
        candidates,
        output_dir,
        catalog_cap=config.catalog_cap,
        draw_examples=draw_examples,
    )
    result["num_steps"] = len(steps)
    result.update(mining_stats)
    return result


def mine_visual_skills_from_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v2 mining over multiple trajectory files."""
    config = config or VisualSkillMiningConfig()
    steps: list[VisualSkillStep] = []
    for input_path in input_paths:
        steps.extend(load_offline_steps(input_path, dataset=dataset, image_root=image_root))
    candidates = mine_visual_skill_candidates(steps, config=config)
    result = write_visual_skill_store(
        candidates,
        output_dir,
        catalog_cap=config.catalog_cap,
        draw_examples=draw_examples,
    )
    result["num_steps"] = len(steps)
    result["num_inputs"] = len(input_paths)
    return result


def mine_visual_skill_v3_from_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v3 mining over multiple trajectory files."""
    config = config or VisualSkillMiningConfig()
    steps: list[VisualSkillStep] = []
    for input_path in input_paths:
        steps.extend(load_offline_steps(input_path, dataset=dataset, image_root=image_root))
    candidates, mining_stats = mine_visual_skill_v3_family_candidates(steps, config=config)
    result = write_visual_skill_v3_store(
        candidates,
        output_dir,
        catalog_cap=config.catalog_cap,
        draw_examples=draw_examples,
    )
    result["num_steps"] = len(steps)
    result["num_inputs"] = len(input_paths)
    result.update(mining_stats)
    return result


def mine_visual_skill_v4_episodes(
    steps: Sequence[VisualSkillStep],
    config: VisualSkillMiningConfig | None = None,
) -> list[VisualSkillCandidate]:
    """Mine v4 visual episode memories: trigger visual state → concrete action plan.

    Reuses the v2 sliding-window support calculation with multi-step enforcement.
    The resulting candidates carry real trajectory steps as the plan payload
    instead of abstract family procedures.
    """
    cfg = config or VisualSkillMiningConfig()
    v4_cfg = VisualSkillMiningConfig(
        min_support=cfg.min_support,
        min_tasks=cfg.min_tasks,
        min_domains=cfg.min_domains,
        max_segment_len=cfg.max_segment_len,
        max_examples_per_skill=cfg.max_examples_per_skill,
        catalog_cap=cfg.catalog_cap,
        include_single_step_skills=False,
        filter_low_information_skills=cfg.filter_low_information_skills,
    )
    return mine_visual_skill_candidates(steps, v4_cfg)


def mine_visual_skill_v4_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v4 mining helper."""
    config = config or VisualSkillMiningConfig()
    steps = load_offline_steps(input_path, dataset=dataset, image_root=image_root)
    candidates = mine_visual_skill_v4_episodes(steps, config=config)
    result = write_visual_skill_v4_store(candidates, output_dir, catalog_cap=config.catalog_cap, draw_examples=draw_examples)
    result["num_steps"] = len(steps)
    return result


def mine_visual_skill_v4_from_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    dataset: str = "auto",
    image_root: str | Path | None = None,
    config: VisualSkillMiningConfig | None = None,
    draw_examples: bool = True,
) -> dict[str, str | int]:
    """End-to-end offline v4 mining over multiple trajectory files."""
    config = config or VisualSkillMiningConfig()
    steps: list[VisualSkillStep] = []
    for input_path in input_paths:
        steps.extend(load_offline_steps(input_path, dataset=dataset, image_root=image_root))
    candidates = mine_visual_skill_v4_episodes(steps, config=config)
    result = write_visual_skill_v4_store(candidates, output_dir, catalog_cap=config.catalog_cap, draw_examples=draw_examples)
    result["num_steps"] = len(steps)
    result["num_inputs"] = len(input_paths)
    return result


def infer_value_kind(value: str, task: str = "") -> str:
    text = f"{value} {task}".lower()
    if not value:
        return "none"
    if re.search(r"\b\d{1,2}[:/.-]\d{1,2}\b|\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec\b", text):
        return "date_or_time"
    if re.search(r"\b(search|find|look up|query|keyword|news about)\b", task.lower()):
        return "query"
    if len(value.split()) <= 4:
        return "option"
    return "free_text"


def infer_target_role(
    *,
    action_type: str,
    target_text: str = "",
    action_description: str = "",
    task: str = "",
    node: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Heuristic target-role labeler for v2 candidate mining."""
    node = node or {}
    class_name = str(node.get("class_name") or "").lower()
    resource = str(node.get("resource_name") or node.get("resource_id") or "").lower()
    hint = str(node.get("hint_text") or "").lower()
    content = str(node.get("content_description") or "").lower()
    target_context = _target_context_for_role(
        target_text=target_text,
        action_description=action_description,
        class_name=class_name,
        resource=resource,
        hint=hint,
        content=content,
    )
    quoted_label_context = " ".join(_quoted_phrases(target_text) + _quoted_phrases(action_description)).lower()
    combined = " ".join(
        [
            target_text.lower(),
            action_description.lower(),
            class_name,
            resource,
            hint,
            content,
        ]
    )

    cues: list[str] = []
    if target_text:
        cues.append(f"text={target_text[:80]}")
    if hint:
        cues.append(f"hint={hint[:80]}")
    if content:
        cues.append(f"content_description={content[:80]}")

    field_context = " ".join([target_text.lower(), action_description.lower(), resource, hint, content])

    if action_type == "input_text":
        if _has_any(field_context, ["date", "time", "calendar", "pickup", "return"]):
            return "date_or_time_field", cues + ["editable date/time field"]
        if _has_any(
            field_context,
            [
                "first name",
                "last name",
                "full name",
                "recipient",
                "email",
                "phone",
                "card number",
                "company",
                "short bio",
                "message field",
                "tracking number",
                "age field",
            ],
        ):
            return "text_field", cues + ["editable text field"]
        if _has_any(
            field_context,
            [
                "search",
                "find",
                "lookup",
                "look up",
                "query",
                "keyword",
                "where to",
                "destination",
                "location input",
                "city, state, or zip",
            ],
        ):
            return "search_bar", cues + ["editable query field"]
        return "text_field", cues + ["editable text field"]

    if _has_any(target_context, ["search result", "suggested search", "suggestion"]):
        return "text_option", cues + ["search suggestion/result option"]
    if _has_any(target_context, ["more actions", "more options", "navigation drawer"]):
        return "menu_control", cues + ["menu/navigation control"]
    if _has_any(target_context, ["top navigation", "navigation menu", "main navigation", "nav bar", "navigation bar"]):
        if _has_any(target_context, ["tab"]):
            return "tab", cues + ["navigation tab"]
        if action_type == "hover" or _has_any(target_context, ["menu", "category", "section"]):
            return "menu_control", cues + ["navigation/menu control"]
        return "text_option", cues + ["navigation option"]
    if _has_any(target_context, [" tab", "tab ", " tab in", " tab under"]) or _has_any(class_name, ["tab"]):
        return "tab", cues + ["tab control"]
    if _has_any(target_context, ["link", "option under", "option in", "menu option", "category"]):
        if _has_any(target_context, ["dropdown", "drop down"]):
            return "dropdown_or_select", cues + ["option selector"]
        return "text_option", cues + ["visible link/option"]
    if _has_any(target_context, ["search bar", "search box", "search field", "search input"]):
        return "search_bar", cues + ["visible query input"]
    if _has_any(target_context, ["date input", "time input", "date field", "time field", "calendar icon"]):
        return "date_or_time_field", cues + ["date/time entry control"]
    if _has_any(target_context, ["input field", "text field", "textbox", "text box"]):
        return "text_field", cues + ["text entry control"]
    if _has_any(target_context, ["search icon", "magnifying", "search button"]) or (
        action_type == "click"
        and _has_any(quoted_label_context or target_context, ["search cars", "search flights", "search hotels", "find your", "find flights", "find trains"])
    ):
        return "search_button", cues + ["search affordance"]
    if _has_any(target_context, ["search icon", "magnifying", "search button"]) or (
        action_type == "click" and (target_text.strip().lower() == "search" or quoted_label_context.strip() == "search")
    ):
        return "search_button", cues + ["search affordance"]
    if _has_any(class_name, ["checkbox"]) or bool(node.get("is_checkable")):
        return "checkbox", cues + ["checkable control"]
    if _has_any(target_context, ["dropdown", "drop down", "spinner", "select"]):
        return "dropdown_or_select", cues + ["option selector"]
    if _has_any(target_context, ["calendar", "date picker"]) or re.fullmatch(r"\d{1,2}", target_text.strip()):
        return "date_cell", cues + ["calendar/date target"]
    commit_context = quoted_label_context or target_context
    if _has_any(commit_context, ["apply", "done", "save", "confirm", "submit", "ok", "continue", "next", "refine results"]):
        return "confirm_button", cues + ["commit/confirmation control"]
    if _has_any(target_context, ["close", "cancel", "dismiss", "back", "navigate up"]):
        return "dismiss_or_back_control", cues + ["dismiss/navigation control"]
    if _has_any(class_name, ["button"]) or _has_any(resource, ["button", "btn"]):
        return "button", cues + ["button-like target"]
    if _has_any(target_context, ["menu"]):
        return "menu_control", cues + ["menu/navigation control"]
    if target_text:
        return "text_option", cues + ["visible text option"]
    return "target_element", cues


def normalize_action_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"type", "type_text", "input", "input_text", "text"}:
        return "input_text"
    if value in {"tap", "click"}:
        return "click"
    if value in {"hover"}:
        return "hover"
    if value in {"swipe", "scroll"}:
        return "scroll"
    if value in {"long_press", "long_click"}:
        return "long_click"
    if value in {"select"}:
        return "select"
    if value in {"press", "key", "press_key"}:
        return "press_key"
    return value or "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine visual skills from offline expert trajectories.")
    parser.add_argument("--input", required=True, nargs="+", help="Input JSON/JSONL trajectory file(s).")
    parser.add_argument("--output-dir", required=True, help="Output visual skill store directory.")
    parser.add_argument("--version", default="v2", choices=["v2", "v3", "v4"], help="Output skill schema version.")
    parser.add_argument("--dataset", default="auto", choices=["auto", "standard", "android_control", "mind2web"])
    parser.add_argument("--image-root", default="", help="Optional root for relative screenshot paths.")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-tasks", type=int, default=3)
    parser.add_argument("--min-domains", type=int, default=1)
    parser.add_argument("--max-segment-len", type=int, default=3)
    parser.add_argument("--max-examples-per-skill", type=int, default=40)
    parser.add_argument("--catalog-cap", type=int, default=200)
    parser.add_argument("--no-single-step", action="store_true")
    parser.add_argument("--keep-low-information", action="store_true")
    parser.add_argument("--no-v3-compact", action="store_true", help="Disable v3 maximal-segment compaction.")
    parser.add_argument("--v3-keep-single-step", action="store_true", help="Keep single-step skills in v3 output.")
    parser.add_argument("--v3-subsequence-task-coverage-threshold", type=float, default=0.65)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)

    config = VisualSkillMiningConfig(
        min_support=args.min_support,
        min_tasks=args.min_tasks,
        min_domains=args.min_domains,
        max_segment_len=args.max_segment_len,
        max_examples_per_skill=args.max_examples_per_skill,
        catalog_cap=args.catalog_cap,
        include_single_step_skills=not args.no_single_step,
        filter_low_information_skills=not args.keep_low_information,
        compact_v3_skills=not args.no_v3_compact,
        v3_keep_single_step_skills=args.v3_keep_single_step,
        v3_subsequence_task_coverage_threshold=args.v3_subsequence_task_coverage_threshold,
    )
    if args.version == "v4":
        result = mine_visual_skill_v4_from_files(
            args.input,
            args.output_dir,
            dataset=args.dataset,
            image_root=args.image_root or None,
            config=config,
            draw_examples=not args.no_images,
        )
    elif args.version == "v3":
        result = mine_visual_skill_v3_from_files(
            args.input,
            args.output_dir,
            dataset=args.dataset,
            image_root=args.image_root or None,
            config=config,
            draw_examples=not args.no_images,
        )
    else:
        result = mine_visual_skills_from_files(
            args.input,
            args.output_dir,
            dataset=args.dataset,
            image_root=args.image_root or None,
            config=config,
            draw_examples=not args.no_images,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _read_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield record
    elif isinstance(data, dict):
        rows = data.get("steps") or data.get("records") or data.get("data") or []
        if isinstance(rows, list):
            for record in rows:
                if isinstance(record, dict):
                    yield record
        else:
            yield data


def _detect_dataset(records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return "standard"
    first = records[0]
    if "accessibility_tree" in first and "episode_id" in first:
        return "android_control"
    if "annotation_id" in first and ("operation" in first or "action_uid" in first):
        return "mind2web"
    return "standard"


def _normalize_standard(record: dict[str, Any], idx: int, image_root: Path | None) -> VisualSkillStep:
    action_type = normalize_action_type(
        str(record.get("action_type") or record.get("operation") or record.get("action", ""))
    )
    action_value = str(record.get("action_value") or record.get("value") or record.get("text") or "")
    task = str(record.get("task") or record.get("goal") or record.get("instruction") or "")
    bbox = _bbox_from_record(record)
    target_text = str(record.get("target_text") or record.get("description") or record.get("element_text") or "")
    role, cues = infer_target_role(
        action_type=action_type,
        target_text=target_text,
        action_description=str(record.get("action_description") or ""),
        task=task,
    )
    return VisualSkillStep(
        source_id=str(_first_present(record, ("source_id", "action_uid"), idx)),
        task_id=str(_first_present(record, ("task_id", "annotation_id", "episode_id"), idx)),
        step_index=int(_first_present(record, ("step", "step_index"), idx)),
        task=task,
        action_type=action_type,
        action_value=action_value,
        bbox=bbox,
        screenshot_path=_resolve_image_path(record, image_root),
        screenshot_width=_optional_int(record.get("screenshot_width") or record.get("width")),
        screenshot_height=_optional_int(record.get("screenshot_height") or record.get("height")),
        target_role=role,
        target_text=target_text,
        visual_cues=cues,
        domain=str(record.get("domain") or "generic"),
        app=str(record.get("app") or record.get("website") or "generic"),
        dataset=str(record.get("dataset") or "standard"),
        postcondition_hint=str(record.get("postcondition") or record.get("state_after_summary") or ""),
    )


def _normalize_mind2web(record: dict[str, Any], idx: int, image_root: Path | None) -> VisualSkillStep:
    action_type = normalize_action_type(str(record.get("operation") or ""))
    action_value = str(record.get("value") or "")
    task = str(record.get("task") or "")
    target_text = str(record.get("description") or record.get("action_description") or "")
    action_description = str(record.get("action_description") or "")
    if action_type == "click" and action_description.strip().lower().startswith("hover"):
        action_type = "hover"
    role, cues = infer_target_role(
        action_type=action_type,
        target_text=target_text,
        action_description=action_description,
        task=task,
    )
    return VisualSkillStep(
        source_id=str(_first_present(record, ("action_uid",), idx)),
        task_id=str(_first_present(record, ("annotation_id",), idx)),
        step_index=int(_first_present(record, ("step",), idx)),
        task=task,
        action_type=action_type,
        action_value=action_value,
        bbox=_bbox_from_mind2web_record(record),
        screenshot_path=_resolve_mind2web_image_path(record, image_root),
        target_role=role,
        target_text=target_text,
        visual_cues=cues,
        domain=str(record.get("domain") or "generic"),
        app=str(record.get("website") or "generic"),
        dataset="mind2web",
    )


def _bbox_from_mind2web_record(record: dict[str, Any]) -> BBox | None:
    precise_bbox = _bbox_from_record(record)
    if precise_bbox:
        return precise_bbox

    target_blocks = record.get("target_blocks") or {}
    if isinstance(target_blocks, dict) and target_blocks:
        first_key = sorted(target_blocks, key=lambda item: int(item) if str(item).isdigit() else str(item))[0]
        block_bboxes = target_blocks.get(first_key) or []
        if block_bboxes:
            return _bbox_from_any(block_bboxes[0])
    return None


def _resolve_mind2web_image_path(record: dict[str, Any], image_root: Path | None) -> str:
    existing = _resolve_image_path(record, image_root)
    if existing:
        return existing
    if image_root is None or not record.get("blocks_path"):
        return ""
    target_blocks = record.get("target_blocks") or {}
    block_id = "0"
    if isinstance(target_blocks, dict) and target_blocks:
        block_id = sorted(target_blocks, key=lambda item: int(item) if str(item).isdigit() else str(item))[0]
    block_rel_path = Path(str(record["blocks_path"])) / f"{block_id}.png"
    direct_path = image_root / block_rel_path
    if direct_path.exists():
        return str(direct_path)

    split = str(record.get("split") or "").strip()
    if split and not split.startswith("cross_"):
        split = f"cross_{split}"
    if split:
        split_path = image_root / split / block_rel_path
        if split_path.exists():
            return str(split_path)
        return str(split_path)

    return str(direct_path)


def _normalize_android_control(record: dict[str, Any], idx: int, image_root: Path | None) -> VisualSkillStep:
    action = record.get("action") or {}
    action_type = normalize_action_type(str(action.get("action_type") or record.get("action_type") or ""))
    action_value = str(action.get("text") or record.get("value") or "")
    task = str(record.get("goal") or record.get("task") or "")
    node = _find_android_target_node(record, action)
    bbox = _bbox_from_android_node(node) or _bbox_from_action_point(action)
    target_text = _android_node_text(node)
    role, cues = infer_target_role(
        action_type=action_type,
        target_text=target_text,
        action_description=str(record.get("step_instruction") or ""),
        task=task,
        node=node,
    )
    return VisualSkillStep(
        source_id=f"{_first_present(record, ('episode_id',), 'android')}_{_first_present(record, ('step',), idx)}",
        task_id=str(_first_present(record, ("episode_id",), idx)),
        step_index=int(_first_present(record, ("step",), idx)),
        task=task,
        action_type=action_type,
        action_value=action_value,
        bbox=bbox,
        screenshot_path=_resolve_image_path(record, image_root),
        screenshot_width=_optional_int(record.get("screenshot_width")),
        screenshot_height=_optional_int(record.get("screenshot_height")),
        target_role=role,
        target_text=target_text,
        visual_cues=cues,
        domain=_android_domain(record, node),
        app=_android_app(node),
        dataset="android_control",
    )


def _bbox_from_record(record: dict[str, Any]) -> BBox | None:
    raw = record.get("bbox") or record.get("target_bbox") or record.get("box")
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    return _bbox_from_any(raw)


def _bbox_from_any(raw: Any) -> BBox | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    vals = [float(raw[i]) for i in range(4)]
    x1, y1, a, b = vals
    # Many GUI datasets use x/y/w/h; xyxy boxes usually have x2>x1 and y2>y1.
    if a <= x1 or b <= y1:
        return BBox(x1, y1, x1 + max(0.0, a), y1 + max(0.0, b))
    # Treat very small third/fourth values as width/height when x/y are large.
    if a < 400 and b < 400 and (x1 > a or y1 > b):
        return BBox(x1, y1, x1 + a, y1 + b)
    return BBox(x1, y1, a, b)


def _bbox_from_android_node(node: dict[str, Any] | None) -> BBox | None:
    if not node:
        return None
    pixels = node.get("bbox_pixels") or {}
    try:
        return BBox(
            float(pixels["x_min"]),
            float(pixels["y_min"]),
            float(pixels["x_max"]),
            float(pixels["y_max"]),
        )
    except Exception:
        return None


def _bbox_from_action_point(action: dict[str, Any]) -> BBox | None:
    if "x" not in action or "y" not in action:
        return None
    x = float(action["x"])
    y = float(action["y"])
    return BBox(x - 8, y - 8, x + 8, y + 8)


def _find_android_target_node(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    tree = record.get("accessibility_tree") or []
    if "x" not in action or "y" not in action or not isinstance(tree, list):
        return {}
    x = float(action["x"])
    y = float(action["y"])
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, node in enumerate(tree):
        bbox = _bbox_from_android_node(node)
        if not bbox or bbox.area <= 0 or not bbox.contains(x, y):
            continue
        priority = 0
        if node.get("is_clickable") or node.get("is_editable") or node.get("is_checkable"):
            priority -= 10
        if _android_node_text(node):
            priority -= 2
        candidates.append((bbox.area, priority + idx, node))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    nearest: list[tuple[float, dict[str, Any]]] = []
    for node in tree:
        bbox = _bbox_from_android_node(node)
        if not bbox or bbox.area <= 0:
            continue
        cx, cy = bbox.center
        nearest.append(((cx - x) ** 2 + (cy - y) ** 2, node))
    nearest.sort(key=lambda item: item[0])
    return nearest[0][1] if nearest else {}


def _android_node_text(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    for key in ("text", "content_description", "hint_text", "resource_name"):
        value = node.get(key)
        if value:
            return str(value)
    return ""


def _android_domain(record: dict[str, Any], node: dict[str, Any]) -> str:
    package = str(node.get("package_name") or "")
    if package:
        return package.split(".")[1] if "." in package else package
    return str(record.get("domain") or "android")


def _android_app(node: dict[str, Any]) -> str:
    package = str(node.get("package_name") or "")
    return package or "android"


def _resolve_image_path(record: dict[str, Any], image_root: Path | None) -> str:
    for key in ("screenshot_path", "screenshot_before", "image", "screenshot"):
        value = record.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute() and image_root is not None:
            path = image_root / path
        return str(path)
    return ""


def _group_steps_by_task(steps: Sequence[VisualSkillStep]) -> dict[str, list[VisualSkillStep]]:
    grouped: dict[str, list[VisualSkillStep]] = defaultdict(list)
    for step in steps:
        grouped[step.task_id].append(step)
    return grouped


def _is_low_information_occurrence(occurrence: VisualSkillOccurrence) -> bool:
    steps = occurrence.steps
    if not steps:
        return True
    if any(step.action_type in {"input_text", "select", "hover", "long_click"} for step in steps):
        return False
    if any(step.target_role in {"checkbox", "date_cell"} for step in steps):
        return False
    roles = [step.target_role for step in steps]
    if any(
        left.target_role == "dropdown_or_select" and right.target_role == "text_option"
        for left, right in zip(steps, steps[1:])
    ):
        return False
    if "date_or_time_field" in roles and "date_cell" in roles:
        return False
    if len(steps) == 1 and steps[0].action_type == "click" and steps[0].target_role in _GENERIC_SINGLE_CLICK_ROLES:
        return True
    if all(step.action_type == "click" and step.value_kind == "none" for step in steps):
        return True
    return False


def _trim_occurrences(
    occurrences: Sequence[VisualSkillOccurrence],
    max_examples: int,
) -> list[VisualSkillOccurrence]:
    if len(occurrences) <= max_examples:
        return list(occurrences)
    by_task: dict[str, VisualSkillOccurrence] = {}
    for occurrence in occurrences:
        by_task.setdefault(occurrence.task_id, occurrence)
    selected = list(by_task.values())[:max_examples]
    if len(selected) < max_examples:
        selected.extend(list(occurrences)[: max_examples - len(selected)])
    return selected[:max_examples]


def _choose_representative(occurrences: Sequence[VisualSkillOccurrence]) -> VisualSkillOccurrence:
    def score(occurrence: VisualSkillOccurrence) -> tuple[int, float, float, int, int]:
        first = occurrence.steps[0]
        first_image_size = _image_size(first.screenshot_path)
        first_quality = _bbox_quality(first.bbox, first_image_size) if first.bbox and first_image_size else 0.0
        has_image = 1 if first_image_size else 0
        total_quality = 0.0
        text = 0
        grounded_steps = 0
        for step in occurrence.steps:
            image_size = _image_size(step.screenshot_path)
            if step.target_text:
                text = 1
            if step.bbox and image_size:
                step_quality = _bbox_quality(step.bbox, image_size)
                if step_quality > 0:
                    grounded_steps += 1
                    total_quality += step_quality
        return (has_image, first_quality, total_quality, grounded_steps, text)

    return max(occurrences, key=score)


@lru_cache(maxsize=4096)
def _image_size(path: str) -> tuple[int, int] | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def _bbox_quality(bbox: BBox, image_size: tuple[int, int]) -> float:
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0
    clamped = bbox.clamp(width, height)
    if clamped.area <= 0:
        return 0.0
    area_ratio = clamped.area / float(width * height)
    if area_ratio > 0.35:
        return 0.0
    if area_ratio < 0.00002:
        return 0.0
    edge_penalty = 0.85 if clamped.to_list() != bbox.to_list() else 1.0
    if 0.0005 <= area_ratio <= 0.12:
        area_score = 1.0
    elif area_ratio <= 0.20:
        area_score = 0.65
    else:
        area_score = 0.35
    return area_score * edge_penalty


def _counter_top(values: Iterable[str], cap: int = 4) -> list[str]:
    return [value for value, _ in Counter(v for v in values if v).most_common(cap)]


def _skill_id_for_signature(signature: str) -> str:
    parts = []
    for motif in signature.split(" -> "):
        role, action, value = (motif.split(":") + ["", "", ""])[:3]
        parts.append("_".join(x for x in (action, role, value) if x and x != "none"))
    slug = re.sub(r"[^a-z0-9]+", "_", "_then_".join(parts).lower()).strip("_")
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"v2_{slug[:56]}_{digest}"


def _title_for_signature(signature: str) -> str:
    motifs = signature.split(" -> ")
    readable = []
    for motif in motifs:
        role, action, value = (motif.split(":") + ["", "", ""])[:3]
        phrase = f"{action.replace('_', ' ')} {role.replace('_', ' ')}"
        if value and value != "none":
            phrase += f" with {value.replace('_', ' ')}"
        readable.append(phrase)
    title = " Then ".join(part.title() for part in readable)
    return title[:96]


def _confidence(
    *,
    occurrences: Sequence[VisualSkillOccurrence],
    support_tasks: int,
    support_domains: int,
) -> float:
    support_term = min(0.45, len(occurrences) / 100.0)
    task_term = min(0.35, support_tasks / 40.0)
    domain_term = min(0.20, support_domains / 10.0)
    return round(0.2 + support_term + task_term + domain_term, 3)


def _write_annotated_example(candidate: VisualSkillCandidate, output_dir: Path) -> str:
    occurrence = candidate.representative
    first = occurrence.steps[0]
    if not first.screenshot_path or not Path(first.screenshot_path).exists():
        return ""

    image_path = Path(first.screenshot_path)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning("Could not open representative image %s: %s", image_path, exc)
        return ""

    draw = ImageDraw.Draw(image)
    colors = ["#ff3333", "#1f77ff", "#00a36c"]
    drawn = 0
    for idx, step in enumerate(occurrence.steps, start=1):
        if Path(step.screenshot_path or "") != image_path:
            continue
        if not step.bbox or step.action_type in _LOCALIZATION_FREE_ACTIONS:
            continue
        bbox = step.bbox.clamp(*image.size)
        if _bbox_quality(bbox, image.size) <= 0:
            continue
        color = colors[(idx - 1) % len(colors)]
        width = max(3, int(min(image.size) / 220))
        draw.rectangle(bbox.to_list(), outline=color, width=width)
        label = f"{idx}. {step.target_role}"
        _draw_label(draw, label, bbox, color)
        drawn += 1

    if drawn == 0:
        return ""

    out_path = output_dir / "images" / f"{candidate.skill_id}.png"
    image.save(out_path)
    return str(out_path)


def _draw_label(draw: ImageDraw.ImageDraw, label: str, bbox: BBox, color: str) -> None:
    font = ImageFont.load_default()
    x = int(bbox.x1)
    y = max(0, int(bbox.y1) - 14)
    text_bbox = draw.textbbox((x, y), label, font=font)
    pad = 3
    bg = [
        text_bbox[0] - pad,
        text_bbox[1] - pad,
        text_bbox[2] + pad,
        text_bbox[3] + pad,
    ]
    draw.rectangle(bg, fill=color)
    draw.text((x, y), label, fill="white", font=font)


def _catalog_entry(record: dict[str, Any]) -> dict[str, Any]:
    context = record.get("applicable_context") or {}
    support = record.get("support") or {}
    roles, actions, value_kinds = _parse_signature_parts(record.get("signature") or "")
    return {
        "id": record["skill_id"],
        "experience_id": record["skill_id"],
        "skill_id": record["skill_id"],
        "version": "visual_skill_v2",
        "title": record.get("title", ""),
        "when": context.get("when", ""),
        "visual_cues": context.get("visual_cues", [])[:5],
        "signature": record.get("signature", ""),
        "roles": roles,
        "actions": actions,
        "value_kinds": value_kinds,
        "sequence_len": len(roles),
        "requires_bbox": any(action not in _LOCALIZATION_FREE_ACTIONS for action in actions),
        "support": {
            "num_occurrences": support.get("num_occurrences", 0),
            "num_tasks": support.get("num_tasks", 0),
            "num_domains": support.get("num_domains", 0),
        },
    }


def _catalog_v3_entry(record: dict[str, Any]) -> dict[str, Any]:
    context = record.get("applicable_context") or {}
    planning = record.get("planning") or {}
    retrieval = record.get("retrieval") or {}
    support = record.get("support") or {}
    roles, actions, value_kinds = _parse_signature_parts(record.get("signature") or "")
    return {
        "id": record["skill_id"],
        "experience_id": record["skill_id"],
        "skill_id": record["skill_id"],
        "version": "visual_skill_v3",
        "skill_type": record.get("skill_type", "ui_planning_skill"),
        "family": record.get("family", ""),
        "variant": record.get("variant", ""),
        "title": record.get("title", ""),
        "intent": record.get("intent", ""),
        "when": context.get("when", ""),
        "preconditions": context.get("preconditions", [])[:6],
        "page_state_cues": context.get("page_state_cues", [])[:6],
        "task_goal_cues": context.get("task_goal_cues", [])[:6],
        "visual_cues": context.get("page_state_cues", [])[:5],
        "procedure": planning.get("procedure", [])[:6],
        "failure_modes": planning.get("failure_modes", [])[:4],
        "signature": record.get("signature", ""),
        "roles": roles,
        "actions": actions,
        "value_kinds": value_kinds,
        "sequence_len": len(roles),
        "requires_bbox": any(action not in _LOCALIZATION_FREE_ACTIONS for action in actions),
        "retrieval": {
            "query_terms": retrieval.get("query_terms", [])[:12],
            "page_state_summary": retrieval.get("page_state_summary", ""),
            "text_evidence": retrieval.get("text_evidence", [])[:4],
        },
        "support": {
            "num_occurrences": support.get("num_occurrences", 0),
            "num_trajectory_segments": support.get("num_trajectory_segments", 0),
            "num_tasks": support.get("num_tasks", 0),
            "num_domains": support.get("num_domains", 0),
            "action_pattern_agreement": support.get("action_pattern_agreement", 0),
        },
    }


def _catalog_v4_entry(record: dict[str, Any]) -> dict[str, Any]:
    action_plan = record.get("action_plan") or {}
    context = record.get("applicable_context") or {}
    retrieval = record.get("retrieval") or {}
    support = record.get("support") or {}
    trigger = record.get("trigger_visual_state") or {}
    roles, actions, value_kinds = _parse_signature_parts(record.get("signature") or "")
    return {
        "id": record["skill_id"],
        "experience_id": record["skill_id"],
        "skill_id": record["skill_id"],
        "version": "visual_skill_v4",
        "skill_type": "visual_episode_memory",
        "title": record.get("title", ""),
        "signature": record.get("signature", ""),
        "when": context.get("when", ""),
        "visual_triggers": context.get("visual_triggers", [])[:5],
        "task_keywords": context.get("task_keywords", [])[:10],
        "plan_description": action_plan.get("plan_description", ""),
        "num_plan_steps": action_plan.get("num_steps", 0),
        "example_tasks": retrieval.get("example_tasks", [])[:3],
        "trigger_state_summary": retrieval.get("trigger_state_summary", ""),
        "trigger_role": trigger.get("target_role", ""),
        "roles": roles,
        "actions": actions,
        "value_kinds": value_kinds,
        "sequence_len": len(roles),
        "requires_bbox": any(a not in _LOCALIZATION_FREE_ACTIONS for a in actions),
        "retrieval": {
            "query_terms": retrieval.get("query_terms", [])[:12],
            "trigger_state_summary": retrieval.get("trigger_state_summary", ""),
            "example_tasks": retrieval.get("example_tasks", [])[:3],
        },
        "support": {
            "num_occurrences": support.get("num_occurrences", 0),
            "num_tasks": support.get("num_tasks", 0),
            "num_domains": support.get("num_domains", 0),
        },
    }


def _parse_signature_parts(signature: str) -> tuple[list[str], list[str], list[str]]:
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


def _extract_v3_episode_at(
    steps: Sequence[VisualSkillStep],
    idx: int,
) -> tuple[str, str, VisualSkillOccurrence, int] | None:
    step = steps[idx]
    if _is_hover_menu_start(step):
        end = _find_next_matching(
            steps,
            idx,
            lambda item: item.target_role in {"text_option", "menu_control", "tab", "dropdown_or_select"}
            and item.action_type in {"click", "select"},
            max_lookahead=4,
        )
        if end is not None:
            occurrence = VisualSkillOccurrence(list(steps[idx : end + 1]))
            return "hover_menu_navigation", "generic", occurrence, end + 1

    if _is_date_related(step):
        end = _extend_date_episode(steps, idx)
        if end > idx:
            occurrence = VisualSkillOccurrence(list(steps[idx : end + 1]))
            return "date_picker_selection", "date_or_time", occurrence, end + 1

    if _is_dropdown_related(step):
        end = _extend_dropdown_episode(steps, idx)
        if end > idx:
            occurrence = VisualSkillOccurrence(list(steps[idx : end + 1]))
            dropdown_count = sum(1 for item in occurrence.steps if item.target_role == "dropdown_or_select")
            family = "dependent_or_multi_dropdown_form" if dropdown_count >= 2 else "dropdown_option_selection"
            return family, _episode_value_variant(occurrence), occurrence, end + 1

    if _is_text_input(step):
        option_end = _find_next_matching(
            steps,
            idx,
            lambda item: item.target_role == "text_option" and item.action_type in {"click", "select"},
            max_lookahead=3,
        )
        if option_end is not None:
            occurrence = VisualSkillOccurrence(list(steps[idx : option_end + 1]))
            return "autocomplete_search_selection", _input_variant(step), occurrence, option_end + 1

        submit_end = _find_next_matching(
            steps,
            idx,
            lambda item: item.target_role in {"search_button", "confirm_button"} and item.action_type == "click",
            max_lookahead=4,
        )
        if submit_end is not None:
            occurrence = VisualSkillOccurrence(list(steps[idx : submit_end + 1]))
            family = "search_then_submit" if step.target_role == "search_bar" else "form_fill_and_submit"
            return family, _episode_value_variant(occurrence), occurrence, submit_end + 1

    if _is_form_field(step):
        end = _extend_form_submit_episode(steps, idx)
        if end > idx:
            occurrence = VisualSkillOccurrence(list(steps[idx : end + 1]))
            return "form_fill_and_submit", _episode_value_variant(occurrence), occurrence, end + 1

    if _is_navigation_start(step):
        end = _find_next_matching(
            steps,
            idx,
            lambda item: item.target_role in {"text_option", "tab", "menu_control", "search_bar"}
            and item.action_type in {"click", "input_text", "hover"},
            max_lookahead=3,
        )
        if end is not None:
            occurrence = VisualSkillOccurrence(list(steps[idx : end + 1]))
            return "tab_or_menu_navigation", "generic", occurrence, end + 1

    return None


def _is_hover_menu_start(step: VisualSkillStep) -> bool:
    return step.action_type == "hover" or (
        step.target_role == "menu_control" and _has_any(step.target_text.lower(), ["menu", "navigation", "reservations"])
    )


def _is_date_related(step: VisualSkillStep) -> bool:
    return step.target_role in {"date_or_time_field", "date_cell"} or step.value_kind == "date_or_time"


def _is_dropdown_related(step: VisualSkillStep) -> bool:
    return step.target_role == "dropdown_or_select" and step.action_type in {"click", "select"}


def _is_text_input(step: VisualSkillStep) -> bool:
    return step.action_type == "input_text" and step.target_role in {"search_bar", "text_field", "date_or_time_field"}


def _is_form_field(step: VisualSkillStep) -> bool:
    return step.action_type in {"input_text", "select"} and step.target_role in {
        "text_field",
        "search_bar",
        "dropdown_or_select",
        "date_or_time_field",
    }


def _is_navigation_start(step: VisualSkillStep) -> bool:
    return step.action_type in {"click", "hover"} and step.target_role in {"tab", "menu_control", "text_option"}


def _find_next_matching(
    steps: Sequence[VisualSkillStep],
    idx: int,
    predicate,
    *,
    max_lookahead: int,
) -> int | None:
    limit = min(len(steps), idx + max_lookahead + 1)
    for pos in range(idx + 1, limit):
        if predicate(steps[pos]):
            return pos
        if _is_hard_episode_boundary(steps[pos]):
            return None
    return None


def _extend_dropdown_episode(steps: Sequence[VisualSkillStep], idx: int) -> int:
    end = idx
    limit = min(len(steps), idx + 6)
    for pos in range(idx + 1, limit):
        item = steps[pos]
        if item.target_role in {"dropdown_or_select", "text_option"} and item.action_type in {"click", "select"}:
            end = pos
            continue
        if item.target_role in {"confirm_button", "search_button"} and item.action_type == "click" and end > idx:
            return pos
        break
    return end


def _extend_date_episode(steps: Sequence[VisualSkillStep], idx: int) -> int:
    end = idx
    limit = min(len(steps), idx + 6)
    for pos in range(idx + 1, limit):
        item = steps[pos]
        if item.target_role in {"date_or_time_field", "date_cell", "dropdown_or_select", "text_option"} or item.value_kind == "date_or_time":
            end = pos
            continue
        if item.target_role in {"confirm_button", "search_button"} and item.action_type == "click" and end > idx:
            return pos
        break
    return end


def _extend_form_submit_episode(steps: Sequence[VisualSkillStep], idx: int) -> int:
    field_count = 1
    end = idx
    limit = min(len(steps), idx + 7)
    for pos in range(idx + 1, limit):
        item = steps[pos]
        if _is_form_field(item) or (item.target_role == "text_option" and item.action_type in {"click", "select"}):
            field_count += 1 if _is_form_field(item) else 0
            end = pos
            continue
        if item.target_role in {"confirm_button", "search_button"} and item.action_type == "click":
            return pos if field_count >= 1 else end
        if _is_hard_episode_boundary(item):
            break
    return end if field_count >= 2 else idx


def _is_hard_episode_boundary(step: VisualSkillStep) -> bool:
    return step.action_type in {"scroll", "press_key"} or step.target_role in {"dismiss_or_back_control"}


def _is_low_information_v3_episode(family: str, occurrence: VisualSkillOccurrence) -> bool:
    if family in {
        "autocomplete_search_selection",
        "dependent_or_multi_dropdown_form",
        "dropdown_option_selection",
        "date_picker_selection",
        "form_fill_and_submit",
        "search_then_submit",
        "hover_menu_navigation",
    }:
        return False
    if all(step.action_type == "click" and step.value_kind == "none" for step in occurrence.steps):
        return True
    return False


def _input_variant(step: VisualSkillStep) -> str:
    return f"{step.target_role}:{_value_family(step.value_kind)}"


def _episode_value_variant(occurrence: VisualSkillOccurrence) -> str:
    value_families = [_value_family(step.value_kind) for step in occurrence.steps if _value_family(step.value_kind) != "none"]
    if "date_or_time" in value_families:
        return "date_or_time"
    if "query" in value_families:
        return "query"
    if "option" in value_families:
        return "option"
    return "generic"


def _value_family(value_kind: str) -> str:
    if value_kind == "date_or_time":
        return "date_or_time"
    if value_kind == "query":
        return "query"
    if value_kind in {"option", "number"}:
        return "option"
    return value_kind or "none"


def _family_signature_for(
    family: str,
    variant: str,
    occurrences: Sequence[VisualSkillOccurrence],
) -> str:
    signatures = Counter(occurrence.signature for occurrence in occurrences)
    if signatures:
        return signatures.most_common(1)[0][0]
    return f"{family}:{variant}"


def _v3_skill_id_for_family(family: str, variant: str) -> str:
    key = f"{family}:{variant}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return f"v3_{slug[:64]}_{digest}"


def _v3_family_from_signature(signature: str) -> str:
    roles, actions, value_kinds = _parse_signature_parts(signature)
    if "date_cell" in roles or "date_or_time_field" in roles or "date_or_time" in value_kinds:
        return "date_picker_selection"
    if "dropdown_or_select" in roles and "text_option" in roles:
        return "dropdown_option_selection"
    if "search_bar" in roles and "input_text" in actions and "text_option" in roles:
        return "autocomplete_search_selection"
    if "search_button" in roles:
        return "search_then_submit"
    if "confirm_button" in roles:
        return "form_fill_and_submit"
    if "menu_control" in roles and "hover" in actions:
        return "hover_menu_navigation"
    return "tab_or_menu_navigation"


def _v3_family_title(family: str, variant: str) -> str:
    titles = {
        "autocomplete_search_selection": "Autocomplete Search Selection",
        "dropdown_option_selection": "Dropdown Option Selection",
        "dependent_or_multi_dropdown_form": "Dependent Or Multi-Dropdown Form",
        "date_picker_selection": "Date Picker Selection",
        "form_fill_and_submit": "Form Fill And Submit",
        "search_then_submit": "Search Then Submit",
        "hover_menu_navigation": "Hover Menu Navigation",
        "tab_or_menu_navigation": "Tab Or Menu Navigation",
    }
    title = titles.get(family, family.replace("_", " ").title())
    if variant and variant not in {"generic", "none"}:
        title += f" ({variant.replace(':', ', ').replace('_', ' ')})"
    return title


def _v3_family_intent(family: str, variant: str) -> str:
    intents = {
        "autocomplete_search_selection": "Enter a task value into a search or text field and select the matching suggestion/result.",
        "dropdown_option_selection": "Open or use a dropdown/select control and choose the option that matches the task.",
        "dependent_or_multi_dropdown_form": "Fill related dropdown controls in a form, respecting dependencies between broad and specific fields.",
        "date_picker_selection": "Choose a requested date or time through a date/time field, calendar, or related selector.",
        "form_fill_and_submit": "Fill visible form fields or filters and commit them with the relevant apply/search/submit control.",
        "search_then_submit": "Enter a query into a search field and submit it through the page's search affordance.",
        "hover_menu_navigation": "Expose menu choices by hovering a navigation/menu control and then choose the relevant option.",
        "tab_or_menu_navigation": "Navigate through visible tabs, menu items, or category options to reach the requested section.",
    }
    return intents.get(family, "Reuse a recurring UI interaction strategy for the current page state.")


def _v3_family_when(family: str, variant: str) -> str:
    return f"Use when the current task and visible page state match the {family.replace('_', ' ')} interaction family."


def _v3_family_preconditions(family: str, variant: str) -> list[str]:
    mapping = {
        "autocomplete_search_selection": [
            "A search/text field is visible and the task provides a value to enter.",
            "The page is likely to show suggestions, options, or matching results after typing.",
        ],
        "dropdown_option_selection": [
            "A dropdown/select control is visible or already open.",
            "The task specifies an option that should be chosen from that control.",
        ],
        "dependent_or_multi_dropdown_form": [
            "Multiple related dropdown/select controls appear in the same form or filter panel.",
            "Later fields may depend on earlier selections.",
        ],
        "date_picker_selection": [
            "A date/time field, calendar, date cell, or date/time selector is visible.",
            "The task contains a date or time constraint.",
        ],
        "form_fill_and_submit": [
            "A form or filter panel contains fields matching the task constraints.",
            "A Search, Apply, Submit, Done, Continue, or Confirm control is available after filling values.",
        ],
        "search_then_submit": [
            "A search field is visible and the task provides a query.",
            "The page has a search icon/button or submit affordance.",
        ],
        "hover_menu_navigation": [
            "A navigation/menu control is visible and can expose more options on hover.",
            "The target section is likely nested under that menu.",
        ],
        "tab_or_menu_navigation": [
            "Tabs, menu items, categories, or section links are visible.",
            "The task requires moving to a specific section before acting.",
        ],
    }
    return mapping.get(family, ["The current page state matches the historical interaction evidence."])


def _v3_family_procedure(family: str, variant: str) -> list[str]:
    mapping = {
        "autocomplete_search_selection": [
            "Focus the search/text field that corresponds to the task slot.",
            "Enter the task-provided query or value.",
            "Wait for suggestions, autocomplete options, or matching results to appear.",
            "Select the suggestion/result that best matches the task value.",
            "If no suggestion appears, retry the query or submit it directly if the page supports that.",
        ],
        "dropdown_option_selection": [
            "Open the dropdown/select control if it is not already open.",
            "Inspect the visible options and choose the one matching the task value.",
            "If the desired option is not visible, scroll within the dropdown or reopen it after a short wait.",
            "Verify the chosen value is displayed before moving on.",
        ],
        "dependent_or_multi_dropdown_form": [
            "Identify the related dropdowns in the same form or filter panel.",
            "Fill broad/category fields before dependent or more specific fields.",
            "After each selection, wait briefly for downstream options to refresh or become enabled.",
            "Continue selecting values from general to specific.",
            "Commit the form only after the visible selected values match the task.",
        ],
        "date_picker_selection": [
            "Open the date/time field or calendar control if needed.",
            "Navigate to the requested month/time range when it is not visible.",
            "Select the date/time cell or option matching the task.",
            "Click Done, Apply, or Confirm if the date picker requires an explicit commit.",
            "Verify the selected date/time appears in the field or page state.",
        ],
        "form_fill_and_submit": [
            "Map each task constraint to the visible form field or filter control.",
            "Fill text fields and select dropdown/filter values in the form.",
            "Check that required values are visible and no required field is still empty.",
            "Click the Search, Apply, Submit, Done, Continue, or Confirm control.",
            "Verify the page advances to results or a committed state.",
        ],
        "search_then_submit": [
            "Focus the search field matching the task.",
            "Enter the query from the task goal.",
            "Submit using the search icon/button or equivalent keyboard/page affordance.",
            "Verify results or a search results page appears.",
        ],
        "hover_menu_navigation": [
            "Hover the relevant top-level menu or navigation control.",
            "Wait for the submenu or flyout options to appear.",
            "Click the option that matches the requested section or subgoal.",
            "If the menu disappears, hover again and choose the option more directly.",
        ],
        "tab_or_menu_navigation": [
            "Identify the visible tab/menu/category matching the current subgoal.",
            "Click that navigation option.",
            "If a second-level option appears, choose the one matching the task.",
            "Continue only after the page/section visibly changes.",
        ],
    }
    return mapping.get(family, ["Apply the reusable interaction strategy only when the current page matches the preconditions."])


def _v3_family_postcondition_checks(family: str) -> list[str]:
    mapping = {
        "autocomplete_search_selection": ["A matching suggestion/result is selected or the query is accepted."],
        "dropdown_option_selection": ["The selected dropdown value is visible in the control or filter state."],
        "dependent_or_multi_dropdown_form": ["Dependent controls refresh and all required selected values are visible."],
        "date_picker_selection": ["The requested date/time appears in the field or committed page state."],
        "form_fill_and_submit": ["The page advances to results, filtered content, or a confirmed state."],
        "search_then_submit": ["Search results or a query-specific page appears."],
        "hover_menu_navigation": ["The target submenu option is clicked and the page/section changes."],
        "tab_or_menu_navigation": ["The requested section/tab/category becomes active."],
    }
    return mapping.get(family, ["The page state changes in the direction implied by the task."])


def _v3_family_failure_modes(family: str) -> list[str]:
    mapping = {
        "autocomplete_search_selection": [
            "Suggestions do not appear after typing.",
            "Multiple suggestions look similar.",
            "A typed value is accepted but not selected from the suggestion list.",
        ],
        "dropdown_option_selection": [
            "The dropdown closes before selection.",
            "The desired option is below the visible dropdown area.",
            "The control looks like a dropdown but is disabled.",
        ],
        "dependent_or_multi_dropdown_form": [
            "A dependent dropdown is disabled or empty until an earlier value is selected.",
            "Options refresh asynchronously after each selection.",
            "A downstream value is reset when an upstream field changes.",
        ],
        "date_picker_selection": [
            "The calendar opens on the wrong month.",
            "The date picker requires a separate Done/Apply click.",
            "The desired date/time is hidden behind calendar navigation.",
        ],
        "form_fill_and_submit": [
            "The submit/apply button is below the fold.",
            "A required field remains empty.",
            "A modal, consent banner, or loading state blocks the form.",
        ],
        "search_then_submit": [
            "The search icon/button is not immediately visible.",
            "Typing updates suggestions but does not submit.",
        ],
        "hover_menu_navigation": [
            "The submenu disappears when the pointer leaves the menu.",
            "The desired option is nested one level deeper.",
        ],
        "tab_or_menu_navigation": [
            "The clicked tab/menu item opens a submenu instead of navigating.",
            "The page changes sections without a full navigation.",
        ],
    }
    return mapping.get(family, ["The current page no longer matches the skill preconditions."])


def _v3_family_recovery_steps(family: str) -> list[str]:
    mapping = {
        "autocomplete_search_selection": [
            "Refocus the input and type the query again if suggestions vanish.",
            "Use the closest exact-text suggestion when several options are similar.",
        ],
        "dropdown_option_selection": [
            "Reopen the dropdown and scroll inside it if the option is not visible.",
            "Wait briefly if the control appears disabled or still loading.",
        ],
        "dependent_or_multi_dropdown_form": [
            "Return to the broadest field and refill dependent fields in order.",
            "Wait and reopen downstream dropdowns after each upstream selection.",
        ],
        "date_picker_selection": [
            "Use calendar next/previous controls to reach the requested month.",
            "Click Done/Apply after selecting the date if the field did not update.",
        ],
        "form_fill_and_submit": [
            "Scroll within the form/page to find the commit button.",
            "Review visible required fields before submitting again.",
        ],
        "search_then_submit": [
            "Try the visible search icon/button, then keyboard submit if appropriate.",
        ],
        "hover_menu_navigation": [
            "Hover the top-level menu again and move directly to the target option.",
        ],
        "tab_or_menu_navigation": [
            "If the first click only expands choices, select the matching child option.",
        ],
    }
    return mapping.get(family, ["Stop using this skill if the current page state no longer matches."])


def _v3_family_negative_conditions(family: str) -> list[str]:
    return [
        "Do not use this skill when the current page lacks the stated UI family.",
        "Do not use this skill only because the task text is similar; the visible page state must also match.",
    ]


def _v3_action_pattern_agreement(candidate: VisualSkillCandidate) -> float:
    if not candidate.occurrences:
        return 0.0
    counts = Counter(occurrence.signature for occurrence in candidate.occurrences)
    return round(counts.most_common(1)[0][1] / len(candidate.occurrences), 3)


def _signature_motifs(signature: str) -> list[str]:
    return [part for part in str(signature or "").split(" -> ") if part]


def _candidate_task_ids(candidate: VisualSkillCandidate) -> set[str]:
    return {occurrence.task_id for occurrence in candidate.occurrences}


def _contains_contiguous_subsequence(longer: Sequence[str], shorter: Sequence[str]) -> bool:
    if not shorter or len(shorter) >= len(longer):
        return False
    short_len = len(shorter)
    for start in range(0, len(longer) - short_len + 1):
        if list(longer[start : start + short_len]) == list(shorter):
            return True
    return False


def _v3_skill_id_for_signature(signature: str) -> str:
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", signature.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return f"v3_{slug[:56]}_{digest}"


def _v3_title_for_candidate(candidate: VisualSkillCandidate) -> str:
    roles = [role.replace("_", " ").title() for role in candidate.dominant_roles[:3]]
    actions = [action.replace("_", " ").title() for action in candidate.dominant_actions[:2]]
    if roles and actions:
        return f"Plan {', '.join(actions)} Through {', '.join(roles)}"
    return candidate.title


def _v3_intent_for(candidate: VisualSkillCandidate) -> str:
    roles = ", ".join(role.replace("_", " ") for role in candidate.dominant_roles[:3])
    actions = ", ".join(action.replace("_", " ") for action in candidate.dominant_actions[:3])
    return f"Transfer a recurring UI interaction plan for {actions} actions over {roles} states."


def _v3_when_for_candidate(candidate: VisualSkillCandidate) -> str:
    return (
        "Use when the current task goal and visible page state match the mined "
        f"pattern: {candidate.signature}."
    )


def _v3_preconditions_for(candidate: VisualSkillCandidate) -> list[str]:
    roles = candidate.dominant_roles
    preconditions: list[str] = []
    if "search_bar" in roles:
        preconditions.append("A search input or search affordance is visible and the task requires a query.")
    if "text_field" in roles:
        preconditions.append("A form field is visible and the task provides the value to enter.")
    if "dropdown_or_select" in roles:
        preconditions.append("One or more dropdown/select controls are visible and the task requires choosing values.")
    if "text_option" in roles:
        preconditions.append("A visible option, suggestion, menu item, or dropdown item should be selected.")
    if "date_cell" in roles or "date_or_time_field" in roles:
        preconditions.append("A date/time field or calendar state is visible and the task contains a date/time constraint.")
    if "confirm_button" in roles or "search_button" in roles:
        preconditions.append("The form or selection is ready to be committed with a search/apply/confirm control.")
    if not preconditions:
        preconditions.append("The current UI exposes the same target roles and action sequence as the support segments.")
    return _dedupe_keep_order(preconditions)


def _v3_procedure_for(candidate: VisualSkillCandidate) -> list[str]:
    steps = candidate.representative.steps
    procedure: list[str] = []
    if _looks_like_dependent_dropdown(candidate):
        procedure.extend(
            [
                "Select the broadest/highest-level dropdown first.",
                "Wait for dependent dropdown options to refresh or become enabled.",
                "Select dependent dropdown values from general to specific.",
                "Verify the selected values are visible before committing the form.",
                "Click Search, Apply, Done, or the nearest commit button if present.",
            ]
        )
        return procedure

    for idx, step in enumerate(steps, start=1):
        role = step.target_role.replace("_", " ")
        action = step.action_type.replace("_", " ")
        value = step.value_kind.replace("_", " ")
        if step.action_type == "input_text":
            procedure.append(f"Step {idx}: enter the task-provided {value or 'text'} into the matching {role}.")
        elif step.action_type == "select":
            procedure.append(f"Step {idx}: select the matching option in the {role}; wait if options refresh.")
        elif step.action_type == "click":
            procedure.append(f"Step {idx}: click the {role} that matches the current subgoal.")
        elif step.action_type == "hover":
            procedure.append(f"Step {idx}: hover the {role} and inspect the newly exposed choices.")
        elif step.action_type == "scroll":
            procedure.append(f"Step {idx}: scroll only enough to expose the next required control.")
        else:
            procedure.append(f"Step {idx}: perform {action} on the matching {role}.")
    if len(steps) > 1:
        procedure.append("After each action, check whether the page state advanced before continuing the pattern.")
    return procedure


def _v3_postcondition_checks_for(candidate: VisualSkillCandidate) -> list[str]:
    checks = [_postcondition_for(candidate)]
    roles = set(candidate.dominant_roles)
    actions = set(candidate.dominant_actions)
    if "dropdown_or_select" in roles or "select" in actions:
        checks.append("The chosen value is visible or the dependent control updates.")
    if "input_text" in actions:
        checks.append("The entered text remains in the intended field or suggestions/results appear.")
    if "confirm_button" in roles or "search_button" in roles:
        checks.append("The page advances to results, filtered content, or a confirmed state.")
    return _dedupe_keep_order(checks)


def _v3_failure_modes_for(candidate: VisualSkillCandidate) -> list[str]:
    roles = set(candidate.dominant_roles)
    failures: list[str] = []
    if "dropdown_or_select" in roles:
        failures.extend(
            [
                "A dependent dropdown is disabled or empty until an earlier value is selected.",
                "The dropdown closes or refreshes before the desired option is selected.",
            ]
        )
    if "text_option" in roles:
        failures.append("Autocomplete or dropdown suggestions are not visible yet.")
    if "confirm_button" in roles or "search_button" in roles:
        failures.append("The commit button is below the fold or hidden until required fields are filled.")
    if "date_cell" in roles or "date_or_time_field" in roles:
        failures.append("The calendar opens on a different month or requires navigation before the date is visible.")
    failures.append("A modal, consent banner, or loading state blocks the intended control.")
    return _dedupe_keep_order(failures)


def _v3_recovery_steps_for(candidate: VisualSkillCandidate) -> list[str]:
    roles = set(candidate.dominant_roles)
    recovery: list[str] = []
    if "dropdown_or_select" in roles:
        recovery.extend(["Wait briefly, reopen the dropdown, and re-check whether options refreshed.", "If a dependent field is disabled, fill the preceding broader field first."])
    if "text_option" in roles:
        recovery.append("If suggestions are missing, refocus the input or type the query again before selecting.")
    if "confirm_button" in roles or "search_button" in roles:
        recovery.append("Scroll within the form/page to find Search, Apply, Done, or Confirm.")
    recovery.append("If the page state no longer matches the preconditions, stop using this skill.")
    return _dedupe_keep_order(recovery)


def _v3_task_goal_cues_for(candidate: VisualSkillCandidate) -> list[str]:
    phrases: list[str] = []
    for occurrence in candidate.occurrences[:20]:
        for step in occurrence.steps:
            phrases.extend(_quoted_phrases(step.task, step.action_value))
            if step.value_kind != "none":
                phrases.append(step.value_kind)
    if not phrases:
        phrases.extend(candidate.dominant_actions)
    return _dedupe_keep_order([str(item) for item in phrases if str(item).strip()])[:8]


def _v3_query_terms_for(candidate: VisualSkillCandidate) -> list[str]:
    terms = []
    terms.extend(candidate.dominant_roles)
    terms.extend(candidate.dominant_actions)
    for value in _parse_signature_parts(candidate.signature)[2]:
        if value != "none":
            terms.append(value)
    terms.extend(_v3_task_goal_cues_for(candidate))
    return _dedupe_keep_order(term.replace("_", " ") for term in terms if term)


def _v3_page_state_summary_for(candidate: VisualSkillCandidate) -> str:
    roles = ", ".join(role.replace("_", " ") for role in candidate.dominant_roles[:4])
    cues = "; ".join(_visual_cues_for(candidate)[:4])
    return f"Historical states show {roles}. Representative cues: {cues}."


def _v3_visual_evidence_for(candidate: VisualSkillCandidate, *, example_image: str = "") -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if example_image:
        first = candidate.representative.steps[0]
        evidence.append(
            {
                "image_path": example_image,
                "target_role": first.target_role,
                "target_text": first.target_text,
                "bbox": first.bbox.to_list() if first.bbox else None,
                "source": "annotated_representative",
            }
        )
    seen_paths = {example_image} if example_image else set()
    for occurrence in candidate.occurrences:
        for step in occurrence.steps:
            if not step.screenshot_path or step.screenshot_path in seen_paths:
                continue
            evidence.append(
                {
                    "image_path": step.screenshot_path,
                    "target_role": step.target_role,
                    "target_text": step.target_text,
                    "bbox": step.bbox.to_list() if step.bbox else None,
                    "source": step.source_id,
                }
            )
            seen_paths.add(step.screenshot_path)
            if len(evidence) >= 4:
                return evidence
    return evidence


def _v3_text_evidence_for(candidate: VisualSkillCandidate) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for occurrence in candidate.occurrences[:6]:
        evidence.append(
            {
                "task_id": occurrence.task_id,
                "task": occurrence.steps[0].task,
                "domain": occurrence.domain,
                "app": occurrence.app,
                "actions": [
                    {
                        "role": step.target_role,
                        "action": step.action_type,
                        "value_kind": step.value_kind,
                        "target_text": step.target_text,
                    }
                    for step in occurrence.steps
                ],
            }
        )
    return evidence


def _looks_like_dependent_dropdown(candidate: VisualSkillCandidate) -> bool:
    roles = candidate.dominant_roles
    actions = candidate.dominant_actions
    dropdown_count = sum(1 for occurrence in candidate.occurrences for step in occurrence.steps if step.target_role == "dropdown_or_select")
    return dropdown_count >= 2 and ("select" in actions or "click" in actions) and "dropdown_or_select" in roles


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _when_for_candidate(candidate: VisualSkillCandidate) -> str:
    roles = ", ".join(role.replace("_", " ") for role in candidate.dominant_roles[:2])
    actions = ", ".join(action.replace("_", " ") for action in candidate.dominant_actions[:2])
    return f"The current UI contains a {roles} target and the task needs a {actions} interaction."


def _visual_cues_for(candidate: VisualSkillCandidate) -> list[str]:
    cues: list[str] = []
    for occurrence in candidate.occurrences[:10]:
        for step in occurrence.steps:
            cues.extend(step.visual_cues)
    compact = []
    seen = set()
    for cue in cues:
        key = cue.lower()
        if key not in seen:
            compact.append(cue)
            seen.add(key)
        if len(compact) >= 8:
            break
    return compact or [role.replace("_", " ") for role in candidate.dominant_roles]


def _domain_hint_for(candidate: VisualSkillCandidate) -> str:
    domains = Counter(item.domain for item in candidate.occurrences)
    if len(domains) > 3:
        return "generic"
    return ", ".join(domain for domain, _ in domains.most_common(3)) or "generic"


def _target_instruction_for(candidate: VisualSkillCandidate) -> str:
    first = candidate.representative.steps[0]
    return (
        f"Select the {first.target_role.replace('_', ' ')} region shown by the bounding box "
        "in the example image, then follow the action template."
    )


def _action_guidance_for(candidate: VisualSkillCandidate) -> str:
    templates = "; ".join(_action_templates_for(candidate))
    return f"Use this visual pattern when the target role and task slot match. Action pattern: {templates}."


def _action_templates_for(candidate: VisualSkillCandidate) -> list[str]:
    templates = []
    for step in candidate.representative.steps:
        slot = step.value_kind if step.value_kind != "none" else ""
        if step.action_type == "input_text":
            value = f"<{slot or 'text'}>"
            templates.append(f"input_text(target_bbox, {value})")
        elif step.action_type == "click":
            templates.append("click(target_bbox)")
        elif step.action_type == "scroll":
            templates.append("scroll(direction)")
        elif step.action_type == "select":
            templates.append("select(target_bbox)")
        else:
            templates.append(f"{step.action_type}(target_bbox)")
    return templates


def _value_slots_for(candidate: VisualSkillCandidate) -> list[dict[str, str]]:
    slots = []
    seen = set()
    for occurrence in candidate.occurrences:
        for step in occurrence.steps:
            kind = step.value_kind
            if kind == "none" or kind in seen:
                continue
            seen.add(kind)
            slots.append({"name": kind, "source": "task instruction or current UI"})
    return slots


def _postcondition_for(candidate: VisualSkillCandidate) -> str:
    actions = set(candidate.dominant_actions)
    if "input_text" in actions:
        return "The requested value appears in the target field or the UI advances to matching results."
    if "click" in actions:
        return "The UI visibly commits the selected target, opens the intended panel, or advances to the next state."
    return "The UI changes in the direction implied by the action template."


def _avoid_for(candidate: VisualSkillCandidate) -> str:
    if any(role == "search_bar" for role in candidate.dominant_roles):
        return "Do not click unrelated result cards or navigation links before entering the query."
    if any(role == "confirm_button" for role in candidate.dominant_roles):
        return "Do not leave the current panel before committing the visible change."
    return "Do not use this skill when the target role or value slot is only a weak visual match."


def _v4_title(candidate: VisualSkillCandidate) -> str:
    roles = " + ".join(r.replace("_", " ").title() for r in candidate.dominant_roles[:2])
    actions = " + ".join(a.replace("_", " ").title() for a in candidate.dominant_actions[:2])
    num_steps = len(candidate.representative.steps)
    return f"{actions} via {roles} ({num_steps}-step plan, {candidate.support_tasks} tasks)"


def _v4_when(candidate: VisualSkillCandidate) -> str:
    trigger = candidate.representative.steps[0]
    cue = trigger.visual_cues[0][:60] if trigger.visual_cues else trigger.target_role.replace("_", " ")
    num_steps = len(candidate.representative.steps)
    roles = " + ".join(candidate.dominant_roles[:2])
    return (
        f"Apply when the current screen shows '{cue}' and the task requires "
        f"a {num_steps}-step sequence involving {roles}."
    )


def _v4_visual_triggers(candidate: VisualSkillCandidate) -> list[str]:
    trigger = candidate.representative.steps[0]
    cues: list[str] = list(trigger.visual_cues[:4])
    if trigger.target_role not in cues:
        cues.insert(0, trigger.target_role)
    return cues[:5]


def _v4_task_keywords(candidate: VisualSkillCandidate) -> list[str]:
    token_counter: Counter[str] = Counter()
    _stopwords = {"the", "and", "for", "with", "from", "that", "this", "into", "then", "click", "find"}
    for occ in candidate.occurrences:
        if occ.steps:
            for tok in re.findall(r"[a-z][a-z0-9_-]{2,}", occ.steps[0].task.lower()):
                if tok not in _stopwords:
                    token_counter[tok] += 1
    return [tok for tok, _ in token_counter.most_common(10)]


def _v4_example_tasks(candidate: VisualSkillCandidate) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for occ in candidate.occurrences:
        task = (occ.steps[0].task if occ.steps else "")[:120]
        if task and task not in seen:
            seen.add(task)
            result.append(task)
        if len(result) >= 4:
            break
    return result


def _v4_trigger_state_summary(candidate: VisualSkillCandidate) -> str:
    trigger = candidate.representative.steps[0]
    cue = trigger.visual_cues[0][:60] if trigger.visual_cues else trigger.target_text[:60]
    return f"{trigger.target_role} in {trigger.app} ({trigger.domain}): {cue}"


def _v4_query_terms(candidate: VisualSkillCandidate) -> list[str]:
    terms: set[str] = set()
    for role in candidate.dominant_roles[:3]:
        terms.update(role.split("_"))
    for action in candidate.dominant_actions[:3]:
        terms.add(action)
    _stopwords = {"the", "and", "for", "with", "from"}
    for occ in candidate.occurrences[:5]:
        if occ.steps:
            for tok in re.findall(r"[a-z][a-z0-9]{2,}", occ.steps[0].task.lower()):
                if tok not in _stopwords:
                    terms.add(tok)
    return sorted(terms)[:16]


def _v4_negative_conditions(candidate: VisualSkillCandidate) -> list[str]:
    conditions = ["Do not apply if task requires only a single click or simple navigation."]
    if "input_text" in candidate.dominant_actions:
        conditions.append("Do not apply if no text input is needed for this step.")
    return conditions


def _v4_action_plan_steps(occurrence: VisualSkillOccurrence) -> list[dict[str, Any]]:
    result = []
    for i, step in enumerate(occurrence.steps):
        value = step.action_value or ""
        target_desc = (step.target_text or step.target_role)[:80]
        if step.action_type == "input_text":
            slot_hint = f"[{step.value_kind.upper()}]" if step.value_kind not in ("none", "") else "[TEXT]"
            description = f"Type {slot_hint} into {step.target_role}: {target_desc}"
        elif step.action_type in ("click", "select"):
            description = f"Click {step.target_role}: {target_desc}"
            slot_hint = ""
        elif step.action_type == "hover":
            description = f"Hover {step.target_role}: {target_desc}"
            slot_hint = ""
        elif step.action_type == "scroll":
            description = f"Scroll ({value or 'direction'})"
            slot_hint = ""
        elif step.action_type == "press_key":
            description = f"Press {value or 'key'}"
            slot_hint = ""
        else:
            description = f"{step.action_type} {step.target_role}: {target_desc}"
            slot_hint = ""
        result.append({
            "step_index": i,
            "action_type": step.action_type,
            "target_role": step.target_role,
            "example_value": value[:60] or None,
            "slot_hint": slot_hint,
            "description": description,
        })
    return result


def _v4_plan_description(occurrence: VisualSkillOccurrence) -> str:
    parts = []
    for step in occurrence.steps:
        value = step.action_value or ""
        if step.action_type == "input_text":
            parts.append(f"type {repr(value[:30]) if value else '[text]'} into {step.target_role}")
        elif step.action_type in ("click", "select"):
            label = (step.target_text or step.target_role)[:40]
            parts.append(f"click {label}")
        elif step.action_type == "hover":
            label = (step.target_text or step.target_role)[:40]
            parts.append(f"hover {label}")
        elif step.action_type == "scroll":
            parts.append(f"scroll ({value or ''})")
        elif step.action_type == "press_key":
            parts.append(f"press {value or 'key'}")
        else:
            parts.append(f"{step.action_type} {step.target_role}")
    return " → ".join(parts)


def _v4_slot_analysis(candidate: VisualSkillCandidate) -> list[dict[str, Any]]:
    num_steps = len(candidate.representative.steps)
    result = []
    for i in range(num_steps):
        step = candidate.representative.steps[i]
        values_at_step = [
            occ.steps[i].action_value
            for occ in candidate.occurrences
            if len(occ.steps) > i
        ]
        unique_values = list({v for v in values_at_step if v})
        is_slot = step.action_type == "input_text" and len(unique_values) > 1
        slot_name = f"{step.target_role.upper()}_VALUE" if is_slot else ""
        result.append({
            "step_index": i,
            "action_type": step.action_type,
            "target_role": step.target_role,
            "is_slot": is_slot,
            "slot_name": slot_name,
            "example_values": unique_values[:4] if is_slot else [],
            "fixed_example": values_at_step[0] if not is_slot and values_at_step else None,
        })
    return result


def _v4_visual_evidence(candidate: VisualSkillCandidate, *, example_image: str = "") -> list[dict[str, Any]]:
    evidence = []
    rep = candidate.representative
    if rep.steps:
        evidence.append({
            "image_path": example_image or rep.steps[0].screenshot_path or None,
            "trigger_role": rep.steps[0].target_role,
            "task": rep.steps[0].task[:80],
            "domain": rep.domain,
            "app": rep.app,
        })
    for occ in candidate.occurrences[1:3]:
        if occ is rep or not occ.steps:
            continue
        evidence.append({
            "image_path": occ.steps[0].screenshot_path or None,
            "trigger_role": occ.steps[0].target_role,
            "task": occ.steps[0].task[:80],
            "domain": occ.domain,
            "app": occ.app,
        })
    return evidence


def _has_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _quoted_phrases(*texts: str) -> list[str]:
    phrases: list[str] = []
    for text in texts:
        phrases.extend(match.strip() for match in re.findall(r'"([^"]+)"', text or "") if match.strip())
    return phrases


def _target_context_for_role(
    *,
    target_text: str,
    action_description: str,
    class_name: str,
    resource: str,
    hint: str,
    content: str,
) -> str:
    text = (action_description or target_text or "").lower()
    # Mind2Web descriptions often append purpose clauses such as "to confirm
    # the city"; those clauses describe intent, not the clicked element type.
    text = re.split(r"\s+to\s+(?:access|apply|begin|confirm|continue|find|initiate|open|proceed|search|select|show|sort|submit|view)\b", text, maxsplit=1)[0]
    quoted = " ".join(phrase.lower() for phrase in _quoted_phrases(target_text, action_description))
    return " ".join([quoted, text, class_name, resource, hint, content]).strip()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _first_present(record: dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


if __name__ == "__main__":
    raise SystemExit(main())
