"""Task-specific and cross-task memory for GUI-agent evaluation.

The package is intentionally policy-agnostic. Evaluation runners call
``GUIMemorySystem.prepare_step`` before each policy decision and
``GUIMemorySystem.commit_action`` after executing the chosen action.
"""

from guimemorysystem.cross_task_memory import (
    ExperienceSelection,
    build_selector_context,
    load_catalog,
    load_library_by_id,
    render_experience_slot,
    select_experience,
)
from guimemorysystem.engine import EngineProtocol, OpenAICompatibleEngine
from guimemorysystem.system import GUIMemorySystem, MemoryStepContext
from guimemorysystem.task_memory import MemoryState, render_recent_item

_VISUAL_SKILL_EXPORTS = {
    "BBox",
    "RecentFrameContext",
    "SelectedVisualSkill",
    "VisualSkillCandidate",
    "VisualSkillMiningConfig",
    "VisualSkillOccurrence",
    "VisualSkillSelection",
    "VisualSkillStep",
    "build_visual_skill_record",
    "load_offline_steps",
    "load_visual_skill_store",
    "mine_visual_skill_candidates",
    "mine_visual_skill_v3_from_file",
    "mine_visual_skill_v3_from_files",
    "mine_visual_skills_from_file",
    "mine_visual_skills_from_files",
    "normalize_records",
    "render_selected_visual_skills",
    "retrieve_visual_skill_catalog_candidates",
    "select_visual_skills",
    "write_visual_skill_store",
    "write_visual_skill_v3_store",
}

__all__ = [
    "EngineProtocol",
    "ExperienceSelection",
    "GUIMemorySystem",
    "MemoryState",
    "MemoryStepContext",
    "OpenAICompatibleEngine",
    "BBox",
    "RecentFrameContext",
    "SelectedVisualSkill",
    "VisualSkillCandidate",
    "VisualSkillMiningConfig",
    "VisualSkillOccurrence",
    "VisualSkillSelection",
    "VisualSkillStep",
    "build_visual_skill_record",
    "build_selector_context",
    "load_offline_steps",
    "load_catalog",
    "load_library_by_id",
    "load_visual_skill_store",
    "mine_visual_skill_candidates",
    "mine_visual_skill_v3_from_file",
    "mine_visual_skill_v3_from_files",
    "mine_visual_skills_from_file",
    "mine_visual_skills_from_files",
    "normalize_records",
    "render_selected_visual_skills",
    "render_experience_slot",
    "render_recent_item",
    "retrieve_visual_skill_catalog_candidates",
    "select_experience",
    "select_visual_skills",
    "write_visual_skill_store",
    "write_visual_skill_v3_store",
]


def __getattr__(name: str):
    if name in _VISUAL_SKILL_EXPORTS:
        import importlib

        module_name = "visual_skill_selector" if name in {
            "RecentFrameContext",
            "SelectedVisualSkill",
            "VisualSkillSelection",
            "load_visual_skill_store",
            "render_selected_visual_skills",
            "retrieve_visual_skill_catalog_candidates",
            "select_visual_skills",
        } else "visual_skill_memory"
        module = importlib.import_module(f"guimemorysystem.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
