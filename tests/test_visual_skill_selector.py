from __future__ import annotations

import json

from PIL import Image

from guimemorysystem.visual_skill_selector import (
    RecentFrameContext,
    retrieve_visual_skill_catalog_candidates,
    select_visual_skills,
)


class FakeVisualSelectorEngine:
    def __init__(self) -> None:
        self.messages = None
        self.response_format = None

    def chat(self, messages, max_tokens=300, temperature=None, response_format=None):
        self.messages = messages
        self.response_format = response_format
        user_content = messages[1]["content"]
        text = "\n".join(item.get("text", "") for item in user_content if item.get("type") == "text")
        assert "candidate_targets" not in text
        assert "Current frame image:" in text
        assert "Recent frame 1 image:" in text
        return json.dumps(
            {
                "selected_skills": [
                    {
                        "skill_id": "skill_search",
                        "confidence": 0.87,
                        "matched_visual_evidence": "current frame has a visible search input",
                        "expected_target_description": "editable search bar near the top of the page",
                        "slot_values": {"query": "Donnie Darko"},
                        "reason": "The task requires entering a query and the current frame shows a search box.",
                    }
                ],
                "no_skill_needed": False,
                "selector_notes": "search flow likely applies",
            }
        )


def _catalog():
    return [
        {
            "skill_id": "skill_search",
            "title": "Input Query Into Search Bar",
            "signature": "search_bar:input_text:query",
            "when": "The current UI contains an editable search field and the task needs a query.",
            "visual_cues": ["placeholder contains search", "magnifying icon near input"],
            "support": {"num_occurrences": 100, "num_tasks": 80, "num_domains": 5},
        },
        {
            "skill_id": "skill_dropdown",
            "title": "Open Dropdown Then Select Option",
            "signature": "dropdown_or_select:click:none -> text_option:click:none",
            "when": "The current UI contains a dropdown and a selectable option.",
            "visual_cues": ["option selector"],
            "support": {"num_occurrences": 50, "num_tasks": 40, "num_domains": 4},
        },
    ]


def _library():
    return {
        "skill_search": {
            "skill_id": "skill_search",
            "experience_id": "skill_search",
            "title": "Input Query Into Search Bar",
            "applicable_context": {"when": "A search bar is visible and query input is needed."},
            "target_instruction": "Find the search bar in the current frame.",
            "action_guidance": "Input the query into the search bar.",
            "action_templates": ["input_text(target_bbox, <query>)"],
            "value_slots": [{"name": "query", "source": "task"}],
            "example": {"target_role": "search_bar"},
        },
        "skill_dropdown": {
            "skill_id": "skill_dropdown",
            "experience_id": "skill_dropdown",
            "title": "Open Dropdown Then Select Option",
            "applicable_context": {"when": "A dropdown needs an option selection."},
            "target_instruction": "Find the dropdown.",
            "action_guidance": "Click dropdown then option.",
            "action_templates": ["click(dropdown_bbox)", "click(option_bbox)"],
            "example": {"target_role": "dropdown_or_select"},
        },
    }


def _image():
    return Image.new("RGB", (80, 60), "white")


def test_retrieve_visual_skill_catalog_candidates_prefers_search_query():
    candidates = retrieve_visual_skill_catalog_candidates(
        _catalog(),
        task="Search IMDb for Donnie Darko",
        current_observation="A page with a Search IMDb input is visible.",
        recent_frames=[],
        max_candidates=2,
    )

    assert candidates[0]["skill_id"] == "skill_search"
    assert candidates[0]["_retrieval_score"] > candidates[1]["_retrieval_score"]


def test_select_visual_skills_uses_current_and_recent_frames_without_candidate_targets():
    engine = FakeVisualSelectorEngine()
    selection = select_visual_skills(
        engine=engine,
        task="Search IMDb for Donnie Darko",
        current_frame=_image(),
        catalog=_catalog(),
        library=_library(),
        current_observation="Search IMDb input is visible.",
        recent_frames=[
            RecentFrameContext(
                image=_image(),
                action="CLICK search icon",
                result_summary="search field is now focused",
            )
        ],
        max_catalog_candidates=2,
        max_selected_skills=2,
    )

    assert selection.matched
    assert engine.response_format == {"type": "json_object"}
    assert selection.selected_skills[0].skill_id == "skill_search"
    assert "input_text(target_bbox, <query>)" in selection.injection
    assert "Donnie Darko" in selection.injection


class FakeV3SelectorEngine:
    def __init__(self) -> None:
        self.response_format = None

    def chat(self, messages, max_tokens=300, temperature=None, response_format=None):
        self.response_format = response_format
        user_content = messages[1]["content"]
        text = "\n".join(item.get("text", "") for item in user_content if item.get("type") == "text")
        assert "version=visual_skill_v3" in text
        assert "procedure=" in text
        return json.dumps(
            {
                "selected_skills": [
                    {
                        "skill_id": "v3_dropdown",
                        "confidence": 0.91,
                        "matched_visual_evidence": "current page shows Make and Model dropdowns",
                        "matched_skill_evidence": "historical evidence has dependent vehicle dropdowns",
                        "expected_target_description": "Make dropdown followed by Model dropdown",
                        "suggested_plan": "select Make, wait for Model, then select Model",
                        "slot_values": {"make": "Honda", "model": "Civic"},
                        "reason": "The task and current form match the dependent dropdown planning skill.",
                    }
                ],
                "no_skill_needed": False,
            }
        )


class FakeMalformedV3SelectorEngine:
    def __init__(self) -> None:
        self.response_format = None

    def chat(self, messages, max_tokens=300, temperature=None, response_format=None):
        self.response_format = response_format
        return """
{
  "selected_skills": [
    {
      "skill_id": "v3_dropdown",
      "confidence": 0.91,
      "matched_visual_evidence": "current page shows Make and Model dropdowns"
      "suggested_plan": "select Make first then Model"
    }
  ],
  "no_skill_needed": false
}
"""


def test_select_visual_skill_v3_renders_procedural_planning():
    catalog = [
        {
            "skill_id": "v3_dropdown",
            "version": "visual_skill_v3",
            "title": "Plan Select Through Dropdowns",
            "intent": "Transfer dependent dropdown search planning.",
            "signature": "dropdown_or_select:select:query -> dropdown_or_select:select:query",
            "when": "Use when a page has dependent dropdowns.",
            "preconditions": ["Make/Model dropdowns are visible."],
            "procedure": ["Select Make first.", "Wait for Model to refresh.", "Select Model."],
            "retrieval": {"page_state_summary": "Historical states show vehicle dropdown forms."},
            "support": {"num_trajectory_segments": 5, "num_tasks": 5, "num_domains": 1},
        }
    ]
    library = {
        "v3_dropdown": {
            "skill_id": "v3_dropdown",
            "experience_id": "v3_dropdown",
            "version": "visual_skill_v3",
            "title": "Plan Select Through Dropdowns",
            "intent": "Transfer dependent dropdown search planning.",
            "applicable_context": {"preconditions": ["Make/Model dropdowns are visible."]},
            "planning": {
                "procedure": ["Select Make first.", "Wait for Model to refresh.", "Select Model."],
                "postcondition_checks": ["Model options become available."],
                "failure_modes": ["Model dropdown is disabled until Make is selected."],
                "recovery_steps": ["Fill Make first, then reopen Model."],
            },
        }
    }

    engine = FakeV3SelectorEngine()
    selection = select_visual_skills(
        engine=engine,
        task="Find a Honda Civic near me",
        current_frame=_image(),
        catalog=catalog,
        library=library,
        current_observation="Make and Model dropdowns are visible.",
        max_catalog_candidates=1,
        max_selected_skills=1,
    )

    assert selection.matched
    assert engine.response_format == {"type": "json_object"}
    assert "planning_procedure" in selection.injection
    assert "Select Make first." in selection.injection
    assert "Model dropdown is disabled" in selection.injection


def test_select_visual_skill_v3_recovers_malformed_selector_json():
    catalog = [
        {
            "skill_id": "v3_dropdown",
            "version": "visual_skill_v3",
            "title": "Plan Select Through Dropdowns",
            "intent": "Transfer dependent dropdown search planning.",
            "signature": "dropdown_or_select:select:query -> dropdown_or_select:select:query",
            "when": "Use when a page has dependent dropdowns.",
            "preconditions": ["Make/Model dropdowns are visible."],
            "procedure": ["Select Make first.", "Wait for Model to refresh.", "Select Model."],
            "support": {"num_trajectory_segments": 5, "num_tasks": 5, "num_domains": 1},
        }
    ]
    library = {
        "v3_dropdown": {
            "skill_id": "v3_dropdown",
            "experience_id": "v3_dropdown",
            "version": "visual_skill_v3",
            "title": "Plan Select Through Dropdowns",
            "intent": "Transfer dependent dropdown search planning.",
            "applicable_context": {"preconditions": ["Make/Model dropdowns are visible."]},
            "planning": {"procedure": ["Select Make first.", "Wait for Model to refresh.", "Select Model."]},
        }
    }

    engine = FakeMalformedV3SelectorEngine()
    selection = select_visual_skills(
        engine=engine,
        task="Find a Honda Civic near me",
        current_frame=_image(),
        catalog=catalog,
        library=library,
        current_observation="Make and Model dropdowns are visible.",
        max_catalog_candidates=1,
        max_selected_skills=1,
    )

    assert selection.matched
    assert engine.response_format == {"type": "json_object"}
    assert selection.selector_notes == "recovered from malformed selector JSON"
    assert "Select Make first." in selection.injection
