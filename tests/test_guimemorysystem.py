from __future__ import annotations

import json

from PIL import Image

from guimemorysystem import GUIMemorySystem


class FakeEngine:
    def chat(self, messages, max_tokens=300, temperature=None):
        system = messages[0]["content"]
        if "experience selector" in system:
            user = messages[1]["content"]
            if "Apply button visible" not in user:
                return json.dumps({"experience_id": None, "reason": "no concrete trigger"})
            return json.dumps({"experience_id": "exp_apply", "reason": "Apply is visible after a filter change."})
        return json.dumps({"older_summary": "Earlier: opened filters and changed one value."})

    def chat_with_images(
        self,
        system_prompt,
        user_text,
        current_image,
        previous_image=None,
        max_tokens=300,
        lossy=True,
    ):
        return json.dumps(
            {
                "item_type": "recent_change_keyframe",
                "change": "filter value changed and apply remains visible",
                "element": "price filter",
                "action_type": "CLICK",
                "action_value": "",
                "focus_after": "filter modal",
                "next_goal": "apply the filter",
                "keep_image": True,
                "action": "CLICK price filter",
            }
        )


def _image():
    return Image.new("RGB", (64, 64), "white")


def test_prepare_step_updates_task_memory_and_selects_experience():
    engine = FakeEngine()
    system = GUIMemorySystem(
        memory_engine=engine,
        selector_engine=engine,
        experience_catalog=[{"id": "exp_apply", "title": "confirm filters", "when": "Apply is visible"}],
        experience_library={
            "exp_apply": {
                "experience_id": "exp_apply",
                "title": "confirm filters",
                "applicable_context": {"when": "Apply is visible"},
                "action_guidance": "Click Apply before leaving the modal.",
                "action_templates": ["CLICK [button: Apply]"],
            }
        },
        recent_k=3,
    )
    system.reset("Find listings with the requested filters")

    first = system.prepare_step(_image(), current_page_title="Search")
    assert first.context_text == ""

    system.commit_action("CLICK price filter")
    second = system.prepare_step(_image(), observation_summary="Apply button visible")

    assert "[recent_buffer]" in second.task_memory_text
    assert second.history_image_count == 1
    assert second.experience_id == "exp_apply"
    assert "[active_experience]" in second.active_experience_text


def test_recent_buffer_rolls_into_older_summary():
    engine = FakeEngine()
    system = GUIMemorySystem(memory_engine=engine, recent_k=1)
    system.reset("Complete the task")
    system.prepare_step(_image())

    system.commit_action("CLICK one")
    system.prepare_step(_image())
    system.commit_action("CLICK two")
    context = system.prepare_step(_image())

    assert "Earlier:" in context.task_memory_text
    assert len(system.task_memory.recent_buffer) == 1
