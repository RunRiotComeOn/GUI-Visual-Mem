from __future__ import annotations

import json

from PIL import Image

from guimemorysystem.visual_skill_memory import (
    VisualSkillMiningConfig,
    infer_target_role,
    load_offline_steps,
    mine_visual_skill_candidates,
    mine_visual_skill_v3_from_file,
    mine_visual_skills_from_file,
    normalize_records,
)


def test_mine_visual_skills_from_standard_jsonl(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (200, 120), "white").save(image_root / "search.png")

    rows = []
    for idx in range(3):
        rows.append(
            {
                "task_id": f"task_{idx}",
                "source_id": f"task_{idx}_0",
                "step": 0,
                "task": f"Search for item {idx}",
                "screenshot": "search.png",
                "action_type": "input_text",
                "action_value": f"item {idx}",
                "bbox": [20, 30, 160, 24],
                "target_text": "Search",
                "domain": f"domain_{idx % 2}",
                "app": "browser",
            }
        )
    input_path = tmp_path / "steps.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = mine_visual_skills_from_file(
        input_path,
        tmp_path / "skills",
        image_root=image_root,
        config=VisualSkillMiningConfig(min_support=3, min_tasks=3, min_domains=2, max_segment_len=1),
    )

    assert result["num_steps"] == 3
    assert result["num_skills"] == 1

    catalog = json.loads((tmp_path / "skills" / "catalog.json").read_text(encoding="utf-8"))
    assert catalog[0]["version"] == "visual_skill_v2"
    assert "search_bar:input_text:query" in catalog[0]["signature"]

    library_lines = (tmp_path / "skills" / "skill_library.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(library_lines[0])
    assert record["example"]["image_path"]
    assert record["example"]["bbox"] == [20.0, 30.0, 180.0, 54.0]


def test_android_control_normalization_finds_clicked_node():
    rows = [
        {
            "step": 2,
            "episode_id": 42,
            "goal": "Search for news about the stock price of Apple.",
            "screenshot": "42/screenshot_2.png",
            "screenshot_width": 1080,
            "screenshot_height": 2400,
            "action": {
                "action_type": "type_text",
                "x": 100,
                "y": 220,
                "text": "Stock price of Apple",
            },
            "accessibility_tree": [
                {
                    "text": "",
                    "content_description": "",
                    "hint_text": "Search",
                    "class_name": "android.widget.EditText",
                    "is_editable": True,
                    "is_visible": True,
                    "package_name": "com.browser",
                    "bbox_pixels": {
                        "x_min": 20,
                        "x_max": 500,
                        "y_min": 180,
                        "y_max": 260,
                    },
                }
            ],
        }
    ]

    steps = normalize_records(rows, dataset="android_control")

    assert len(steps) == 1
    assert steps[0].action_type == "input_text"
    assert steps[0].target_role == "search_bar"
    assert steps[0].bbox is not None
    assert steps[0].bbox.to_list() == [20.0, 180.0, 500.0, 260.0]


def test_segment_mining_counts_contiguous_motifs(tmp_path):
    rows = []
    for idx in range(3):
        rows.extend(
            [
                {
                    "task_id": f"task_{idx}",
                    "source_id": f"task_{idx}_0",
                    "step": 0,
                    "task": f"Search for item {idx}",
                    "action_type": "input_text",
                    "action_value": f"item {idx}",
                    "bbox": [10, 10, 100, 20],
                    "target_text": "Search",
                    "domain": "generic",
                },
                {
                    "task_id": f"task_{idx}",
                    "source_id": f"task_{idx}_1",
                    "step": 1,
                    "task": f"Search for item {idx}",
                    "action_type": "click",
                    "bbox": [120, 10, 50, 20],
                    "target_text": "Search",
                    "domain": "generic",
                },
            ]
        )
    input_path = tmp_path / "steps.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    steps = load_offline_steps(input_path)

    candidates = mine_visual_skill_candidates(
        steps,
        VisualSkillMiningConfig(min_support=3, min_tasks=3, max_segment_len=2),
    )

    signatures = {candidate.signature for candidate in candidates}
    assert "search_bar:input_text:query -> search_button:click:none" in signatures


def test_v3_skill_requires_supported_segments_and_writes_planning_schema(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (240, 160), "white").save(image_root / "form.png")

    rows = []
    for idx in range(3):
        rows.extend(
            [
                {
                    "task_id": f"car_task_{idx}",
                    "source_id": f"car_task_{idx}_make",
                    "step": 0,
                    "task": f"Search for Honda Civic 202{idx}",
                    "screenshot": "form.png",
                    "action_type": "select",
                    "action_value": "Honda",
                    "bbox": [10, 20, 80, 24],
                    "target_text": "Make dropdown",
                    "domain": "shopping",
                    "app": "cars",
                },
                {
                    "task_id": f"car_task_{idx}",
                    "source_id": f"car_task_{idx}_model",
                    "step": 1,
                    "task": f"Search for Honda Civic 202{idx}",
                    "screenshot": "form.png",
                    "action_type": "select",
                    "action_value": "Civic",
                    "bbox": [100, 20, 80, 24],
                    "target_text": "Model dropdown",
                    "domain": "shopping",
                    "app": "cars",
                },
            ]
        )
    input_path = tmp_path / "steps.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = mine_visual_skill_v3_from_file(
        input_path,
        tmp_path / "skills_v3",
        image_root=image_root,
        config=VisualSkillMiningConfig(
            min_support=3,
            min_tasks=3,
            min_domains=1,
            max_segment_len=2,
            include_single_step_skills=False,
        ),
    )

    assert result["version"] == "visual_skill_v3"
    assert result["num_skills"] == 1

    catalog = json.loads((tmp_path / "skills_v3" / "catalog.json").read_text(encoding="utf-8"))
    assert catalog[0]["version"] == "visual_skill_v3"
    assert catalog[0]["skill_id"].startswith("v3_")
    assert catalog[0]["support"]["num_trajectory_segments"] == 3

    record = json.loads((tmp_path / "skills_v3" / "skill_library.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["skill_type"] == "ui_planning_skill"
    assert "planning" in record
    assert any("dependent dropdown" in item.lower() for item in record["planning"]["failure_modes"])
    assert record["support"]["num_tasks"] == 3


def test_mind2web_role_inference_ignores_task_purpose_text():
    role, _ = infer_target_role(
        action_type="click",
        target_text='Click on the "Map search" option under the "BOOK A FLIGHT" section to access the interactive map.',
        action_description='Click on the "Map search" option under the "BOOK A FLIGHT" section to access the interactive map.',
        task="Show me options for a roundtrip flight.",
    )
    assert role == "text_option"

    role, _ = infer_target_role(
        action_type="click",
        target_text='Click on the "Check-in" tab in the top navigation menu to proceed with the check-in process.',
        action_description='Click on the "Check-in" tab in the top navigation menu to proceed with the check-in process.',
        task="Check in with confirmation number 10987654.",
    )
    assert role == "tab"

    role, _ = infer_target_role(
        action_type="click",
        target_text='Click on the suggested search result "Sacramento, CA, US (SMF)" to confirm the departure city.',
        action_description='Click on the suggested search result "Sacramento, CA, US (SMF)" to confirm the departure city.',
        task="Find flights from Sacramento.",
    )
    assert role == "text_option"
