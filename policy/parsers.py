"""Parsers and converters for GUIPivot-style policy outputs.

Mirrors GUI-Libra/evaluation/online-mind2web-eval/syn/prompts.py. Kept here so
that all four bench adapters can share a single canonical parser.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

try:
    import json_repair as _json_repair  # type: ignore
except ImportError:  # bare envs without GUI-Libra installed
    _json_repair = None

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
SMART_RESIZE_MIN_PIXELS = 3136
SMART_RESIZE_MAX_PIXELS = 2109744
SMART_RESIZE_MAX_RATIO = 200


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > SMART_RESIZE_MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {SMART_RESIZE_MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_plan_guipivot(plan: str) -> tuple[str, str, str]:
    think_match = re.search(r"<think>(.*?)</think>", plan, re.DOTALL)
    if not think_match:
        think_match = re.search(r"<thinking>(.*?)</thinking>", plan, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""

    answer_match = re.search(r"<answer>(.*?)</answer>", plan, re.DOTALL)
    action = answer_match.group(1).strip() if answer_match else ""
    if not action:
        try:
            data = _loads_tolerant(plan)
            if isinstance(data, dict) and len(data) > 0:
                action = json.dumps(data)
        except Exception:
            action = ""

    operation = ""
    if action:
        op_match = re.search(r'"action_description"\s*:\s*"([^"]+)"', action)
        if op_match:
            operation = op_match.group(1)
        else:
            type_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', action)
            if type_match:
                operation = type_match.group(1)

    return thought, operation, action


def _extract_fields_via_regex(raw: str) -> dict:
    result: dict[str, Any] = {}
    for field in ("action_type", "action_description", "value"):
        m = re.search(rf"<{field}>(.*?)</{field}>", raw, re.DOTALL)
        if m:
            val = m.group(1).strip()
            if val.lower() == "none":
                val = None
            result[field] = val

    p2d_match = re.search(
        r"<point_2d>\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]+\s*(-?\d+(?:\.\d+)?)\s*\]?\s*</point_2d>",
        raw,
    )
    if not p2d_match:
        p2d_match = re.search(
            r'"point_2d"\s*:\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]+\s*(-?\d+(?:\.\d+)?)',
            raw,
        )
    if p2d_match:
        result["point_2d"] = [float(p2d_match.group(1)), float(p2d_match.group(2))]

    if "action_type" not in result:
        m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
        if m:
            result["action_type"] = m.group(1)
    if "action_description" not in result:
        m = re.search(r'"action_description"\s*:\s*"([^"]*)"', raw)
        if m:
            result["action_description"] = m.group(1)
    if "value" not in result:
        m = re.search(r'"value"\s*:\s*"([^"]*)"', raw)
        if m:
            result["value"] = m.group(1)
    return result


def _loads_tolerant(text: str):
    if _json_repair is not None:
        return _json_repair.loads(text)
    return json.loads(text)


def parse_guipivot_json(json_str: str) -> dict:
    try:
        data = _loads_tolerant(json_str)
        if isinstance(data, dict) and len(data) > 0:
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and len(item) > 0:
                    return item
    except Exception:
        pass

    cleaned = json_str
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    if cleaned != json_str:
        try:
            data = _loads_tolerant(cleaned)
            if isinstance(data, dict) and len(data) > 0:
                return data
        except Exception:
            pass

    fixed = re.sub(r":\s*None\b", ": null", json_str)
    fixed = re.sub(r":\s*True\b", ": true", fixed)
    fixed = re.sub(r":\s*False\b", ": false", fixed)
    try:
        data = json.loads(fixed)
        if isinstance(data, dict) and len(data) > 0:
            return data
    except Exception:
        pass

    data = _extract_fields_via_regex(json_str)
    if data and "action_type" in data:
        return data

    raise ValueError(f"Failed to parse GUIPivot JSON with all strategies: {json_str[:300]}")


_CANONICAL_ACTION_MAP = {
    "click": "Click",
    "leftclick": "Click",
    "left": "Click",
    "tap": "Click",
    "longpress": "LongPress",
    "long_press": "LongPress",
    "select": "Select",
    "write": "Write",
    "type": "Write",
    "input": "Write",
    "keyboardpress": "KeyboardPress",
    "keypress": "KeyboardPress",
    "press": "KeyboardPress",
    "scroll": "Scroll",
    "swipe": "Scroll",
    "scrolldown": "Scroll",
    "scrollup": "Scroll",
    "answer": "Answer",
    "back": "Back",
    "goback": "Back",
    "navigateback": "NavigateBack",
    "navigate_back": "NavigateBack",
    "navigatehome": "NavigateHome",
    "navigate_home": "NavigateHome",
    "navigate": "Navigate",
    "openwebsite": "Navigate",
    "openapp": "OpenApp",
    "open_app": "OpenApp",
    "terminate": "terminate",
    "stop": "terminate",
    "wait": "wait",
    "call": "Answer",
}


def normalize_action_type(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    s = raw.strip()
    for prefix in ("action_type:", "action:", "action_type :", "action :"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
            break
    key = s.lower().replace(" ", "").replace("_", "")
    return _CANONICAL_ACTION_MAP.get(key, s)


def convert_to_exec_actions(
    data: dict,
    screen_width: int,
    screen_height: int,
    model_coord_size: int = 1000,
    use_smart_resize: bool = False,
) -> list[dict]:
    """Convert parsed GUIPivot JSON into the bench-neutral exec_action list.

    The format here is the WebArenaLiteV2 / online-mind2web exec_action format:
    [{"name": "<click|write|press|swipe|back|response|terminate|wait>", "parameters": {...}}].
    """
    action_type = normalize_action_type(data.get("action_type", ""))
    value = data.get("value", "")
    point_2d = data.get("point_2d")

    if not point_2d or not isinstance(point_2d, list):
        action_target = data.get("action_target", {})
        if isinstance(action_target, dict):
            if "point_2d" in action_target:
                point_2d = action_target["point_2d"]
            elif "coordinates" in action_target:
                point_2d = action_target["coordinates"]
        if not point_2d or not isinstance(point_2d, list):
            for alt_key in ("coordinates", "position", "coord", "point"):
                alt = data.get(alt_key)
                if isinstance(alt, list) and len(alt) == 2:
                    point_2d = alt
                    break

    if value == "None" or value is None:
        value = ""

    def scale_coords(x: Any, y: Any) -> tuple[float | None, float | None]:
        if x is None or y is None:
            return None, None
        if x == -100 and y == -100:
            return None, None
        if use_smart_resize:
            resize_h, resize_w = smart_resize(screen_height, screen_width)
            return (x / resize_w) * screen_width, (y / resize_h) * screen_height
        return (x / model_coord_size) * screen_width, (y / model_coord_size) * screen_height

    exec_actions: list[dict] = []
    try:
        if action_type in ("Click", "Select", "LongPress"):
            if isinstance(point_2d, list) and len(point_2d) == 2:
                sx, sy = scale_coords(point_2d[0], point_2d[1])
                if sx is None or sy is None:
                    raise ValueError(f"Invalid point_2d after scaling: {point_2d}")
                name = "long_press" if action_type == "LongPress" else "click"
                exec_actions = [{"name": name, "parameters": {"x": sx, "y": sy, "clicks": 1, "button": "left"}}]
            else:
                raise ValueError(f"Invalid point_2d: {point_2d}")

        elif action_type == "Write":
            if isinstance(point_2d, list) and len(point_2d) == 2:
                sx, sy = scale_coords(point_2d[0], point_2d[1])
                if sx is not None and sy is not None:
                    exec_actions.append({"name": "click", "parameters": {"x": sx, "y": sy, "clicks": 1, "button": "left"}})
            exec_actions.append({"name": "write", "parameters": {"message": value}})
            exec_actions.append({"name": "press", "parameters": {"keys": "enter"}})

        elif action_type == "KeyboardPress":
            exec_actions = [{"name": "press", "parameters": {"keys": (value or "").lower()}}]

        elif action_type == "Scroll":
            direction = (value or "down").lower()
            mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
            target_direction = mapping.get(direction, "down")
            exec_actions = [{"name": "swipe", "parameters": {"direction": target_direction, "amount": 0.5}}]

        elif action_type in ("Back", "NavigateBack") or (action_type == "Navigate" and value == "Back"):
            exec_actions = [{"name": "back", "parameters": {}}]

        elif action_type == "NavigateHome":
            exec_actions = [{"name": "home", "parameters": {}}]

        elif action_type == "OpenApp":
            exec_actions = [{"name": "open_app", "parameters": {"app_name": value}}]

        elif action_type == "Answer":
            exec_actions = [{"name": "response", "parameters": {"answer": value or "Task Completed"}}]

        elif action_type == "terminate":
            exec_actions = [{"name": "terminate", "parameters": {"status": "success", "info": value or "Task Completed"}}]

        elif action_type == "wait":
            exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

        else:
            logger.warning("Unknown action_type %r in policy output: %s", action_type, data)
            exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

    except Exception as exc:
        logger.error("Error converting action %s: %s", data, exc)
        exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

    return exec_actions


def action_to_history_repr(exec_action: dict) -> str:
    """Format an exec_action as a Mind2Web-style history line."""
    name = exec_action.get("name", "")
    params = exec_action.get("parameters", {})
    if name == "click":
        x, y = params.get("x"), params.get("y")
        if x is not None and y is not None:
            return f"CLICK at ({int(x)}, {int(y)})"
        return "CLICK"
    if name == "long_press":
        x, y = params.get("x"), params.get("y")
        if x is not None and y is not None:
            return f"LONG PRESS at ({int(x)}, {int(y)})"
        return "LONG PRESS"
    if name == "write":
        return f'TYPE "{params.get("message", "")}"'
    if name == "press":
        return f"PRESS {params.get('keys', '')}"
    if name == "swipe":
        direction = params.get("direction", "")
        display = {"up": "down", "down": "up", "left": "right", "right": "left"}.get(direction, direction)
        return f"SCROLL {display}"
    if name == "back":
        return "GO BACK"
    if name == "home":
        return "GO HOME"
    if name == "open_app":
        return f"OPEN APP {params.get('app_name', '')}"
    if name == "response":
        return f"ANSWER -> {params.get('answer', '')}"
    if name == "terminate":
        return f"TERMINATE -> {params.get('info', '')}"
    if name == "wait":
        return "WAIT"
    return name.upper()
