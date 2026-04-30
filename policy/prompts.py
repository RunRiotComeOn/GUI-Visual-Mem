"""GUIPivot-style policy prompts with optional memory injection.

The base prompts (system + user template + output format) come from
GUI-Libra/evaluation/online-mind2web-eval/syn/prompts.py. We add an
`extra_context_text` slot that holds our memory-augmented blocks
(`older_summary`, `recent_buffer`, `active_experience`).
"""
from __future__ import annotations

from typing import Any

from policy.images import image_to_chat_content


GUIPIVOT_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to choose actions from the the following list:
action_type: Click, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Tap or click a specific UI element and provide its coordinates

action_type: Select, action_target: Element description, value: Value to select, point_2d: [x, y] or None
    ## Explanation: Select an item from a list or dropdown menu

action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y] or None
    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None

action_type: KeyboardPress, action_target: None, value: Key name (e.g., "enter"), point_2d: None
    ## Explanation: Press a specified key on the keyboard

action_type: Scroll, action_target: None, value: "up" | "down" | "left" | "right", point_2d: None
    ## Explanation: Scroll a view or container in the specified direction

action_type: Answer, action_target: None, value: Answer, point_2d: None
    ## Explanation: Return the final answer to the user's question

action_type: Wait, action_target: None, value: None, point_2d: None
    ## Explanation: Wait briefly when the page is loading or a verification check is in progress

Before working on the main task, handle blocking UI if it is visible. If a cookie,
privacy, or consent banner blocks the page, click the accept/allow/agree button,
or close it if accepting is not available. If a Cloudflare or similar "verify you
are human" interstitial shows a simple checkbox or button, click it first; if the
page says it is checking or verifying, wait. If a CAPTCHA puzzle, login wall,
paywall, or "Access Denied"/permission-denied page prevents progress and there is
no obvious verification or close button, do not click random locations; answer
that the page is blocked or inaccessible.
"""

GUIPIVOT_USER_PROMPT_TEMPLATE = """Please generate the next move according to the UI screenshot {screensize}, instruction and previous actions.

Instruction: {instruction}

Interaction History: {actions}
"""

GUIPIVOT_OUTPUT_FORMAT = """

The response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}
</answer>
"""


def format_previous_actions(previous_actions: list[str], last_k: int = 15) -> str:
    if not previous_actions:
        return "None"
    parts: list[str] = []
    for i, action in enumerate(previous_actions[-last_k:], start=1):
        parts.append(f"Step {i}\n Action: {action}\n")
    return "".join(parts) if parts else "None"


def build_policy_messages(
    task: str,
    screenshot: Any,
    image_size: tuple[int, int],
    previous_actions: list[str],
    *,
    history_text: str = "",
    active_experience_text: str = "",
    history_images: list[Any] | None = None,
    last_k: int = 15,
    history_text_char_budget: int | None = 24000,
) -> list[dict]:
    """Compose policy messages.

    The ordering inside the user content list is:
      [history_image_1, ..., history_image_k, current_screenshot, text]
    so that the policy LLM sees keyframes before the current screenshot.
    """
    image_width, image_height = image_size
    img_size_string = f"(original image size {image_width}x{image_height})"
    actions_str = format_previous_actions(previous_actions, last_k=last_k)
    user_prompt_core = GUIPIVOT_USER_PROMPT_TEMPLATE.format(
        screensize=img_size_string,
        instruction=task,
        actions=actions_str,
    )

    extra_blocks = [block for block in (history_text, active_experience_text) if block]
    extra_text = ""
    if extra_blocks:
        extra_text = "\n\n" + "\n\n".join(extra_blocks)
        if history_text_char_budget and len(extra_text) > history_text_char_budget:
            extra_text = extra_text[:history_text_char_budget].rstrip() + "\n[history truncated]"

    user_text = user_prompt_core + extra_text + GUIPIVOT_OUTPUT_FORMAT

    content: list[dict] = []
    if history_images:
        for img in history_images:
            if img is None:
                continue
            content.append(image_to_chat_content(img, lossy=True))
    content.append(image_to_chat_content(screenshot, lossy=False))
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": GUIPIVOT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
