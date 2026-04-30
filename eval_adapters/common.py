"""Shared helpers for GUI-Libra benchmark adapters.

The benchmark adapters are intentionally thin: they translate each benchmark's
observation/action interface to the repo's benchmark-neutral
``MemoryAugmentedPolicy`` contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from policy import MemoryAugmentedPolicy, OpenAICompatibleEngine, PolicyStepResult
from policy.images import image_size
from policy.parsers import action_to_history_repr, convert_to_exec_actions


def _get_any(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            value = mapping[key]
            if isinstance(value, str):
                return os.path.expandvars(value)
            return value
    return default


def _default_api_key(api_base: str) -> str:
    if "localhost" in api_base or "127.0.0.1" in api_base:
        return "token-abc123"
    return ""


@dataclass
class MemoryAdapterConfig:
    """Model and memory options shared by non-Mind2Web adapters."""

    policy_model: str
    policy_api_base: str = "https://api.openai.com/v1"
    policy_api_key: str = ""
    policy_temperature: float = 0.0
    policy_rate_limit: int = -1

    memory_model: str = ""
    memory_api_base: str = "https://api.openai.com/v1"
    memory_api_key: str = ""
    memory_rate_limit: int = -1
    memory_recent_k: int = 3

    selector_model: str = ""
    selector_api_base: str = "https://api.openai.com/v1"
    selector_api_key: str = ""
    selector_rate_limit: int = -1
    experience_catalog_path: str = ""
    experience_library_path: str = ""
    visual_skill_store_path: str = ""

    history_last_k: int = 15
    max_policy_tokens: int = 4096

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MemoryAdapterConfig":
        """Build from a YAML/JSON config dict plus environment fallbacks."""
        policy_model = _get_any(
            data,
            "policy_model",
            "model",
            "model_name",
            default=os.getenv("OPENAI_MODEL", ""),
        )
        if not policy_model:
            raise ValueError("Missing policy model. Set policy_model/model or OPENAI_MODEL.")
        policy_api_base = str(
            _get_any(
                data,
                "policy_api_base",
                "api_base",
                "base_url",
                "openai_api_base",
                default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        )

        return cls(
            policy_model=str(policy_model),
            policy_api_base=policy_api_base,
            policy_api_key=str(
                _get_any(
                    data,
                    "policy_api_key",
                    "api_key",
                    "openai_api_key",
                    default=os.getenv("OPENAI_API_KEY", _default_api_key(policy_api_base)),
                )
            ),
            policy_temperature=float(_get_any(data, "policy_temperature", "temperature", default=0.0)),
            policy_rate_limit=int(_get_any(data, "policy_rate_limit", "rate_limit", default=-1)),
            memory_model=str(_get_any(data, "memory_model", default=os.getenv("MEMORY_MODEL", ""))),
            memory_api_base=str(
                _get_any(data, "memory_api_base", default=os.getenv("MEMORY_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
            ),
            memory_api_key=str(_get_any(data, "memory_api_key", default=os.getenv("MEMORY_API_KEY", os.getenv("OPENAI_API_KEY", "")))),
            memory_rate_limit=int(_get_any(data, "memory_rate_limit", default=os.getenv("MEMORY_RATE_LIMIT", -1))),
            memory_recent_k=int(_get_any(data, "memory_recent_k", default=os.getenv("MEMORY_RECENT_K", 3))),
            selector_model=str(_get_any(data, "selector_model", default=os.getenv("SELECTOR_MODEL", ""))),
            selector_api_base=str(
                _get_any(data, "selector_api_base", default=os.getenv("SELECTOR_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
            ),
            selector_api_key=str(_get_any(data, "selector_api_key", default=os.getenv("SELECTOR_API_KEY", os.getenv("OPENAI_API_KEY", "")))),
            selector_rate_limit=int(_get_any(data, "selector_rate_limit", default=os.getenv("SELECTOR_RATE_LIMIT", -1))),
            experience_catalog_path=str(_get_any(data, "experience_catalog_path", default=os.getenv("EXPERIENCE_CATALOG_PATH", ""))),
            experience_library_path=str(_get_any(data, "experience_library_path", default=os.getenv("EXPERIENCE_LIBRARY_PATH", ""))),
            visual_skill_store_path=str(_get_any(data, "visual_skill_store_path", default=os.getenv("VISUAL_SKILL_STORE_PATH", ""))),
            history_last_k=int(_get_any(data, "history_last_k", default=os.getenv("HISTORY_LAST_K", 15))),
            max_policy_tokens=int(_get_any(data, "max_policy_tokens", default=os.getenv("MAX_POLICY_TOKENS", 4096))),
        )


def build_policy(config: MemoryAdapterConfig) -> MemoryAugmentedPolicy:
    policy_engine = OpenAICompatibleEngine(
        model=config.policy_model,
        api_base=config.policy_api_base,
        api_key=config.policy_api_key or os.getenv("OPENAI_API_KEY", ""),
        temperature=config.policy_temperature,
        rate_limit=config.policy_rate_limit,
    )

    memory_engine = None
    if config.memory_model:
        memory_engine = OpenAICompatibleEngine(
            model=config.memory_model,
            api_base=config.memory_api_base,
            api_key=config.memory_api_key or os.getenv("OPENAI_API_KEY", ""),
            rate_limit=config.memory_rate_limit,
        )

    selector_engine = None
    experience_catalog = None
    experience_library = None
    use_visual_skill = False
    if config.selector_model and config.visual_skill_store_path:
        from guimemorysystem import load_visual_skill_store

        selector_engine = OpenAICompatibleEngine(
            model=config.selector_model,
            api_base=config.selector_api_base,
            api_key=config.selector_api_key or os.getenv("OPENAI_API_KEY", ""),
            rate_limit=config.selector_rate_limit,
        )
        experience_catalog, experience_library = load_visual_skill_store(config.visual_skill_store_path)
        use_visual_skill = True
    elif config.selector_model and config.experience_catalog_path and config.experience_library_path:
        from guimemorysystem import load_catalog, load_library_by_id

        selector_engine = OpenAICompatibleEngine(
            model=config.selector_model,
            api_base=config.selector_api_base,
            api_key=config.selector_api_key or os.getenv("OPENAI_API_KEY", ""),
            rate_limit=config.selector_rate_limit,
        )
        experience_catalog = load_catalog(config.experience_catalog_path)
        experience_library = load_library_by_id(config.experience_library_path)

    return MemoryAugmentedPolicy(
        policy_engine=policy_engine,
        memory_engine=memory_engine,
        selector_engine=selector_engine,
        experience_catalog=experience_catalog,
        experience_library=experience_library,
        recent_k=config.memory_recent_k,
        last_k_actions=config.history_last_k,
        max_policy_tokens=config.max_policy_tokens,
        use_visual_skill=use_visual_skill,
    )


def exec_actions_from_policy_result(
    result: PolicyStepResult,
    screenshot: Any,
    *,
    use_smart_resize: bool = False,
) -> list[dict]:
    width, height = image_size(screenshot)
    return convert_to_exec_actions(
        result.action_data,
        screen_width=width,
        screen_height=height,
        use_smart_resize=use_smart_resize,
    )


def history_repr_for_exec_actions(exec_actions: list[dict], fallback: str = "") -> str:
    if not exec_actions:
        return fallback or "(no executable action)"
    return "; ".join(action_to_history_repr(action) for action in exec_actions)


def policy_debug_info(result: PolicyStepResult) -> dict[str, Any]:
    return {
        "thought": result.thought,
        "operation": result.operation,
        "actions": result.action_data,
        "raw_response": result.raw_response,
        "raw_input_messages": result.raw_messages,
        "memory_history_text": result.history_text,
        "memory_history_image_count": result.history_image_count,
        "experience_id": result.experience_id,
        "experience_reason": result.experience_reason,
    }
