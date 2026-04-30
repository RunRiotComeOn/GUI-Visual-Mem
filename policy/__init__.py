__all__ = [
    "OpenAICompatibleEngine",
    "MemoryAugmentedPolicy",
    "PolicyStepResult",
]


def __getattr__(name: str):
    if name == "OpenAICompatibleEngine":
        from policy.engines import OpenAICompatibleEngine

        return OpenAICompatibleEngine
    if name in {"MemoryAugmentedPolicy", "PolicyStepResult"}:
        from policy.memory_policy import MemoryAugmentedPolicy, PolicyStepResult

        return {
            "MemoryAugmentedPolicy": MemoryAugmentedPolicy,
            "PolicyStepResult": PolicyStepResult,
        }[name]
    raise AttributeError(name)
