# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GUIMem is a Python framework that augments GUI agents (LLM-based agents interacting with graphical interfaces) with two types of memory:

- **Task-specific memory**: Rolling summary + recent action buffer + optional keyframe screenshots for the current task
- **Cross-task memory**: Reusable experience rules selected at runtime and injected into the agent prompt

## Commands

```bash
# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_guimemorysystem.py -v

# Launch evaluation via root launcher (recommended)
python evaluate.py --config configs/offline_multimodal_mind2web.yaml
python evaluate.py --benchmark webarena_lite -- <adapter args>
python evaluate.py --dry-run --config configs/android_world.yaml   # resolve cmd without executing

# Direct adapter invocation (sets PYTHONPATH internally)
python eval_adapters/online_mind2web/run_memory.py ...
python eval_adapters/webarena_lite/run_memory.py ...
```

## Architecture

The codebase has three layers:

### 1. Memory Layer (`guimemorysystem/`)

The core library. `GUIMemorySystem` (`system.py`) is the unified entry point, orchestrating:
- `MemoryState` (`task_memory.py`): Maintains `older_summary` + `recent_buffer` + optional keyframes per task
- `CrossTaskMemory` (`cross_task_memory.py`): Selects and injects relevant past experiences
- `OpenAICompatibleEngine` (`engine.py`): Generic LLM client for any OpenAI-compatible endpoint
- `OnlineExperience` (`online_experience.py`): Stage-A/B trajectory summarization for writing new experiences
- `VisualSkillMemory` / `VisualSkillSelector` (`visual_skill_memory.py`, `visual_skill_selector.py`): Optional visual UI pattern mining (v2/v3)

**Memory lifecycle per task:**
```python
memory.reset(task)                        # start fresh
context = memory.prepare_step(screenshot) # get context before action
# policy chooses and executes action
memory.commit_action(action_repr)         # record what happened
```

`prepare_step()` returns: `task_memory_text`, `task_memory_images`, `active_experience_text`, `experience_id`, `experience_reason`, `previous_actions`, `current_observation`.

### 2. Policy Layer (`policy/`)

`MemoryAugmentedPolicy` (`memory_policy.py`) wraps a base GUI agent, injecting memory context into prompts before each step. It calls `prepare_step()` / `commit_action()` around every agent action and handles action parsing (`parsers.py`) and prompt construction (`prompts.py`).

### 3. Adapter Layer (`eval_adapters/`)

Benchmark-specific glue code. Each adapter translates a benchmark's screenshot/action interface into the memory contract. Adapters:
- `offline_evaluation/` — static trajectory replay
- `online_mind2web/` — live browser agent (uses multiprocessing spawn mode)
- `webarena_lite/` — WebArena Lite benchmark
- `android_world/` — Android World (also `run_docker_memory.py` for Docker gateway)

All adapters share `MemoryAdapterConfig` from `eval_adapters/common.py`.

## Configuration

YAML/JSON configs live in `configs/`. Structure: `benchmark` key, `env` block (env vars with `${VAR}` expansion), `args` block (CLI args forwarded to adapter).

Key environment variables (typically set via configs or `.env`):
- **Policy LLM**: `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`
- **Task memory LLM**: `MEMORY_MODEL`, `MEMORY_API_BASE`, `MEMORY_API_KEY`, `MEMORY_RECENT_K`
- **Experience selector**: `SELECTOR_MODEL`, `SELECTOR_API_BASE`, `SELECTOR_API_KEY`, `EXPERIENCE_CATALOG_PATH`, `EXPERIENCE_LIBRARY_PATH`
- **Visual skills** (optional): `VISUAL_SKILL_STORE_PATH`

## Key Design Decisions

- The memory system is **action-format agnostic**: adapters own action parsing and execution; `commit_action()` receives a plain string representation
- `evaluate.py` is the **canonical launcher**: it resolves configs, sets env vars, and subprocess-invokes the correct `run_memory.py`
- Outputs and data directories are gitignored; benchmark evaluation code lives under `evaluation/`
