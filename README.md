# GUIMem

GUIMem is a memory-augmented GUI agent evaluation framework. It adds an
explicit memory layer around a GUI policy so the agent can use both:

- **Task-specific memory**: rolling memory for the current task, with an
  `older_summary`, a structured `recent_buffer`, and optional keyframe
  screenshots.
- **Cross-task memory**: reusable experience rules selected at step time from
  an experience catalog/library and injected as an `[active_experience]` block.

The memory system is benchmark-agnostic. Evaluation adapters translate each
environment's screenshot/action interface into the same policy and memory
contract.

## Project Layout

```text
GUIMem/
  guimemorysystem/        # task-specific memory + cross-task selector
  policy/                 # GUIPivot-style policy wrapper with memory injection
  eval_adapters/          # benchmark adapters that run the memory policy
  evaluation/             # benchmark code, configs, and task definitions
  tests/                  # lightweight unit tests for the memory system
```

Important modules:

- `guimemorysystem.system.GUIMemorySystem`: unified memory API for eval loops.
- `guimemorysystem.task_memory.MemoryState`: online `older_summary` +
  `recent_buffer` state.
- `guimemorysystem.cross_task_memory.select_experience`: Stage-C cross-task
  experience selector.
- `policy.memory_policy.MemoryAugmentedPolicy`: policy wrapper that calls
  `GUIMemorySystem` before each action.

## Memory Contract

For each task, the evaluation loop follows this lifecycle:

1. `reset(task)` starts a fresh task memory state.
2. `prepare_step(screenshot, ...)` updates memory from the last transition and
   returns prompt context for the current step.
3. The policy chooses an action.
4. The environment executes the action.
5. `commit_action(action_repr)` records the executed action so the next
   screenshot can be compared against the current one.

Minimal usage:

```python
from guimemorysystem import GUIMemorySystem, OpenAICompatibleEngine

memory_engine = OpenAICompatibleEngine(
    model="gpt-4o-mini",
    api_base="https://api.openai.com/v1",
    api_key=api_key,
)
selector_engine = OpenAICompatibleEngine(
    model="gpt-4o-mini",
    api_base="https://api.openai.com/v1",
    api_key=api_key,
)

memory = GUIMemorySystem.from_paths(
    memory_engine=memory_engine,
    selector_engine=selector_engine,
    experience_catalog_path="outputs/cross_task_experience/catalog.json",
    experience_library_path="outputs/cross_task_experience/experience_library.jsonl",
    recent_k=3,
)

memory.reset("Book a hotel with the requested filters", task_id="task_001")

context = memory.prepare_step(
    screenshot,
    current_url=current_url,
    current_page_title=current_title,
    observation_summary="date picker modal is open",
)

# Inject these into the policy prompt.
context.context_text
context.task_memory_images

memory.commit_action("CLICK [button: Apply]")
```

`GUIMemorySystem` does not parse, convert, or execute actions. The adapter owns
that benchmark-specific work and passes the executed action back as text.

## Configuration

Ready-to-edit launcher configs live in `configs/`:

```text
configs/
  offline_multimodal_mind2web.yaml
  offline_android_control.yaml
  online_mind2web.yaml
  webarena_lite.yaml
  webarena_lite/agent_memory.yaml
  webarena_lite/env.yaml
  android_world.yaml
  android_world_docker.yaml
```

Run a config with:

```bash
python /u/yhuang48/GUIMem/evaluate.py --config /u/yhuang48/GUIMem/configs/offline_multimodal_mind2web.yaml
```

Dry-run first to inspect the resolved adapter command:

```bash
python /u/yhuang48/GUIMem/evaluate.py --config /u/yhuang48/GUIMem/configs/webarena_lite.yaml --dry-run
```

Common environment variables:

```bash
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=...

export MEMORY_MODEL=gpt-4o-mini
export MEMORY_API_BASE=$OPENAI_BASE_URL
export MEMORY_API_KEY=$OPENAI_API_KEY
export MEMORY_RECENT_K=3

export SELECTOR_MODEL=gpt-4o-mini
export SELECTOR_API_BASE=$OPENAI_BASE_URL
export SELECTOR_API_KEY=$OPENAI_API_KEY
export EXPERIENCE_CATALOG_PATH=outputs/cross_task_experience/catalog.json
export EXPERIENCE_LIBRARY_PATH=outputs/cross_task_experience/experience_library.jsonl
```

If `MEMORY_MODEL` is empty, task-specific memory is disabled. If
`SELECTOR_MODEL` or either experience path is empty, cross-task memory is
disabled.

## Running Evaluations

The recommended entrypoint is the root launcher:

```bash
python /u/yhuang48/GUIMem/evaluate.py --benchmark <name> -- <benchmark args>
```

or:

```bash
python /u/yhuang48/GUIMem/evaluate.py --config <config.yaml>
```

The launcher sets `PYTHONPATH=/u/yhuang48/GUIMem`, chooses the corresponding
memory-enabled adapter, and forwards config/CLI args to that adapter.

Available benchmark names:

- `online_mind2web`
- `webarena_lite`
- `android_world`
- `android_world_docker`
- `offline`

Dry-run a command without launching the benchmark:

```bash
python /u/yhuang48/GUIMem/evaluate.py --benchmark offline --dry-run -- \
  --benchmark multimodal_mind2web \
  --input_file path/to/input.jsonl \
  --output_file outputs/offline_predictions.jsonl \
  --blocks path/to/blocks
```

You can also run adapters directly. Use `PYTHONPATH` so the local packages are
importable:

```bash
export PYTHONPATH=/u/yhuang48/GUIMem
```

Online-Mind2Web:

```bash
python /u/yhuang48/GUIMem/eval_adapters/online_mind2web/run_memory.py ...
```

WebArenaLiteV2:

```bash
python /u/yhuang48/GUIMem/eval_adapters/webarena_lite/run_memory.py ...
```

Android World:

```bash
python /u/yhuang48/GUIMem/eval_adapters/android_world/run_memory.py ...
```

Android World through the Docker gateway:

```bash
python /u/yhuang48/GUIMem/eval_adapters/android_world/run_docker_memory.py ...
```

Static offline evaluation:

```bash
python /u/yhuang48/GUIMem/evaluate.py --benchmark offline -- \
  --benchmark multimodal_mind2web \
  --input_file path/to/input.jsonl \
  --output_file outputs/offline_predictions.jsonl \
  --blocks path/to/blocks
```

## Outputs

Adapters include memory debug fields in their per-step records when available:

- `memory_history_text`: rendered task-specific memory.
- `memory_history_image_count`: number of keyframe screenshots injected.
- `experience_id`: selected cross-task experience, or `null`.
- `experience_reason`: selector reason grounded in the current step.

Online experience writing, when enabled by an adapter, writes:

- summary buffer JSONL
- experience library JSONL
- catalog JSON

## Tests

Run the lightweight tests:

```bash
PYTHONPATH=/u/yhuang48/GUIMem python -m pytest -q /u/yhuang48/GUIMem/tests
```
