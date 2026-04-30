"""Unified evaluation launcher for GUIMem.

This script is intentionally a thin dispatcher. Each benchmark already owns a
different CLI stack, so the launcher selects the right memory-enabled adapter,
sets ``PYTHONPATH`` to the GUIMem root, and forwards all remaining arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

BENCHMARKS = {
    "online_mind2web": ROOT / "eval_adapters" / "online_mind2web" / "run_memory.py",
    "webarena_lite": ROOT / "eval_adapters" / "webarena_lite" / "run_memory.py",
    "android_world": ROOT / "eval_adapters" / "android_world" / "run_memory.py",
    "android_world_docker": ROOT / "eval_adapters" / "android_world" / "run_docker_memory.py",
    "offline": ROOT / "eval_adapters" / "offline_evaluation" / "run_memory.py",
}

ALIASES = {
    "online-mind2web": "online_mind2web",
    "mind2web": "online_mind2web",
    "webarena": "webarena_lite",
    "webarena-lite": "webarena_lite",
    "android": "android_world",
    "android-docker": "android_world_docker",
    "offline_evaluation": "offline",
}


def _parse_env(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--env expects KEY=VALUE, got: {value}")
        key, val = value.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env expects non-empty KEY, got: {value}")
        parsed[key] = val
    return parsed


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    return data


def _args_from_config(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [os.path.expandvars(item) for item in shlex.split(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [os.path.expandvars(str(item)) for item in value]
    if isinstance(value, Mapping):
        args: list[str] = []
        for key, raw in value.items():
            flag = str(key)
            if not flag.startswith("-"):
                flag = "--" + flag
            if isinstance(raw, bool):
                if raw:
                    args.append(flag)
                continue
            if raw is None:
                continue
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                for item in raw:
                    args.extend([flag, os.path.expandvars(str(item))])
                continue
            args.extend([flag, os.path.expandvars(str(raw))])
        return args
    raise SystemExit("Config field 'args' must be a string, list, or mapping.")


def _env_from_config(value) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SystemExit("Config field 'env' must be a mapping.")
    return {str(key): os.path.expandvars(str(val)) for key, val in value.items() if val is not None}


def _available_benchmarks() -> str:
    return "\n".join(f"  - {name}" for name in sorted(BENCHMARKS))


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run GUIMem memory-enabled evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Available benchmarks:
{_available_benchmarks()}

Config file shape:
  benchmark: offline
  env:
    OPENAI_MODEL: gpt-4o-mini
  args:
    benchmark: multimodal_mind2web
    input_file: data.jsonl

Examples:
  python evaluate.py --config configs/offline_multimodal_mind2web.yaml --dry-run
  python evaluate.py --benchmark offline -- --benchmark multimodal_mind2web --input_file data.jsonl --output_file out.jsonl --blocks blocks/
  python evaluate.py --benchmark webarena_lite -- --agent_config_path config/agent/guilibra_native_agent_qwen25vl.yaml --env_config_path config/envs/web.yaml
  python evaluate.py --benchmark online_mind2web -- --tasks_path evaluation/online-mind2web-eval/configs/mind2web.300.jsonl
""",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Benchmark name. Use --list to show canonical names.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="YAML/JSON launcher config. CLI benchmark/env/adapter args override or append to this config.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the benchmark adapter.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable for the adapter process. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List canonical benchmark names and exit.",
    )

    args, rest = parser.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def resolve_benchmark(name: str) -> tuple[str, Path]:
    canonical = ALIASES.get(name, name)
    if canonical not in BENCHMARKS:
        valid = ", ".join(sorted(BENCHMARKS))
        raise SystemExit(f"Unknown benchmark {name!r}. Valid benchmarks: {valid}")
    script = BENCHMARKS[canonical]
    if not script.exists():
        raise SystemExit(f"Adapter script not found: {script}")
    return canonical, script


def build_env(extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(extra_env)
    return env


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def redact_command(parts: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in parts:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        lower = part.lower()
        if lower in {
            "--api_key",
            "--api-key",
            "--openai_api_key",
            "--openai-api-key",
            "--gpt.openai_api_key",
            "--memory_api_key",
            "--selector_api_key",
            "--online_experience_api_key",
        }:
            redacted.append(part)
            redact_next = True
            continue
        if "api_key=" in lower or "apikey=" in lower:
            key, _, _ = part.partition("=")
            redacted.append(f"{key}=[REDACTED]")
            continue
        if part.startswith("sk-"):
            redacted.append("[REDACTED]")
            continue
        redacted.append(part)
    return redacted


def main(argv: list[str] | None = None) -> int:
    args, adapter_args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        print(_available_benchmarks())
        return 0

    config = _load_config(args.config)
    benchmark = args.benchmark or config.get("benchmark")
    if not benchmark:
        raise SystemExit("--benchmark is required unless provided by --config.")

    config_adapter_args = _args_from_config(config.get("args"))
    config_env = _env_from_config(config.get("env"))
    config_env.update(_parse_env(args.env))

    canonical, script = resolve_benchmark(str(benchmark))
    cmd = [args.python, str(script), *config_adapter_args, *adapter_args]
    env = build_env(config_env)

    print(f"GUIMem benchmark: {canonical}")
    if args.config:
        print(f"Config: {Path(args.config).resolve()}")
    print(f"Command: {shell_join(redact_command(cmd))}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
