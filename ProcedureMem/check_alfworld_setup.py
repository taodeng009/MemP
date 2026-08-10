"""Validate the stage-1 ALFWorld runtime without making model API calls."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ProcedureMem.runtime_config import (
    DEFAULT_ALFWORLD_CONFIG,
    configure_runtime,
    load_alfworld_config,
    validate_alfworld_data,
)


REQUIRED_IMPORTS = (
    "alfworld",
    "faiss",
    "yaml",
    "litellm",
    "langchain_openai",
)
EXPECTED_ALFWORLD_VERSION = "0.4.2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfworld-data", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_ALFWORLD_CONFIG)
    parser.add_argument(
        "--init-env",
        action="store_true",
        help="Initialize and reset one TextWorld environment (requires downloaded data).",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = configure_runtime(alfworld_data=args.alfworld_data)

    failures: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:  # diagnostic command: report every missing dependency
            failures.append(f"import {module_name}: {exc}")
            print(f"[FAIL] import {module_name}: {exc}")

    try:
        installed_version = version("alfworld")
        if installed_version != EXPECTED_ALFWORLD_VERSION:
            raise RuntimeError(
                f"expected {EXPECTED_ALFWORLD_VERSION}, found {installed_version}"
            )
        print(f"[OK] alfworld version {installed_version}")
    except (PackageNotFoundError, RuntimeError) as exc:
        failures.append(f"alfworld version: {exc}")
        print(f"[FAIL] alfworld version: {exc}")

    try:
        config = load_alfworld_config(args.config, validate_data=False)
        print(f"[OK] config {args.config.resolve()}")
    except Exception as exc:
        config = None
        failures.append(f"config: {exc}")
        print(f"[FAIL] config: {exc}")

    missing_data = validate_alfworld_data(settings.alfworld_data)
    if missing_data:
        failures.append("ALFWorld data: " + ", ".join(missing_data))
        print(f"[FAIL] ALFWorld data at {settings.alfworld_data}")
        for entry in missing_data:
            print(f"       missing {entry}")
    else:
        print(f"[OK] ALFWorld data {settings.alfworld_data}")

    if args.init_env and config is not None and not missing_data:
        try:
            from alfworld.agents.environment import get_environment

            split = "eval_in_distribution" if args.split == "dev" else "eval_out_of_distribution"
            env = get_environment(config["env"]["type"])(config, train_eval=split)
            env = env.init_env(batch_size=1)
            observations, info = env.reset()
            if not observations:
                raise RuntimeError("env.reset() returned no observations")
            print(f"[OK] initialized {split}: {len(env.gamefiles)} games")
            print(f"[OK] reset task {info['extra.gamefile'][0]}")
        except Exception as exc:
            failures.append(f"environment init: {exc}")
            print(f"[FAIL] environment init: {exc}")

    if failures:
        print("\nStage-1 setup check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nStage-1 setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
