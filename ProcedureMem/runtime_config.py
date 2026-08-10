"""Shared runtime configuration for the ALFWorld experiment entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
ALFWORLD_ASSETS_DIR = PACKAGE_DIR / "Alfworld"
DEFAULT_ALFWORLD_CONFIG = ALFWORLD_ASSETS_DIR / "base_config.yaml"
DEFAULT_MEMORY_CONFIG = PACKAGE_DIR / "config.yaml"
DEFAULT_EXAMPLES_PATH = ALFWORLD_ASSETS_DIR / "alfworld_examples.json"
DEFAULT_TRAJECTORY_PATH = ALFWORLD_ASSETS_DIR / "alfworld_format_traj.json"
DEFAULT_RESULTS_DIR = ALFWORLD_ASSETS_DIR / "results"
DEFAULT_MEMORY_DIR = PACKAGE_DIR / "memory" / "alfworld"


@dataclass(frozen=True)
class RuntimeSettings:
    model_name: str | None
    api_key: str | None
    api_base_url: str | None
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    alfworld_data: Path


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _set_if_value(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


def configure_runtime(
    *,
    model_name: str | None = None,
    alfworld_data: str | Path | None = None,
    require_llm: bool = False,
    require_embedding: bool = False,
) -> RuntimeSettings:
    """Resolve runtime values, publish compatible aliases, and validate secrets."""
    api_key = _first_env("OPENAI_API_KEY", "API_KEY")
    api_base_url = _first_env(
        "OPENAI_API_BASE", "OPENAI_BASE_URL", "API_BASE_URL"
    )
    resolved_model = model_name or _first_env("MODEL_NAME")

    embedding_api_key = _first_env("EMBEDDING_MODEL_KEY") or api_key
    embedding_base_url = _first_env("EMBEDDING_MODEL_BASE_URL") or api_base_url
    embedding_model = _first_env("EMBEDDING_MODEL_NAME") or "text-embedding-3-small"

    data_value = alfworld_data or _first_env("ALFWORLD_DATA")
    data_path = Path(data_value).expanduser() if data_value else Path.home() / ".cache" / "alfworld"
    data_path = data_path.resolve()

    _set_if_value("OPENAI_API_KEY", api_key)
    _set_if_value("API_KEY", api_key)
    _set_if_value("OPENAI_API_BASE", api_base_url)
    _set_if_value("OPENAI_BASE_URL", api_base_url)
    _set_if_value("API_BASE_URL", api_base_url)
    _set_if_value("MODEL_NAME", resolved_model)
    _set_if_value("EMBEDDING_MODEL_KEY", embedding_api_key)
    _set_if_value("EMBEDDING_MODEL_BASE_URL", embedding_base_url)
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
    os.environ["ALFWORLD_DATA"] = str(data_path)

    missing: list[str] = []
    if require_llm and not api_key:
        missing.append("OPENAI_API_KEY")
    if require_llm and not resolved_model:
        missing.append("MODEL_NAME or --model")
    if require_embedding and not embedding_api_key:
        missing.append("EMBEDDING_MODEL_KEY (or OPENAI_API_KEY)")
    if missing:
        raise RuntimeError("Missing runtime configuration: " + ", ".join(missing))

    return RuntimeSettings(
        model_name=resolved_model,
        api_key=api_key,
        api_base_url=api_base_url,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        alfworld_data=data_path,
    )


def validate_alfworld_data(data_path: Path) -> list[str]:
    """Return missing ALFWorld data entries required by the text environment."""
    required = (
        "json_2.1.1/train",
        "json_2.1.1/valid_seen",
        "json_2.1.1/valid_unseen",
        "logic/alfred.pddl",
        "logic/alfred.twl2",
    )
    return [entry for entry in required if not (data_path / entry).exists()]


def require_alfworld_data(data_path: Path) -> None:
    missing = validate_alfworld_data(data_path)
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"ALFWorld data is incomplete at {data_path}. Missing:\n  - {formatted}\n"
            "Run `alfworld-download` on the server, or pass `--alfworld-data`."
        )


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_alfworld_config(
    config_path: str | Path | None = None,
    *,
    validate_data: bool = True,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_ALFWORLD_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"ALFWorld config not found: {path}")

    data_path = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
    if validate_data:
        require_alfworld_data(data_path)

    with path.open("r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    if not isinstance(config, dict) or "env" not in config or "dataset" not in config:
        raise ValueError(f"Invalid ALFWorld config: {path}")
    return _expand_env(config)


def resolve_repo_path(value: str | Path, *, base: Path = PACKAGE_DIR) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_memory_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_MEMORY_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Memory config not found: {path}")
    with path.open("r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    if not isinstance(config, dict) or "policy" not in config:
        raise ValueError(f"Invalid memory config: {path}")
    for key in ("traj_file_path", "memory_dir"):
        if config.get(key):
            config[key] = str(resolve_repo_path(config[key]))
    return config
