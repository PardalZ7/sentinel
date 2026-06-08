# Copyright 2026 Alexandre Cardoso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
from typing import Any

from sentinel.config.migrations.registry import apply_migrations
from sentinel.config.schema import SentinelConfig
from sentinel.domain.errors import ConfigValidationError


def _load_file(path: str) -> dict:
    """Load JSON or YAML config file based on extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigValidationError(f"Config file not found: {path}")

    suffix = file_path.suffix.lower()
    with open(file_path) as f:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml

                return yaml.safe_load(f)
            except ImportError as e:
                raise ConfigValidationError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                ) from e
        elif suffix == ".json":
            return json.load(f)
        else:
            raise ConfigValidationError(
                f"Unsupported config file format: {suffix}. Use .json or .yaml"
            )


def _interpolate_env_vars(obj: Any) -> Any:
    """Replace ${VAR} and ${VAR:-default} placeholders with environment variables."""
    if isinstance(obj, str):
        import re

        def _replace(m: re.Match) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var, default)
            return os.environ.get(expr, m.group(0))

        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env_vars(i) for i in obj]
    return obj


def _apply_env_overrides(raw: dict) -> dict:
    """Override config values from environment variables with prefix SENTINEL_."""
    sentinel_block = raw.get("sentinel", raw)

    prefix = "SENTINEL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        # Convert SENTINEL_REDIS_HOST -> redis.host
        parts = key[len(prefix):].lower().split("_")
        _set_nested(sentinel_block, parts, value)

    return raw


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a nested dict value using a list of keys."""
    # Try progressively longer key prefixes to find a matching nested key
    for i in range(len(keys) - 1, 0, -1):
        parent_key = "_".join(keys[:i])
        child_key = "_".join(keys[i:])
        if parent_key in d and isinstance(d[parent_key], dict):
            d[parent_key][child_key] = _coerce(value)
            return

    # Fallback: set top-level key as underscore-joined
    d["_".join(keys)] = _coerce(value)


def _coerce(value: str) -> Any:
    """Try to coerce string env var to appropriate type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str) -> SentinelConfig:
    """Load, migrate, apply env overrides, and validate config."""
    try:
        raw = _load_file(path)
        raw = apply_migrations(raw)
        raw = _interpolate_env_vars(raw)
        raw = _apply_env_overrides(raw)

        # Unwrap the sentinel block if present
        sentinel_data = raw.get("sentinel", raw)

        return SentinelConfig.model_validate(sentinel_data)
    except ConfigValidationError:
        raise
    except Exception as e:
        raise ConfigValidationError(f"Failed to load config from {path}: {e}") from e
