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

from typing import Callable

from sentinel.config.migrations.v1_to_v2 import migrate as migrate_v1_to_v2

# List of (from_version, to_version, migrate_fn) tuples — ordered
MIGRATIONS: list[tuple[int, int, Callable[[dict], dict]]] = [
    (1, 2, migrate_v1_to_v2),
]


def apply_migrations(raw: dict) -> dict:
    """Apply all needed migrations in order to bring config up to latest version."""
    sentinel_block = raw.get("sentinel", raw)
    current_version = sentinel_block.get("schema_version", 1)

    for from_ver, to_ver, migrate_fn in MIGRATIONS:
        if current_version == from_ver:
            raw = migrate_fn(raw)
            current_version = to_ver

    return raw
