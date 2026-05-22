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

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from sentinel.domain.errors import ModelNotTrainedError
from sentinel.domain.models import VersionMeta
from sentinel.domain.ports.model_store import IModelStore
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class DiskModelStore(IModelStore):
    """IModelStore implementation using joblib for persistence and a manifest.json.

    Directory layout:
      {base_path}/
        agents/
          {agent_name}/
            {detector_name}/          ← per-detector subdirectory
              manifest.json
              v20250101T120000.joblib
        cortex/
          {cortex_name}/
            manifest.json
            v20250101T120000.pt
            v20250101T120000.meta.json

    Legacy paths (agents/{agent_name}/*.joblib without detector subdirectory) are
    supported via the original save_version/load methods so existing installations
    continue to work during migration.
    """

    def __init__(self, base_path: str, max_versions: int = 1) -> None:
        self._base_path = Path(base_path)
        self._max_versions = max_versions

    # ── Legacy agent methods (single-model per agent, no detector subdir) ────

    def _model_dir(self, name: str) -> Path:
        return self._base_path / "agents" / name

    def _manifest_path(self, name: str) -> Path:
        return self._model_dir(name) / "manifest.json"

    def _read_manifest(self, name: str) -> dict:
        path = self._manifest_path(name)
        if not path.exists():
            return {"current": None, "versions": []}
        with open(path) as f:
            return json.load(f)

    def _write_manifest(self, name: str, manifest: dict) -> None:
        path = self._manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    async def save_version(self, name: str, model: Any) -> str:
        """Save a model to the legacy (no detector subdir) path."""
        model_dir = self._model_dir(name)
        model_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc)
        version_id = f"v{now.strftime('%Y%m%dT%H%M%S')}"
        filepath = model_dir / f"{version_id}.joblib"

        await asyncio.get_event_loop().run_in_executor(
            None, joblib.dump, model, str(filepath)
        )

        manifest = self._read_manifest(name)
        manifest["versions"].append(
            {"id": version_id, "created_at": now.isoformat(), "performance_score": None}
        )
        manifest["current"] = version_id

        while len(manifest["versions"]) > self._max_versions:
            oldest = manifest["versions"].pop(0)
            old_file = model_dir / f"{oldest['id']}.joblib"
            if old_file.exists():
                old_file.unlink()
                logger.info("pruned_model_version", agent=name, version=oldest["id"])

        self._write_manifest(name, manifest)
        logger.info("saved_model_version", agent=name, version=version_id)
        return version_id

    async def load(self, name: str, version: str | None = None) -> Any:
        manifest = self._read_manifest(name)
        if not manifest["versions"]:
            raise ModelNotTrainedError(f"No trained model found for agent: {name}")

        target_version = version or manifest["current"]
        if target_version is None:
            raise ModelNotTrainedError(f"No current model version for agent: {name}")

        filepath = self._model_dir(name) / f"{target_version}.joblib"
        if not filepath.exists():
            raise ModelNotTrainedError(f"Model file not found: {filepath}")

        return await asyncio.get_event_loop().run_in_executor(
            None, joblib.load, str(filepath)
        )

    async def list_versions(self, name: str) -> list[VersionMeta]:
        manifest = self._read_manifest(name)
        return [
            VersionMeta(
                version_id=v["id"],
                created_at=datetime.fromisoformat(v["created_at"]),
                performance_score=v.get("performance_score"),
            )
            for v in manifest["versions"]
        ]

    async def rollback(self, name: str) -> None:
        manifest = self._read_manifest(name)
        versions = manifest["versions"]
        if len(versions) < 2:
            logger.warning("rollback_no_previous", agent=name)
            return

        current = manifest["current"]
        ids = [v["id"] for v in versions]
        try:
            idx = ids.index(current)
        except ValueError:
            idx = len(versions) - 1

        if idx == 0:
            logger.warning("rollback_already_at_oldest", agent=name)
            return

        previous = versions[idx - 1]["id"]
        manifest["current"] = previous
        self._write_manifest(name, manifest)
        logger.info("rolled_back_model", agent=name, from_version=current, to_version=previous)

    # ── Per-detector methods (new, recommended path) ─────────────────────────

    def _detector_dir(self, agent_name: str, detector_name: str) -> Path:
        return self._base_path / "agents" / agent_name / detector_name

    def _detector_manifest_path(self, agent_name: str, detector_name: str) -> Path:
        return self._detector_dir(agent_name, detector_name) / "manifest.json"

    def _read_detector_manifest(self, agent_name: str, detector_name: str) -> dict:
        path = self._detector_manifest_path(agent_name, detector_name)
        if not path.exists():
            return {"current": None, "versions": []}
        with open(path) as f:
            return json.load(f)

    def _write_detector_manifest(
        self, agent_name: str, detector_name: str, manifest: dict
    ) -> None:
        path = self._detector_manifest_path(agent_name, detector_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    async def save_detector_version(
        self, agent_name: str, detector_name: str, model: Any
    ) -> str:
        """Persist a detector model under agents/{agent_name}/{detector_name}/."""
        det_dir = self._detector_dir(agent_name, detector_name)
        det_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc)
        version_id = f"v{now.strftime('%Y%m%dT%H%M%S')}"
        filepath = det_dir / f"{version_id}.joblib"

        await asyncio.get_event_loop().run_in_executor(
            None, joblib.dump, model, str(filepath)
        )

        manifest = self._read_detector_manifest(agent_name, detector_name)
        manifest["versions"].append(
            {"id": version_id, "created_at": now.isoformat(), "performance_score": None}
        )
        manifest["current"] = version_id

        while len(manifest["versions"]) > self._max_versions:
            oldest = manifest["versions"].pop(0)
            old_file = det_dir / f"{oldest['id']}.joblib"
            if old_file.exists():
                old_file.unlink()
                logger.info(
                    "pruned_detector_version",
                    agent=agent_name,
                    detector=detector_name,
                    version=oldest["id"],
                )

        self._write_detector_manifest(agent_name, detector_name, manifest)
        logger.info(
            "saved_detector_version",
            agent=agent_name,
            detector=detector_name,
            version=version_id,
        )
        return version_id

    async def load_detector(
        self, agent_name: str, detector_name: str, version: str | None = None
    ) -> Any:
        """Load a detector model from agents/{agent_name}/{detector_name}/."""
        manifest = self._read_detector_manifest(agent_name, detector_name)
        if not manifest["versions"]:
            raise ModelNotTrainedError(
                f"No trained model for agent '{agent_name}', detector '{detector_name}'"
            )

        target = version or manifest["current"]
        if target is None:
            raise ModelNotTrainedError(
                f"No current version for agent '{agent_name}', detector '{detector_name}'"
            )

        filepath = self._detector_dir(agent_name, detector_name) / f"{target}.joblib"
        if not filepath.exists():
            raise ModelNotTrainedError(f"Model file not found: {filepath}")

        return await asyncio.get_event_loop().run_in_executor(
            None, joblib.load, str(filepath)
        )

    async def list_detector_versions(
        self, agent_name: str, detector_name: str
    ) -> list[VersionMeta]:
        manifest = self._read_detector_manifest(agent_name, detector_name)
        return [
            VersionMeta(
                version_id=v["id"],
                created_at=datetime.fromisoformat(v["created_at"]),
                performance_score=v.get("performance_score"),
            )
            for v in manifest["versions"]
        ]

    # ── Cortex (PyTorch autoencoder) ─────────────────────────────────────────

    def _cortex_dir(self, name: str) -> Path:
        return self._base_path / "cortex" / name

    async def save_cortex_version(self, name: str, model: Any, metadata: dict) -> str:
        import torch

        cortex_dir = self._cortex_dir(name)
        cortex_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc)
        version_id = f"v{now.strftime('%Y%m%dT%H%M%S')}"
        model_path = cortex_dir / f"{version_id}.pt"
        meta_path = cortex_dir / f"{version_id}.meta.json"

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: torch.save(model.state_dict(), str(model_path))
        )

        full_meta = {
            "id": version_id,
            "created_at": now.isoformat(),
            "model_class": type(model).__name__,
            **metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(full_meta, f, indent=2, default=str)

        manifest_path = cortex_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"current": None, "versions": []}
        )
        manifest["versions"].append({"id": version_id, "created_at": now.isoformat()})
        manifest["current"] = version_id

        while len(manifest["versions"]) > self._max_versions:
            oldest = manifest["versions"].pop(0)
            for ext in (".pt", ".meta.json"):
                old_file = cortex_dir / f"{oldest['id']}{ext}"
                if old_file.exists():
                    old_file.unlink()

        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("saved_cortex_version", cortex=name, version=version_id)
        return version_id

    async def load_cortex_version(
        self, name: str, model_factory: Any, version: str | None = None
    ) -> tuple[Any, dict]:
        import torch

        cortex_dir = self._cortex_dir(name)
        manifest_path = cortex_dir / "manifest.json"
        if not manifest_path.exists():
            raise ModelNotTrainedError(f"No trained cortex model found for: {name}")

        manifest = json.loads(manifest_path.read_text())
        target = version or manifest.get("current")
        if not target:
            raise ModelNotTrainedError(f"No current cortex model for: {name}")

        model_path = cortex_dir / f"{target}.pt"
        meta_path = cortex_dir / f"{target}.meta.json"
        if not model_path.exists():
            raise ModelNotTrainedError(f"Cortex model file not found: {model_path}")

        state_dict = await asyncio.get_event_loop().run_in_executor(
            None, lambda: torch.load(str(model_path), map_location="cpu", weights_only=True)
        )
        model = model_factory()
        model.load_state_dict(state_dict)
        model.eval()

        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        logger.info("loaded_cortex_version", cortex=name, version=target)
        return model, metadata

    def cortex_manifest(self, name: str) -> dict:
        p = self._cortex_dir(name) / "manifest.json"
        if not p.exists():
            return {"current": None, "versions": []}
        return json.loads(p.read_text())
