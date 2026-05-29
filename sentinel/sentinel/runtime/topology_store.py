import json
from pathlib import Path

from sentinel.config.loader import _interpolate_env_vars
from sentinel.config.schema import SentinelConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class TopologyStore:
    """Persists topology configs to disk so Sentinel can restore them after restart."""

    def __init__(self, base_path: str) -> None:
        self._dir = Path(base_path).expanduser() / "topologies"
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, topology_id: str, config: SentinelConfig) -> None:
        path = self._dir / f"{topology_id}.json"
        path.write_text(config.model_dump_json())
        logger.info("topology_persisted", topology=topology_id, path=str(path))

    def load(self, topology_id: str) -> SentinelConfig | None:
        path = self._dir / f"{topology_id}.json"
        if not path.exists():
            return None
        try:
            raw = _interpolate_env_vars(json.loads(path.read_text()))
            return SentinelConfig.model_validate(raw)
        except Exception as exc:
            logger.warning("topology_load_failed", topology=topology_id, error=str(exc))
            return None

    def load_all(self) -> dict[str, SentinelConfig]:
        result: dict[str, SentinelConfig] = {}
        for path in self._dir.glob("*.json"):
            try:
                raw = _interpolate_env_vars(json.loads(path.read_text()))
                result[path.stem] = SentinelConfig.model_validate(raw)
            except Exception as exc:
                logger.warning("topology_load_failed", path=str(path), error=str(exc))
        return result

    def delete(self, topology_id: str) -> None:
        path = self._dir / f"{topology_id}.json"
        path.unlink(missing_ok=True)
        logger.info("topology_deleted", topology=topology_id)

    def list_ids(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]
