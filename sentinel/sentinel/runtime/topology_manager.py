import asyncio
from dataclasses import dataclass, field

from sentinel.config.schema import SentinelConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _Topology:
    config: SentinelConfig
    # Maps component name → running asyncio Task
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)


class TopologyManager:
    """Manages the lifecycle of dynamically deployed Agent and Cortex runners.

    Static components (from sentinel.json) are started directly by launcher.py.
    Dynamic topologies (from POST /topologies) are managed here.

    When a topology is applied, the manager diffs desired vs. running components
    and creates or cancels asyncio tasks accordingly.
    """

    def __init__(
        self,
        base_config: SentinelConfig,
        dashboard_state,  # DashboardState | None — avoid circular import
        engines_enabled: bool = True,
    ) -> None:
        self._base_config = base_config
        self._dashboard_state = dashboard_state
        self._engines_enabled = engines_enabled
        self._topologies: dict[str, _Topology] = {}

    async def apply(self, topology_id: str, topology_config: SentinelConfig) -> None:
        """Reconcile desired topology with currently running tasks.

        - Components in desired but not running → started.
        - Components running but removed from desired → cancelled.
        - Infrastructure (redis, aws, storage_path) is always taken from base_config.
        """
        from sentinel.runtime.launcher import build_agent, build_cortex, build_correlation_engine, build_temporal_layer

        merged = self._merge_infra(topology_config)
        desired_engines = {e.name: e for e in merged.correlation_engines}
        desired_agents = {a.name: a for a in merged.agents}
        desired_cortex = {c.name: c for c in merged.cortex}
        desired_temporal = {t.name: t for t in (merged.temporal_layers or [])}
        desired_all = {**desired_engines, **desired_agents, **desired_cortex, **desired_temporal}

        existing = self._topologies.get(topology_id)
        if existing:
            existing_agent_names = {a.name for a in existing.config.agents}
            existing_cortex_names = {c.name for c in existing.config.cortex}
            for name in list(existing.tasks.keys()):
                if name not in desired_all:
                    existing.tasks[name].cancel()
                    del existing.tasks[name]
                    if self._dashboard_state:
                        if name in existing_agent_names:
                            self._dashboard_state.unregister_agent(name)
                        elif name in existing_cortex_names:
                            self._dashboard_state.unregister_cortex(name)
                    logger.info("topology_component_removed", name=name, topology=topology_id)

        if topology_id not in self._topologies:
            self._topologies[topology_id] = _Topology(config=merged)
        else:
            self._topologies[topology_id].config = merged

        topo = self._topologies[topology_id]

        if self._engines_enabled:
            for name, engine_config in desired_engines.items():
                if name not in topo.tasks or topo.tasks[name].done():
                    runner = build_correlation_engine(engine_config, merged)
                    task = asyncio.create_task(runner.run(), name=f"dyn-engine-{name}")
                    topo.tasks[name] = task
                    logger.info("topology_engine_started", engine=name, topology=topology_id)
        elif desired_engines:
            logger.info("topology_engines_skipped", count=len(desired_engines), topology=topology_id, reason="agent-only mode")

        for name, agent_config in desired_agents.items():
            if name not in topo.tasks or topo.tasks[name].done():
                runner = build_agent(agent_config, merged, self._dashboard_state)
                task = asyncio.create_task(runner.run(), name=f"dyn-agent-{name}")
                topo.tasks[name] = task
                logger.info("topology_agent_started", agent=name, topology=topology_id)

        for name, cortex_config in desired_cortex.items():
            if name not in topo.tasks or topo.tasks[name].done():
                runner = build_cortex(cortex_config, merged, self._dashboard_state)
                task = asyncio.create_task(runner.run(), name=f"dyn-cortex-{name}")
                topo.tasks[name] = task
                logger.info("topology_cortex_started", cortex=name, topology=topology_id)

        for name, tl_config in desired_temporal.items():
            if name not in topo.tasks or topo.tasks[name].done():
                runner = build_temporal_layer(tl_config, merged, self._dashboard_state)
                task = asyncio.create_task(runner.run(), name=f"dyn-temporal-{name}")
                topo.tasks[name] = task
                logger.info("topology_temporal_started", temporal=name, topology=topology_id)

    async def remove(self, topology_id: str) -> None:
        topo = self._topologies.pop(topology_id, None)
        if topo is None:
            return
        agent_names = {a.name for a in topo.config.agents}
        cortex_names = {c.name for c in topo.config.cortex}
        for name, task in topo.tasks.items():
            task.cancel()
            if self._dashboard_state:
                if name in agent_names:
                    self._dashboard_state.unregister_agent(name)
                elif name in cortex_names:
                    self._dashboard_state.unregister_cortex(name)
        logger.info("topology_removed", topology=topology_id)

    def get_status(self) -> dict:
        result: dict = {}
        for tid, topo in self._topologies.items():
            result[tid] = {
                "engines": [e.name for e in topo.config.correlation_engines],
                "agents": [a.name for a in topo.config.agents],
                "cortex": [c.name for c in topo.config.cortex],
                "tasks": {
                    name: "running" if not task.done() else "stopped"
                    for name, task in topo.tasks.items()
                },
            }
        return result

    async def restore_from_store(self, store) -> None:
        """Re-apply all persisted topologies on process restart."""
        for topology_id, config in store.load_all().items():
            logger.info("topology_restoring", topology=topology_id)
            await self.apply(topology_id, config)

    def _merge_infra(self, topology_config: SentinelConfig) -> SentinelConfig:
        """Return topology config with infra settings overridden by base_config."""
        return topology_config.model_copy(update={
            "redis": self._base_config.redis,
            "aws": self._base_config.aws,
            "storage_path": self._base_config.storage_path,
        })
