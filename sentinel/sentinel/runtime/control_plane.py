from aiohttp import web

from sentinel.config.loader import _interpolate_env_vars
from sentinel.config.schema import SentinelConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def register_routes(
    app: web.Application,
    topology_manager,  # TopologyManager
    topology_store,    # TopologyStore
) -> None:
    """Mount Control Plane routes onto an existing aiohttp app under /cp."""

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def list_topologies(_request: web.Request) -> web.Response:
        return web.json_response(topology_manager.get_status())

    async def get_topology(request: web.Request) -> web.Response:
        tid = request.match_info["id"]
        status = topology_manager.get_status()
        if tid not in status:
            raise web.HTTPNotFound(reason=f"Topology '{tid}' not found")
        config = topology_store.load(tid)
        return web.json_response({
            "id": tid,
            "status": status[tid],
            "config": config.model_dump(mode="json") if config else None,
        })

    async def deploy_topology(request: web.Request) -> web.Response:
        tid = request.match_info["id"]
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(reason="Request body must be valid JSON")

        config_data = _interpolate_env_vars(body.get("config", body))
        try:
            config = SentinelConfig.model_validate(config_data)
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid config: {exc}")

        topology_store.save(tid, config)
        await topology_manager.apply(tid, config)

        logger.info(
            "topology_deployed",
            topology=tid,
            agents=[a.name for a in config.agents],
            cortex=[c.name for c in config.cortex],
        )
        return web.json_response(
            {
                "id": tid,
                "status": "applied",
                "agents": [a.name for a in config.agents],
                "cortex": [c.name for c in config.cortex],
            },
            status=201,
        )

    async def delete_topology(request: web.Request) -> web.Response:
        tid = request.match_info["id"]
        topology_store.delete(tid)
        await topology_manager.remove(tid)
        return web.json_response({"id": tid, "status": "removed"})

    app.router.add_get("/cp/health", health)
    app.router.add_get("/cp/topologies", list_topologies)
    app.router.add_get("/cp/topologies/{id}", get_topology)
    app.router.add_put("/cp/topologies/{id}", deploy_topology)
    app.router.add_delete("/cp/topologies/{id}", delete_topology)
