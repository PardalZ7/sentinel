import asyncio

from aiohttp import web

from sentinel.config.schema import SentinelConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def create_control_plane_app(
    topology_manager,  # TopologyManager
    topology_store,    # TopologyStore
) -> web.Application:
    """Build the aiohttp application for the Sentinel Control Plane."""

    app = web.Application()

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

        # Accept either {"config": {...}} or the raw config dict directly
        config_data = body.get("config", body)

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

    app.router.add_get("/health", health)
    app.router.add_get("/topologies", list_topologies)
    app.router.add_get("/topologies/{id}", get_topology)
    app.router.add_put("/topologies/{id}", deploy_topology)
    app.router.add_delete("/topologies/{id}", delete_topology)

    return app


async def start_control_plane(app: web.Application, host: str, port: int) -> None:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("control_plane_started", host=host, port=port)
    while True:
        await asyncio.sleep(3600)
