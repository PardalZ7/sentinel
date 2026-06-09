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

"""Dashboard HTTP server using aiohttp.

Serves the static HTML dashboard and provides:
- GET  /                          → index.html
- GET  /api/status                → full state snapshot (JSON)
- GET  /api/events                → SSE stream of real-time events
- GET  /api/events/buffer         → last 100 buffered events (JSON, on-demand)
- GET  /api/config                → parsed sentinel.json
- POST /api/agents/{name}/phase   → change agent phase
- GET  /api/models/export         → download all trained models as .tar.gz
- POST /api/models/import         → restore models from a .tar.gz export
- POST /api/models/clear          → delete all model files from disk
- POST /api/redis/purge           → delete all Sentinel correlation keys from Redis
- POST /api/queues/purge          → purge all agent input/output SQS queues
"""

import asyncio
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sentinel.dashboard.state import DashboardState
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_PUBLIC_DIR = Path(__file__).parent / "public"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_storage_path(state: DashboardState) -> Path:
    cfg = state.config_data.get("sentinel", state.config_data)
    return Path(cfg.get("storage_path", "~/.sentinel")).expanduser()


def _config_fingerprint(config: dict) -> str:
    """SHA-256 of the fields that define what a model learned.

    Intentionally excludes infra details (redis, queue URLs, ports) since
    those don't affect model compatibility.
    """
    sentinel = config.get("sentinel", config)
    relevant = {
        "model": sentinel.get("model", {}),
        "agents": sorted(
            [
                {
                    "name": a["name"],
                }
                for a in sentinel.get("agents", [])
            ],
            key=lambda x: x["name"],
        ),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


# ── Route handlers ────────────────────────────────────────────────────────────

async def _handle_index(request):
    from aiohttp import web
    return web.FileResponse(_PUBLIC_DIR / "index.html")


async def _handle_status(request):
    from aiohttp import web
    state: DashboardState = request.app["sentinel_state"]
    return web.json_response(state.to_status_dict())


async def _handle_sse(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
    try:
        await response.prepare(request)

        snapshot = json.dumps({"type": "snapshot", "data": state.to_status_dict()})
        await response.write(f"data: {snapshot}\n\n".encode())

        async for chunk in state.sse_stream():
            await response.write(chunk.encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as exc:
        from aiohttp.client_exceptions import ClientConnectionResetError
        if isinstance(exc, ClientConnectionResetError):
            pass
        else:
            raise

    return response


async def _handle_config(request):
    from aiohttp import web
    state: DashboardState = request.app["sentinel_state"]
    return web.json_response(state.config_data)


async def _handle_phase_change(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    try:
        body = await request.json()
        new_phase = body.get("phase", "").upper()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if new_phase not in ("TRAINING", "INFERENCE"):
        return web.json_response({"error": "phase must be TRAINING or INFERENCE"}, status=400)

    ok = await state.change_agent_phase(agent_name, new_phase)
    if not ok:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("dashboard_phase_change", agent=agent_name, phase=new_phase)
    return web.json_response({"agent": agent_name, "phase": new_phase})


async def _handle_agent_test_rate(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    try:
        body = await request.json()
        rate = float(body.get("rate", 0)) / 100.0  # UI sends 0–100
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    ok = await state.set_agent_test_rate(agent_name, rate)
    if not ok:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("agent_test_rate_set", agent=agent_name, rate=rate)
    return web.json_response({"agent": agent_name, "rate": rate})


async def _handle_agent_run_test(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    result = await state.run_agent_test(agent_name)
    if result is None:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("agent_test_run", agent=agent_name, result=result)
    return web.json_response(result)


async def _handle_detector_phase_change(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    detector_name = request.match_info["detector_name"]
    try:
        body = await request.json()
        new_phase = body.get("phase", "").upper()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if new_phase not in ("TRAINING", "INFERENCE"):
        return web.json_response({"error": "phase must be TRAINING or INFERENCE"}, status=400)

    ok = await state.change_agent_phase(agent_name, new_phase, detector_name=detector_name)
    if not ok:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("dashboard_detector_phase_change", agent=agent_name, detector=detector_name, phase=new_phase)
    return web.json_response({"agent": agent_name, "detector": detector_name, "phase": new_phase})


async def _handle_detector_test_rate(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    detector_name = request.match_info["detector_name"]
    try:
        body = await request.json()
        rate = float(body.get("rate", 0)) / 100.0
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    ok = await state.set_agent_test_rate(agent_name, rate, detector_name=detector_name)
    if not ok:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("detector_test_rate_set", agent=agent_name, detector=detector_name, rate=rate)
    return web.json_response({"agent": agent_name, "detector": detector_name, "rate": rate})


async def _handle_detector_run_test(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    detector_name = request.match_info["detector_name"]
    result = await state.run_agent_test(agent_name, detector_name=detector_name)
    if result is None:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    # run_agent_test returns {detector_name: result_dict} — extract the single detector's result
    detector_result = result.get(detector_name, result)
    logger.info("detector_test_run", agent=agent_name, detector=detector_name, result=detector_result)
    return web.json_response(detector_result)


async def _handle_detector_auto_infer(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agent_name = request.match_info["name"]
    detector_name = request.match_info["detector_name"]
    try:
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        threshold = float(body.get("threshold", 0.05)) if enabled else None
        if threshold is not None:
            threshold = max(0.0, min(1.0, threshold))
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    ok = await state.set_detector_auto_infer(agent_name, detector_name, threshold)
    if not ok:
        return web.json_response({"error": f"agent '{agent_name}' not found"}, status=404)

    logger.info("detector_auto_infer_set", agent=agent_name, detector=detector_name, threshold=threshold)
    return web.json_response({"agent": agent_name, "detector": detector_name, "auto_infer_fp_threshold": threshold})


async def _handle_cortex_phase_change(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    cortex_name = request.match_info["name"]
    try:
        body = await request.json()
        new_phase = body.get("phase", "").upper()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    if new_phase not in ("TRAINING", "INFERENCE"):
        return web.json_response({"error": "phase must be TRAINING or INFERENCE"}, status=400)

    ok = await state.change_cortex_phase(cortex_name, new_phase)
    if not ok:
        return web.json_response({"error": f"cortex '{cortex_name}' not found"}, status=404)

    logger.info("dashboard_cortex_phase_change", cortex=cortex_name, phase=new_phase)
    return web.json_response({"cortex": cortex_name, "phase": new_phase})


async def _handle_cortex_test_rate(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    cortex_name = request.match_info["name"]
    try:
        body = await request.json()
        rate = float(body.get("rate", 0)) / 100.0
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    ok = await state.set_cortex_test_rate(cortex_name, rate)
    if not ok:
        return web.json_response({"error": f"cortex '{cortex_name}' not found"}, status=404)

    logger.info("cortex_test_rate_set", cortex=cortex_name, rate=rate)
    return web.json_response({"cortex": cortex_name, "rate": rate})


async def _handle_cortex_run_test(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    cortex_name = request.match_info["name"]
    result = state.run_cortex_test(cortex_name)
    if result is None:
        return web.json_response({"error": f"cortex '{cortex_name}' not found"}, status=404)

    logger.info("cortex_test_run", cortex=cortex_name, result=result)
    return web.json_response(result)


def _build_state_snapshot(state, storage_path: Path, fingerprint: str, exported_at: str) -> dict:
    """Build a full state snapshot describing every agent and cortex."""
    sentinel_cfg = state.config_data.get("sentinel", state.config_data)
    snapshot: dict = {
        "exported_at": exported_at,
        "config_fingerprint": fingerprint,
        "agents": {},
        "cortex": {},
    }

    for agent in sentinel_cfg.get("agents", []):
        name = agent["name"]
        agent_dir = storage_path / "agents" / name
        dash = state.to_status_dict().get("agents", {}).get(name, {})

        # Collect per-detector manifests (new layout: agents/{agent}/{detector}/manifest.json)
        detector_versions: dict = {}
        if agent_dir.exists():
            for item in sorted(agent_dir.iterdir()):
                if item.is_dir():
                    det_manifest_path = item / "manifest.json"
                    if det_manifest_path.exists():
                        det_manifest = json.loads(det_manifest_path.read_text())
                        detector_versions[item.name] = {
                            "current_model_version": det_manifest.get("current"),
                            "model_versions": [v["id"] for v in det_manifest.get("versions", [])],
                        }

        # Fall back to legacy manifest (agents/{agent}/manifest.json) when no detector subdirs exist
        if detector_versions:
            current_model_version = next(
                (d["current_model_version"] for d in detector_versions.values() if d["current_model_version"]),
                None,
            )
            model_versions = sorted({
                vid
                for d in detector_versions.values()
                for vid in d["model_versions"]
            })
        else:
            legacy_manifest_path = agent_dir / "manifest.json"
            legacy_manifest = json.loads(legacy_manifest_path.read_text()) if legacy_manifest_path.exists() else {"current": None, "versions": []}
            current_model_version = legacy_manifest.get("current")
            model_versions = [v["id"] for v in legacy_manifest.get("versions", [])]

        snapshot["agents"][name] = {
            "phase": dash.get("phase", "COLD"),
            "total_messages": dash.get("total_messages", 0),
            "total_anomalies": dash.get("total_anomalies", 0),
            "current_model_version": current_model_version,
            "model_versions": model_versions,
            "detectors": detector_versions,
        }

    for cx in sentinel_cfg.get("cortex", []):
        name = cx["name"]
        manifest_path = storage_path / "cortex" / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"current": None, "versions": []}
        # Read baseline_error from latest meta file if available
        baseline_error = None
        current = manifest.get("current")
        if current:
            meta_path = storage_path / "cortex" / name / f"{current}.meta.json"
            if meta_path.exists():
                baseline_error = json.loads(meta_path.read_text()).get("baseline_error")
        dash_cx = state.to_status_dict().get("cortex", {}).get(name, {})
        snapshot["cortex"][name] = {
            "phase": "INFERENCE" if manifest.get("current") else "TRAINING",
            "total_alerts": dash_cx.get("total_alerts", 0),
            "current_model_version": current,
            "model_versions": [v["id"] for v in manifest.get("versions", [])],
            "baseline_error": baseline_error,
        }

    return snapshot


async def _handle_models_export(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    storage_path = _get_storage_path(state)

    fingerprint = _config_fingerprint(state.config_data)
    exported_at = datetime.now(timezone.utc).isoformat()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def _add_bytes(arcname: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        # meta.json — fingerprint for import validation
        _add_bytes("meta.json", json.dumps(
            {"exported_at": exported_at, "config_fingerprint": fingerprint}, indent=2
        ).encode())

        # sentinel.json — full config snapshot
        _add_bytes("sentinel.json", json.dumps(state.config_data, indent=2).encode())

        # state.json — training state of all agents and cortex
        snapshot = _build_state_snapshot(state, storage_path, fingerprint, exported_at)
        _add_bytes("state.json", json.dumps(snapshot, indent=2).encode())

        # agents/ — joblib files + manifests, supports both legacy (2-level) and
        # new per-detector (3-level) layouts
        agents_dir = storage_path / "agents"
        if agents_dir.exists():
            for agent_dir in sorted(agents_dir.iterdir()):
                if not agent_dir.is_dir():
                    continue
                for item in sorted(agent_dir.iterdir()):
                    if item.is_file():
                        # Legacy: agents/{agent}/{file}
                        tar.add(str(item), arcname=f"agents/{agent_dir.name}/{item.name}")
                    elif item.is_dir():
                        # New: agents/{agent}/{detector}/{file}
                        for f in sorted(item.iterdir()):
                            if f.is_file():
                                tar.add(str(f), arcname=f"agents/{agent_dir.name}/{item.name}/{f.name}")

        # cortex/ — PyTorch autoencoder .pt files + .meta.json + manifests
        cortex_dir = storage_path / "cortex"
        if cortex_dir.exists():
            for cx_dir in sorted(cortex_dir.iterdir()):
                if not cx_dir.is_dir():
                    continue
                for f in sorted(cx_dir.iterdir()):
                    if f.is_file():
                        tar.add(str(f), arcname=f"cortex/{cx_dir.name}/{f.name}")

    buf.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"sentinel-export-{ts}.tar.gz"

    logger.info("models_exported", storage_path=str(storage_path))
    return web.Response(
        body=buf.read(),
        content_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _handle_models_import(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    storage_path = _get_storage_path(state)

    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "archive":
            return web.json_response(
                {"error": "Expected multipart field named 'archive'"}, status=400
            )
        file_bytes = await field.read()
    except Exception as exc:
        return web.json_response({"error": f"Failed to read upload: {exc}"}, status=400)

    try:
        buf = io.BytesIO(file_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # Validate meta.json
            try:
                meta = json.loads(tar.extractfile("meta.json").read())
            except KeyError:
                return web.json_response(
                    {"error": "Archive is missing meta.json — not a valid Sentinel export."},
                    status=422,
                )

            archive_fp = meta.get("config_fingerprint", "")
            current_fp = _config_fingerprint(state.config_data)

            if archive_fp != current_fp:
                try:
                    archived_cfg = json.loads(tar.extractfile("sentinel.json").read())
                    arch_sentinel = archived_cfg.get("sentinel", archived_cfg)
                    arch_agents = [a["name"] for a in arch_sentinel.get("agents", [])]
                    cur_sentinel = state.config_data.get("sentinel", state.config_data)
                    cur_agents = [a["name"] for a in cur_sentinel.get("agents", [])]
                except Exception:
                    arch_agents, cur_agents = [], []

                return web.json_response(
                    {
                        "error": (
                            "Incompatible configuration: this archive was exported from a different "
                            "Sentinel setup. Restoring it would corrupt your current training data."
                        ),
                        "detail": {
                            "archive_agents": arch_agents,
                            "current_agents": cur_agents,
                            "archive_fingerprint": archive_fp,
                            "current_fingerprint": current_fp,
                        },
                    },
                    status=409,
                )

            # Extract agents/ and cortex/ atomically via temp dir
            restore_members = [
                m for m in tar.getmembers()
                if m.name.startswith("agents/") or m.name.startswith("cortex/")
            ]
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                tar.extractall(path=tmp_path, members=restore_members)
                for subdir in ("agents", "cortex"):
                    tmp_sub = tmp_path / subdir
                    target_sub = storage_path / subdir
                    if tmp_sub.exists():
                        if target_sub.exists():
                            shutil.rmtree(target_sub)
                        shutil.copytree(tmp_sub, target_sub)

    except tarfile.TarError as exc:
        return web.json_response({"error": f"Invalid archive: {exc}"}, status=422)

    logger.info("models_imported", storage_path=str(storage_path))
    return web.json_response({
        "ok": True,
        "message": "Models restored successfully. Restart Sentinel to load the imported models.",
    })


async def _handle_models_clear(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    agents_dir = _get_storage_path(state) / "agents"

    cleared = []
    if agents_dir.exists():
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir():
                shutil.rmtree(agent_dir)
                cleared.append(agent_dir.name)

    for agent_name in cleared:
        snap = state.agents.get(agent_name)
        if snap:
            snap.total_messages = 0
            snap.total_anomalies = 0

    logger.info("models_cleared", agents=cleared)
    return web.json_response({"ok": True, "cleared": cleared})


async def _handle_redis_purge(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    sentinel_cfg = state.config_data.get("sentinel", state.config_data)
    redis_cfg = sentinel_cfg.get("redis", {})

    try:
        import redis.asyncio as aioredis
        client = aioredis.Redis(
            host=redis_cfg.get("host", "localhost"),
            port=redis_cfg.get("port", 6379),
            db=redis_cfg.get("db", 0),
            password=redis_cfg.get("password") or None,
        )
        agent_names = [a["name"] for a in sentinel_cfg.get("agents", [])]
        deleted = 0
        for name in agent_names:
            pattern = f"{name}:*"
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=500)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break
        await client.aclose()
    except Exception as exc:
        logger.error("redis_purge_failed", error=str(exc))
        return web.json_response({"error": f"Redis purge failed: {exc}"}, status=500)

    logger.info("redis_purged", deleted=deleted)
    return web.json_response({"ok": True, "deleted_keys": deleted})


async def _handle_queues_purge(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    sentinel_cfg = state.config_data.get("sentinel", state.config_data)
    aws_cfg = sentinel_cfg.get("aws", {})

    queue_urls: list[str] = []
    seen: set[str] = set()
    for agent in sentinel_cfg.get("agents", []):
        for side in ("input", "output"):
            url = agent.get(side, {}).get("resource", "")
            if url and url not in seen:
                seen.add(url)
                queue_urls.append(url)

    if not queue_urls:
        return web.json_response({"ok": True, "purged": [], "skipped": [], "errors": []})

    purged, skipped, errors = [], [], []
    try:
        import aiobotocore.session as abcs
        session = abcs.get_session()
        async with session.create_client(
            "sqs",
            region_name=aws_cfg.get("region", "us-east-1"),
            endpoint_url=aws_cfg.get("endpoint_url") or None,
        ) as sqs:
            for url in queue_urls:
                try:
                    await sqs.purge_queue(QueueUrl=url)
                    purged.append(url.split("/")[-1])
                    logger.info("queue_purged", queue=url)
                except Exception as exc:
                    err_str = str(exc)
                    if "PurgeQueueInProgress" in err_str:
                        skipped.append(url.split("/")[-1])
                    else:
                        errors.append({"queue": url.split("/")[-1], "error": err_str})
                        logger.warning("queue_purge_failed", queue=url, error=err_str)
    except Exception as exc:
        logger.error("queues_purge_failed", error=str(exc))
        return web.json_response({"error": f"SQS purge failed: {exc}"}, status=500)

    return web.json_response({"ok": True, "purged": purged, "skipped": skipped, "errors": errors})


async def _handle_topology_reset(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    sentinel_cfg = state.config_data.get("sentinel", state.config_data)
    redis_cfg = sentinel_cfg.get("redis", {})
    storage_path = _get_storage_path(state)

    # 1. Reset all agents and cortex in-memory (models, buffers, counters)
    await state.reset_topology()

    # 2. Clear model files from disk (agents + cortex)
    cleared_agents: list[str] = []
    agents_dir = storage_path / "agents"
    if agents_dir.exists():
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir():
                shutil.rmtree(agent_dir)
                cleared_agents.append(agent_dir.name)

    cleared_cortex: list[str] = []
    cortex_dir = storage_path / "cortex"
    if cortex_dir.exists():
        for cx_dir in cortex_dir.iterdir():
            if cx_dir.is_dir():
                shutil.rmtree(cx_dir)
                cleared_cortex.append(cx_dir.name)

    # 3. Purge all Sentinel keys from Redis
    redis_deleted = 0
    redis_error: str | None = None
    try:
        import redis.asyncio as aioredis
        client = aioredis.Redis(
            host=redis_cfg.get("host", "localhost"),
            port=redis_cfg.get("port", 6379),
            db=redis_cfg.get("db", 0),
            password=redis_cfg.get("password") or None,
        )
        agent_names = [a["name"] for a in sentinel_cfg.get("agents", [])]
        cortex_names = [c["name"] for c in sentinel_cfg.get("cortex", [])]
        for name in agent_names + cortex_names:
            pattern = f"{name}:*"
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=500)
                if keys:
                    redis_deleted += await client.delete(*keys)
                if cursor == 0:
                    break
        # Also purge cortex grouping keys
        for name in cortex_names:
            pattern = f"cortex:{name}:*"
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=500)
                if keys:
                    redis_deleted += await client.delete(*keys)
                if cursor == 0:
                    break
        await client.aclose()
    except Exception as exc:
        redis_error = str(exc)
        logger.warning("topology_reset_redis_failed", error=redis_error)

    logger.info(
        "topology_reset",
        agents=cleared_agents,
        cortex=cleared_cortex,
        redis_deleted=redis_deleted,
    )
    return web.json_response({
        "ok": True,
        "cleared_agents": cleared_agents,
        "cleared_cortex": cleared_cortex,
        "redis_deleted_keys": redis_deleted,
        "redis_error": redis_error,
    })


async def _handle_events_buffer(request):
    from aiohttp import web
    from dataclasses import asdict

    state: DashboardState = request.app["sentinel_state"]
    events = [asdict(e) for e in list(state.recent_events)]
    return web.json_response({"events": events})


async def _handle_errors_buffer(request):
    from aiohttp import web
    from dataclasses import asdict

    state: DashboardState = request.app["sentinel_state"]
    errors = [asdict(e) for e in list(state.recent_errors)]
    return web.json_response({"events": errors})


async def _handle_temporal_anomalies_buffer(request):
    from aiohttp import web
    from dataclasses import asdict

    state: DashboardState = request.app["sentinel_state"]
    records = [asdict(r) for r in list(state.recent_temporal_anomalies)]
    return web.json_response({"temporal_anomalies": records})


async def _handle_event_payload(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    correlation_id = request.match_info["correlation_id"]
    payload = state.get_payload(correlation_id)
    if payload is None:
        return web.json_response({"error": "Payload not found"}, status=404)
    return web.json_response(payload)


async def _handle_alert_payload(request):
    from aiohttp import web

    state: DashboardState = request.app["sentinel_state"]
    alert_id = request.match_info["alert_id"]
    payload = state.get_payload(alert_id)
    if payload is None:
        return web.json_response({"error": "Payload not found"}, status=404)
    return web.json_response(payload)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(state: DashboardState, topology_manager=None, topology_store=None):
    from aiohttp import web

    app = web.Application(client_max_size=256 * 1024 * 1024)  # 256 MB upload limit
    app["sentinel_state"] = state

    if topology_manager is not None and topology_store is not None:
        from sentinel.runtime.control_plane import register_routes
        register_routes(app, topology_manager, topology_store)

    app.router.add_get("/", _handle_index)
    app.router.add_get("/api/config", _handle_config)
    app.router.add_get("/api/status", _handle_status)
    app.router.add_get("/api/events", _handle_sse)
    app.router.add_get("/api/events/buffer", _handle_events_buffer)
    app.router.add_get("/api/errors/buffer", _handle_errors_buffer)
    app.router.add_get("/api/temporal-anomalies/buffer", _handle_temporal_anomalies_buffer)
    app.router.add_get("/api/events/{correlation_id}/payload", _handle_event_payload)
    app.router.add_get("/api/alerts/{alert_id}/payload", _handle_alert_payload)
    app.router.add_post("/api/agents/{name}/phase", _handle_phase_change)
    app.router.add_post("/api/agents/{name}/test-rate", _handle_agent_test_rate)
    app.router.add_post("/api/agents/{name}/run-test", _handle_agent_run_test)
    app.router.add_post("/api/agents/{name}/detectors/{detector_name}/phase", _handle_detector_phase_change)
    app.router.add_post("/api/agents/{name}/detectors/{detector_name}/test-rate", _handle_detector_test_rate)
    app.router.add_post("/api/agents/{name}/detectors/{detector_name}/run-test", _handle_detector_run_test)
    app.router.add_post("/api/agents/{name}/detectors/{detector_name}/auto-infer", _handle_detector_auto_infer)
    app.router.add_post("/api/cortex/{name}/phase", _handle_cortex_phase_change)
    app.router.add_post("/api/cortex/{name}/test-rate", _handle_cortex_test_rate)
    app.router.add_post("/api/cortex/{name}/run-test", _handle_cortex_run_test)
    app.router.add_get("/api/models/export", _handle_models_export)
    app.router.add_post("/api/models/import", _handle_models_import)
    app.router.add_post("/api/models/clear", _handle_models_clear)
    app.router.add_post("/api/redis/purge", _handle_redis_purge)
    app.router.add_post("/api/queues/purge", _handle_queues_purge)
    app.router.add_post("/api/topology/reset", _handle_topology_reset)

    return app


async def start_dashboard(
    state: DashboardState,
    host: str = "0.0.0.0",
    port: int = 8888,
    topology_manager=None,
    topology_store=None,
) -> None:
    """Start the aiohttp dashboard server as an asyncio task."""
    try:
        from aiohttp import web
    except ImportError:
        logger.warning("aiohttp_not_installed", msg="Dashboard disabled. Install aiohttp to enable.")
        return

    app = create_app(state, topology_manager=topology_manager, topology_store=topology_store)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("dashboard_started", url=f"http://{host}:{port}")
    await asyncio.Event().wait()
