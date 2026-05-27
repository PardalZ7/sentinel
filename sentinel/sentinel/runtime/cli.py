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
import sys
from pathlib import Path

import click

from sentinel.logging.logger import configure_logging, get_logger

logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """Sentinel — anomaly detection for event-driven architectures."""
    configure_logging()


@cli.command()
@click.option(
    "--config",
    default="sentinel.json",
    show_default=True,
    help="Path to the Sentinel config file (JSON or YAML).",
)
@click.option(
    "--mode",
    default="all",
    show_default=True,
    type=click.Choice(["all", "engine", "agent"], case_sensitive=False),
    help=(
        "Which components to start. "
        "'engine' runs only CorrelationEngines (use-case side). "
        "'agent' runs only Agents and Cortex (Sentinel cloud side). "
        "'all' runs everything."
    ),
)
def start(config: str, mode: str) -> None:
    """Start Sentinel components defined in the config."""
    from sentinel.config.loader import load_config
    from sentinel.runtime.launcher import launch

    try:
        sentinel_config = load_config(config)
    except Exception as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        sys.exit(1)

    logger.info("sentinel_starting", config=config, mode=mode)
    asyncio.run(launch(sentinel_config, mode=mode))


@cli.command("set-phase")
@click.argument("agent_name")
@click.argument("phase")
def set_phase_cmd(agent_name: str, phase: str) -> None:
    """Transition an agent to TRAINING or INFERENCE phase.

    AGENT_NAME is the name of the agent as defined in the config.
    PHASE must be one of: TRAINING, INFERENCE, COLD.
    """
    from sentinel.domain.models import ModelPhase

    try:
        new_phase = ModelPhase(phase.upper())
    except ValueError:
        valid = [p.value for p in ModelPhase]
        click.echo(f"Invalid phase '{phase}'. Valid values: {valid}", err=True)
        sys.exit(1)

    # For a running Sentinel, you would send a signal or use an admin API.
    # This CLI stub logs the intent; full implementation requires a running process.
    click.echo(f"Setting agent '{agent_name}' to phase '{new_phase.value}'")
    logger.info("cli_set_phase", agent=agent_name, phase=new_phase.value)
    click.echo(
        "Note: To apply this change to a running Sentinel instance, "
        "ensure your deployment supports live phase transitions via the admin API."
    )


@cli.command()
@click.option(
    "--config",
    default="sentinel.json",
    show_default=True,
    help="Path to the sentinel.json to deploy.",
)
@click.option(
    "--sentinel-url",
    required=True,
    envvar="SENTINEL_URL",
    help="Base URL of the Sentinel Control Plane (e.g. https://sentinel.example.com).",
)
@click.option(
    "--id",
    "topology_id",
    default=None,
    help="Topology ID (default: stem of the config filename, e.g. 'sentinel' for sentinel.json).",
)
@click.option(
    "--no-engines",
    is_flag=True,
    default=False,
    help="Register the topology with Sentinel but do NOT start engines locally.",
)
def deploy(config: str, sentinel_url: str, topology_id: str | None, no_engines: bool) -> None:
    """Deploy a topology to the Sentinel Control Plane and start local engines.

    Reads the sentinel.json, POSTs the topology to the remote Sentinel instance,
    then launches CorrelationEngines locally (--mode engine) unless --no-engines is set.

    The SENTINEL_URL environment variable can be used instead of --sentinel-url.
    """
    import aiohttp

    from sentinel.config.loader import load_config
    from sentinel.runtime.launcher import launch

    try:
        sentinel_config = load_config(config)
    except Exception as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        sys.exit(1)

    tid = topology_id or Path(config).stem

    async def _run() -> None:
        url = f"{sentinel_url.rstrip('/')}/cp/topologies/{tid}"
        click.echo(f"Deploying topology '{tid}' to {url} ...")
        config_payload = sentinel_config.model_dump(mode="json")
        if no_engines:
            config_payload["correlation_engines"] = []
        async with aiohttp.ClientSession() as session:
            async with session.put(
                url,
                json={"config": config_payload},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    click.echo(f"Sentinel returned {resp.status}: {body}", err=True)
                    sys.exit(1)
                result = await resp.json()

        click.echo(
            f"Topology '{tid}' applied — "
            f"agents: {result.get('agents')}, cortex: {result.get('cortex')}"
        )
        logger.info("deploy_complete", topology=tid, **result)

        if not no_engines:
            click.echo("Starting correlation engines locally ...")
            await launch(sentinel_config, mode="engine")

    asyncio.run(_run())


@cli.command()
def status() -> None:
    """Display the current status of all configured agents and cortex nodes."""
    click.echo("Sentinel Status")
    click.echo("=" * 40)
    click.echo(
        "Note: Live status requires a running Sentinel instance with the admin API enabled.\n"
        "To check status, connect to the gRPC endpoint or inspect model manifests "
        "in the storage_path directory."
    )
