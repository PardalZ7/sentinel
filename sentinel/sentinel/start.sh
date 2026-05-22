#!/usr/bin/env bash
set -e
python -m pip install -e . --quiet
exec python -m sentinel.runtime.cli start --config sentinel-cloud.json --mode agent
