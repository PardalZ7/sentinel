#!/usr/bin/env bash
set -e
pip install -e . --quiet
exec sentinel start --config sentinel-cloud.json --mode agent
