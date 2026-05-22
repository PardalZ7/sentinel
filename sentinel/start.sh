#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
CONFIG="${1:-$SCRIPT_DIR/../usecase/sentinel.json}"

if [ ! -d "$VENV" ]; then
  echo "[sentinel] Criando venv..."
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

if ! pip show sentinel > /dev/null 2>&1; then
  echo "[sentinel] Instalando dependências..."
  pip install -e "$SCRIPT_DIR/.[dev]"
fi

echo "[sentinel] Iniciando com config: $CONFIG"
sentinel start --config "$CONFIG"
