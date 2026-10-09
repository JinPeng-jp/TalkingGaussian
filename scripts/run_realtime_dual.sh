#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Listen on the campus-network interface by default so other authenticated
# campus-network users can open the demo without an SSH tunnel.  Keeping the
# address configurable avoids exposing it on the server's other interfaces.
host="${TG_HOST:-192.168.167.5}"
port="${TG_PORT:-8080}"
https_port="${TG_HTTPS_PORT:-8443}"
cert_dir="${PROJECT_DIR}/realtime/certs"

exec python realtime_server.py \
  -s data/may \
  -m output/may \
  --macron_source data/macron \
  --macron_model output/macron \
  --host "${host}" \
  --port "${port}" \
  --https_port "${https_port}" \
  --ssl_cert "${cert_dir}/server.crt" \
  --ssl_key "${cert_dir}/server.key" \
  --ssl_ca "${cert_dir}/campus-ca.crt" \
  --render_gpu 0 \
  --asr_gpu 1 \
  "$@"
