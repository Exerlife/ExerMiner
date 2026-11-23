#!/usr/bin/env bash
# HiveOS custom miner config hook. Supports AriaPool JSON-RPC and Vela Stratum pools.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/h-manifest.conf"

POOL_URL=${CUSTOM_URL:-vela.suprnova.cc:7601}
LOGIN=${CUSTOM_TEMPLATE:-${WALLET:-}}
PASS=${CUSTOM_PASS:-x}
EXTRA=${CUSTOM_USER_CONFIG:-}

if [[ -z "$LOGIN" ]]; then
  echo "vela-custom-hive: Wallet/worker is empty. Set Wallet template to %WAL%.%WORKER_NAME%." >&2
  exit 1
fi
if [[ "$LOGIN" != vela1* ]]; then
  echo "vela-custom-hive: Wallet/worker must start with a mainnet VELA address (vela1...)." >&2
  exit 1
fi
if [[ "$LOGIN" != *.* && -n "${WORKER_NAME:-}" ]]; then
  LOGIN="${LOGIN}.${WORKER_NAME}"
fi

mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
python3 - "$CUSTOM_CONFIG_FILENAME" "$POOL_URL" "$LOGIN" "$PASS" "$EXTRA" <<'PY'
import json, sys
path, pool, login, password, extra = sys.argv[1:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"pool_url": pool, "login": login, "password": password, "extra": extra}, f)
    f.write("\n")
PY
chmod 600 "$CUSTOM_CONFIG_FILENAME"
echo "vela-custom-hive: HiveOS pool configuration saved (credentials are not printed)."
