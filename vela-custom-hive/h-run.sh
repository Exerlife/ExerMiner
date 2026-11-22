#!/usr/bin/env bash
# ExerMiner — HiveOS launcher (AriaPool JSON-RPC or Vela named-params Stratum).
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
. ./h-manifest.conf

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
exec > >(tee -i --append "${CUSTOM_LOG_BASENAME}.log") 2>&1

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
fail() { log "ERROR: $*"; exit 1; }
log "ExerMiner HiveOS package $CUSTOM_VERSION"

[[ -f "$CUSTOM_CONFIG_FILENAME" ]] || fail "No flight-sheet configuration at $CUSTOM_CONFIG_FILENAME; save the flight sheet again."

# Parse the saved Flight Sheet. Plain host:port endpoints are treated as Stratum;
# http(s) endpoints select the AriaPool JSON-RPC path. Secrets never enter logs.
eval "$(python3 - "$CUSTOM_CONFIG_FILENAME" <<'PY'
import json, shlex, sys
from urllib.parse import urlsplit
with open(sys.argv[1], encoding='utf-8') as f:
    c=json.load(f)
pool=str(c.get('pool_url','')).strip()
login=str(c.get('login','')).strip()
password=str(c.get('password','x') or 'x')
extra=str(c.get('extra','') or '')
if not login.startswith('vela1'):
    raise SystemExit('invalid VELA wallet/worker; expected vela1...')
if not pool:
    raise SystemExit('pool URL is empty')
if '://' not in pool:
    pool='stratum://' + pool
p=urlsplit(pool)
values={'LOGIN':login,'WALLET_ADDRESS':login.split('.',1)[0],
        'POOL_PASSWORD':password,'EXTRA_ARGS':extra}
if p.scheme in ('http','https'):
    if not p.hostname or not p.port or p.username or p.password:
        raise SystemExit('AriaPool URL must be http(s)://host:port without credentials')
    from urllib.parse import quote, urlunsplit
    netloc=f"{quote(login,safe='')}:{quote(password,safe='')}@{p.hostname}:{p.port}"
    if ':' in p.hostname and not p.hostname.startswith('['):
        netloc=f"{quote(login,safe='')}:{quote(password,safe='')}@[{p.hostname}]:{p.port}"
    rpc=urlunsplit((p.scheme,netloc,p.path or '/',p.query,''))
    values.update({'POOL_MODE':'rpc','POOL_RPC_URL':rpc,'POOL_ENDPOINT':pool})
elif p.scheme in ('stratum','stratum+tcp','stratum+ssl','stratum+tls'):
    if not p.hostname or not p.port or p.username or p.password or p.path not in ('','/') or p.query or p.fragment:
        raise SystemExit('Stratum URL must be host:port; put credentials in Wallet/Pass fields')
    values.update({'POOL_MODE':'stratum','POOL_ENDPOINT':f"{p.scheme}://{p.netloc}"})
else:
    raise SystemExit('unsupported pool URL scheme; use http(s):// for AriaPool or host:port for Stratum')
for key,value in values.items():
    print(f'export {key}={shlex.quote(value)}')
PY
)" || fail "Could not parse Flight Sheet configuration."

# HiveOS mounts /hive-config on a tiny configuration partition (often only 40 MB).
# Keep only Flight Sheet JSON there. Python, PyTorch wheels, caches and logs belong
# on the root filesystem, which is normally the large /dev/sda4 partition.
STATE_DIR=/var/lib/vela-custom-hive
LEGACY_STATE_DIR=/hive-config/vela-custom-hive
MODEL_DIR="$STATE_DIR/models"
VENV="$STATE_DIR/venv"
MODEL="$MODEL_DIR/sd-turbo-int.poisd"
MODEL_SHA=ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36
MODEL_URLS=(
  "https://vela.suprnova.cc/dl/sd-turbo-int.poisd"
  "https://huggingface.co/valereseach/vela-sd-turbo-int-v2/resolve/main/sd-turbo-int.poisd"
  "https://dl.ariabrain.com/vela-dl/sd-turbo-int.poisd"
)
mkdir -p "$MODEL_DIR" "$STATE_DIR/logs" "$STATE_DIR/tmp" "$STATE_DIR/pip-cache" "$STATE_DIR/cache"

# Keep pip's temporary files and wheel cache on the large root filesystem too.
export TMPDIR="$STATE_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="$STATE_DIR/pip-cache"
export XDG_CACHE_HOME="$STATE_DIR/cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Free only known incomplete artefacts left by v0.2.1 in the 40 MB config
# partition. Do not remove the old log or any user-supplied model/configuration.
if [[ -d "$LEGACY_STATE_DIR/venv" ]]; then
  log "Removing legacy ExerMiner Python environment from the small /hive-config partition."
  rm -rf -- "$LEGACY_STATE_DIR/venv"
fi
rm -f -- "$LEGACY_STATE_DIR/get-pip.py.part"
if [[ -s "$LEGACY_STATE_DIR/logs/vela-miner.log" && ! -e "$STATE_DIR/logs/vela-miner.log" ]]; then
  cp -p -- "$LEGACY_STATE_DIR/logs/vela-miner.log" "$STATE_DIR/logs/vela-miner.log" || true
fi

# Refuse early if the root filesystem cannot hold CUDA PyTorch, the model and
# temporary wheel extraction. This avoids half-filled filesystems on first run.
AVAILABLE_KB=$(df -Pk "$STATE_DIR" | awk 'END {print $4}')
MINIMUM_KB=$((8 * 1024 * 1024))
if [[ ! "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || (( AVAILABLE_KB < MINIMUM_KB )); then
  fail "Only $(( ${AVAILABLE_KB:-0} / 1024 )) MiB free under $STATE_DIR; at least 8192 MiB is required for the first CUDA runtime install."
fi
log "Persistent runtime storage: $STATE_DIR ($(awk -v k="$AVAILABLE_KB" 'BEGIN {printf "%.1f GiB free", k/1048576}'))"

# Reuse an already-installed model before downloading. Verify bytes, not filename.
model_ok=0
for candidate in \
  "$MODEL" \
  /hive/miners/custom/vela-model/sd-turbo-int.poisd \
  /hive-config/models/sd-turbo-int.poisd \
  /hive/models/sd-turbo-int.poisd; do
  [[ -f "$candidate" ]] || continue
  got=$(sha256sum "$candidate" | awk '{print $1}')
  if [[ "$got" == "$MODEL_SHA" ]]; then
    MODEL="$candidate"
    model_ok=1
    log "Reusing existing verified PoI model at $MODEL"
    break
  fi
  log "Ignoring model with wrong SHA-256 at $candidate"
done
if [[ "$model_ok" != 1 ]]; then
  rm -f "$MODEL.part"
  for url in "${MODEL_URLS[@]}"; do
    log "Downloading the official Vela PoI v2 model from $url"
    part="$MODEL.part"
    if curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 -C - -o "$part" "$url"; then
      got=$(sha256sum "$part" | awk '{print $1}')
      if [[ "$got" == "$MODEL_SHA" ]]; then
        mv -f "$part" "$MODEL"
        model_ok=1
        break
      fi
      log "Model SHA-256 mismatch from this source; trying the next mirror. actual=$got"
      rm -f "$part"
    else
      log "Download failed from this source; trying the next mirror."
      rm -f "$part"
    fi
  done
fi
[[ "$model_ok" == 1 ]] || fail "Could not download/verify sd-turbo-int.poisd. Expected SHA-256 $MODEL_SHA."
printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | sha256sum -c - || fail "Model integrity check failed."

# One persistent venv; bootstrap pip ourselves because some HiveOS images ship
# Python's venv module but omit ensurepip. A failed `python -m venv` can leave a
# usable interpreter without pip, so validate both the venv and pip separately.
PYTHON_BIN=$(command -v python3 || true)
[[ -n "$PYTHON_BIN" ]] || fail "python3 is not installed on this HiveOS image."
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c 'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)' >/dev/null 2>&1; then
  log "Creating persistent Python environment at $VENV (without ensurepip)"
  rm -rf -- "$VENV"
  "$PYTHON_BIN" -m venv --without-pip "$VENV" || fail "Python venv module is unavailable. Install the matching HiveOS python3-venv package."
fi
VPY="$VENV/bin/python"

# Install pip into the venv if the base HiveOS Python has no ensurepip/pip.
# The old config-partition venv is no longer used; all runtime state is stored
# under /var/lib/vela-custom-hive on the larger root filesystem.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  GET_PIP="$STATE_DIR/get-pip.py"
  if [[ ! -s "$GET_PIP" ]]; then
    log "Bootstrapping pip (HiveOS Python does not include ensurepip)."
    GET_PIP_TMP="$GET_PIP.part"
    rm -f -- "$GET_PIP_TMP"
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 \
      -o "$GET_PIP_TMP" "https://bootstrap.pypa.io/get-pip.py" || fail "Could not download the official pip bootstrap script. Check HiveOS internet access."
    [[ -s "$GET_PIP_TMP" ]] || fail "Downloaded pip bootstrap script is empty."
    mv -f -- "$GET_PIP_TMP" "$GET_PIP"
  fi
  "$VPY" "$GET_PIP" --disable-pip-version-check --no-warn-script-location || fail "Could not bootstrap pip inside the Vela virtualenv."
fi
"$VPY" -m pip --version >/dev/null 2>&1 || fail "pip is still unavailable inside $VENV."
DEPS_MARKER="$STATE_DIR/dependencies-cu128-torch271.ok"
if [[ ! -f "$DEPS_MARKER" ]] || ! "$VPY" -c 'import torch, numpy, cryptography; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  log "Installing CUDA PyTorch 2.7.1 (cu128), NumPy and cryptography; wheel cache and temporary files are under $STATE_DIR."
  "$VPY" -m pip install --upgrade pip || fail "pip upgrade failed inside the Vela virtualenv."
  "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1' || fail "CUDA PyTorch installation failed. Check network access and Python compatibility."
  "$VPY" -m pip install 'numpy==1.26.4' 'cryptography>=42,<47' || fail "Python dependencies failed to install."
  "$VPY" - <<'PY' || fail "PyTorch CUDA self-check failed."
import torch, numpy, cryptography
print('torch=', torch.__version__, 'torch_cuda=', torch.version.cuda,
      'cuda_available=', torch.cuda.is_available(), 'gpu_count=', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA is unavailable: refusing CPU mining')
for i in range(torch.cuda.device_count()):
    print('GPU', i, torch.cuda.get_device_name(i), 'capability', torch.cuda.get_device_capability(i))
PY
  touch "$DEPS_MARKER"
fi

# Never silently fall back to CPU on a mining rig.
"$VPY" - <<'PY' || fail "No usable CUDA GPU was detected; refusing to mine on CPU."
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(1)
PY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
LOG_FILE="$STATE_DIR/logs/vela-miner.log"

# Defaults favor safe VRAM use on 8 GB RTX 3070 cards. Extra config can override these options.
read -r -a extra_array <<< "${EXTRA_ARGS:-}"
log "Starting Vela PoI v2 miner; mode=$POOL_MODE endpoint=$POOL_ENDPOINT wallet=${WALLET_ADDRESS:0:10}... model=$MODEL"
log "Default tuning: one CUDA stream per GPU; auto batch 1/2/4/8 with exact reference checks. Add --lanes 2 in Extra config only after a safe comparison."
if [[ "$POOL_MODE" == "stratum" ]]; then
  exec "$VPY" -u "$SCRIPT_DIR/legacy-python/sd_miner.py" \
    --stratum-url "$POOL_ENDPOINT" \
    --wallet "$LOGIN" \
    --password "$POOL_PASSWORD" \
    --model "$MODEL" \
    --devices auto \
    --lanes 1 \
    --auto-batch --max-batch 8 \
    --blocks 0 \
    --canary-interval 120 \
    --status-interval 30 \
    --log-file "$LOG_FILE" \
    "${extra_array[@]}"
else
  exec "$VPY" -u "$SCRIPT_DIR/legacy-python/sd_miner.py" \
    --rpc "$POOL_RPC_URL" \
    --address "$WALLET_ADDRESS" \
    --model "$MODEL" \
    --devices auto \
    --lanes 1 \
    --auto-batch --max-batch 8 \
    --blocks 0 \
    --tip-poll 0.5 \
    --canary-interval 120 \
    --status-interval 30 \
    --log-file "$LOG_FILE" \
    "${extra_array[@]}"
fi
