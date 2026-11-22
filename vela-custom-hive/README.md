# ExerMiner for HiveOS — Vela PoI v2

ExerMiner v0.2.2 supports two Vela pool interfaces while retaining the same CUDA
PoI v2 inference engine and the final independent reference check:

- **SuprNova named-parameter Stratum** (`vela.suprnova.cc:7601`, TLS on `:7604`).
- **AriaPool JSON-RPC** (`http://host:3349`) through the existing
  `getblocktemplate` / `submitblock` path.

The Stratum implementation is an independent Python client in
`legacy-python/stratum_client.py`. It subscribes and authorizes, receives Vela
PoI job notifications and variable share difficulty, reconstructs the pool's
coinbase/Merkle root/header, evaluates the PoI v2 work with the existing CUDA
engine, verifies a found candidate with the independent `sdint.py` reference,
and submits `x0` plus `hashpoi` to the pool. AriaPool continues to use the
original JSON-RPC mining path.

## SuprNova Flight Sheet

| Field | Value |
|---|---|
| Coin | Custom VELA |
| Miner | Custom → `vela-custom-hive` |
| Installation URL | direct URL to `vela-custom-hive-0.2.2.tar.gz` |
| Algorithm | `vela` |
| Wallet/worker template | `%WAL%.%WORKER_NAME%` |
| Pool URL | `vela.suprnova.cc:7601` |
| Password | `x` |
| Extra config | leave empty for the safe starting profile |

For TLS, use `stratum+ssl://vela.suprnova.cc:7604` as the Pool URL. Plain
`vela.suprnova.cc:7604` is also treated as TLS automatically. Keep the full
wallet plus worker suffix in the Wallet/worker field; the Stratum authorization
and share submissions use that same identity.

## AriaPool Flight Sheet

Use `http://159.69.194.46:3349` (or your AriaPool JSON-RPC endpoint) as the Pool
URL. The wrapper detects `http://` or `https://` and continues to use the
original RPC-based mining path. Do not use a bare `host:port` form for AriaPool;
bare `host:port` is intentionally interpreted as Stratum.

## Existing PoI model reuse

Before downloading, the launcher searches for a SHA-256-verified model at:

1. `/hive-config/vela-custom-hive/models/sd-turbo-int.poisd`
2. `/hive/miners/custom/vela-model/sd-turbo-int.poisd`
3. `/hive-config/models/sd-turbo-int.poisd`
4. `/hive/models/sd-turbo-int.poisd`

The expected model-file SHA-256 is
`ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36`.
A file is reused only when its bytes match this digest; a same-named but
mismatched file is ignored. The first valid candidate is passed directly to
the miner, so no duplicate model download is needed.

The CUDA-enabled Python environment, pip bootstrap, wheel cache, temporary
install files and detailed logs are kept under `/var/lib/vela-custom-hive` on
the large root filesystem. `/hive-config` is typically only 40 MB and is used
only for small Hive configuration data. The launcher removes only the known
incomplete v0.2.1 virtualenv and partial `get-pip.py` download from the old
config-partition location; it preserves the old log. It refuses to start the
first CUDA runtime install unless at least 8 GiB is free on the runtime
filesystem. The launcher refuses to mine when CUDA is unavailable; it does not
silently switch to CPU. It currently installs PyTorch 2.7.1 cu128 and NumPy
into that persistent environment.

## Runtime defaults and safeguards

- All CUDA-visible NVIDIA GPUs, ordered by `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- One stream per GPU initially; `--lanes 2` remains an experimental option.
- Per-GPU adaptive batch probing over 1/2/4/8, with exact comparison against
  the reference implementation before selecting a batch.
- Periodic consensus canaries and a fresh reference inference for every
  candidate share before submission.
- Work is abandoned when the Stratum job/connection changes; shares from an
  obsolete connection are discarded instead of being submitted.
- Model and pool-provided token IDs are checked; a model-hash mismatch refuses
  mining.

SuprNova Stratum mode currently implements PoI payload v2 (`.poisd`). It does
not implement the separate Fork-M/VIR payload v3 path. The AriaPool JSON-RPC
mode retains the original v2/v3 node-template path.

## Logs and HiveOS stats

- HiveOS log: `/var/log/miner/vela-custom-hive/vela-custom-hive.log`
- Detailed miner log: `/var/lib/vela-custom-hive/logs/vela-miner.log`
- HiveOS stats callback: `h-stats.sh`; reports per-GPU inference rates and
  accepted/rejected share counters for Stratum, as well as RPC block results.

## Verification status

The Stratum client includes offline tests for URL parsing, target calculation,
named-parameter subscribe/authorize, job parsing, Merkle/header reconstruction,
and share submission/acknowledgement. These tests do not replace a live pool
session or a real CUDA/HiveOS run. Verify the first submitted shares and the
per-GPU `inf/s` rate on the rig before considering the new mode production-ready.


### HiveOS runtime storage fix (v0.2.2)

HiveOS commonly mounts `/hive-config` as a 40 MB partition. v0.2.1 incorrectly
placed the Python virtualenv and `get-pip.py` there, which can cause
`curl: (23) Failed writing body` when the partition fills. v0.2.2 stores the
venv, pip cache, temporary install files, model fallback directory and detailed
logs under `/var/lib/vela-custom-hive` on the large root filesystem. It also
removes only the incomplete v0.2.1 venv and partial bootstrap file from the old
location, retaining logs and configuration. At least 8 GiB must be free for
the first CUDA-enabled PyTorch install.
