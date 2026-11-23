# Vela miner

Standalone miners for **Vela**, a fork of Bitcoin Core in which every proof-of-work
attempt runs a deterministic text-to-image model (Proof-of-Inference). The block header
commits to the model output (`hashPoI`) and every node verifies a block by recomputing
the inference bit for bit.

Protocol reference: <https://velaresearch.ai/docs/#mining-flow> · whitepaper: <https://velaresearch.ai/>

| File | Purpose |
| --- | --- |
| `sd_miner.py` | **PoI GPU miner**: v2 (`sd-turbo-int.poisd`) and, from the Fork-M height, v3 (`sd-turbo-int.vir`) — one process mines across the activation |
| `sdfast.py` | Batched/vectorised CUDA engine used by `sd_miner.py` for v2 (`--engine fast`, default) |
| `sdint.py` | Executable specification of PoI v2; loads the model file and exports it (`sdint.py export`) |
| `poi_miner.py` | CPU miner for the v1 toy model, and the CPU reference path for v3 (`--model3`); also provides the RPC, coinbase and serialisation helpers used by `sd_miner.py` |
| `vir.py` | Normative Python implementation of the `.vir` format (VIR, opset 1): every operator, rounding rule and container check |
| `vir_mine.py` | Fork-M mining helpers: leaf digests, `R = POI/leaves(...)`, `hashPoI = POI/out5(seed || x0 || R)`, the v3 payload, and the CPU/GPU engine adapters |
| `vir_torch.py` | GPU VIR engine (`TorchVM`): the v3 graph on CUDA, bit-identical to the node |
| `noderoot.py` | GPU digest kernel (CuPy) that produces the leaf digests while the output is still on the device |
| `render_poi.py` | Renders a block's latent `x0` as a PNG (v1 preview matrix / v2 sd-turbo VAE) |

## How mining works

```
getblocktemplate
  -> coinbase: height, extranonce, payout, devfee output,
               OP_RETURN "POI1"||tokens, witness commitment
  -> ctx = TextEncoder(tokens)                                  once per template
  -> for each nonce:
        seed    = TaggedHash("POI/seed", header[0..80) || model_hash)
        x0      = UNet(noise(seed), ctx)                        one inference per attempt
        hashPoI = TaggedHash("POI/out", seed || x0)
        if SHA256d(112-byte header) <= target:
            submitblock(header || txs || version || x0)
```

One attempt costs one inference: that is the useful work of the chain. The header is
112 bytes (Bitcoin's 80 plus `hashPoI`). The full byte layout and the integer
arithmetic are in whitepaper Section 4 and Appendix A.

From the **Fork-M** height the network switches to PoI payload **v3**: the same model
expressed as a `.vir` file, run by the node's VIR runtime, and the payload also commits
to the digest of every 256 KiB leaf of every node output:

```
        x0, leaves = VIR(seed, ctx)                             payload v3
        hashPoI    = TaggedHash("POI/out5", seed || x0 || R)     R = TaggedHash("POI/leaves", leaves)
```

`sd_miner.py` reads the payload version from each template and picks its engine
accordingly, so one process mines v2 before the activation and v3 after it. Spec:
<https://velaresearch.ai/research/difficulty-lanes/> and the Whitepaper's Fork-M section.

## Install

```sh
pip install -r requirements.txt          # torch, numpy, cryptography
```

A CUDA build of PyTorch is required for the fast engine; `sd_miner.py` falls back to the
slow reference engine on CPU (`--device cpu`). Rendering a v2 block with `--vae` and
re-exporting the model file additionally need `pip install diffusers transformers`.

**v3 mining** additionally needs CuPy for the digest kernel — `pip install cupy-cuda12x` —
and the `.vir` model (§ The model files). The GPU VIR engine ships in this repository
(`vir_torch.py`, `noderoot.py`); `--engine-dir` exists only to point at a different
checkout. Before mining, confirm the engine reproduces the reference output on your
machine:

```sh
python vir_torch.py --model sd-turbo-int.vir
# x0 sha256 edc9d82935eb5f91912a3f91ee1e4e887f68e5c5eead409c5cc8d47cfe7aac8e
```

A different digest means a broken environment (driver, PyTorch build, float64 matmul),
not a different model.

## The model files

PoI v2 needs `sd-turbo-int.poisd` (1,187,585,285 bytes). It is not in git.

```sh
huggingface-cli download valereseach/vela-sd-turbo-int-v2 sd-turbo-int.poisd SHA256SUMS --local-dir .
sha256sum -c SHA256SUMS
```

| | |
| --- | --- |
| File SHA-256 | `ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36` |
| Model hash | `a7fd64b04c8bf36c1b74a629e32f093ce10e364787292f53fa1337557eb9776a` |
| License | Stability AI Community License (derivative of sd-turbo) |

PoI v3 (Fork-M) needs the same weights as a `.vir` file, `sd-turbo-int.vir`
(1,187,715,968 bytes). The registry entry that activates it on testnet4 pins it:

| | |
| --- | --- |
| File SHA-256 | `80c548386b613378cd6377319e28c38e8881487551ea6606611ed189381ddf17` |
| Model hash (`POI/model`) | `4539e064ca8b7197e5c39e86a3e1454b632c62faade38b2ffb13b3875c08e6ad` |
| Output | 32,768 bytes; 9,415 leaf digests per block |

A node refuses a model file whose hash differs. To regenerate the v2 file from the
floating-point model instead of downloading it:

```sh
huggingface-cli download stabilityai/sd-turbo --local-dir sd-turbo \
    --include "model_index.json" "scheduler/*" "tokenizer/*" "*/config.json" "*.fp16.safetensors"
python sdint.py export sd-turbo sd-turbo-int.poisd
```

## GPU miner

The custom V2 runtime verifies the fast CUDA engine against the specification vector at
start-up and repeats a consensus canary on every active CUDA stream (default: every
120 seconds). A candidate share is recomputed with the independent `sdint.py` reference
path and re-hashed against the block target before `submitblock`; any mismatch stops the
miner without submitting the block.

It now supports multiple GPUs and optional independent CUDA streams per GPU for V2.
Each GPU has an independent scheduler (a slow card does not hold back the others); streams
on the same card synchronize at batch boundaries. Nonce ranges are interleaved and disjoint
across all GPU/stream workers. The batch tuner
measures candidate sizes on each GPU, checks **every sample** against the reference
implementation, and retains only a passing configuration. The noise tensor is generated
in a single batched host-to-device transfer rather than one transfer per nonce.

### Scheduling design borrowed from FFGV 0.4.6

The V2 runtime adopts the *ideas* behind FFGV's measured lane work, rather than
copying its binary-only engine: optional two-stream execution per GPU, a host-side
lane-start barrier and batch-completion barrier, per-device batch-size measurement,
periodic per-lane consensus canaries, and independent scheduling per GPU. Every
selected batch is checked against the reference implementation before mining; the
winning nonce is independently recomputed before submission. Unlike FFGV's CUDA
engine, these lanes share one PyTorch model/weight cache, and the launch barrier is a
host scheduling barrier (not a custom CUDA event/kernel barrier). The expected speedup
must therefore be measured on the actual cards; the FFGV 0.4.6 RTX 3070 result is not
assumed to transfer to this engine.

The logger writes timestamped console and rotating file records to
`logs/vela-miner.log` by default (10 MiB per file, five backups). Job and progress logs
include the payload version, height, previous block hash, bits, full target, model hash,
Merkle root, nonce, last full block hash, `hashPoI`, attempts and measured inference rate.
Candidate logs additionally include the `x0` SHA-256, block size and final reference
check result. RPC credentials and cookie contents are never written to the log.

```sh
# offline: load the model, run the self-check, time N inferences
python sd_miner.py --model sd-turbo-int.poisd --bench 300

# safest first run: benchmark all 4 RTX 3070 cards, one CUDA stream per card
python sd_miner.py --model sd-turbo-int.poisd --devices 0,1,2,3 --bench 300

# experimental: 2 CUDA streams per card; select the best exact batch up to 8
python sd_miner.py --model sd-turbo-int.poisd --devices 0,1,2,3 \
    --lanes 2 --auto-batch --max-batch 8 --bench 600

# mine on testnet4 (RPC 39732), forever; the prompt rotates every block
python sd_miner.py --rpc http://vela:<pass>@127.0.0.1:39732 \
    --model sd-turbo-int.poisd --address <tvela1...> --blocks 0

# ... or mine with your own prompt list
python sd_miner.py --rpc http://vela:<pass>@127.0.0.1:39732 \
    --model sd-turbo-int.poisd --address <tvela1...> \
    --prompts-file prompts.txt --blocks 0

# mainnet/local node: cookie overrides credentials in --rpc; select GPU workers explicitly
python sd_miner.py --rpc http://u:p@127.0.0.1:19732 --cookie ~/.vela/.cookie \
    --model sd-turbo-int.poisd --address <vela1...> --devices 0,1,2,3 \
    --lanes 2 --auto-batch --max-batch 8 --tip-poll 0.5 --canary-interval 120 --blocks 0

# across the Fork-M activation: give both models, one process
python sd_miner.py --rpc http://vela:<pass>@127.0.0.1:39732 \
    --model sd-turbo-int.poisd --model3 sd-turbo-int.vir \
    --address <tvela1...> --blocks 0
```

`--model3` alone also works on a chain that is already past its Fork-M height.

**Runtime scope:** the multi-GPU/stream scheduler, adaptive batch tuning, periodic canaries,
final candidate recheck and rotating diagnostics described below apply to the PoI v2
(`.poisd`) CUDA path. The existing Fork-M v3 VIR adapter remains on its original execution
path in this release; its GPU selection and multi-stream scheduling are not changed here.
The mining engine now has two pool paths. Local-node and AriaPool jobs continue to use
Vela JSON-RPC (`getblocktemplate` / `submitblock`). HiveOS SuprNova jobs use the separate
`stratum_client.py` named-parameter Stratum transport (`mining.subscribe`, `mining.authorize`,
`mining.notify`, and `mining.submit`). The Stratum route reconstructs the pool-provided coinbase,
Merkle branch and 80-byte header before reusing the same PoI v2 CUDA inference and final `sdint.py`
reference check. Stratum support is PoI v2 only in this package; the Fork-M/VIR v3 route remains
JSON-RPC-only.

### HiveOS pool package

The HiveOS wrapper selects Stratum for a bare `host:port` or `stratum*://` Pool URL and selects the
original RPC route for an explicit `http(s)://host:port` URL. The default is SuprNova at
`vela.suprnova.cc:7601`; AriaPool can be selected explicitly with its HTTP JSON-RPC endpoint.
The wrapper verifies and reuses an existing PoI model from the common HiveOS model locations before
downloading, creates a persistent CUDA-enabled Python environment, and reports per-GPU inference
rates and Stratum accepted/rejected shares through `h-stats.sh`.

| Option | Meaning |
| --- | --- |
| `--stratum-url <host:port>` | named-parameter Vela Stratum endpoint; `:7604` or `stratum+ssl://` enables TLS |
| `--wallet <addr[.worker]>` | Stratum authorization and share identity; mainnet VELA address with optional worker suffix |
| `--password <text>` | Stratum password (default `x`) |
| `--model <file>` | `*.poisd` model file (payload v2); required unless `--model3` is enough |
| `--model-sha256 <hex>` | expected full model-file SHA-256 (official V2 digest by default) |
| `--model3 <file>` | `*.vir` model file: mine payload v3 from the fork height on |
| `--engine-dir <dir>` | GPU VIR engine directory for v3 (default: this repository; or `VIR_GPU_DIR`) |
| `--rpc <url>` | node RPC URL, `http://user:pass@host:port` (default `http://u:p@127.0.0.1:18743`) |
| `--address <addr>` | required on non-regtest networks; `OP_TRUE` is permitted only on regtest |
| `--prompt <text>` | fix one prompt for every block (default: rotate the built-in list, `prompts[height % 10]`) |
| `--prompts-file <f>` | one prompt per line, replacing the built-in ten (blank lines and `#` comments ignored) |
| `--prompt-index <n>` | pin prompt `n` (0-based) from the list instead of rotating by height |
| `--node-prompt` | use the node's `-poiprompt` instead of the local list |
| `--blocks <n>` | blocks to mine, `0` = forever (default `1`) |
| `--device <dev>` | One device (`cuda:0` or `cpu`); overrides `--devices` |
| `--devices <list>` | `auto` (all visible CUDA GPUs) or comma-separated indices such as `0,1,2,3` |
| `--engine <e>` | `fast` (default, `sdfast.py`) or `ref` (CPU specification engine) |
| `--batch <n>` | nonces per lane per pass; fixed sizes are reference-checked before mining |
| `--auto-batch` | test 1/2/4/8 independently on every GPU and pick the fastest exact batch |
| `--max-batch <n>` | maximum auto-batch candidate: 1, 2, 4 or 8 |
| `--lanes <n>` | CUDA streams per GPU: 1 (default, safer) or 2 (experimental; more VRAM) |
| `--canary-interval <sec>` | repeat the exact consensus vector on every stream (120 default, 0 disables) |
| `--tip-poll <sec>` | background active-tip poll (0.5 default) |
| `--status-interval <sec>` | V2 progress log cadence (30 seconds by default) |
| `--log-file <path>` | rotating log file (default `logs/vela-miner.log`; `none` disables file logging) |
| `--log-level <level>` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--bench <n>` | offline: time N inferences per GPU, no node needed; uses all configured streams |
| `--throttle <F>` | idle `F` x the inference time after each attempt (e.g. `1.0` ≈ 50 % GPU) so a desktop GPU stays responsive |
| `--low-priority` | run the miner process at below-normal CPU priority |
| `--verbose` | extra v3 progress output; V2 always logs at `--status-interval` |

### Prompts

Every block commits its own prompt tokens in the coinbase (`OP_RETURN "POI1" || tokens`), so the
network accepts any prompt of up to 300 tokens that the model's tokenizer can encode — the choice
belongs to the miner. By default this miner rotates ten prompts, one per block, by height:

| Height | Prompt |
| --- | --- |
| `h % 10 == 0` | a lighthouse on a granite cliff at dawn |
| `h % 10 == 1` | a koi pond in autumn rain |
| `h % 10 == 2` | a snow leopard crossing a frozen ridge at noon |
| `h % 10 == 3` | a night market of paper lanterns reflected in wet asphalt |
| `h % 10 == 4` | a bioluminescent tide retreating over black sand |
| `h % 10 == 5` | a desert observatory dome under the milky way |
| `h % 10 == 6` | a misty tea plantation terraced into a hillside |
| `h % 10 == 7` | a cargo airship docking above a stormy harbour |
| `h % 10 == 8` | a coral reef seen through the glass floor of a boat |
| `h % 10 == 9` | a lone cabin in a birch forest after the first snow |

Rotating by height rather than by "the Nth block I mined" keeps every miner in step: all blocks at
one height carry the same prompt, so the rendered block images follow the chain. The same ten lines
are in `prompts.txt` — copy it, edit it, pass it with `--prompts-file`.

Avoid punctuation the tokenizer re-spaces: a hyphen or comma comes back as `storm - lit` / `rain ,`
in the decoded prompt, because those are separate tokens. `python -c` round-trips a candidate:

```sh
python -c "import sys;sys.path.insert(0,'.');import sdint;n,_=sdint.load('sd-turbo-int.poisd','cpu');t=sdint.tokenizer(n);p='your prompt here';print(repr(t.decode(t.encode(p))))"
```

The text encoder and the cross-attention K/V run once per template; the tip is watched
in a background thread so the GPU loop never blocks on RPC.

## CPU miner (v1 toy model, v3 reference path)

`poi_miner.py` mines the v1 toy model on multiple CPU processes, and is the CPU
reference path for v3 (`--model3`, about ten minutes per sd-turbo inference: a
cross-check, not a mining engine). It shares its RPC, coinbase and serialisation helpers
with `sd_miner.py`, so it must stay in the same directory.

```sh
python poi_miner.py --rpc http://u:p@127.0.0.1:18743 --address <rvela1...> --blocks 1
python poi_miner.py --rpc http://127.0.0.1:39732 --cookie <datadir>/testnet4/.cookie \
    --model3 sd-turbo-int.vir --address <tvela1...> --blocks 1
python poi_miner.py --bench 500              # offline throughput benchmark
python poi_miner.py --selftest               # cross-check against the node's reference
```

| Option | Meaning |
| --- | --- |
| `--workers <n>` | worker processes (default: cores − 1) |
| `--chunk <n>` | nonces per worker task (default `64`) |
| `--model3 <file>` | `.vir` model: mine payload v3 with the reference implementation |
| `--cookie <file>` | node `.cookie` for RPC auth, instead of `user:pass` in `--rpc` |
| `--blocks <n>` | blocks to mine, `0` = forever (default `1`) |
| `--bench <n>` | offline: time N inferences |
| `--selftest` | compare the numpy model with the reference implementation |

CPU mining of **v2** costs one CPU inference (8–23 s) per attempt — use the GPU miner for
real work. `--selftest` needs the node's reference model from a vela checkout
(`test/functional/test_framework/poi.py`); run it from there if this repository is
checked out on its own.

## Block images

A block stores the latent `x0` (v2: 4×64×64 int16, 32 KiB). Decoding it to a 512×512
image with the sd-turbo VAE is **not** part of consensus.

```sh
python render_poi.py --rpc http://vela:<pass>@127.0.0.1:39732 <height> \
    --vae <sd-turbo dir> --no-verify -o block.png
python render_poi.py --hex <x0 hex> -o x0.png          # v1 latent, no node
```

| Option | Meaning |
| --- | --- |
| `--vae <dir>` | sd-turbo directory, required for v2 blocks (or `POI_SD_DIR`) |
| `--scale <n>` | nearest-neighbour upscale for the v1 preview (default `16`) |
| `--no-verify` | do not ask the node to recompute the inference |
| `--hex <x0>` | render a raw v1 latent instead of a block |

## Networks & ports

| Network | Flag | P2P | RPC |
| --- | --- | --- | --- |
| Main (not launched) | (default) | `19733` | `19732` |
| Testnet (PoI v2 from block 1, Fork-M at height 10,000) | `-testnet4` | `39733` | `39732` |
| Signet | `-signet` | `49733` | `49732` |
| Regtest | `-regtest` | `18744` | `18743` |

| | |
| --- | --- |
| DNS seed | `tseed1.velaresearch.ai` |
| Fixed seed | `56.10.177.122:39733` |
| Developer fund | `tvela1qegx4wllff9p2hp0e4xjut9uma3lf9p5wrhvm32` |

A node that provides `getblocktemplate` is required. For v2 it runs with
`velad -testnet4 -poimodel=sd-turbo-int.poisd`; for v3 it also needs the VIR weights,
`-poimodel3=sd-turbo-int.vir` (a node without them light-verifies and cannot mine).
`getblocktemplate` returns `poi` (payload version, model hash, prompt tokens and
script, vocabulary, max tokens, and for v3 the leaf count and the lane) and
`devfee {script, amount}`; `coinbasevalue` is the miner's share, and a coinbase that does
not pay the developer fund is rejected (`bad-cb-devfee`).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `fast engine does not reproduce the specification vector` | The GPU is producing wrong integer results (driver / PyTorch build); try `--engine ref`. |
| `model hash differs from the node's` | Not the canonical export; download the file again or regenerate it with `sdint.py export`. |
| `node expects PoI payload v<N>` | The template asks for a payload this miner was not given: pass `--model3` for v3 (and `--model` for v2 on a chain that still has pre-fork blocks). |
| `the v3 GPU miner needs the VIR engine directory` | `pip install cupy-cuda12x`, or point `--engine-dir` / `VIR_GPU_DIR` at a checkout that holds `vir_torch.py` and `noderoot.py`. |
| `GPU engine directory not found` | Same: the directory must exist and contain the two engine files. |
| `bad-poi-inference` | The submitted `x0` (or the leaf digests) is not what the model produces; check the model hash and the prompt. |
| Desktop stutters while GPU mining | `--throttle 1.0 --low-priority`. |
| `rpc error: ... ; retrying in 10s` | The node is unreachable; check `-server=1`, the RPC port and the credentials. |

## Provenance

These scripts are the miner programs of the [vela](https://git.velaresearch.ai/velaresearch/vela)
repository, copied from `contrib/poi-sd/`, `contrib/poi-miner/` and `contrib/vir/` at
branch `fork-m/integration` (commit `c7e552d`), which adds Fork-M mining: payload v3, the
leaf commitments, and the GPU VIR engine (`vir_torch.py`, `noderoot.py`) that used to
live outside the repository. They are collected here so that a miner can be cloned and
run without building the node.

The prompt rotation (`DEFAULT_PROMPTS`, `--prompts-file`) exists only here; it was kept
when the Fork-M changes were merged in. `vir.py` and `sdint.py` are consensus
specifications: **any change to them, or to the model files they describe, is a consensus
change (hard fork)**. Changes belong in the `vela` repository first, then flow back here.

## License

MIT — see [LICENSE](LICENSE). The PoI v2 model file is a derivative of
[sd-turbo](https://huggingface.co/stabilityai/sd-turbo) and is distributed under the
Stability AI Community License; review its terms before commercial use.
