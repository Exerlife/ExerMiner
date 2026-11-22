#!/usr/bin/env python3
"""PoI GPU miner: integer sd-turbo on CUDA, bit-identical to the node.

  python sd_miner.py --rpc http://u:p@127.0.0.1:18743 --model sd-turbo-int.poisd \\
      [--prompt "a koi pond in autumn rain"] [--address ADDR] [--blocks 1] [--batch 1]
  python sd_miner.py --model sd-turbo-int.poisd --bench 3

Flow: getblocktemplate -> coinbase (BIP34 height, extranonce, payout, prompt,
developer fund, witness commitment) -> merkle root -> for each nonce: seed ->
integer sd-turbo -> hashPoI -> SHA256d(112-byte header) <= target -> submitblock.

Fork-M (PoI payload v3, doc/fork-m.md): pass --model3 <sd-turbo-int.vir> as well.
The miner switches per template: before the fork height the node asks for v2 and
the .poisd engine (sdint/sdfast) mines it; from the fork height the node asks for
v3 and the VIR engine (--engine-dir, vir_torch.py + noderoot.py) mines it, with
the leaf commitments and hashPoI = POI/out5(seed || x0 || R). One process can run
across the activation. --model3 alone also works on a Fork-M-only chain.

Engines: "fast" (sdfast.py, batched/vectorised CUDA, default) or "ref"
(sdint.py, the specification). The fast engine is checked against the
specification's test vector at start-up and refuses to mine on mismatch.
The text encoder and the cross-attention K/V run once per template.
"""
import argparse, concurrent.futures, gc, hashlib, logging, os, signal, struct, sys, threading, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "poi-miner"))
import sdint  # noqa: E402
import sdfast  # noqa: E402
import vir_mine  # noqa: E402
from stratum_client import StratumClient, build_stratum_header76  # noqa: E402
from poi_miner import RPC, build_coinbase, merkle_root, prompt_script, ser_compact, sha256d, tagged_hash  # noqa: E402

VECTOR_SEED = hashlib.sha256(b"x").digest()
VECTOR_TOKENS = [320, 736, 3240]
VECTOR_SHA256 = "bef682cc9651fe8d42f075e98b907432f86a0b9e9a40e772fef99fc424f5c518"
V2_MODEL_FILE_SHA256 = "ba38ab96b3398e22ea631d8a457350f91833e57463199d063663308513651f36"
LOG = logging.getLogger("vela-miner")


class ConsensusError(RuntimeError):
    """Fatal local mismatch against the PoI reference/consensus vector."""


def setup_logging(path, level="INFO"):
    """Console + bounded rotating file log. Credentials are never logged."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise SystemExit(f"invalid --log-level: {level}")
    LOG.setLevel(numeric)
    LOG.handlers.clear()
    LOG.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(numeric)
    LOG.addHandler(console)
    if path and path.lower() != "none":
        log_path = Path(path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric)
        LOG.addHandler(file_handler)
        LOG.info("log_file=%s rotation=10MiB backups=5", log_path.resolve())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_v2_model_file(path, expected=None):
    expected = (expected or V2_MODEL_FILE_SHA256).lower()
    actual = sha256_file(path)
    LOG.info("model_file=%s file_sha256=%s expected_sha256=%s", path, actual, expected)
    if actual != expected:
        raise SystemExit(
            "V2 model file SHA-256 mismatch; refusing to mine. "
            f"actual={actual} expected={expected}"
        )
    return actual


def resolve_devices(args):
    """Resolve one or more CUDA devices; auto means every visible GPU for V2."""
    if args.device and args.devices != "auto":
        raise SystemExit("use either --device or --devices, not both")
    if args.engine == "ref":
        if args.device and args.device != "cpu":
            raise SystemExit("--engine ref is CPU-only; use --device cpu")
        if args.devices != "auto":
            raise SystemExit("--engine ref is CPU-only; remove --devices")
        return ["cpu"]

    if args.device:
        requested = args.device.strip().lower()
        if requested == "cpu":
            return ["cpu"]
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA device '{requested}' requested, but this PyTorch build has no CUDA")
        if requested.isdigit():
            requested = f"cuda:{requested}"
        try:
            parsed = torch.device(requested)
        except Exception as exc:
            raise SystemExit(f"invalid --device '{args.device}': {exc}")
        if parsed.type != "cuda" or parsed.index is None or not 0 <= parsed.index < torch.cuda.device_count():
            raise SystemExit(f"invalid or unavailable --device '{args.device}'")
        return [str(parsed)]

    if args.devices.lower() == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        LOG.warning("cuda_unavailable_falling_back_to_cpu_reference; mining speed will be very low")
        return ["cpu"]

    values = [item.strip().lower() for item in args.devices.split(",") if item.strip()]
    if not values:
        raise SystemExit("--devices must be 'auto' or a comma-separated GPU list")
    devices = []
    for value in values:
        if value == "cpu":
            devices.append("cpu")
        elif value.isdigit():
            devices.append(f"cuda:{value}")
        elif value.startswith("cuda:"):
            devices.append(value)
        else:
            raise SystemExit(f"invalid device '{value}'; use indices, cuda:N, or cpu")
    if len(set(devices)) != len(devices):
        raise SystemExit("--devices contains duplicates")
    if "cpu" in devices and len(devices) > 1:
        raise SystemExit("CPU cannot be mixed with CUDA devices in one mining run")
    if any(device.startswith("cuda:") for device in devices) and not torch.cuda.is_available():
        raise SystemExit("CUDA devices requested, but this PyTorch build has no CUDA")
    for device in devices:
        if device == "cpu":
            continue
        try:
            index = torch.device(device).index
        except Exception as exc:
            raise SystemExit(f"invalid CUDA device '{device}': {exc}")
        if index is None or index < 0 or index >= torch.cuda.device_count():
            raise SystemExit(f"CUDA device '{device}' is not visible")
    return devices

DEFAULT_PROMPTS = (
    "a lighthouse on a granite cliff at dawn",
    "a koi pond in autumn rain",
    "a snow leopard crossing a frozen ridge at noon",
    "a night market of paper lanterns reflected in wet asphalt",
    "a bioluminescent tide retreating over black sand",
    "a desert observatory dome under the milky way",
    "a misty tea plantation terraced into a hillside",
    "a cargo airship docking above a stormy harbour",
    "a coral reef seen through the glass floor of a boat",
    "a lone cabin in a birch forest after the first snow",
)


def load_prompts(path):
    with open(path, encoding="utf-8") as fh:
        prompts = tuple(line.strip() for line in fh
                        if line.strip() and not line.lstrip().startswith("#"))
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def pick_prompt(args, height):
    """Which prompt this block commits: --prompt, then --prompt-index, then rotation by height."""
    if args.prompt:
        return args.prompt
    if args.node_prompt:
        return None                                     # whatever the node hands out
    if args.prompt_index is not None:
        return args.prompts[args.prompt_index % len(args.prompts)]
    return args.prompts[height % len(args.prompts)]


class StratumJobWatcher:
    """Expose the current Stratum job as the stale-work guard used by the GPU loop."""
    def __init__(self, client, job):
        self.client = client
        self.job = dict(job)

    def set_expected_tip(self, expected):
        # The selected immutable Stratum job is the expected work generation.
        return True

    def is_stale(self, expected=None):
        current = self.client.current_job()
        return (current is None
                or current.get("generation") != self.job.get("generation")
                or current.get("connection_id") != self.job.get("connection_id"))


class TipWatcher(threading.Thread):
    """Poll the node tip without blocking CUDA work; synchronized against template changes."""
    def __init__(self, rpc, interval=2.0):
        super().__init__(daemon=True, name="vela-tip-watcher")
        self.rpc = rpc
        self.interval = max(0.25, float(interval))
        self._lock = threading.Lock()
        self._tip = None
        self._expected = None
        self._stale = False
        self._job_generation = 0
        self._stop_event = threading.Event()
        self._last_error_log = 0.0

    def set_expected_tip(self, expected):
        with self._lock:
            self._expected = expected
            self._job_generation += 1
            # The cached tip may belong to the previous job and can be older
            # than the template we just fetched. Only a poll started after
            # this point may declare this new job stale.
            self._stale = False
            return True

    def is_stale(self, expected=None):
        with self._lock:
            if expected is not None and self._expected != expected:
                self._expected = expected
                self._stale = self._tip is not None and self._tip != expected
            return self._stale

    def stop(self):
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=self.interval + 2.0)

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                request_generation = self._job_generation
            try:
                tip = self.rpc("getbestblockhash")
                with self._lock:
                    self._tip = tip
                    # Ignore a response to an RPC begun before the current
                    # template was selected: it may describe the old parent.
                    if request_generation == self._job_generation and self._expected is not None:
                        self._stale = tip != self._expected
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error_log >= 30.0:
                    LOG.warning("tip_poll_failed error=%s", exc)
                    self._last_error_log = now
            self._stop_event.wait(self.interval)


class Miner:
    def __init__(self, path, device, engine="fast", lanes=1, model_data=None):
        t0 = time.perf_counter()
        if model_data is None:
            self.net, mh = sdint.load(path, device)
        else:
            tensors, mh = model_data
            self.net = sdint.Net(tensors, device)
        self.model_hash = bytes.fromhex(mh)[::-1]   # uint256 byte order
        self.tok = sdint.tokenizer(self.net)
        self.device = device
        self.device_obj = torch.device(device)
        self.is_cuda = self.device_obj.type == "cuda" and torch.cuda.is_available()
        self.engine = engine if self.is_cuda else "ref"
        self.fast = sdfast.Fast(self.net) if self.engine == "fast" else None
        self._key, self._ctx, self._kv = None, None, None
        self.batch = 1
        self.streams = []
        if self.fast:
            self.streams = [torch.cuda.Stream(device=self.device_obj) for _ in range(max(1, lanes))]
        if self.is_cuda:
            name = torch.cuda.get_device_name(self.device_obj)
            free_mem, total_mem = torch.cuda.mem_get_info(self.device_obj)
            device_info = f"{name}, VRAM free={free_mem / 2**30:.2f}/{total_mem / 2**30:.2f} GiB"
        else:
            device_info = "CPU reference"
        LOG.info("model_loaded device=%s hardware=%s model_hash=%s load_seconds=%.2f engine=%s lanes=%d",
                 device, device_info, mh, time.perf_counter() - t0, self.engine, len(self.streams) or 1)
        if self.engine == "fast":
            t0 = time.perf_counter()
            out = self.x0s([VECTOR_SEED], VECTOR_TOKENS, stream_id=0)[0]
            if hashlib.sha256(out).hexdigest() != VECTOR_SHA256:
                raise SystemExit(f"{device}: fast engine differs from the consensus vector - refusing to mine")
            LOG.info("consensus_selftest=PASS device=%s vector_sha256=%s seconds=%.2f",
                     device, hashlib.sha256(out).hexdigest(), time.perf_counter() - t0)
            if len(self.streams) > 1:
                primary_stream = self.streams[0]
                for stream in self.streams[1:]:
                    stream.wait_stream(primary_stream)
                torch.cuda.synchronize(self.device_obj)
        self._prep_ready = False

    def _prep(self, tokens):
        key = tuple(tokens)
        if self._key != key:
            ids = torch.tensor(sdint.context_ids(tokens), device=self.device)
            self._ctx = self.net.text(ids)
            self._kv = self.fast.cross_kv(self._ctx) if self.fast else None
            self._key = key
            self._prep_ready = False

    def prepare(self, tokens):
        """Prepare prompt context on the default stream before worker streams start."""
        self._prep(tokens)
        if self.fast and not self._prep_ready:
            default_stream = torch.cuda.default_stream(self.device_obj)
            for stream in self.streams:
                stream.wait_stream(default_stream)
            self._prep_ready = True

    def x0s(self, seeds, tokens, stream_id=0):
        self.prepare(tokens)
        if self.fast:
            stream = self.streams[stream_id % len(self.streams)]
            with torch.cuda.stream(stream), torch.inference_mode():
                return self.fast.run_batch(seeds, self._kv)
        return [sdint.run(self.net, seed, tokens, self._ctx) for seed in seeds]

    def check_canaries(self, restore_tokens):
        """Run the consensus vector on every stream, then restore the live prompt."""
        if not self.fast:
            return
        for stream_id in range(len(self.streams)):
            output = self.x0s([VECTOR_SEED], VECTOR_TOKENS, stream_id=stream_id)[0]
            digest = hashlib.sha256(output).hexdigest()
            if digest != VECTOR_SHA256:
                LOG.critical("consensus_canary=FAIL device=%s lane=%d got=%s expected=%s",
                             self.device, stream_id, digest, VECTOR_SHA256)
                raise ConsensusError(f"consensus canary failed on {self.device} lane {stream_id}")
            LOG.info("consensus_canary=PASS device=%s lane=%d vector_sha256=%s",
                     self.device, stream_id, digest)
        self.prepare(restore_tokens)


def build_lane_specs(miners):
    """Allocate disjoint interleaved nonce ranges to every GPU stream."""
    lane_specs = []
    first_nonce = 0
    for miner in miners:
        lane_count = len(miner.streams) if miner.fast else 1
        for lane_id in range(lane_count):
            batch_size = miner.batch if miner.fast else 1
            lane_specs.append({"miner": miner, "stream_id": lane_id, "batch": batch_size,
                               "next_nonce": first_nonce})
            first_nonce += batch_size
    if first_nonce <= 0:
        raise ValueError("no nonce lanes were configured")
    return lane_specs, first_nonce


def lane_nonce_sequence(spec, stride, rounds):
    """Small deterministic helper used by tests to prove no lane overlap."""
    return [spec["next_nonce"] + round_index * stride + offset
            for round_index in range(rounds) for offset in range(spec["batch"])]


def make_bench_tasks(total, batch_size, lane_count):
    """Split an exact inference count across independent CUDA streams."""
    remaining = int(total)
    tasks = []
    for lane_id in range(int(lane_count)):
        count = min(int(batch_size), remaining)
        if count <= 0:
            break
        tasks.append((lane_id, count))
        remaining -= count
    return tasks


def auto_tune_batch(miner, tokens, max_batch=8):
    """Find the fastest exact batch for this GPU, including its configured streams."""
    if not miner.fast:
        miner.batch = 1
        LOG.info("auto_batch device=%s selected=1 reason=CPU_reference", miner.device)
        return 1
    miner.prepare(tokens)
    lane_count = max(1, len(miner.streams))
    candidates = [b for b in (1, 2, 4, 8) if b <= max_batch]
    seed_count = max(candidates) * lane_count
    seeds = [hashlib.sha256(b"vela-custom-batch-test" + i.to_bytes(4, "little")).digest()
             for i in range(seed_count)]
    best_batch, best_rate = None, -1.0

    def run_all(batch_size):
        with concurrent.futures.ThreadPoolExecutor(max_workers=lane_count) as executor:
            futures = [executor.submit(miner.x0s, seeds[lane * batch_size:(lane + 1) * batch_size],
                                       tokens, lane) for lane in range(lane_count)]
            return [future.result() for future in futures]

    for batch_size in candidates:
        try:
            outputs = run_all(batch_size)  # warm-up and populate immutable weight caches
            torch.cuda.synchronize(miner.device_obj)
            # Validate both ends of every stream's batch against the specification path.
            for lane, lane_outs in enumerate(outputs):
                lane_seeds = seeds[lane * batch_size:(lane + 1) * batch_size]
                for index, candidate in enumerate(lane_outs):
                    ref = sdint.run(miner.net, lane_seeds[index], tokens, miner._ctx)
                    if candidate != ref:
                        raise RuntimeError(f"batch {batch_size} lane {lane} consensus mismatch at sample {index}")
            del outputs
            torch.cuda.synchronize(miner.device_obj)
            elapsed = []
            for _ in range(3):
                t0 = time.perf_counter()
                run_all(batch_size)
                torch.cuda.synchronize(miner.device_obj)
                elapsed.append(time.perf_counter() - t0)
            median = sorted(elapsed)[len(elapsed) // 2]
            rate = (batch_size * lane_count) / max(median, 1e-9)
            LOG.info("auto_batch_test device=%s lanes=%d batch_per_lane=%d aggregate_inf_per_second=%.3f "
                     "median_seconds=%.4f exact=PASS", miner.device, lane_count, batch_size, rate, median)
            if rate > best_rate:
                best_batch, best_rate = batch_size, rate
        except Exception as exc:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_oom = (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()
            is_mismatch = "consensus mismatch" in str(exc)
            if not is_oom and not is_mismatch:
                raise
            LOG.warning("auto_batch_rejected device=%s lanes=%d batch_per_lane=%d reason=%s",
                        miner.device, lane_count, batch_size, exc)
            try:
                torch.cuda.synchronize(miner.device_obj)
            except Exception:
                pass
            gc.collect()
            with torch.cuda.device(miner.device_obj):
                torch.cuda.empty_cache()
            if is_oom:
                LOG.info("auto_batch_probe_stopped device=%s reason=larger_batches_need_more_memory", miner.device)
                break
    if best_batch is None:
        raise SystemExit(f"{miner.device}: no batch size passed exactness and memory checks")
    miner.batch = best_batch
    LOG.info("auto_batch_selected device=%s lanes=%d batch_per_lane=%d aggregate_inf_per_second=%.3f",
             miner.device, lane_count, best_batch, best_rate)
    return best_batch


def serialize_v2_block(header80, hash_poi, coinbase_witness, transactions, x0):
    """Serialize a V2 block with the consensus 112-byte header (80 + hashPoI)."""
    if len(header80) != 80:
        raise ValueError(f"V2 base header must be 80 bytes, got {len(header80)}")
    if len(hash_poi) != 32:
        raise ValueError(f"V2 hashPoI must be 32 bytes, got {len(hash_poi)}")
    header112 = header80 + hash_poi
    tx_count = 1 + len(transactions)
    block = (header112 + ser_compact(tx_count) + coinbase_witness
             + b"".join(bytes.fromhex(tx["data"]) for tx in transactions)
             + bytes([sdint.VERSION]) + ser_compact(len(x0)) + x0)
    return block


def mine_one_v3(rpc, engine, args, payout_spk, extranonce, watcher, tmpl=None):
    """Fork-M template (payload v3): VIR inference + leaf digests + POI/out5."""
    tmpl = tmpl or rpc("getblocktemplate", {"rules": ["segwit"]})
    poi = tmpl["poi"]
    if poi["version"] != vir_mine.PAYLOAD_VERSION:
        raise SystemExit(f"node expects PoI payload v{poi['version']}, the v3 path mines "
                         f"v{vir_mine.PAYLOAD_VERSION}")
    if poi["model_hash"].lower() != engine.model_hash:
        raise SystemExit(f"model hash differs from the node's: {poi['model_hash']} vs {engine.model_hash}")
    model_hash = bytes.fromhex(engine.model_hash)[::-1]
    text = pick_prompt(args, tmpl["height"])
    tokens = engine.encode(text) if text else list(poi["prompt_tokens"])
    wc = bytes.fromhex(tmpl["default_witness_commitment"]) if "default_witness_commitment" in tmpl else None
    cb_legacy, cb_wit = build_coinbase(tmpl["height"], tmpl["coinbasevalue"], payout_spk,
                                       prompt_script(tokens), wc, extranonce, tmpl.get("devfee"))
    root = merkle_root([sha256d(cb_legacy)] + [bytes.fromhex(t["txid"])[::-1] for t in tmpl["transactions"]])
    target = int(tmpl["target"], 16)
    target_hex = tmpl["target"].lower().zfill(64)
    header76 = (struct.pack("<i", tmpl["version"]) + bytes.fromhex(tmpl["previousblockhash"])[::-1] + root
                + struct.pack("<I", max(tmpl["curtime"], tmpl["mintime"])) + bytes.fromhex(tmpl["bits"])[::-1])
    prev = tmpl["previousblockhash"]
    watcher.set_expected_tip(prev)
    t0, tries, nonce = time.perf_counter(), 0, 0
    last_status = t0
    found = None
    LOG.info("job_start payload=3 height=%s prev_hash=%s bits=%s target=%s model_hash=%s batch=1 lanes=1",
             tmpl["height"], prev, tmpl["bits"], target_hex, engine.model_hash)
    while found is None and nonce < (1 << 32):
        if watcher.is_stale(prev):
            LOG.warning("job_stale height=%s prev_hash=%s attempts=%d", tmpl["height"], prev, tries)
            return None, tries, time.perf_counter() - t0
        h80 = header76 + struct.pack("<I", nonce)
        seed = tagged_hash("POI/seed", h80 + model_hash)
        t_inf = time.perf_counter()
        x0, leaves = engine.attempt(seed, tokens)
        tries += 1
        if args.throttle > 0:
            time.sleep((time.perf_counter() - t_inf) * args.throttle)
        hpoi = vir_mine.poi_output_hash_v5(seed, x0, vir_mine.leaves_commitment(leaves))
        raw_hash = sha256d(h80 + hpoi)
        if int.from_bytes(raw_hash, "little") <= target:
            found = (h80, hpoi, x0, leaves, nonce, raw_hash[::-1].hex())
            break
        nonce += 1
        now = time.perf_counter()
        if args.verbose and now - last_status >= args.status_interval:
            rate = tries / max(now - t0, 1e-9)
            LOG.info("mining_progress payload=3 height=%s attempts=%d inf_per_second=%.3f nonce=%d "
                     "last_hash=%s target=%s", tmpl["height"], tries, rate, nonce - 1,
                     raw_hash[::-1].hex(), target_hex)
            last_status = now
    if found is None:
        return None, tries, time.perf_counter() - t0
    h80, hpoi, x0, leaves, nonce, bhash = found
    header = h80 + hpoi
    if rpc("getbestblockhash") != prev:
        LOG.warning("candidate_discarded reason=stale_tip height=%s candidate_hash=%s prev_hash=%s",
                    tmpl["height"], bhash, prev)
        return None, tries, time.perf_counter() - t0
    block = (header + ser_compact(1 + len(tmpl["transactions"])) + cb_wit
             + b"".join(bytes.fromhex(t["data"]) for t in tmpl["transactions"])
             + vir_mine.build_payload(x0, leaves))
    LOG.info("candidate_found payload=3 height=%s block_hash=%s hash_poi=%s target=%s nonce=%d "
             "x0_sha256=%s leaves=%d block_bytes=%d", tmpl["height"], bhash, hpoi[::-1].hex(), target_hex,
             nonce, hashlib.sha256(x0).hexdigest(), len(leaves), len(block))
    res = rpc("submitblock", block.hex())
    LOG.info("submit_result payload=3 height=%s block_hash=%s result=%s",
             tmpl["height"], bhash, "accepted_by_rpc" if res is None else str(res))
    return (bhash, tmpl["height"], res, nonce, f"{len(leaves)} leaves; target={target_hex}"), tries, time.perf_counter() - t0


def mine_one(rpc, miners, args, payout_spk, extranonce, watcher, tmpl=None, stratum_client=None):
    """PoI v2 CUDA mining via node JSON-RPC or a named-params Stratum job."""
    stratum_mode = stratum_client is not None
    tmpl = tmpl or rpc("getblocktemplate", {"rules": ["segwit"]})
    poi = tmpl["poi"]
    if poi["version"] != sdint.VERSION:
        raise SystemExit(f"node expects PoI v{poi['version']} at height {tmpl['height']}; this miner implements v{sdint.VERSION}")
    if not isinstance(miners, (list, tuple)):
        miners = [miners]
    if not miners:
        raise SystemExit("no V2 mining devices were initialized")
    primary = miners[0]
    node_model_hash = poi["model_hash"].lower()
    local_model_hash = primary.model_hash[::-1].hex()
    if node_model_hash != local_model_hash:
        LOG.critical("model_hash_mismatch height=%s node_model_hash=%s local_model_hash=%s",
                     tmpl["height"], node_model_hash, local_model_hash)
        raise SystemExit("model hash differs from the node's; refusing to mine")
    if any(m.model_hash != primary.model_hash for m in miners):
        raise SystemExit("GPU workers loaded different model hashes")

    if stratum_mode:
        # The pool commits the exact token IDs in its notify; do not re-tokenize or
        # substitute a locally selected prompt, otherwise the PoI share is invalid.
        tokens = list(poi["prompt_tokens"])
    else:
        text = pick_prompt(args, tmpl["height"])
        tokens = primary.tok.encode(text) if text else list(poi["prompt_tokens"])
    for miner in miners:
        miner.prepare(tokens)
        if args.auto_batch:
            if not getattr(miner, "_batch_tuned", False):
                auto_tune_batch(miner, tokens, args.max_batch)
                miner._batch_tuned = True
        else:
            miner.batch = args.batch
            if miner.fast and getattr(miner, "_validated_batch", None) != miner.batch:
                check_seeds = [hashlib.sha256(b"vela-fixed-batch-check" + i.to_bytes(4, "little")).digest()
                               for i in range(miner.batch)]
                for lane_id in range(len(miner.streams)):
                    try:
                        check_outputs = miner.x0s(check_seeds, tokens, stream_id=lane_id)
                    except Exception as exc:
                        if "out of memory" in str(exc).lower():
                            gc.collect()
                            if miner.is_cuda:
                                with torch.cuda.device(miner.device_obj):
                                    torch.cuda.empty_cache()
                            raise SystemExit(
                                f"requested --batch {miner.batch} does not fit on {miner.device}; "
                                "reduce --batch or use --auto-batch"
                            ) from exc
                        raise
                    for sample_index, candidate in enumerate(check_outputs):
                        reference = sdint.run(miner.net, check_seeds[sample_index], tokens, miner._ctx)
                        if candidate != reference:
                            raise ConsensusError(f"fixed batch failed exactness: device={miner.device} lane={lane_id} sample={sample_index}")
                miner._validated_batch = miner.batch
                LOG.info("fixed_batch_integrity=PASS device=%s lanes=%d batch_per_lane=%d samples_per_lane=%d",
                         miner.device, len(miner.streams), miner.batch, len(check_seeds))

    if stratum_mode:
        # Rebuild the exact pool-issued coinbase, Merkle root and block-header prefix.
        header76, root, target = build_stratum_header76(tmpl)
        target_hex = target.to_bytes(32, "big").hex()
        prev = tmpl["prev_hash"]
    else:
        wc = bytes.fromhex(tmpl["default_witness_commitment"]) if "default_witness_commitment" in tmpl else None
        cb_legacy, cb_wit = build_coinbase(tmpl["height"], tmpl["coinbasevalue"], payout_spk,
                                           prompt_script(tokens), wc, extranonce, tmpl.get("devfee"))
        root = merkle_root([sha256d(cb_legacy)] + [bytes.fromhex(t["txid"])[::-1] for t in tmpl["transactions"]])
        target = int(tmpl["target"], 16)
        target_hex = tmpl["target"].lower().zfill(64)
        header76 = (struct.pack("<i", tmpl["version"]) + bytes.fromhex(tmpl["previousblockhash"])[::-1] + root
                    + struct.pack("<I", max(tmpl["curtime"], tmpl["mintime"])) + bytes.fromhex(tmpl["bits"])[::-1])
        prev = tmpl["previousblockhash"]
    watcher.set_expected_tip(prev)

    # Each lane owns a disjoint interleaved nonce interval. Different GPUs and
    # streams never evaluate the same nonce, even when their batch sizes differ.
    lane_specs, nonce_stride = build_lane_specs(miners)

    model_hash = primary.model_hash
    devices_text = ",".join(m.device for m in miners)
    LOG.info("job_start payload=2 height=%s prev_hash=%s bits=%s target=%s model_hash=%s "
             "devices=%s lanes_per_gpu=%d batches=%s merkle_root=%s prompt_tokens=%d extranonce=%d",
             tmpl["height"], prev, tmpl["bits"], target_hex, model_hash[::-1].hex(), devices_text,
             args.lanes, ",".join(str(m.batch) for m in miners), root[::-1].hex(), len(tokens), extranonce)

    stop_event = threading.Event()
    stale_event = threading.Event()
    found_box = {"candidate": None}
    candidate_lock = threading.Lock()
    group_specs = {miner.device: [spec for spec in lane_specs if spec["miner"] is miner]
                   for miner in miners}
    states = {miner.device: {"tries": 0, "last_nonce": -1, "last_hash": "-", "last_hash_poi": "-",
                             "last_updated": 0.0}
              for miner in miners}

    def run_lane(spec, nonces, headers, seeds, start_barrier=None):
        # FFGV-inspired host-side launch barrier: dispatch sibling streams of a
        # GPU together, while retaining independent scheduling across GPUs.
        if start_barrier is not None:
            try:
                start_barrier.wait(timeout=10.0)
            except threading.BrokenBarrierError as exc:
                raise RuntimeError("GPU lane launch barrier broke before batch dispatch") from exc
        t_inf = time.perf_counter()
        outputs = spec["miner"].x0s(seeds, tokens, stream_id=spec["stream_id"])
        if args.throttle > 0:
            time.sleep((time.perf_counter() - t_inf) * args.throttle)
        return nonces, headers, seeds, outputs

    def run_gpu_group(miner, specs):
        """One independent scheduler per GPU; streams on a card share a round barrier."""
        state = states[miner.device]
        last_canary = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(specs)), thread_name_prefix=f"vela-{miner.device.replace(':', '-')}-lane"
            ) as lane_executor:
                while not stop_event.is_set():
                    if watcher.is_stale(prev):
                        stale_event.set()
                        stop_event.set()
                        return
                    future_map = {}
                    active_specs = [spec for spec in specs if spec["next_nonce"] <= 0xFFFFFFFF]
                    if not active_specs:
                        return
                    start_barrier = threading.Barrier(len(active_specs)) if len(active_specs) > 1 else None
                    for spec in active_specs:
                        start_nonce = spec["next_nonce"]
                        count = min(spec["batch"], 0x100000000 - start_nonce)
                        nonces = list(range(start_nonce, start_nonce + count))
                        headers = [header76 + struct.pack("<I", n) for n in nonces]
                        seeds = [tagged_hash("POI/seed", header + model_hash) for header in headers]
                        future = lane_executor.submit(run_lane, spec, nonces, headers, seeds, start_barrier)
                        future_map[future] = spec

                    completed = []
                    for future in concurrent.futures.as_completed(future_map):
                        spec = future_map[future]
                        nonces, headers, seeds, outputs = future.result()
                        completed.append((spec, nonces, headers, seeds, outputs))
                    completed.sort(key=lambda item: item[0]["stream_id"])

                    # Count all work already completed, even when another GPU
                    # found a candidate while this card was finishing its round.
                    state["tries"] += sum(len(item[4]) for item in completed)
                    round_last_nonce = max((item[1][-1] for item in completed if item[1]), default=-1)
                    if round_last_nonce >= 0:
                        state["last_nonce"] = round_last_nonce
                    if stop_event.is_set():
                        return
                    if watcher.is_stale(prev):
                        stale_event.set()
                        stop_event.set()
                        return

                    for spec, nonces, headers, seeds, outputs in completed:
                        for nonce, header, seed, x0 in zip(nonces, headers, seeds, outputs):
                            hpoi = tagged_hash("POI/out", seed + x0)
                            raw_hash = sha256d(header + hpoi)
                            state["last_hash"] = raw_hash[::-1].hex()
                            state["last_hash_poi"] = hpoi[::-1].hex()
                            state["last_updated"] = time.monotonic()
                            if int.from_bytes(raw_hash, "little") <= target:
                                candidate = (header, hpoi, x0, nonce, raw_hash[::-1].hex())
                                with candidate_lock:
                                    if found_box["candidate"] is None:
                                        found_box["candidate"] = candidate
                                        stop_event.set()
                                return
                        spec["next_nonce"] += nonce_stride

                    now = time.perf_counter()
                    if args.canary_interval > 0 and now - last_canary >= args.canary_interval:
                        LOG.info("consensus_canary_round_start device=%s lanes=%d interval_seconds=%d",
                                 miner.device, len(miner.streams), args.canary_interval)
                        miner.check_canaries(tokens)
                        LOG.info("consensus_canary_round=PASS device=%s lanes=%d",
                                 miner.device, len(miner.streams))
                        last_canary = time.perf_counter()
        except BaseException:
            stop_event.set()
            raise

    t0 = time.perf_counter()
    last_status = t0
    last_status_tries = {device: 0 for device in states}
    last_status_at = t0
    gpu_futures = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(miners), thread_name_prefix="vela-gpu-scheduler"
    ) as gpu_executor:
        for miner in miners:
            gpu_futures[gpu_executor.submit(run_gpu_group, miner, group_specs[miner.device])] = miner.device
        pending = set(gpu_futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=0.25, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    future.result()
                except BaseException:
                    stop_event.set()
                    raise
            if not stop_event.is_set() and watcher.is_stale(prev):
                stale_event.set()
                stop_event.set()
            now = time.perf_counter()
            if now - last_status >= args.status_interval:
                tries = sum(state["tries"] for state in states.values())
                elapsed = max(now - t0, 1e-9)
                interval = max(now - last_status_at, 1e-9)
                rate = tries / elapsed
                latest_state = max(states.values(), key=lambda state: state["last_updated"])
                last_nonce = latest_state["last_nonce"]
                last_hash = latest_state["last_hash"]
                last_hpoi = latest_state["last_hash_poi"]
                per_gpu_parts = []
                for device, state in states.items():
                    gpu_rate = max(0, state["tries"] - last_status_tries.get(device, 0)) / interval
                    per_gpu_parts.append(
                        f"{device}:n={state['tries']}:nonce={state['last_nonce']}:hash={state['last_hash']}:"
                        f"hash_poi={state['last_hash_poi']}:inf_s={gpu_rate:.3f}"
                    )
                    last_status_tries[device] = state["tries"]
                per_gpu = ";".join(per_gpu_parts)
                LOG.info("mining_progress payload=2 height=%s attempts=%d inf_per_second=%.3f elapsed=%.1f "
                         "last_nonce=%d last_hash=%s last_hash_poi=%s target=%s per_gpu=%s",
                         tmpl["height"], tries, rate, elapsed, last_nonce, last_hash, last_hpoi,
                         target_hex, per_gpu)
                last_status = now
                last_status_at = now

    tries = sum(state["tries"] for state in states.values())
    if stale_event.is_set() or watcher.is_stale(prev):
        LOG.warning("job_stale height=%s prev_hash=%s attempts=%d", tmpl["height"], prev, tries)
        return None, tries, time.perf_counter() - t0
    found = found_box["candidate"]
    if found is None:
        LOG.warning("nonce_space_exhausted height=%s attempts=%d", tmpl["height"], tries)
        return None, tries, time.perf_counter() - t0

    header, hpoi, x0, nonce, bhash = found
    # FFGV-style last-line integrity guard: recompute a found x0 with the
    # independent specification implementation before a network submission.
    candidate_seed = tagged_hash("POI/seed", header + model_hash)
    reference_x0 = sdint.run(primary.net, candidate_seed, tokens, primary._ctx)
    if reference_x0 != x0:
        LOG.critical("candidate_rejected reason=reference_mismatch height=%s nonce=%d gpu_x0_sha256=%s "
                     "reference_x0_sha256=%s; block_not_submitted",
                     tmpl["height"], nonce, hashlib.sha256(x0).hexdigest(),
                     hashlib.sha256(reference_x0).hexdigest())
        raise ConsensusError("candidate x0 differs from the PoI v2 reference implementation")
    reference_hpoi = tagged_hash("POI/out", candidate_seed + reference_x0)
    if reference_hpoi != hpoi:
        LOG.critical("candidate_rejected reason=hash_poi_mismatch height=%s nonce=%d gpu_hash_poi=%s "
                     "reference_hash_poi=%s; block_not_submitted",
                     tmpl["height"], nonce, hpoi[::-1].hex(), reference_hpoi[::-1].hex())
        raise ConsensusError("candidate hashPoI differs from the recomputed reference")
    checked_hash = sha256d(header + reference_hpoi)
    if checked_hash[::-1].hex() != bhash or int.from_bytes(checked_hash, "little") > target:
        LOG.critical("candidate_rejected reason=header_hash_or_target_check_failed height=%s nonce=%d "
                     "calculated_hash=%s logged_hash=%s target=%s; block_not_submitted",
                     tmpl["height"], nonce, checked_hash[::-1].hex(), bhash, target_hex)
        raise ConsensusError("candidate block hash failed its final target check")
    LOG.info("candidate_reference_check=PASS height=%s nonce=%d x0_sha256=%s hash_poi=%s block_hash=%s target=%s",
             tmpl["height"], nonce, hashlib.sha256(reference_x0).hexdigest(),
             reference_hpoi[::-1].hex(), bhash, target_hex)
    # Never submit a result built on a parent that is no longer the active tip.
    if stratum_mode:
        # A share is bound to the connection and job that generated it. The client
        # checks both again immediately before writing the submit request.
        if watcher.is_stale(prev):
            LOG.warning("candidate_discarded reason=stratum_job_changed height=%s candidate_hash=%s",
                        tmpl["height"], bhash)
            return None, tries, time.perf_counter() - t0
        nonce_hex = f"{nonce:08x}"
        sent = stratum_client.submit_share(
            tmpl, nonce_hex, reference_x0.hex(), reference_hpoi.hex()
        )
        LOG.info("stratum_candidate height=%s block_hash=%s nonce=%s hash_poi=%s target=%s submitted=%s",
                 tmpl["height"], bhash, nonce_hex, reference_hpoi[::-1].hex(), target_hex, sent)
        info = (f"share_target={target_hex}; difficulty={tmpl['difficulty']}; "
                f"devices={devices_text}; lanes={args.lanes}; "
                f"batch={','.join(str(m.batch) for m in miners)}")
        return (bhash, tmpl["height"], None if sent else "discarded", nonce, info), tries, time.perf_counter() - t0

    current_tip = rpc("getbestblockhash")
    if current_tip != prev:
        LOG.warning("candidate_discarded reason=stale_tip height=%s candidate_hash=%s prev_hash=%s current_tip=%s",
                    tmpl["height"], bhash, prev, current_tip)
        return None, tries, time.perf_counter() - t0

    block = serialize_v2_block(header, reference_hpoi, cb_wit, tmpl["transactions"], x0)
    header112 = block[:112]
    if len(header112) != 112 or header112[:80] != header or header112[80:] != reference_hpoi:
        raise ConsensusError("serialized V2 block header is not the expected 80-byte header + hashPoI")
    LOG.info("candidate_found payload=2 height=%s block_hash=%s hash_poi=%s target=%s nonce=%d "
             "header_bytes=%d x0_sha256=%s block_bytes=%d attempts=%d", tmpl["height"], bhash,
             reference_hpoi[::-1].hex(), target_hex, nonce, len(header112), hashlib.sha256(x0).hexdigest(),
             len(block), tries)
    res = rpc("submitblock", block.hex())
    LOG.info("submit_result payload=2 height=%s block_hash=%s result=%s",
             tmpl["height"], bhash, "accepted_by_rpc" if res is None else str(res))
    info = f"target={target_hex}; devices={devices_text}; lanes={args.lanes}; batch={','.join(str(m.batch) for m in miners)}"
    return (bhash, tmpl["height"], res, nonce, info), tries, time.perf_counter() - t0


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpc", default="http://u:p@127.0.0.1:18743")
    ap.add_argument("--stratum-url", help="Vela named-params Stratum endpoint, e.g. vela.suprnova.cc:7601")
    ap.add_argument("--wallet", help="Stratum username: VELA address optionally followed by .worker")
    ap.add_argument("--password", default="x", help="Stratum password (SuprNova currently ignores it)")
    ap.add_argument("--cookie", help="path to the node's .cookie file (instead of user:password in --rpc)")
    ap.add_argument("--model", help="*.poisd model file (payload v2; required unless --model3 only)")
    ap.add_argument("--model-sha256", default=V2_MODEL_FILE_SHA256,
                    help="expected whole-file SHA-256 for the V2 model (default: official model hash)")
    ap.add_argument("--model3", metavar="FILE", help="Fork-M .vir model (payload v3 from the fork height)")
    ap.add_argument("--engine-dir", default=os.environ.get("VIR_GPU_DIR", ""),
                    help="GPU VIR engine checkout (vir_torch.py + noderoot.py) for the v3 path; "
                         "defaults to the copy shipped in contrib/vir, or VIR_GPU_DIR if set")
    ap.add_argument("--address", help="payout address; required on public networks (OP_TRUE only on regtest)")
    ap.add_argument("--prompt", help="fix the prompt for every block (default: rotate the list per block)")
    ap.add_argument("--prompts-file", help="file with one prompt per line, replacing the built-in ten "
                                          "(blank lines and # comments are ignored)")
    ap.add_argument("--prompt-index", type=int, metavar="N",
                    help="pin prompt N (0-based) from the list instead of rotating by height")
    ap.add_argument("--node-prompt", action="store_true",
                    help="use the node's -poiprompt instead of the local list")
    ap.add_argument("--blocks", type=int, default=1, help="blocks to mine (0 = forever)")
    ap.add_argument("--device", help="one device, e.g. cuda:0 or cpu (overrides --devices)")
    ap.add_argument("--devices", default="auto",
                    help="V2 CUDA devices: auto = all visible GPUs, or a list such as 0,1,2,3 (default: auto)")
    ap.add_argument("--engine", choices=["fast", "ref"], default="fast")
    ap.add_argument("--batch", type=int, default=1, help="nonces per lane per GPU pass (default: 1)")
    ap.add_argument("--auto-batch", action="store_true",
                    help="benchmark 1/2/4/8 batches per GPU, verify against the reference, and select the fastest safe size")
    ap.add_argument("--max-batch", type=int, choices=[1, 2, 4, 8], default=8,
                    help="largest batch probed by --auto-batch (default: 8)")
    ap.add_argument("--lanes", type=int, choices=[1, 2], default=1,
                    help="independent CUDA streams per GPU; 2 is experimental and uses extra VRAM (default: 1)")
    ap.add_argument("--bench", type=int, metavar="N", help="offline: time N inferences per selected GPU")
    ap.add_argument("--status-interval", type=int, default=30, metavar="SEC",
                    help="mining progress log interval (default: 30 seconds)")
    ap.add_argument("--tip-poll", type=float, default=0.5, metavar="SEC",
                    help="background getbestblockhash polling interval (default: 0.5 seconds)")
    ap.add_argument("--canary-interval", type=int, default=120, metavar="SEC",
                    help="repeat the known consensus vector on every active GPU lane (0 disables; default: 120 seconds)")
    ap.add_argument("--log-file", default="logs/vela-miner.log",
                    help="rotating log path; use 'none' to disable file logging (default: logs/vela-miner.log)")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    ap.add_argument("--verbose", action="store_true",
                    help="extra v3 progress output (v2 always logs at --status-interval)")
    ap.add_argument("--throttle", type=float, default=0.0, metavar="F",
                    help="idle F x inference time after each batch to keep a display GPU responsive")
    ap.add_argument("--low-priority", action="store_true", help="run the miner process at below-normal CPU priority")
    args = ap.parse_args()

    setup_logging(args.log_file, args.log_level)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigterm)
    if args.blocks < 0:
        raise SystemExit("--blocks must be >= 0")
    if args.batch < 1 or args.batch > 16:
        raise SystemExit("--batch must be in [1, 16]")
    if args.status_interval < 5 or args.status_interval > 3600:
        raise SystemExit("--status-interval must be in [5, 3600]")
    if args.tip_poll < 0.25 or args.tip_poll > 60:
        raise SystemExit("--tip-poll must be in [0.25, 60] seconds")
    if args.canary_interval != 0 and not 30 <= args.canary_interval <= 3600:
        raise SystemExit("--canary-interval must be 0 or in [30, 3600] seconds")
    if args.throttle < 0:
        raise SystemExit("--throttle must be >= 0")
    if not args.model and not args.model3:
        raise SystemExit("give --model (v2 .poisd) and/or --model3 (Fork-M .vir)")
    args.prompts = load_prompts(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS

    if args.low_priority:
        try:
            if os.name == "nt":
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
            else:
                os.nice(10)
        except Exception as exc:
            LOG.warning("low_priority_unavailable error=%s", exc)

    v2_devices = resolve_devices(args) if args.model else []
    if args.model and args.engine == "ref" and (len(v2_devices) != 1 or args.lanes != 1):
        raise SystemExit("--engine ref supports one CPU worker; use --lanes 1")
    if args.model and args.lanes > 1 and v2_devices == ["cpu"]:
        raise SystemExit("--lanes applies to CUDA streams only; use --lanes 1 for the CPU reference")
    if args.model and not args.model3:
        LOG.info("v2_devices=%s lanes_per_gpu=%d batch=%s", ",".join(v2_devices), args.lanes,
                 "auto" if args.auto_batch else args.batch)

    def get_v2():
        if get_v2.engines is None:
            if not args.model:
                raise SystemExit("the node asks for payload v2 but no --model (.poisd) was given")
            verify_v2_model_file(args.model, args.model_sha256)
            LOG.info("loading_model_once_for_devices count=%d devices=%s", len(v2_devices), ",".join(v2_devices))
            tensors, model_hash = sdint.read_model(args.model)
            get_v2.model_data = (tensors, model_hash)
            engines = []
            for device in v2_devices:
                lane_count = args.lanes if torch.device(device).type == "cuda" else 1
                engines.append(Miner(args.model, device, args.engine, lanes=lane_count,
                                     model_data=get_v2.model_data))
            get_v2.engines = engines
        return get_v2.engines
    get_v2.engines = None
    get_v2.model_data = None

    def get_v3():
        if get_v3.engine is None:
            if not args.model3:
                raise SystemExit("the node asks for payload v3 (Fork-M) but no --model3 (.vir) was given")
            if args.devices != "auto" or args.device:
                LOG.warning("payload_v3_gpu_selection_is_managed_by_vir_engine; --devices/--device only applies to V2")
            get_v3.engine = vir_mine.GpuEngine(args.model3, args.engine_dir)
        return get_v3.engine
    get_v3.engine = None

    if args.bench is not None:
        if args.bench < 1:
            raise SystemExit("--bench must be >= 1")
        if args.model3 and not args.model:
            toks = get_v3().encode(args.prompt) if args.prompt else []
            t0 = time.perf_counter()
            for _ in range(args.bench):
                get_v3().attempt(os.urandom(32), toks)
            dt = time.perf_counter() - t0
            LOG.info("bench payload=3 inferences=%d seconds=%.3f seconds_per_inference=%.3f",
                     args.bench, dt, dt / args.bench)
            return
        miners = get_v2()
        toks = miners[0].tok.encode(args.prompt or "benchmark prompt")
        for miner in miners:
            miner.prepare(toks)
            if args.auto_batch:
                auto_tune_batch(miner, toks, args.max_batch)
            else:
                miner.batch = args.batch
        def bench_device(miner):
            total = 0
            lane_count = len(miner.streams) if miner.fast else 1
            t0 = time.perf_counter()
            while total < args.bench:
                tasks = []
                for lane_id, count in make_bench_tasks(args.bench - total, miner.batch, lane_count):
                    seeds = [os.urandom(32) for _ in range(count)]
                    tasks.append((lane_id, count, seeds))
                if lane_count > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=lane_count) as lane_executor:
                        futures = [lane_executor.submit(miner.x0s, seeds, toks, lane_id)
                                   for lane_id, _, seeds in tasks]
                        for future in futures:
                            future.result()
                else:
                    for lane_id, _, seeds in tasks:
                        miner.x0s(seeds, toks, stream_id=lane_id)
                total += sum(count for _, count, _ in tasks)
            elapsed = time.perf_counter() - t0
            return miner.device, total / max(elapsed, 1e-9), elapsed
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(miners)) as executor:
            results = list(executor.map(bench_device, miners))
        for device, rate, elapsed in results:
            LOG.info("bench payload=2 device=%s inferences=%d seconds=%.3f inf_per_second=%.3f",
                     device, args.bench, elapsed, rate)
        LOG.info("bench_aggregate devices=%d inf_per_second=%.3f", len(results), sum(r[1] for r in results))
        return

    if args.stratum_url:
        if args.model3:
            raise SystemExit("Stratum mode currently supports PoI payload v2 only; --model3 is not supported")
        if not args.model:
            raise SystemExit("--model (.poisd) is required for Stratum mode")
        if not args.wallet:
            raise SystemExit("--wallet is required for Stratum mode")
        client = StratumClient(args.stratum_url, args.wallet, args.password)
        mined = 0
        client.start()
        try:
            LOG.info("stratum_mode_enabled endpoint=%s wallet=%s", args.stratum_url, args.wallet)
            while args.blocks == 0 or mined < args.blocks:
                job = client.current_job()
                if job is None:
                    time.sleep(0.25)
                    continue
                if job["poi_version"] != sdint.VERSION:
                    raise SystemExit(f"pool job requires PoI payload v{job['poi_version']}; this runtime implements v{sdint.VERSION}")
                engines = get_v2()
                local_hash = engines[0].model_hash[::-1].hex()
                if job["model_hash"] and job["model_hash"] != local_hash:
                    raise SystemExit(f"pool model hash mismatch: job={job['model_hash']} local={local_hash}; refusing to mine")
                # A fresh 4-byte extranonce2 changes the coinbase root each pass,
                # so restarting the 32-bit nonce scan does not duplicate inference work.
                import secrets
                work_job = dict(job)
                work_job["extranonce2"] = secrets.token_hex(4)
                watcher = StratumJobWatcher(client, job)
                LOG.info("stratum_job_selected height=%s id=%s generation=%s difficulty=%s extranonce2=%s",
                         job["height"], job["job_id"], job["generation"], job["difficulty"],
                         work_job["extranonce2"])
                stratum_template = {**work_job, "poi": {
                    "version": job["poi_version"],
                    "model_hash": job["model_hash"] or local_hash,
                    "prompt_tokens": job["prompt_tokens"],
                }}
                result_tuple = mine_one(None, engines, args, None, 0, watcher,
                                        tmpl=stratum_template, stratum_client=client)
                if result_tuple is None:
                    continue
                result, tries, elapsed = result_tuple
                if result is None:
                    LOG.info("stratum_job_refresh height=%s attempts=%d elapsed_seconds=%.2f",
                             job["height"], tries, elapsed)
                    continue
                _, height, submit_result, nonce, info = result
                if submit_result is None:
                    mined += 1
                    LOG.info("stratum_share_cycle_complete height=%s nonce=%d attempts=%d elapsed_seconds=%.2f shares_found=%d info=%s",
                             height, nonce, tries, elapsed, mined, info)
                else:
                    LOG.warning("stratum_share_not_sent height=%s nonce=%d reason=%s", height, nonce, submit_result)
        except KeyboardInterrupt:
            LOG.warning("shutdown_requested signal=KeyboardInterrupt")
        finally:
            client.stop()
            LOG.info("stratum_miner_stopped accepted=%d rejected=%d stale=%d submitted=%d discarded=%d",
                     client.accepted, client.rejected, client.stale, client.submitted, client.discarded)
        return

    rpc = RPC(args.rpc, args.cookie)
    chain_info = rpc("getblockchaininfo")
    chain_name = chain_info.get("chain", "unknown")
    if not args.address and chain_name != "regtest":
        raise SystemExit("--address is required outside regtest; refusing an OP_TRUE payout script")
    if args.address:
        address_info = rpc("validateaddress", args.address)
        if address_info.get("isvalid") is False or not address_info.get("scriptPubKey"):
            raise SystemExit(f"invalid or unsupported payout address: {args.address}")
        payout_spk = bytes.fromhex(address_info["scriptPubKey"])
    else:
        payout_spk = b"\x51"
    LOG.info("miner_start chain=%s rpc_endpoint=%s payloads=%s blocks=%s cuda_available=%s visible_gpus=%d",
             chain_name, args.rpc.rsplit("@", 1)[-1], "v2+v3" if args.model and args.model3 else
             "v2" if args.model else "v3", args.blocks, torch.cuda.is_available(), torch.cuda.device_count())
    LOG.info("runtime_safety tip_poll_seconds=%.2f canary_interval_seconds=%d final_share_reference_check=enabled",
             args.tip_poll, args.canary_interval)

    watcher_rpc_timeout = max(2.0, args.tip_poll * 4.0)
    watcher = TipWatcher(RPC(args.rpc, args.cookie, timeout=watcher_rpc_timeout), args.tip_poll)
    watcher.start()
    mined = 0
    extranonce = int.from_bytes(os.urandom(4), "little") << 32
    try:
        while args.blocks == 0 or mined < args.blocks:
            extranonce += 1
            try:
                # Fetch exactly one template; pass the same immutable job to the selected engine.
                tmpl = rpc("getblocktemplate", {"rules": ["segwit"]})
                version = tmpl["poi"]["version"]
                if version == vir_mine.PAYLOAD_VERSION:
                    result_tuple = mine_one_v3(rpc, get_v3(), args, payout_spk, extranonce, watcher, tmpl=tmpl)
                elif version == sdint.VERSION:
                    result_tuple = mine_one(rpc, get_v2(), args, payout_spk, extranonce, watcher, tmpl=tmpl)
                else:
                    raise SystemExit(f"node requests unsupported PoI payload version {version}")
            except ConsensusError as exc:
                LOG.critical("fatal_consensus_integrity_error=%s mining_stopped_without_submit", exc)
                raise SystemExit(3) from exc
            except FileNotFoundError as exc:
                LOG.critical("required_local_file_missing=%s", exc)
                raise SystemExit(2) from exc
            except (OSError, RuntimeError) as exc:
                LOG.exception("rpc_or_mining_error retry_in_seconds=10 error=%s", exc)
                time.sleep(10)
                continue
            if result_tuple is None:
                LOG.info("job_refresh reason=tip_changed")
                continue
            result, tries, elapsed = result_tuple
            if result is None:
                LOG.info("job_refresh reason=no_candidate attempts=%d elapsed_seconds=%.2f", tries, elapsed)
                continue
            block_hash, height, submit_result, nonce, info = result
            if submit_result is None:
                active_tip = rpc("getbestblockhash")
                if active_tip != block_hash:
                    LOG.warning("block_submitted_but_not_active height=%s block_hash=%s active_tip=%s",
                                height, block_hash, active_tip)
                    continue
                mined += 1
                LOG.info("block_accepted height=%s block_hash=%s nonce=%d target_info=%s total_accepted=%d",
                         height, block_hash, nonce, info, mined)
            else:
                LOG.error("block_rejected height=%s block_hash=%s nonce=%d reason=%s info=%s",
                          height, block_hash, nonce, submit_result, info)
    except KeyboardInterrupt:
        LOG.warning("shutdown_requested signal=KeyboardInterrupt")
    finally:
        watcher.stop()
        LOG.info("miner_stopped accepted_blocks=%d", mined)


if __name__ == "__main__":
    main()
