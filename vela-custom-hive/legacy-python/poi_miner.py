#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""External multi-process Proof-of-Inference miner for Vela.

  getblocktemplate -> build coinbase (BIP34 height, payout, POI1 prompt,
  witness commitment) -> N worker processes grind disjoint nonce ranges:
      seed    = TaggedHash("POI/seed")(header80 || model_hash)
      x0      = Model_v1(prompt, ChaCha20(seed))        (numpy, bit-exact)
      hashPoI = TaggedHash("POI/out")(seed || x0)
      accept if SHA256d(header80 || hashPoI) <= target
  -> submitblock.

Every nonce costs one inference: that is the "useful work" of the chain.

Fork-M (payload v3): pass --model3 <file>.vir instead of using the built-in v1
model; the miner then signs each attempt with the leaf commitments
(hashPoI = POI/out5(seed || x0 || R), payload {u8 3, x0, leaves}) from vir.py.
That path is single-process and slow (about ten minutes per sd-turbo inference):
the GPU miner is sd_miner.py (v2 and v3).

Usage:
  poi_miner.py --rpc http://u:p@127.0.0.1:18743 --address <addr> [--prompt TEXT]
               [--workers 4] [--blocks 10] [--selftest]
  poi_miner.py --rpc ... --cookie <datadir>/regtest/.cookie --model3 model.vir
"""

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
import struct
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

# ----------------------------------------------------------------------------
# Toy model v1 (must stay bit-identical to src/poi/poi.cpp)
# ----------------------------------------------------------------------------
POI_VERSION = 1
C, H, W = 4, 16, 16
N = H * W
HIDDEN = 16
STEPS = 2
X0_SIZE = C * N
VOCAB = 4096
MAX_TOKENS = 300
MAGIC = b"POI1"


def sha256(b):
    return hashlib.sha256(b).digest()


def sha256d(b):
    return sha256(sha256(b))


def tagged_hash(tag, data):
    t = sha256(tag.encode())
    return sha256(t + t + data)


def _rotl(v, c):
    return ((v << np.uint32(c)) | (v >> np.uint32(32 - c))).astype(np.uint32)


def chacha20_keystream(key, nbytes):
    """RFC 8439 ChaCha20 keystream, nonce = 0, counter starting at 0 (vectorised over blocks)."""
    nblocks = (nbytes + 63) // 64
    const = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], dtype=np.uint32)
    k = np.frombuffer(key, dtype="<u4").astype(np.uint32)
    st = np.zeros((16, nblocks), dtype=np.uint32)
    st[0:4] = const[:, None]
    st[4:12] = k[:, None]
    st[12] = np.arange(nblocks, dtype=np.uint32)
    x = st.copy()

    def qr(a, b, c, d):
        x[a] += x[b]; x[d] ^= x[a]; x[d] = _rotl(x[d], 16)
        x[c] += x[d]; x[b] ^= x[c]; x[b] = _rotl(x[b], 12)
        x[a] += x[b]; x[d] ^= x[a]; x[d] = _rotl(x[d], 8)
        x[c] += x[d]; x[b] ^= x[c]; x[b] = _rotl(x[b], 7)

    with np.errstate(over="ignore"):
        for _ in range(10):
            qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
            qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
        x += st
    return x.T.astype("<u4").tobytes()[:nbytes]


class Model:
    def __init__(self):
        key = sha256(b"PoIChain toy model v1")
        n = VOCAB * HIDDEN + STEPS * (HIDDEN * C * 2 + HIDDEN * HIDDEN + C * HIDDEN)
        ks = chacha20_keystream(key, n)
        w = np.frombuffer(ks, dtype=np.int8).astype(np.int64)
        pos = VOCAB * HIDDEN
        self.tok = w[:pos].reshape(VOCAB, HIDDEN)
        self.w1x, self.w1n, self.w1e, self.w2 = [], [], [], []
        for _ in range(STEPS):
            self.w1x.append(w[pos:pos + HIDDEN * C].reshape(HIDDEN, C)); pos += HIDDEN * C
            self.w1n.append(w[pos:pos + HIDDEN * C].reshape(HIDDEN, C)); pos += HIDDEN * C
            self.w1e.append(w[pos:pos + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN)); pos += HIDDEN * HIDDEN
            self.w2.append(w[pos:pos + C * HIDDEN].reshape(C, HIDDEN)); pos += C * HIDDEN
        self.hash = tagged_hash("POI/model", bytes([POI_VERSION]) + ks)

    def embed(self, tokens):
        t = np.array(tokens, dtype=np.int64) % VOCAB
        mult = (np.arange(len(tokens)) % 7 + 1)[:, None]
        acc = (self.tok[t] * mult).sum(axis=0)
        return np.clip(acc >> 4, -128, 127)

    def run(self, seed, emb):
        ks = np.frombuffer(chacha20_keystream(seed, 4 * X0_SIZE), dtype=np.uint8).astype(np.int64).reshape(X0_SIZE, 4)
        x = np.clip((ks.sum(axis=1) - 510) >> 2, -128, 127).reshape(C, H, W)
        for s in range(STEPS):
            eterm = self.w1e[s] @ emb                                    # [HIDDEN]
            nb = np.roll(x, 1, 2) + np.roll(x, -1, 2) + np.roll(x, 1, 1) + np.roll(x, -1, 1)
            acc = (np.einsum("jc,chw->jhw", self.w1x[s], x) + np.einsum("jc,chw->jhw", self.w1n[s], nb)
                   + eterm[:, None, None])
            a = np.maximum(np.clip(acc >> 10, -128, 127), 0)
            out = np.einsum("cj,jhw->chw", self.w2[s], a)
            x = np.clip(x + np.clip(out >> 9, -128, 127), -128, 127)
        return (x.reshape(-1) & 0xFF).astype(np.uint8).tobytes()


def tokenize(text):
    return [b + 1 for b in text.encode()[:MAX_TOKENS]] or [1]


# ----------------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------------
def ser_compact(n):
    if n < 253:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    if n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def push_data(data):
    n = len(data)
    if n < 0x4C:
        return bytes([n]) + data
    if n <= 0xFF:
        return b"\x4c" + bytes([n]) + data
    return b"\x4d" + struct.pack("<H", n) + data


def script_num(n):
    if n == 0:
        return b"\x00"
    if 1 <= n <= 16:
        return bytes([0x50 + n])
    out = bytearray()
    v = n
    while v:
        out.append(v & 0xFF)
        v >>= 8
    if out[-1] & 0x80:
        out.append(0)
    return push_data(bytes(out))


def prompt_script(tokens):
    return b"\x6a" + push_data(MAGIC + b"".join(struct.pack("<H", t) for t in tokens))


def build_coinbase(height, value, payout_spk, prompt_spk, witness_commitment, extranonce, devfee=None):
    """devfee: getblocktemplate's {"script", "amount"} (Vela developer fund, consensus)."""
    script_sig = script_num(height) + push_data(struct.pack("<Q", extranonce))
    outs = [(value, payout_spk)]
    if devfee:
        outs.append((devfee["amount"], bytes.fromhex(devfee["script"])))
    outs.append((0, prompt_spk))
    if witness_commitment:
        outs.append((0, witness_commitment))
    body = struct.pack("<i", 2)
    vin = ser_compact(1) + b"\x00" * 32 + b"\xff\xff\xff\xff" + ser_compact(len(script_sig)) + script_sig + b"\xff\xff\xff\xff"
    vout = ser_compact(len(outs)) + b"".join(struct.pack("<q", v) + ser_compact(len(s)) + s for v, s in outs)
    locktime = b"\x00\x00\x00\x00"
    legacy = body + vin + vout + locktime
    # witness serialization: marker/flag + 32-byte zero reserved value
    wit = body + b"\x00\x01" + vin + vout + ser_compact(1) + ser_compact(32) + b"\x00" * 32 + locktime
    return legacy, wit


def merkle_root(txids):
    layer = list(txids)
    if not layer:
        return b"\x00" * 32
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [sha256d(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


# ----------------------------------------------------------------------------
# RPC
# ----------------------------------------------------------------------------
class RPC:
    def __init__(self, url, cookie=None, timeout=120):
        p = urllib.parse.urlparse(url)
        self.url = f"{p.scheme}://{p.hostname}:{p.port}{p.path or '/'}"
        user, password = urllib.parse.unquote(p.username or ''), urllib.parse.unquote(p.password or '')
        if cookie:
            # An explicitly supplied cookie overrides even placeholder credentials
            # in the default URL (the documented --cookie behaviour).
            with open(cookie, encoding="utf-8") as f:
                user, _, password = f.read().strip().partition(":")
            if not user or not password:
                raise ValueError(f"invalid RPC cookie file: {cookie}")
        self.auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.timeout = float(timeout)

    def __call__(self, method, *params):
        req = urllib.request.Request(self.url, data=json.dumps({"jsonrpc": "1.0", "id": 0, "method": method, "params": list(params)}).encode(),
                                     headers={"Authorization": "Basic " + self.auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            resp = json.load(e)
        if resp.get("error"):
            raise RuntimeError(f"{method}: {resp['error']}")
        return resp["result"]


# ----------------------------------------------------------------------------
# Mining
# ----------------------------------------------------------------------------
_MODEL = None


def _worker_init():
    global _MODEL
    _MODEL = Model()


def grind(job):
    """Try nonces [start, start+count). Returns (nonce, hashpoi, x0, tries) or (None, None, None, tries)."""
    header76, model_hash, emb, target, start, count = job
    emb = np.array(emb, dtype=np.int64)
    for i in range(count):
        nonce = (start + i) & 0xFFFFFFFF
        h80 = header76 + struct.pack("<I", nonce)
        seed = tagged_hash("POI/seed", h80 + model_hash)
        x0 = _MODEL.run(seed, emb)
        hpoi = tagged_hash("POI/out", seed + x0)
        if int.from_bytes(sha256d(h80 + hpoi), "little") <= target:
            return nonce, hpoi, x0, i + 1
    return None, None, None, count


def selftest(model):
    """Compare against the reference model in test/functional/test_framework/poi.py if available."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "test", "functional"))
    from test_framework import poi as ref  # noqa: E402
    assert model.hash == ref.model_hash(), "model hash mismatch"
    toks = tokenize("selftest prompt")
    for k in range(3):
        seed = sha256(bytes([k]))
        assert model.run(seed, model.embed(toks)) == ref.run_inference(seed, toks), "x0 mismatch"
    print("selftest OK: numpy model == reference, model_hash", model.hash[::-1].hex())


def mine_one_v3(rpc, engine, args, payout_spk, extranonce):
    """Fork-M (payload v3): VIR inference + leaf digests + POI/out5.

    The model is a .vir file (--model3); leaves and commitments come from
    vir_mine. Slow with the CPU engine (about ten minutes per inference), which
    is why sd_miner.py is the production path.
    """
    import vir_mine  # noqa: E402  (same directory)

    tmpl = rpc("getblocktemplate", {"rules": ["segwit"]})
    poi = tmpl["poi"]
    if poi["version"] != vir_mine.PAYLOAD_VERSION:
        raise SystemExit(f"node requires PoI payload v{poi['version']}, this path mines v{vir_mine.PAYLOAD_VERSION}")
    if poi["model_hash"].lower() != engine.model_hash:
        raise SystemExit(f"model hash mismatch with the node: node {poi['model_hash']}, file {engine.model_hash}")
    model_hash = bytes.fromhex(engine.model_hash)[::-1]   # raw digest for the seed

    tokens = engine.encode(args.prompt) if args.prompt else list(poi["prompt_tokens"])
    wc = bytes.fromhex(tmpl["default_witness_commitment"]) if "default_witness_commitment" in tmpl else None
    cb_legacy, cb_wit = build_coinbase(tmpl["height"], tmpl["coinbasevalue"], payout_spk,
                                       prompt_script(tokens), wc, extranonce, tmpl.get("devfee"))
    root = merkle_root([sha256d(cb_legacy)] + [bytes.fromhex(t["txid"])[::-1] for t in tmpl["transactions"]])
    target = int(tmpl["target"], 16)
    header76 = (struct.pack("<i", tmpl["version"]) + bytes.fromhex(tmpl["previousblockhash"])[::-1] + root
                + struct.pack("<I", max(tmpl["curtime"], tmpl["mintime"])) + bytes.fromhex(tmpl["bits"])[::-1])

    t0, tries, nonce = time.time(), 0, args.start_nonce
    found = None
    while found is None and nonce < (1 << 32):
        h80 = header76 + struct.pack("<I", nonce)
        seed = tagged_hash("POI/seed", h80 + model_hash)
        tries += 1
        x0, leaves = engine.attempt(seed, tokens)
        hpoi = vir_mine.poi_output_hash_v5(seed, x0, vir_mine.leaves_commitment(leaves))
        if int.from_bytes(sha256d(h80 + hpoi), "little") <= target:
            found = (h80, hpoi, x0, leaves, nonce)
            break
        if rpc("getbestblockhash") != tmpl["previousblockhash"]:
            return None, tries, time.time() - t0
        nonce += 1
    if found is None:
        return None, tries, time.time() - t0
    h80, hpoi, x0, leaves, nonce = found
    header = h80 + hpoi
    block = (header + ser_compact(1 + len(tmpl["transactions"])) + cb_wit
             + b"".join(bytes.fromhex(t["data"]) for t in tmpl["transactions"])
             + vir_mine.build_payload(x0, leaves))
    res = rpc("submitblock", block.hex())
    return (sha256d(header)[::-1].hex(), tmpl["height"], res, nonce, len(leaves)), tries, time.time() - t0


def mine_one(rpc, pool, model, args, payout_spk, extranonce):
    tmpl = rpc("getblocktemplate", {"rules": ["segwit"]})
    poi = tmpl["poi"]
    assert poi["version"] == POI_VERSION, f"node requires PoI model v{poi['version']}, miner implements v{POI_VERSION}"
    assert bytes.fromhex(poi["model_hash"])[::-1] == model.hash, "model hash mismatch with node"
    tokens = tokenize(args.prompt) if args.prompt else poi["prompt_tokens"]
    emb = model.embed(tokens).tolist()

    wc = bytes.fromhex(tmpl["default_witness_commitment"]) if "default_witness_commitment" in tmpl else None
    cb_legacy, cb_wit = build_coinbase(tmpl["height"], tmpl["coinbasevalue"], payout_spk, prompt_script(tokens), wc, extranonce, tmpl.get("devfee"))
    txids = [sha256d(cb_legacy)] + [bytes.fromhex(t["txid"])[::-1] for t in tmpl["transactions"]]
    root = merkle_root(txids)
    target = int(tmpl["target"], 16)
    ntime = max(tmpl["curtime"], tmpl["mintime"])
    header76 = (struct.pack("<i", tmpl["version"]) + bytes.fromhex(tmpl["previousblockhash"])[::-1] + root
                + struct.pack("<I", ntime) + bytes.fromhex(tmpl["bits"])[::-1])

    t0 = time.time()
    tries = 0
    start = 0
    while start <= 0xFFFFFFFF:
        jobs = [(header76, model.hash, emb, target, start + w * args.chunk, args.chunk) for w in range(args.workers)]
        start += args.workers * args.chunk
        found = None
        for nonce, hpoi, x0, n in pool.imap_unordered(grind, jobs):
            tries += n
            if nonce is not None and found is None:
                found = (nonce, hpoi, x0)
        if found:
            break
        # refresh if the tip moved
        if rpc("getbestblockhash") != tmpl["previousblockhash"]:
            return None, tries, time.time() - t0
    nonce, hpoi, x0 = found
    header = header76 + struct.pack("<I", nonce) + hpoi
    block = (header + ser_compact(1 + len(tmpl["transactions"])) + cb_wit
             + b"".join(bytes.fromhex(t["data"]) for t in tmpl["transactions"])
             + bytes([POI_VERSION]) + ser_compact(len(x0)) + x0)
    res = rpc("submitblock", block.hex())
    dt = time.time() - t0
    bhash = sha256d(header)[::-1].hex()
    return (bhash, tmpl["height"], res, nonce), tries, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpc", default="http://u:p@127.0.0.1:18743")
    ap.add_argument("--cookie", help="path to the node's .cookie file (instead of user:password in --rpc)")
    ap.add_argument("--address", help="payout address (default: anyone-can-spend OP_TRUE)")
    ap.add_argument("--prompt", help="prompt text (default: node's -poiprompt)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=64, help="nonces per worker task")
    ap.add_argument("--blocks", type=int, default=1, help="blocks to mine (0 = forever)")
    ap.add_argument("--model3", metavar="FILE",
                    help="Fork-M .vir model: mine payload v3 after the fork height (no --workers/batch: "
                         "the reference implementation runs one inference at a time, about 10 min for sd-turbo)")
    ap.add_argument("--start-nonce", type=int, default=0, help="first nonce for the v3 path")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", type=int, metavar="NONCES", help="offline throughput benchmark (no node needed)")
    args = ap.parse_args()

    rpc = RPC(args.rpc, args.cookie)
    if args.model3:
        # Fork-M: the template decides. Before the fork height the node still asks
        # for the older payload, so refuse rather than mine something invalid.
        import vir_mine  # noqa: E402  (same directory)
        engine = vir_mine.CpuEngine(args.model3)
        payout_spk = bytes.fromhex(rpc("validateaddress", args.address)["scriptPubKey"]) if args.address else b"\x51"
        print(f"PoI miner (Fork-M): payload v{vir_mine.PAYLOAD_VERSION} {engine.model_hash[:16]}..")
        mined = 0
        extranonce = int.from_bytes(os.urandom(4), "little") << 32
        while args.blocks == 0 or mined < args.blocks:
            extranonce += 1
            result, tries, dt = mine_one_v3(rpc, engine, args, payout_spk, extranonce)
            if result is None:
                print(f"  tip changed, restarting ({tries} inferences in {dt:.0f}s)")
                continue
            bhash, height, res, nonce, nleaves = result
            status = "accepted" if res is None else f"REJECTED ({res})"
            print(f"  block {height} {bhash[:16]}.. nonce={nonce} leaves={nleaves} {status}  "
                  f"{tries} inferences in {dt:.0f}s")
            if res is not None:
                sys.exit(1)
            mined += 1
        return

    model = Model()
    if args.selftest:
        selftest(model)
        return
    if args.bench:
        emb = model.embed(tokenize(args.prompt or "benchmark")).tolist()
        per = max(1, args.bench // args.workers)
        with mp.Pool(args.workers, initializer=_worker_init) as pool:
            t0 = time.time()
            jobs = [(os.urandom(76), model.hash, emb, 0, w * per, per) for w in range(args.workers)]
            total = sum(r[3] for r in pool.map(grind, jobs))
            dt = time.time() - t0
        print(f"bench: {total} inferences, {args.workers} workers, {dt:.2f}s -> {total / dt:.0f} inf/s")
        return
    payout_spk = bytes.fromhex(rpc("validateaddress", args.address)["scriptPubKey"]) if args.address else b"\x51"
    print(f"PoI miner: model v{POI_VERSION} {model.hash[::-1].hex()[:16]}..  workers={args.workers}")
    mined = 0
    extranonce = int.from_bytes(os.urandom(4), "little") << 32
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        while args.blocks == 0 or mined < args.blocks:
            extranonce += 1
            result, tries, dt = mine_one(rpc, pool, model, args, payout_spk, extranonce)
            rate = tries / dt if dt > 0 else 0
            if result is None:
                print(f"  tip changed, restarting ({tries} inferences, {rate:.0f} inf/s)")
                continue
            bhash, height, res, nonce = result
            status = "accepted" if res is None else f"REJECTED ({res})"
            print(f"  block {height} {bhash[:16]}.. nonce={nonce} {status}  {tries} inferences in {dt:.2f}s ({rate:.0f} inf/s)")
            if res is not None:
                sys.exit(1)
            mined += 1


if __name__ == "__main__":
    main()