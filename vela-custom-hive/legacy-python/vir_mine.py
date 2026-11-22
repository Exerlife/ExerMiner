#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""Fork-M (PoI payload v3) mining helpers shared by the external miners.

A v3 attempt is one VIR inference plus the digest of every 256 KiB leaf of every
node output, and ``hashPoI = POI/out5(seed || x0 || R)`` with
``R = POI/leaves(leaves)`` (doc/fork-m.md sections 5, 9):

    seed    = TaggedHash("POI/seed")(header80 || model_hash)
    x0, leaves = engine.attempt(seed, tokens)
    hashPoI = TaggedHash("POI/out5")(seed || x0 || leaves_commitment(leaves))

Engine adapters:

- ``CpuEngine``   — ``contrib/vir/vir.py`` on the CPU. Works on any machine, about
  ten minutes per inference for sd-turbo: a cross-check, not a mining engine.
- ``GpuEngine``   — the GPU VIR engine that ships with the repository
  (``contrib/vir/vir_torch.py`` + ``contrib/vir/noderoot.py``, bit-identical to
  the CPU one). Sub-chunk digests are hashed on the GPU while the activation is
  still in device memory; ~30-65 s per inference on an RTX 3050. Point
  ``--engine-dir`` (or ``VIR_GPU_DIR``) at another checkout to use a different
  engine.

Both engines take the reference ``vir`` module as the arithmetic authority, so
the two agree byte for byte with the node (checked by mining: the node recomputes
every block it accepts).
"""
import hashlib
import os
import struct
import sys

PAYLOAD_VERSION = 3     # CPoIPayload::LEAVES_VERSION (Fork-M)
SUB_BYTES = 4096
LEAF_SUBS = 64
LEAF_VALUES = LEAF_SUBS * SUB_BYTES // 4     # int32 per leaf
LEAF_TAG = b"VELA/leaf/v1"

# Where vir.py and the GPU engine live: this directory in the standalone miner
# repository, or ../vir for the copy inside the node repository.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _dir_with(filename):
    for candidate in (_HERE, os.path.abspath(os.path.join(_HERE, os.pardir, "vir"))):
        if os.path.isfile(os.path.join(candidate, filename)):
            return candidate
    return os.path.abspath(os.path.join(_HERE, os.pardir, "vir"))


DEFAULT_ENGINE_DIR = _dir_with("vir_torch.py")


# ---------------------------------------------------------------------------
# commitments and the payload
# ---------------------------------------------------------------------------

def tagged_hash(tag: str, msg: bytes) -> bytes:
    t = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(t + t + msg).digest()


def leaves_commitment(leaves) -> bytes:
    """R = TaggedHash("POI/leaves")(leaf digests, node order)."""
    return tagged_hash("POI/leaves", b"".join(leaves))


def poi_output_hash_v5(seed: bytes, x0: bytes, commitment: bytes) -> bytes:
    """hashPoI from Fork-M on: TaggedHash("POI/out5")(seed || x0 || R)."""
    return tagged_hash("POI/out5", seed + x0 + commitment)


def leaf_digest(node: int, leaf: int, values_le_bytes: bytes) -> bytes:
    """``SHA256("VELA/leaf/v1" || be32(node) || be32(leaf) || sub_0 ..)``.

    ``values_le_bytes`` is the node output as int32le bytes (at most 256 KiB); a
    short final 4 KiB sub-chunk is hashed as it is.
    """
    nsub = max(1, -(-len(values_le_bytes) // SUB_BYTES))
    h = hashlib.sha256(LEAF_TAG + node.to_bytes(4, "big") + leaf.to_bytes(4, "big"))
    for s in range(nsub):
        chunk = values_le_bytes[s * SUB_BYTES:(s + 1) * SUB_BYTES]
        h.update(hashlib.sha256(node.to_bytes(4, "big")
                                + (leaf * LEAF_SUBS + s).to_bytes(4, "big") + chunk).digest())
    return h.digest()


def node_leaves(node: int, values_le_bytes: bytes) -> list[bytes]:
    nleaf = max(1, -(-len(values_le_bytes) // (LEAF_VALUES * 4)))
    return [leaf_digest(node, k, values_le_bytes[k * LEAF_VALUES * 4:(k + 1) * LEAF_VALUES * 4])
            for k in range(nleaf)]


def ser_compact(n: int) -> bytes:
    if n < 253:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    if n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def build_payload(x0: bytes, leaves) -> bytes:
    """{u8 nVersion=3, x0, leaves} in the full form (form byte 0 + the digests)."""
    return (bytes([PAYLOAD_VERSION]) + ser_compact(len(x0)) + x0
            + b"\x00" + ser_compact(len(leaves)) + b"".join(leaves))


# ---------------------------------------------------------------------------
# CPU engine: contrib/vir/vir.py
# ---------------------------------------------------------------------------

class _LeafTracer:
    """Collects node outputs as leaf digests (no trace commitment)."""

    def __init__(self, np_module):
        self.np = np_module
        self.leaves: list[bytes] = []

    def node(self, index, node, out, is_last):
        v = self.np.ascontiguousarray(out, dtype="<i4").ravel()
        self.leaves.extend(node_leaves(index, v.tobytes()))

    def finish(self, payload, latent):
        pass


class CpuEngine:
    """VIR reference implementation on the CPU (slow; ~10 min per sd-turbo inference)."""

    def __init__(self, model_path, verbose=True):
        sys.path.insert(0, _dir_with("vir.py"))
        import vir
        import numpy as np
        self.np = np
        self.vir = vir
        with open(model_path, "rb") as f:
            self.model = vir.VirModel.parse(f.read())
        self.model_path = model_path
        self.model_hash = self.model.model_hash()          # display order
        self._tokenizer = vir.tokenizer(self.model) if self.model.header.tok_type else None
        if verbose:
            print(f"engine: CPU VIR reference, {len(self.model.graph.nodes)} nodes, hash {self.model_hash[:16]}..")

    def encode(self, text):
        if self._tokenizer is None:
            raise SystemExit("this model has no tokenizer; drop --prompt")
        return self._tokenizer.encode(text)

    def attempt(self, seed, tokens):
        m, n, vir = self.model, self.model.noise, self.vir
        np = self.np
        noise = m.noise_latent(seed)
        x = vir.sat32(vir.rshr(noise * np.int64(n.cin_q30), 30))
        feed = {n.latent_tensor: x}
        ids = next((t for t in m.graph.inputs if t != n.latent_tensor), None)
        if ids is not None:
            feed[ids] = m.context_ids(tokens)
        tracer = _LeafTracer(np)
        values = m.evaluate(feed, tracer)
        eps = values[n.eps_tensor]
        if n.sampler_kind == 1:
            x0 = vir.sat32(vir.rshr(noise * np.int64(n.sigma_q16) - eps * np.int64(n.eps_q16), n.latent_shift))
        else:
            x0 = vir.sat32(vir.rshr((noise - eps) * np.int64(n.sigma_q16), n.latent_shift))
        q = np.clip(vir.rshr(x0, n.out_shift), -n.clamp_abs, n.clamp_abs)
        return q.astype("<i2").tobytes(), tracer.leaves


# ---------------------------------------------------------------------------
# GPU engine: vir_torch.py + noderoot.py (research checkout)
# ---------------------------------------------------------------------------

class GpuEngine:
    """VIR on CUDA, bit-identical to the CPU reference; leaves hashed on the GPU."""

    def __init__(self, model_path, engine_dir=None, verbose=True):
        import os
        engine_dir = engine_dir or os.environ.get("VIR_GPU_DIR") or DEFAULT_ENGINE_DIR
        if not os.path.isdir(engine_dir):
            raise SystemExit(f"GPU engine directory not found: {engine_dir} (--engine-dir / VIR_GPU_DIR)")
        sys.path.insert(0, engine_dir)
        import vir_torch
        import noderoot
        import vir  # the copy vir_torch just loaded
        import numpy as np
        self.vir, self.vir_torch, self.noderoot, self.np = vir, vir_torch, noderoot, np
        from pathlib import Path
        self.vm = vir_torch.TorchVM(Path(model_path))
        self.model = self.vm.model
        self.model_hash = self.model.model_hash()
        self._tokenizer = vir.tokenizer(self.model) if self.model.header.tok_type else None
        if verbose:
            print(f"engine: GPU VIR, {len(self.model.graph.nodes)} nodes, hash {self.model_hash[:16]}..")

    def encode(self, text):
        if self._tokenizer is None:
            raise SystemExit("this model has no tokenizer; drop --prompt")
        return self._tokenizer.encode(text)

    def attempt(self, seed, tokens):
        m, n, vir, np = self.model, self.model.noise, self.vir, self.np
        noise = m.noise_latent(seed)
        x = vir.sat32(vir.rshr(noise * np.int64(n.cin_q30), 30))
        feed = {n.latent_tensor: x}
        ids = next((t for t in m.graph.inputs if t != n.latent_tensor), None)
        if ids is not None:
            feed[ids] = m.context_ids(tokens)
        nr = self.noderoot.NodeRoots(len(m.graph.nodes))
        self_torch = self.vir_torch
        import torch

        class Tracer:
            def node(self, index, node, out, is_last):
                t = self_torch._LAST_CUDA[0]
                if t is None or t.numel() != np.asarray(out).size:   # CPU fallback op
                    t = torch.from_numpy(np.ascontiguousarray(out, dtype=np.int64)).to(self_torch.DEV)
                nr.add(index, t)
                self_torch._LAST_CUDA[0] = None

        values = m.evaluate(feed, Tracer())
        eps = values[n.eps_tensor]
        if n.sampler_kind == 1:
            x0 = vir.sat32(vir.rshr(noise * np.int64(n.sigma_q16) - eps * np.int64(n.eps_q16), n.latent_shift))
        else:
            x0 = vir.sat32(vir.rshr((noise - eps) * np.int64(n.sigma_q16), n.latent_shift))
        q = np.clip(vir.rshr(x0, n.out_shift), -n.clamp_abs, n.clamp_abs)
        return q.astype("<i2").tobytes(), self._leaves(nr)

    @staticmethod
    def _leaves(nr):
        """Leaf digests from the GPU sub-chunk digests.

        Same result as noderoot.NodeRoots.finish_leaves(), but each node's digests
        are taken as one byte string instead of joining numpy elements one by one
        (that is what makes the difference between ~30 s and ~1 s).
        """
        out = []
        for i, p in enumerate(nr.parts):
            if p is None:
                raise RuntimeError(f"no output recorded for node {i}")
            dig = p.get().astype(">u4").tobytes()        # 32 bytes per sub-chunk digest
            nsub = len(dig) // 32
            for k in range(max(1, -(-nsub // LEAF_SUBS))):
                subs = dig[k * LEAF_SUBS * 32:(k + 1) * LEAF_SUBS * 32]
                out.append(hashlib.sha256(LEAF_TAG + i.to_bytes(4, "big") + k.to_bytes(4, "big")
                                          + subs).digest())
        nr.keep.clear()
        return out


def make_engine(kind, model_path, engine_dir=None, verbose=True):
    if kind == "cpu":
        return CpuEngine(model_path, verbose)
    return GpuEngine(model_path, engine_dir or "D:/zkbench/vir_gpu", verbose)
