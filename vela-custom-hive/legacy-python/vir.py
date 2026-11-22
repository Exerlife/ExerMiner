#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""VIR: Vela Integer Runtime -- normative Python reference implementation.

This file is the executable specification of ``doc/vir-format.md`` (format
v0.1, opset 1).  Every operator below is defined on exact integers, so any
conforming implementation (this file, the C++ interpreter, a CUDA miner) must
produce the same bytes for the same file, seed and prompt.

Numeric model (identical to PoI v2, see vela/contrib/poi-sd/sdint.py)
  * activations are int32 with 16 fractional bits ("Q16"), saturated after
    every operation;
  * weights are int8 with one (mantissa, shift) pair per output channel,
    real = w * m * 2**-s, m in [2**14, 2**15);
  * matrix-product inputs are quantised per row to [..., 32767] with a
    power-of-two shift e = max(0, bitlen(max|x|) - 15), products accumulated
    exactly in int64;
  * rshr(v, s) rounds half up: (v + 2**(s-1)) >> s for s > 0, v for s == 0 and
    a saturating left shift for s < 0;
  * every non-linearity is an interpolated lookup table stored in the file.

Usage
  python vir.py --selftest [--v2-model PATH]
  python vir.py --build-sdxs [--model-dir DIR] [--dump]
  python vir.py --hash FILE [--opset N]
  python vir.py --v2-hash FILE

The module has no third-party dependency except numpy, and it is importable
without numpy for the pure-parsing helpers (numpy is only needed to execute a
graph).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - numpy is required to run a graph, not to parse one
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# ---------------------------------------------------------------------------
# version / layout constants
# ---------------------------------------------------------------------------

MAGIC = b"VIR1"
OPSET = 1
FORMAT_VERSION = 1  # every section carries its own version byte
HEADER_SIZE = 256
ALIGN = 64  # tensor data alignment
ALIGN_LOG2 = 6
NOISE_SIZE = 64

#: activation fractional bits (Q16)
BITS = 16

#: header <-> file kind
KIND_LATENT_1STEP = 0
#: header out_dtype
OUT_I16LE = 0
#: tokenizer kinds
TOK_NONE = 0
TOK_CLIP_BPE = 1
#: padding policies
PAD_USE_PAD_ID = 0
PAD_USE_EOS_ID = 1
PAD_NONE = 2


class VirError(Exception):
    """A malformed or unsupported .vir file (message is the reject reason)."""


class NotSupported(VirError):
    """Valid file, feature outside opset 1 (e.g. an op forward we do not run)."""


# ---------------------------------------------------------------------------
# integer primitives -- the whole numeric contract lives here
# ---------------------------------------------------------------------------

I32_MIN = -(2**31)
I32_MAX = 2**31 - 1


def _i64(v):
    return np.asarray(v, dtype=np.int64)


def sat32(v):
    """Clamp to the int32 range (no-op on Python ints outside it)."""
    return np.clip(_i64(v), I32_MIN, I32_MAX)


def _rshr_scalar(v, s: int):
    v = _i64(v)
    if s >= 0:
        p = 62 if s > 62 else s
        return sat32((v + ((np.int64(1) << p) >> 1)) >> p)
    n = 31 if -s > 31 else -s
    lim = np.int64(1) << (62 - n)
    r = np.clip(v, -lim, lim)
    return sat32(r << n)


def rshr(v, s):
    """Rounding arithmetic right shift, round half up; negative s shifts left.

    Mirrors ``poi::sd::Rshr`` (vela/src/poi/sdmodel.cpp) and ``sdint.rshr``
    bit for bit, including the saturating left-shift branch.
    """
    v = _i64(v)
    sa = np.asarray(s, dtype=np.int64)
    if sa.ndim == 0:
        return _rshr_scalar(v, int(sa))
    if sa.shape == ():
        return _rshr_scalar(v, int(sa))
    pos = np.clip(sa, 0, 62)
    half = np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), np.int64(0))
    r = (v + half) >> pos
    neg = np.clip(-sa, 0, 31)
    lim = np.left_shift(np.int64(1), 62 - neg)
    r = np.where(sa < 0, np.left_shift(np.clip(r, -lim, lim), neg), r)
    return sat32(r)


def floor_div(a, b):
    """Floor division (C++ FloorDiv), not truncation."""
    return np.floor_divide(_i64(a), _i64(b))


def isqrt(v):
    """Exact floor(sqrt(v)) elementwise; v must be >= 0."""
    arr = _i64(v)
    if arr.size == 0:
        return arr
    if int(arr.min()) < 0:
        raise VirError("isqrt of a negative value")
    flat = arr.ravel()
    out = np.fromiter((math.isqrt(int(x)) for x in flat), dtype=np.int64, count=flat.size)
    return out.reshape(arr.shape)


def row_exp(m):
    """e = max(0, bitlen(max|x|) - 15), elementwise (see sdint.row_exp)."""
    m = _i64(m)
    e = np.zeros(m.shape, dtype=np.int64)
    for k in range(15, 32):
        e = e + (m >= (1 << k)).astype(np.int64)
    return e


def quant16(v, e):
    """Per-row 16-bit quantisation with the power-of-two shift e."""
    return np.clip(rshr(v, e), -32767, 32767)


def row_max_abs(x):
    """max|x| over the last axis, keepdims (the row whose scale is used)."""
    return np.abs(_i64(x)).max(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# hashing: ModelHash = TaggedHash("POI/model", u8(opset) || file)
# ---------------------------------------------------------------------------


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def tagged_hash(tag: str, msg: bytes) -> bytes:
    """BIP340-style tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg).

    This is what ``HashWriter{TaggedHash(tag)}`` computes in the node, e.g.
    ``HashWriter hw{TaggedHash("POI/model")}; hw << u8; hw.write(file);``.
    """
    t = _sha256(tag.encode())
    return _sha256(t + t + msg)


class _HashStream:
    """Incremental version of :func:`tagged_hash` (for files of a few GB)."""

    def __init__(self, tag: str):
        t = _sha256(tag.encode())
        self._h = hashlib.sha256()
        self._h.update(t)
        self._h.update(t)

    def update(self, b: bytes) -> None:
        self._h.update(b)

    def digest(self) -> bytes:
        return self._h.digest()


def model_hash_digest(opset: int, file_bytes: bytes) -> bytes:
    """Raw 32-byte digest (internal byte order, as uint256 stores it)."""
    if not 0 <= opset <= 255:
        raise VirError("opset out of range")
    return tagged_hash("POI/model", bytes([opset]) + file_bytes)


def digest_to_display(d: bytes) -> str:
    """uint256 display order: the digest bytes reversed (as sdint.py does)."""
    return d[::-1].hex()


def model_hash(opset: int, file_bytes: bytes) -> str:
    """Human/consensus-facing model hash string, e.g. SD_MODEL_HASH's form."""
    return digest_to_display(model_hash_digest(opset, file_bytes))


# ---------------------------------------------------------------------------
# tensor descriptors, attributes, operators
# ---------------------------------------------------------------------------

DTYPE_CODE = {"i8": 0, "i16": 1, "i32": 2, "u8": 3}
DTYPE_FROM_CODE = {v: k for k, v in DTYPE_CODE.items()}
DTYPE_NP = {"i8": "i1", "i16": "<i2", "i32": "<i4", "u8": "u1"}
DTYPE_SIZE = {"i8": 1, "i16": 2, "i32": 4, "u8": 1}
#: dtypes a GEMM/CONV2D/EMBED weight may use, and the largest |w| each allows
WEIGHT_DTYPES = ("i8", "i16")
WEIGHT_QMAX = {"i8": 127, "i16": 32767}
#: K bound so that sum_k q*w*m cannot overflow int64 with i16 weights:
#: 32767 * 32767 * 32767 * K < 2^63  <=>  K < 2^18; opset 1 allows K <= 2^17.
MAX_K = 1 << 17


class Storage(IntEnum):
    ACTIVATION = 0  # produced by the graph; no bytes in the file
    WEIGHT = 1      # int8 weights, one (m, s, b) per output channel
    PARAM = 2       # int32 scales, biases, eps, positional embeddings
    LUT = 3         # int32 lookup table
    BLOB = 4        # opaque bytes (tokenizer vocab/merges)


class Attr(IntEnum):
    """Attribute keys.  Keys are global (not per-op) and never reused."""

    SHAPE = 1
    PERM = 2
    AXIS = 3
    GROUPS = 4
    NUM_HEADS = 5
    HEAD_DIM = 6
    CAUSAL = 7
    Q_SCALE = 8
    LUT_LO = 9
    LUT_STEP_BITS = 10
    LUT_EDGE_LO = 11
    LUT_EDGE_HI = 12
    POST_SHIFT = 13
    FACTOR = 14
    MODE = 15
    KH = 16
    KW = 17
    STRIDE = 18
    PAD = 19
    MULT = 20
    SHIFT = 21
    WGROUP = 22  # GEMM/CONV2D weight-scale group size along K (0 = one scale per row)


#: attribute value kinds: u32, i32, u32[], i32[] (encoded as LE u32 lanes)
ATTR_KIND = {
    Attr.SHAPE: "u32[]",
    Attr.PERM: "u32[]",
    Attr.AXIS: "u32",
    Attr.GROUPS: "u32",
    Attr.NUM_HEADS: "u32",
    Attr.HEAD_DIM: "u32",
    Attr.CAUSAL: "u32",
    Attr.Q_SCALE: "u32",
    Attr.LUT_LO: "i32",
    Attr.LUT_STEP_BITS: "u32",
    Attr.LUT_EDGE_LO: "i32",
    Attr.LUT_EDGE_HI: "i32",
    Attr.POST_SHIFT: "u32",
    Attr.FACTOR: "u32",
    Attr.MODE: "u32",
    Attr.KH: "u32",
    Attr.KW: "u32",
    Attr.STRIDE: "u32",
    Attr.PAD: "u32",
    Attr.MULT: "i32",
    Attr.SHIFT: "i32",
    Attr.WGROUP: "u32",
}

ATTR_TYPE_CODE = {"u32": 0, "i32": 1, "u32[]": 2, "i32[]": 3}
ATTR_TYPE_FROM_CODE = {v: k for k, v in ATTR_TYPE_CODE.items()}


class Op(IntEnum):
    """Operator set 1.  Codes are append-only for a given opset."""

    GEMM = 1
    CONV2D = 2
    GROUPNORM = 3
    LAYERNORM = 4
    LUT_ACT = 5
    ATTENTION = 6
    UPSAMPLE = 7
    DOWNSAMPLE = 8
    ADD = 9
    CONCAT = 10
    RESHAPE = 11
    SCALE = 12
    MUL = 13
    EMBED = 14


@dataclass(frozen=True)
class OpSchema:
    n_in: int
    n_out: int
    attrs: tuple[Attr, ...]

    def sorted_attrs(self) -> tuple[Attr, ...]:
        return tuple(sorted(self.attrs, key=int))


OP_SPECS: dict[Op, OpSchema] = {
    Op.GEMM: OpSchema(5, 1, (Attr.WGROUP,)),
    Op.CONV2D: OpSchema(5, 1, (Attr.KH, Attr.KW, Attr.STRIDE, Attr.PAD, Attr.WGROUP)),
    Op.GROUPNORM: OpSchema(4, 1, (Attr.GROUPS,)),
    Op.LAYERNORM: OpSchema(4, 1, ()),
    Op.LUT_ACT: OpSchema(2, 1, (Attr.LUT_LO, Attr.LUT_STEP_BITS, Attr.LUT_EDGE_LO,
                                Attr.LUT_EDGE_HI, Attr.POST_SHIFT)),
    Op.ATTENTION: OpSchema(4, 1, (Attr.NUM_HEADS, Attr.HEAD_DIM, Attr.CAUSAL, Attr.Q_SCALE)),
    Op.UPSAMPLE: OpSchema(1, 1, (Attr.FACTOR, Attr.MODE)),
    Op.DOWNSAMPLE: OpSchema(1, 1, (Attr.FACTOR, Attr.MODE)),
    Op.ADD: OpSchema(2, 1, ()),
    Op.CONCAT: OpSchema(2, 1, (Attr.AXIS,)),
    Op.RESHAPE: OpSchema(1, 1, (Attr.PERM, Attr.SHAPE)),
    Op.SCALE: OpSchema(1, 1, (Attr.MULT, Attr.SHIFT)),
    Op.MUL: OpSchema(2, 1, (Attr.SHIFT,)),
    Op.EMBED: OpSchema(5, 1, (Attr.POST_SHIFT,)),
}


def default_attrs(op: Op) -> dict[Attr, Any]:
    """Defaults used by the builder; the writer always emits every attribute."""
    out: dict[Attr, Any] = {}
    for key in OP_SPECS[op].sorted_attrs():
        kind = ATTR_KIND[key]
        if kind in ("u32", "i32"):
            out[key] = 1 if key in (Attr.STRIDE, Attr.Q_SCALE) else 0
        else:
            out[key] = ()
    if op is Op.CONV2D:
        out.update({Attr.KH: 3, Attr.KW: 3, Attr.STRIDE: 1, Attr.PAD: 1})
    elif op is Op.ATTENTION:
        out.update({Attr.CAUSAL: 0})
    elif op is Op.UPSAMPLE:
        out.update({Attr.FACTOR: 2, Attr.MODE: 0})
    elif op is Op.DOWNSAMPLE:
        out.update({Attr.FACTOR: 2, Attr.MODE: 0})
    elif op is Op.LUT_ACT:
        out.update({Attr.LUT_LO: -(16 << BITS), Attr.LUT_STEP_BITS: 10,
                    Attr.LUT_EDGE_LO: 0, Attr.LUT_EDGE_HI: 65536, Attr.POST_SHIFT: BITS})
    elif op is Op.MUL:
        out.update({Attr.SHIFT: BITS})
    elif op is Op.SCALE:
        out.update({Attr.MULT: 1, Attr.SHIFT: 0})
    elif op is Op.EMBED:
        out.update({Attr.POST_SHIFT: BITS})
    elif op is Op.RESHAPE:
        out.update({Attr.PERM: (), Attr.SHAPE: ()})
    return out


@dataclass
class TensorDesc:
    """One entry of the tensor table."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "i32"
    storage: Storage = Storage.ACTIVATION
    data: Any = None  # numpy array for stored tensors, None for activations

    @property
    def count(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        if self.storage == Storage.ACTIVATION:
            return 0
        return self.count * DTYPE_SIZE[self.dtype]

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class Node:
    op: Op
    inputs: list[int]
    outputs: list[int]
    attrs: dict[Attr, Any] = field(default_factory=dict)

    def attr(self, key: Attr, default=None):
        return self.attrs.get(key, default)


def _prod(seq: Iterable[int]) -> int:
    n = 1
    for x in seq:
        n *= int(x)
    return n


class Graph:
    """A DAG of opset-1 nodes over a single tensor table.

    Tensor ids are positions in :attr:`tensors`.  Ids are *not* required to be
    ordered, but every node's inputs must already be defined when the node is
    reached (graph-input tensors or outputs of earlier nodes), which is what
    :meth:`validate` enforces and what makes file evaluation single-pass.
    """

    def __init__(self) -> None:
        self.tensors: list[TensorDesc] = []
        self.nodes: list[Node] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self._by_name: dict[str, int] = {}
        self._anon = 0

    # -- table ---------------------------------------------------------
    def add_tensor(self, desc: TensorDesc) -> int:
        if desc.name in self._by_name:
            raise VirError(f"duplicate tensor name {desc.name!r}")
        self.tensors.append(desc)
        tid = len(self.tensors) - 1
        self._by_name[desc.name] = tid
        return tid

    def find(self, name: str) -> int:
        try:
            return self._by_name[name]
        except KeyError:
            raise VirError(f"unknown tensor {name!r}") from None

    def tensor(self, tid: int) -> TensorDesc:
        try:
            return self.tensors[tid]
        except IndexError:
            raise VirError(f"tensor id {tid} out of range") from None

    def declare_input(self, name: str, shape: Sequence[int], dtype: str = "i32") -> int:
        tid = self.add_tensor(TensorDesc(name, tuple(int(d) for d in shape), dtype))
        self.inputs.append(tid)
        return tid

    def new_activation(self, shape: Sequence[int], op: Op) -> int:
        self._anon += 1
        name = f"a{self._anon:05d}.{op.name.lower()}"
        return self.add_tensor(TensorDesc(name, tuple(int(d) for d in shape), "i32"))

    # -- nodes ---------------------------------------------------------
    def add_node(self, op: Op, inputs: Sequence[int], out_shapes: Sequence[Sequence[int]],
                 attrs: dict[Attr, Any] | None = None) -> list[int]:
        spec = OP_SPECS[op]
        if len(inputs) != spec.n_in:
            raise VirError(f"{op.name}: expected {spec.n_in} inputs, got {len(inputs)}")
        if len(out_shapes) != spec.n_out:
            raise VirError(f"{op.name}: expected {spec.n_out} outputs, got {len(out_shapes)}")
        full = default_attrs(op)
        if attrs:
            for key, value in attrs.items():
                if key not in spec.attrs:
                    raise VirError(f"{op.name}: attribute {key.name} not in schema")
                full[key] = value
        outs = [self.new_activation(shape, op) for shape in out_shapes]
        self.nodes.append(Node(op, [int(i) for i in inputs], outs, full))
        return outs

    def emit(self, op: Op, inputs: Sequence[int], shape: Sequence[int],
             attrs: dict[Attr, Any] | None = None) -> int:
        return self.add_node(op, inputs, [tuple(int(d) for d in shape)], attrs)[0]

    # -- weights -------------------------------------------------------
    def weight(self, name: str, n: int, k: int, data=None, dtype: str = "i8") -> int:
        """Register a weight; dtype is i8 (|w| <= 127) or i16 (|w| <= 32767)."""
        if dtype not in WEIGHT_DTYPES:
            raise VirError(f"weight {name}: dtype must be one of {WEIGHT_DTYPES}")
        return self.add_tensor(TensorDesc(name + ".w", (int(n), int(k)), dtype, Storage.WEIGHT, data))

    def channel_params(self, name: str, n: int, data=None, k: int = 0,
                       wgroup: int = 0) -> tuple[int, int, int]:
        """Register .m / .s / .b for a GEMM or Conv2D weight ([N] or [N, ceil(K/G)] for m)."""
        m_shape = (int(n),) if not wgroup else (int(n), -(-int(k) // int(wgroup)))
        m = self.add_tensor(TensorDesc(name + ".m", m_shape, "i32", Storage.PARAM,
                                       None if data is None else data[0]))
        s = self.add_tensor(TensorDesc(name + ".s", (int(n),), "i32", Storage.PARAM,
                                       None if data is None else data[1]))
        b = self.add_tensor(TensorDesc(name + ".b", (int(n),), "i32", Storage.PARAM,
                                       None if data is None else data[2]))
        return m, s, b

    # -- validation ----------------------------------------------------
    def validate(self, require_data: bool = False) -> None:
        errs: list[str] = []
        if not self.tensors:
            errs.append("empty tensor table")
        if not self.outputs:
            errs.append("graph has no declared output")
        seen: set[int] = set(self.inputs)
        for tid in self.inputs:
            self._check_id(tid, errs, "graph input")
        for tid in self.outputs:
            self._check_id(tid, errs, "graph output")
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            errs.append("duplicate tensor names")
        for ti, node in enumerate(self.nodes):
            where = f"node {ti} ({node.op.name})"
            for tid in node.inputs:
                # Stored tensors (weights, params, tables, blobs) are always
                # available; activations must be produced by an earlier node.
                if self.tensors[tid].storage != Storage.ACTIVATION:
                    continue
                if tid not in seen:
                    errs.append(f"{where}: input tensor {tid} is not defined yet")
            try:
                shapes = infer_shapes(node, self)
                for out, shape in zip(node.outputs, shapes):
                    got = self.tensors[out].shape
                    if tuple(got) != tuple(shape):
                        errs.append(f"{where}: declared output {got} != inferred {shape}")
            except VirError as exc:
                errs.append(f"{where}: {exc}")
            seen.update(node.outputs)
        for tid, t in enumerate(self.tensors):
            if t.storage == Storage.ACTIVATION and tid in seen:
                continue
            if t.storage == Storage.ACTIVATION:
                errs.append(f"tensor {tid} ({t.name}) is an activation but never produced")
        if require_data:
            for tid, t in enumerate(self.tensors):
                if t.storage != Storage.ACTIVATION and t.data is None:
                    errs.append(f"tensor {tid} ({t.name}) has no data")
        if errs:
            raise VirError("; ".join(errs[:8]))

    def _check_id(self, tid: int, errs: list[str], what: str) -> None:
        if not 0 <= tid < len(self.tensors):
            errs.append(f"{what} id {tid} out of range")


# ---------------------------------------------------------------------------
# shape and type inference (also the load-time validator)
# ---------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise VirError(msg)


def _in_shape(node: Node, g: Graph, i: int) -> tuple[int, ...]:
    return tuple(g.tensors[node.inputs[i]].shape)


def _in_dtype(node: Node, g: Graph, i: float) -> str:
    return g.tensors[node.inputs[int(i)]].dtype


def _check_weight_scales(opname: str, node: Node, w, m, s, b) -> None:
    """wgroup = 0: m is [N] (one scale per output row, the v2 form).
    wgroup = G > 0: m is [N, ceil(K/G)] (one mantissa per row per group of G
    consecutive K), s and b stay [N]: the row shares one power-of-two shift."""
    n, k = w[0], w[1]
    _require(k <= MAX_K, f"{opname}: K = {k} exceeds {MAX_K}")
    grp = int(node.attr(Attr.WGROUP) or 0)
    _require(grp == 0 or 1 <= grp <= k, f"{opname}: wgroup must be 0 or in [1, K]")
    want_m = (n,) if grp == 0 else (n, -(-k // grp))
    _require(tuple(m) == want_m, f"{opname}: m must be {list(want_m)}")
    for name, t in (("s", s), ("b", b)):
        _require(tuple(t) == (n,), f"{opname}: {name} must be [{n}]")


def _grouped_acc(cols, w, m, grp: int):
    """sum_j m[n,j] * (sum_{k in group j} q[t,k] * w[n,k]), exact in int64.

    |partial| <= G*32767*127 and m < 2^15, so every term is < 2^37*G; the loader
    bounds K so the sum cannot overflow (doc/vir-format.md section 1)."""
    k = w.shape[1]
    acc = np.zeros((cols.shape[0], w.shape[0]), dtype=np.int64)
    w64 = _i64(w)
    m64 = _i64(m)
    for j, k0 in enumerate(range(0, k, grp)):
        part = cols[:, k0:k0 + grp] @ w64[:, k0:k0 + grp].T
        acc += part * m64[None, :, j]
    return acc


def infer_shapes(node: Node, g: Graph) -> list[tuple[int, ...]]:
    """Return the output shapes of *node*; raise VirError on any violation."""
    op = node.op
    shp = lambda i: _in_shape(node, g, i)  # noqa: E731
    if op is Op.GEMM:
        x, w, m, s, b = shp(0), shp(1), shp(2), shp(3), shp(4)
        _require(len(x) == 2, "GEMM: x must be rank 2 [T,K]")
        _require(len(w) == 2, "GEMM: W must be rank 2 [N,K]")
        _require(x[1] == w[1], f"GEMM: K mismatch {x[1]} != {w[1]}")
        _check_weight_scales("GEMM", node, w, m, s, b)
        _require(_in_dtype(node, g, 1) in WEIGHT_DTYPES, "GEMM: W must be i8 or i16")
        return [(x[0], w[0])]
    if op is Op.CONV2D:
        x, w = shp(0), shp(1)
        kh, kw = int(node.attr(Attr.KH)), int(node.attr(Attr.KW))
        stride, pad = int(node.attr(Attr.STRIDE)), int(node.attr(Attr.PAD))
        _require(len(x) == 3, "CONV2D: x must be rank 3 [C,H,W]")
        _require(len(w) == 2, "CONV2D: W must be rank 2 [N,C*KH*KW]")
        _require(kh >= 1 and kw >= 1 and stride >= 1, "CONV2D: bad geometry")
        _require(w[1] == x[0] * kh * kw, f"CONV2D: K mismatch {w[1]} != {x[0]}*{kh}*{kw}")
        ho = (x[1] + 2 * pad - kh) // stride + 1
        wo = (x[2] + 2 * pad - kw) // stride + 1
        _require(ho > 0 and wo > 0, "CONV2D: empty output")
        _check_weight_scales("CONV2D", node, w, shp(2), shp(3), shp(4))
        _require(_in_dtype(node, g, 1) in WEIGHT_DTYPES, "CONV2D: W must be i8 or i16")
        return [(w[0], ho, wo)]
    if op is Op.GROUPNORM:
        x, gamma, beta, eps = shp(0), shp(1), shp(2), shp(3)
        groups = int(node.attr(Attr.GROUPS))
        _require(len(x) == 3, "GROUPNORM: x must be rank 3 [C,H,W]")
        _require(groups >= 1 and x[0] % groups == 0, "GROUPNORM: groups must divide C")
        _require(gamma == (x[0],) and beta == (x[0],), "GROUPNORM: gamma/beta must be [C]")
        _require(_prod(eps) == 1, "GROUPNORM: eps must hold 1 element")
        return [x]
    if op is Op.LAYERNORM:
        x = shp(0)
        _require(len(x) == 2, "LAYERNORM: x must be rank 2 [T,C]")
        _require(shp(1) == (x[1],) and shp(2) == (x[1],), "LAYERNORM: gamma/beta must be [C]")
        _require(_prod(shp(3)) == 1, "LAYERNORM: eps must hold 1 element")
        return [x]
    if op is Op.LUT_ACT:
        x, table = shp(0), shp(1)
        _require(len(table) == 1 and table[0] >= 2, "LUT_ACT: table must be [n], n >= 2")
        _require(len(x) >= 1, "LUT_ACT: x must have rank >= 1")
        return [x]
    if op is Op.ATTENTION:
        q, k, v, lut = shp(0), shp(1), shp(2), shp(3)
        heads, dh = int(node.attr(Attr.NUM_HEADS)), int(node.attr(Attr.HEAD_DIM))
        _require(len(q) == 2 and len(k) == 2 and len(v) == 2,
                 "ATTENTION: q/k/v must be rank 2 [T,C]")
        _require(heads >= 1 and dh >= 1, "ATTENTION: heads/head_dim must be >= 1")
        _require(q[1] == heads * dh, "ATTENTION: q width != heads*head_dim")
        _require(k[0] == v[0] and k[1] == q[1] and v[1] == q[1],
                 "ATTENTION: k/v must be [Tk, C] with the same C")
        _require(int(node.attr(Attr.Q_SCALE)) >= 1, "ATTENTION: q_scale must be >= 1")
        _require(lut == (1025,), "ATTENTION: exp table must be [1025] (opset 1)")
        if int(node.attr(Attr.CAUSAL)):
            _require(q[0] == k[0], "ATTENTION: causal requires Tq == Tk")
        return [(q[0], q[1])]
    if op is Op.UPSAMPLE:
        x = shp(0)
        f = int(node.attr(Attr.FACTOR))
        _require(len(x) == 3 and f >= 1, "UPSAMPLE: x must be [C,H,W] and factor >= 1")
        return [(x[0], x[1] * f, x[2] * f)]
    if op is Op.DOWNSAMPLE:
        x = shp(0)
        f = int(node.attr(Attr.FACTOR))
        _require(len(x) == 3 and f >= 1, "DOWNSAMPLE: x must be [C,H,W] and factor >= 1")
        return [(x[0], x[1] // f, x[2] // f)]
    if op is Op.ADD:
        a, b = shp(0), shp(1)
        _require(a == b, f"ADD: shape mismatch {a} vs {b}")
        return [a]
    if op is Op.CONCAT:
        a, b = shp(0), shp(1)
        axis = int(node.attr(Attr.AXIS))
        _require(len(a) == len(b) and 0 <= axis < len(a), "CONCAT: bad rank/axis")
        for i, (p, q) in enumerate(zip(a, b)):
            if i != axis:
                _require(p == q, f"CONCAT: axis {i} mismatch {p} != {q}")
        out = list(a)
        out[axis] += b[axis]
        return [tuple(out)]
    if op is Op.RESHAPE:
        x = shp(0)
        shape = tuple(int(d) for d in node.attr(Attr.SHAPE))
        perm = tuple(int(d) for d in node.attr(Attr.PERM))
        _require(shape and all(d >= 1 for d in shape), "RESHAPE: bad target shape")
        _require(_prod(shape) == _prod(x), f"RESHAPE: element count {_prod(shape)} != {_prod(x)}")
        if perm:
            _require(len(perm) == len(x) and sorted(perm) == list(range(len(x))),
                     "RESHAPE: perm must be a permutation of the input rank")
        return [shape]
    if op is Op.SCALE:
        x = shp(0)
        _require(int(node.attr(Attr.MULT)) != 0, "SCALE: mult must be non-zero")
        return [x]
    if op is Op.MUL:
        a, b = shp(0), shp(1)
        _require(a == b, f"MUL: shape mismatch {a} vs {b}")
        return [a]
    if op is Op.EMBED:
        ids, w, m, s, pos = shp(0), shp(1), shp(2), shp(3), shp(4)
        _require(len(ids) == 1, "EMBED: ids must be rank 1 [T]")
        _require(len(w) == 2, "EMBED: W must be [V,C]")
        _require(_in_dtype(node, g, 1) in WEIGHT_DTYPES, "EMBED: W must be i8 or i16")
        _require(m == (w[0],) and s == (w[0],), "EMBED: m/s must be [V]")
        _require(pos == (ids[0], w[1]), "EMBED: pos must be [T,C]")
        return [pos]
    raise VirError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# operator implementations -- the normative integer semantics
# ---------------------------------------------------------------------------


def op_gemm(node: Node, ins: list[Any]) -> Any:
    """y[t,n] = sat32(rshr(acc[t,n] * m[n], s[n] - e[t]) + b[n]).

    acc is the exact int64 product of the per-row 16-bit quantised activations
    with the int8 weights ([N,K], row-major), i.e. sdint.lin().
    """
    x, w, m, s, b = ins
    e = row_exp(row_max_abs(x))
    q = quant16(x, e)
    grp = int(node.attr(Attr.WGROUP) or 0)
    if grp:
        y = rshr(_grouped_acc(q, w, m, grp), _i64(s)[None, :] - e)
    else:
        acc = q @ _i64(w).T
        y = rshr(acc * _i64(m)[None, :], _i64(s)[None, :] - e)
    return sat32(y + _i64(b)[None, :])


def _conv_cols(x, kh: int, kw: int, stride: int, pad: int, ho: int, wo: int):
    """im2col in the C++ (channel, ky, kx) order, with the per-pixel scale."""
    c, h, w = x.shape
    xp = np.pad(_i64(x), ((0, 0), (pad, pad), (pad, pad)))
    # Per output pixel: max |x| over every tap position (C++ Conv builds pm as
    # max over channels and then takes the max over the taps).
    pm = np.abs(_i64(x)).max(axis=0)
    pmp = np.pad(pm, pad)
    win = np.zeros((ho, wo), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            win = np.maximum(win, pmp[ky:ky + stride * (ho - 1) + 1:stride,
                                     kx:kx + stride * (wo - 1) + 1:stride][:ho, :wo])
    e = row_exp(win)
    cols = np.zeros((ho * wo, c * kh * kw), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            block = xp[:, ky:ky + stride * (ho - 1) + 1:stride,
                          kx:kx + stride * (wo - 1) + 1:stride][:, :ho, :wo]
            q = quant16(block, e[None, :, :]).reshape(c, ho * wo)
            for ci in range(c):
                cols[:, (ci * kh + ky) * kw + kx] = q[ci]
    return cols, e


def op_conv2d(node: Node, ins: list[Any]) -> Any:
    """3x3/1x1 convolution as an unfolded GEMM; x is [C,H,W], out [N,Ho,Wo]."""
    x, w, m, s, b = ins
    kh, kw = int(node.attr(Attr.KH)), int(node.attr(Attr.KW))
    stride, pad = int(node.attr(Attr.STRIDE)), int(node.attr(Attr.PAD))
    c, h, wd = x.shape
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    cols, e = _conv_cols(x, kh, kw, stride, pad, ho, wo)
    grp = int(node.attr(Attr.WGROUP) or 0)
    if grp:
        y = rshr(_grouped_acc(cols, w, m, grp), _i64(s)[None, :] - e.reshape(-1, 1))
    else:
        acc = cols @ _i64(w).T
        y = rshr(acc * _i64(m)[None, :], _i64(s)[None, :] - e.reshape(-1, 1))
    y = sat32(y + _i64(b)[None, :])
    return y.T.reshape(w.shape[0], ho, wo)


def _norm_core(v, den: int):
    """NormCore: integer mean, pre-shifted variance, exact integer sqrt.

    v is [G, n] (G groups, n elements each); returns Q16 normalised values.
    This is a line-by-line transcription of poi::sd::NormCore.
    """
    n = v.shape[1]
    nn = np.int64(n)
    mean = floor_div(v.sum(axis=1, keepdims=True) + n // 2, nn)
    d = v - mean
    e = np.maximum(row_exp(np.abs(d).max(axis=1, keepdims=True)) - 5, 0)
    dd = rshr(d, e)
    var = floor_div((dd * dd).sum(axis=1, keepdims=True), nn)
    eps = floor_div(np.left_shift(np.int64(1), 32 - 2 * e) + den // 2, den)
    sd = isqrt((var + eps) << np.int64(BITS))
    return floor_div(np.left_shift(d, 24 - e) * 2 + sd, 2 * sd)


def op_groupnorm(node: Node, ins: list[Any]) -> Any:
    """GroupNorm over groups of channels; x is [C,H,W]."""
    x, gamma, beta, eps = ins
    groups = int(node.attr(Attr.GROUPS))
    den = int(_i64(eps).reshape(-1)[0])
    c, h, w = x.shape
    v = _i64(x).reshape(groups, (c // groups) * h * w)
    y = _norm_core(v, den).reshape(c, h * w)
    y = sat32(rshr(y * _i64(gamma)[:, None], BITS) + _i64(beta)[:, None])
    return y.reshape(c, h, w)


def op_layernorm(node: Node, ins: list[Any]) -> Any:
    """LayerNorm over the last axis; x is [T,C]."""
    x, gamma, beta, eps = ins
    den = int(_i64(eps).reshape(-1)[0])
    y = _norm_core(_i64(x), den)
    y = sat32(rshr(y * _i64(gamma)[None, :], BITS) + _i64(beta)[None, :])
    return y


def lut_value(x, table, lo: int, step_bits: int, edge_lo: int, edge_hi: int):
    """Interpolated table lookup on [lo, lo + (n-1)*2**step_bits), Q16 domain."""
    x = _i64(x)
    table = _i64(table)
    n = table.shape[0]
    span = (n - 1) << step_bits
    mask = (1 << step_bits) - 1
    z = x - lo
    zc = np.clip(z, 0, span - 1)
    i = zc >> step_bits
    f = zc & mask
    t0 = table[i]
    t1 = table[i + 1]
    v = t0 + (((t1 - t0) * f) >> step_bits)
    v = np.where(z < 0, edge_lo, v)
    v = np.where(z >= span, edge_hi, v)
    return v


def op_lut_act(node: Node, ins: list[Any]) -> Any:
    """y = sat32(rshr(x * table(x), post_shift)); SiLU, GELU and variants.

    One op, three tables: silu = x*sigmoid(x), gelu = x*Phi(x), quick_gelu =
    x*sigmoid(1.702x) (the 1.702 is baked into the table, so no
    non-power-of-two arithmetic enters the runtime).
    """
    x, table = ins
    v = lut_value(x, table, int(node.attr(Attr.LUT_LO)), int(node.attr(Attr.LUT_STEP_BITS)),
                  int(node.attr(Attr.LUT_EDGE_LO)), int(node.attr(Attr.LUT_EDGE_HI)))
    return sat32(rshr(_i64(x) * v, int(node.attr(Attr.POST_SHIFT))))


def _exp_table_value(z, table):
    """exp lookup used inside attention: z >= 0, domain [0,16), Q30 output."""
    return lut_value(z, table, 0, 10, 0, 0)


def op_attention(node: Node, ins: list[Any]) -> Any:
    """Softmax attention, head-major, exactly as poi::sd::Attention.

    Inputs q,k,v [T,*], exp table [1025]; attribute q_scale is
    round(2**16 / sqrt(head_dim)) so that a head_dim of 64 gives the v2
    constant 2**13 and the score shift 19 - e_q - e_k.
    """
    q, k, v, exp_lut = ins
    heads = int(node.attr(Attr.NUM_HEADS))
    dh = int(node.attr(Attr.HEAD_DIM))
    causal = bool(int(node.attr(Attr.CAUSAL)))
    qs = int(node.attr(Attr.Q_SCALE))
    tq, c = q.shape
    tk = k.shape[0]
    out = np.zeros((tq, c), dtype=np.int64)
    tri = np.triu(np.ones((tq, tk), dtype=bool), 1) if causal else None
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        qh = _i64(q)[:, sl]
        kh = _i64(k)[:, sl]
        vh = _i64(v)[:, sl]
        eq = row_exp(row_max_abs(qh))
        ek = row_exp(row_max_abs(kh))
        s = (quant16(qh, eq) @ quant16(kh, ek).T) * qs
        s = rshr(s, (32 - eq) - ek.T)
        if tri is not None:
            s = np.where(tri, np.int64(I32_MIN), s)
        mx = s.max(axis=1, keepdims=True)
        z = mx - s
        e = _exp_table_value(z, exp_lut)
        r = floor_div(np.int64(1) << 62, e.sum(axis=1, keepdims=True))
        p = np.minimum((e * r + (np.int64(1) << 46)) >> 47, 32767)
        ev = row_exp(np.abs(vh).max(axis=0, keepdims=True))
        o = p @ quant16(vh, ev)
        out[:, sl] = rshr(o, 15 - ev)
    return out


def op_upsample(node: Node, ins: list[Any]) -> Any:
    x = _i64(ins[0])
    f = int(node.attr(Attr.FACTOR))
    if int(node.attr(Attr.MODE)) != 0:
        raise NotSupported("UPSAMPLE: only mode 0 (nearest) exists in opset 1")
    return np.repeat(np.repeat(x, f, axis=1), f, axis=2)


def op_downsample(node: Node, ins: list[Any]) -> Any:
    """Strided pooling downsample.  Not used by v2 or SDXS (both downsample
    with a stride-2 Conv2d), so opset 1 leaves the modes undefined.

    Signature: ``f(x: int64[C,H,W], factor: int, mode: int) -> int64[C,H//f,W//f]``
      mode 0 = average (round half up on the sum, then divide), 1 = max.
    """
    raise NotImplementedError(
        "DOWNSAMPLE forward is not part of opset 1: SD-class UNets downsample with "
        "CONV2D(stride=2). The signature is f(x[C,H,W], factor, mode) -> [C,H//f,W//f]; "
        "modes 0 (average, round-half-up) and 1 (max) are TBD.")


def op_add(node: Node, ins: list[Any]) -> Any:
    return sat32(_i64(ins[0]) + _i64(ins[1]))


def op_concat(node: Node, ins: list[Any]) -> Any:
    axis = int(node.attr(Attr.AXIS))
    return np.concatenate([_i64(ins[0]), _i64(ins[1])], axis=axis)


def op_reshape(node: Node, ins: list[Any]) -> Any:
    x = _i64(ins[0])
    perm = tuple(int(d) for d in node.attr(Attr.PERM))
    if perm:
        x = np.transpose(x, perm)
    return x.reshape(tuple(int(d) for d in node.attr(Attr.SHAPE)))


def op_scale(node: Node, ins: list[Any]) -> Any:
    """Fixed-point scale by an integer constant: sat32(rshr(x * c, shift))."""
    x = _i64(ins[0])
    return sat32(rshr(x * np.int64(int(node.attr(Attr.MULT))), int(node.attr(Attr.SHIFT))))


def op_mul(node: Node, ins: list[Any]) -> Any:
    """Element-wise Q16 product with rescale: sat32(rshr(a * b, shift))."""
    return sat32(rshr(_i64(ins[0]) * _i64(ins[1]), int(node.attr(Attr.SHIFT))))


def op_embed(node: Node, ins: list[Any]) -> Any:
    """Token embedding with per-row scale: y = sat32(rshr(w[id]*m[id], s[id]-16) + pos)."""
    ids, w, m, s, pos = ins
    ids = _i64(ids).reshape(-1)
    w = _i64(w)
    shift = (_i64(s)[ids] - int(node.attr(Attr.POST_SHIFT)))[:, None]
    y = rshr(w[ids] * _i64(m)[ids][:, None], shift)
    return sat32(y + _i64(pos))


OP_FORWARD = {
    Op.GEMM: op_gemm,
    Op.CONV2D: op_conv2d,
    Op.GROUPNORM: op_groupnorm,
    Op.LAYERNORM: op_layernorm,
    Op.LUT_ACT: op_lut_act,
    Op.ATTENTION: op_attention,
    Op.UPSAMPLE: op_upsample,
    Op.DOWNSAMPLE: op_downsample,
    Op.ADD: op_add,
    Op.CONCAT: op_concat,
    Op.RESHAPE: op_reshape,
    Op.SCALE: op_scale,
    Op.MUL: op_mul,
    Op.EMBED: op_embed,
}


# ---------------------------------------------------------------------------
# file sections: header, graph, tensors, noise
# ---------------------------------------------------------------------------


@dataclass
class VirHeader:
    opset: int = OPSET
    file_kind: int = KIND_LATENT_1STEP
    flags: int = 0
    latent: tuple[int, int, int] = (4, 64, 64)
    out: tuple[int, int, int] = (4, 64, 64)
    out_dtype: int = OUT_I16LE
    clamp_abs: int = 32767
    out_shift: int = 8
    tok_type: int = TOK_NONE
    tok_vocab_tensor: int = 0
    tok_merges_tensor: int = 0
    vocab_size: int = 0
    bos_id: int = 0
    eos_id: int = 0
    pad_id: int = 0
    pad_policy: int = PAD_USE_PAD_ID
    ctx_len: int = 0
    max_tokens: int = 0
    te_hidden: int = 0
    te_layers: int = 0
    input_count: int = 1
    output_count: int = 1
    n_nodes: int = 0
    n_tensors: int = 0

    # section bookkeeping (filled in by the writer, checked by the reader)
    file_size: int = 0
    graph_off: int = HEADER_SIZE
    graph_size: int = 0
    tensors_off: int = 0
    tensors_size: int = 0
    noise_off: int = 0
    noise_size: int = NOISE_SIZE

    @property
    def output_bytes(self) -> int:
        c, h, w = self.out
        return c * h * w * DTYPE_SIZE["i16"]

    @property
    def has_tokenizer(self) -> bool:
        return bool(self.flags & 1)


#: (name, offset, struct code).  Reserved gaps must be zero.
_HDR_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("magic", 0, "4s"),
    ("opset", 4, "B"),
    ("file_kind", 5, "B"),
    ("flags", 6, "B"),
    ("align_log2", 7, "B"),
    ("header_size", 8, "H"),
    ("file_version", 10, "B"),
    ("file_size", 16, "Q"),
    ("graph_off", 24, "Q"),
    ("graph_size", 32, "I"),
    ("tensors_off", 40, "Q"),
    ("tensors_size", 48, "Q"),
    ("noise_off", 56, "Q"),
    ("noise_size", 64, "I"),
    ("latent_c", 72, "H"),
    ("latent_h", 74, "H"),
    ("latent_w", 76, "H"),
    ("out_c", 78, "H"),
    ("out_h", 80, "H"),
    ("out_w", 82, "H"),
    ("out_dtype", 84, "B"),
    ("clamp_abs", 88, "H"),
    ("out_shift", 90, "B"),
    ("tok_vocab", 92, "I"),
    ("tok_merges", 96, "I"),
    ("vocab_size", 100, "I"),
    ("bos_id", 104, "I"),
    ("eos_id", 108, "I"),
    ("pad_id", 112, "I"),
    ("ctx_len", 116, "H"),
    ("max_tokens", 118, "H"),
    ("te_hidden", 120, "H"),
    ("te_layers", 122, "H"),
    ("tok_type", 124, "B"),
    ("pad_policy", 125, "B"),
    ("input_count", 128, "H"),
    ("output_count", 130, "H"),
    ("n_nodes", 136, "I"),
    ("n_tensors", 140, "I"),
)

#: byte ranges of the header that must be zero in a canonical file
_HDR_RESERVED: tuple[tuple[int, int], ...] = (
    (11, 5), (36, 4), (68, 4), (85, 3), (91, 1), (126, 2), (132, 4), (144, 2),
    (146, HEADER_SIZE - 146),
)


@dataclass
class NoiseSpec:
    """Section 4: how the latent noise is derived from the seed, and the
    one-step sampler that turns (noise, eps) into x0."""

    noise_version: int = FORMAT_VERSION
    noise_kind: int = 0          # 0 = ChaCha20 keystream -> u16le -> gauss LUT
    sampler_kind: int = 0        # 0 = one sigma (v2 Euler), 1 = two coefficients (a, b)
    n_samples: int = 4 * 64 * 64
    gauss_tensor: int = 0
    eps_tensor: int = 0
    latent_tensor: int = 0
    cin_q30: int = 0             # sigma / sqrt(sigma^2 + 1) in Q30
    sigma_q16: int = 0           # sigma (kind 0) or a (kind 1) in Q16
    #: kind 1 only: b, the epsilon coefficient. x0 = rshr(n*a - eps*b, latent_shift).
    #: 0 with kind 0 is the v2 form x0 = rshr((n - eps)*sigma, latent_shift).
    eps_q16: int = 0
    latent_shift: int = BITS
    out_shift: int = 8           # out = rshr(x0, out_shift)
    clamp_abs: int = 32767       # then clamp to +/-clamp_abs
    seed_len: int = 32
    nonce_len: int = 12
    counter_start: int = 0
    ks_bytes: int = 2 * 4 * 64 * 64


_NOISE_FMT = "<BBBBIIIIIiiBBHIIIII8x"
_NOISE_FIELDS = (
    "noise_version", "noise_kind", "sampler_kind", "reserved0",
    "n_samples", "gauss_tensor", "eps_tensor", "latent_tensor",
    "ks_bytes", "cin_q30", "sigma_q16",
    "latent_shift", "out_shift", "clamp_abs",
    "seed_len", "nonce_len", "counter_start", "eps_q16", "reserved2",
)


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _pack_header(h: VirHeader) -> bytes:
    buf = bytearray(HEADER_SIZE)
    struct.pack_into("<4s", buf, 0, MAGIC)
    struct.pack_into("<B", buf, 4, h.opset)
    struct.pack_into("<B", buf, 5, h.file_kind)
    struct.pack_into("<B", buf, 6, h.flags)
    struct.pack_into("<B", buf, 7, ALIGN_LOG2)
    struct.pack_into("<H", buf, 8, HEADER_SIZE)
    struct.pack_into("<B", buf, 10, FORMAT_VERSION)
    struct.pack_into("<Q", buf, 16, h.file_size)
    struct.pack_into("<Q", buf, 24, h.graph_off)
    struct.pack_into("<I", buf, 32, h.graph_size)
    struct.pack_into("<Q", buf, 40, h.tensors_off)
    struct.pack_into("<Q", buf, 48, h.tensors_size)
    struct.pack_into("<Q", buf, 56, h.noise_off)
    struct.pack_into("<I", buf, 64, h.noise_size)
    struct.pack_into("<HHH", buf, 72, *h.latent)
    struct.pack_into("<HHH", buf, 78, *h.out)
    struct.pack_into("<B", buf, 84, h.out_dtype)
    struct.pack_into("<H", buf, 88, h.clamp_abs)
    struct.pack_into("<B", buf, 90, h.out_shift)
    struct.pack_into("<IIIIII", buf, 92, h.tok_vocab_tensor, h.tok_merges_tensor,
                     h.vocab_size, h.bos_id, h.eos_id, h.pad_id)
    struct.pack_into("<HH", buf, 116, h.ctx_len, h.max_tokens)
    struct.pack_into("<HH", buf, 120, h.te_hidden, h.te_layers)
    struct.pack_into("<BB", buf, 124, h.tok_type, h.pad_policy)
    struct.pack_into("<HH", buf, 128, h.input_count, h.output_count)
    struct.pack_into("<II", buf, 136, h.n_nodes, h.n_tensors)
    return bytes(buf)


def _unpack_header(data: bytes) -> VirHeader:
    if len(data) < HEADER_SIZE:
        raise VirError(f"file shorter than the {HEADER_SIZE}-byte header")
    magic, = struct.unpack_from("<4s", data, 0)
    if magic != MAGIC:
        raise VirError(f"bad magic {magic!r}, expected {MAGIC!r}")
    h = VirHeader()
    h.opset = data[4]
    if h.opset != OPSET:
        raise VirError(f"unsupported opset {h.opset} (this build implements {OPSET})")
    h.file_kind = data[5]
    h.flags = data[6]
    if data[7] != ALIGN_LOG2:
        raise VirError(f"align_log2 {data[7]} != {ALIGN_LOG2}")
    header_size, = struct.unpack_from("<H", data, 8)
    if header_size != HEADER_SIZE:
        raise VirError(f"header_size {header_size} != {HEADER_SIZE}")
    if data[10] != FORMAT_VERSION:
        raise VirError(f"header version {data[10]} != {FORMAT_VERSION}")
    h.file_size, = struct.unpack_from("<Q", data, 16)
    if h.file_size != len(data):
        raise VirError(f"file_size {h.file_size} != actual size {len(data)}")
    h.graph_off, = struct.unpack_from("<Q", data, 24)
    h.graph_size, = struct.unpack_from("<I", data, 32)
    h.tensors_off, = struct.unpack_from("<Q", data, 40)
    h.tensors_size, = struct.unpack_from("<Q", data, 48)
    h.noise_off, = struct.unpack_from("<Q", data, 56)
    h.noise_size, = struct.unpack_from("<I", data, 64)
    h.latent = struct.unpack_from("<HHH", data, 72)
    h.out = struct.unpack_from("<HHH", data, 78)
    h.out_dtype = data[84]
    h.clamp_abs, = struct.unpack_from("<H", data, 88)
    h.out_shift = data[90]
    (h.tok_vocab_tensor, h.tok_merges_tensor, h.vocab_size, h.bos_id, h.eos_id,
     h.pad_id) = struct.unpack_from("<IIIIII", data, 92)
    h.ctx_len, h.max_tokens = struct.unpack_from("<HH", data, 116)
    h.te_hidden, h.te_layers = struct.unpack_from("<HH", data, 120)
    h.tok_type, h.pad_policy = data[124], data[125]
    h.input_count, h.output_count = struct.unpack_from("<HH", data, 128)
    h.n_nodes, h.n_tensors = struct.unpack_from("<II", data, 136)
    for start, length in _HDR_RESERVED:
        chunk = data[start:start + length]
        if any(chunk):
            raise VirError(f"non-zero reserved bytes in header at {start}")
    if h.out_dtype != OUT_I16LE:
        raise VirError(f"unsupported out_dtype {h.out_dtype}")
    return h


def _pack_graph(g: Graph, idmap: dict[int, int]) -> bytes:
    out = bytearray()
    out += struct.pack("<BBBB", FORMAT_VERSION, 0, 0, 0)  # version + 3 reserved
    out += b"\0" * 4                                     # pad to offset 8
    out += struct.pack("<III", len(g.nodes), len(g.inputs), len(g.outputs))
    for tid in g.inputs:
        out += struct.pack("<I", idmap[tid])
    for tid in g.outputs:
        out += struct.pack("<I", idmap[tid])
    for node in g.nodes:
        spec = OP_SPECS[node.op]
        attrs = list(spec.sorted_attrs())
        out += struct.pack("<HHHH", int(node.op), len(attrs), len(node.inputs), len(node.outputs))
        for tid in node.inputs:
            out += struct.pack("<I", idmap[tid])
        for tid in node.outputs:
            out += struct.pack("<I", idmap[tid])
        for key in attrs:
            kind = ATTR_KIND[key]
            value = node.attrs[key]
            if kind in ("u32", "i32"):
                vals = [int(value)]
            else:
                vals = [int(v) for v in value]
            code = ATTR_TYPE_CODE[kind]
            out += struct.pack("<HBB", int(key), code, len(vals))
            if vals:
                out += struct.pack(f"<{len(vals)}I", *[v & 0xFFFFFFFF for v in vals])
    return bytes(out)


def _read_graph(data: bytes, tensors: list[TensorDesc]) -> tuple[list[Node], list[int], list[int]]:
    if len(data) < 20:
        raise VirError("graph section truncated")
    version, = struct.unpack_from("<B", data, 0)
    if version != FORMAT_VERSION:
        raise VirError(f"graph section version {version} != {FORMAT_VERSION}")
    if any(data[1:8]):
        raise VirError("non-zero reserved bytes in graph header")
    n_nodes, n_in, n_out = struct.unpack_from("<III", data, 8)
    pos = 20
    if len(data) < pos + 4 * (n_in + n_out):
        raise VirError("graph section truncated (io ids)")
    inputs = list(struct.unpack_from(f"<{n_in}I", data, pos)) if n_in else []
    pos += 4 * n_in
    outputs = list(struct.unpack_from(f"<{n_out}I", data, pos)) if n_out else []
    pos += 4 * n_out
    nodes: list[Node] = []
    for i in range(n_nodes):
        if len(data) < pos + 8:
            raise VirError(f"graph truncated at node {i}")
        op_code, n_attr, n_i, n_o = struct.unpack_from("<HHHH", data, pos)
        pos += 8
        try:
            op = Op(op_code)
        except ValueError:
            raise VirError(f"node {i}: unknown op code {op_code}") from None
        spec = OP_SPECS[op]
        if (n_i, n_o) != (spec.n_in, spec.n_out):
            raise VirError(f"node {i}: op {op.name} wants {spec.n_in}/{spec.n_out} "
                           f"inputs/outputs, file says {n_i}/{n_o}")
        if n_attr != len(spec.attrs):
            raise VirError(f"node {i}: op {op.name} wants {len(spec.attrs)} attributes, "
                           f"file says {n_attr}")
        need = 4 * (n_i + n_o)
        if len(data) < pos + need:
            raise VirError(f"graph truncated at node {i} (ids)")
        ids = list(struct.unpack_from(f"<{n_i + n_o}I", data, pos))
        pos += need
        for tid in ids:
            if tid >= len(tensors):
                raise VirError(f"node {i}: tensor id {tid} out of range")
        attrs: dict[Attr, Any] = {}
        for _ in range(n_attr):
            if len(data) < pos + 4:
                raise VirError(f"graph truncated at node {i} (attr)")
            key_code, type_code, count = struct.unpack_from("<HBB", data, pos)
            pos += 4
            try:
                key = Attr(key_code)
            except ValueError:
                raise VirError(f"node {i}: unknown attribute key {key_code}") from None
            if key not in spec.attrs:
                raise VirError(f"node {i}: op {op.name} does not take attribute {key.name}")
            kind = ATTR_TYPE_FROM_CODE.get(type_code)
            if kind is None:
                raise VirError(f"node {i}: bad attribute type {type_code}")
            if kind != ATTR_KIND[key]:
                raise VirError(f"node {i}: attribute {key.name} must be {ATTR_KIND[key]}, "
                               f"file says {kind}")
            if kind in ("u32", "i32") and count != 1:
                raise VirError(f"node {i}: scalar attribute {key.name} with count {count}")
            if len(data) < pos + 4 * count:
                raise VirError(f"graph truncated at node {i} (attr payload)")
            vals = list(struct.unpack_from(f"<{count}I", data, pos)) if count else []
            pos += 4 * count
            if kind == "i32":
                vals = [v - (1 << 32) if v & 0x80000000 else v for v in vals]
            attrs[key] = vals[0] if kind in ("u32", "i32") else tuple(vals)
        if list(attrs.keys()) != list(spec.sorted_attrs()):
            raise VirError(f"node {i}: attributes are not in canonical (ascending) order")
        nodes.append(Node(op, ids[:n_i], ids[n_i:], attrs))
    if pos != len(data):
        raise VirError(f"graph section has {len(data) - pos} trailing bytes")
    return nodes, inputs, outputs


def _pack_tensors(g: Graph, order: Sequence[int]) -> bytes:
    names = bytearray()
    records = bytearray()
    data = bytearray()
    for tid in order:
        t = g.tensors[tid]
        name = t.name.encode("ascii")
        if not 1 <= len(name) <= 255:
            raise VirError(f"tensor name {t.name!r} must be 1..255 ASCII bytes")
        records += struct.pack("<IBBBB", len(names), DTYPE_CODE[t.dtype], t.rank,
                              int(t.storage), len(name))
        if t.rank:
            records += struct.pack(f"<{t.rank}I", *t.shape)
        if t.storage == Storage.ACTIVATION:
            records += struct.pack("<QQ", 0, 0)
        else:
            if t.data is None:
                raise VirError(f"tensor {t.name!r} is stored but has no data")
            arr = np.ascontiguousarray(t.data, dtype=DTYPE_NP[t.dtype])
            if tuple(arr.shape) != tuple(t.shape):
                raise VirError(f"tensor {t.name!r}: data shape {arr.shape} != {t.shape}")
            off = _align_up(len(data))
            data += b"\0" * (off - len(data))
            records += struct.pack("<QQ", off, arr.nbytes)
            data += arr.tobytes()
        names += name
    table_size = len(records)
    name_size = len(names)
    pad = _align_up(32 + table_size + name_size) - (32 + table_size + name_size)
    head = struct.pack("<BBBBIQQQ", FORMAT_VERSION, ALIGN_LOG2, 0, 0, len(order),
                       table_size, name_size, len(data))
    body = head + bytes(records) + bytes(names) + b"\0" * pad + bytes(data)
    return body


def _read_tensors(data: bytes) -> list[TensorDesc]:
    if len(data) < 32:
        raise VirError("tensor section truncated")
    version, align_log2, r0, r1 = struct.unpack_from("<BBBB", data, 0)
    if version != FORMAT_VERSION:
        raise VirError(f"tensor section version {version} != {FORMAT_VERSION}")
    if align_log2 != ALIGN_LOG2:
        raise VirError(f"tensor align_log2 {align_log2} != {ALIGN_LOG2}")
    if r0 or r1:
        raise VirError("non-zero reserved bytes in tensor section")
    count, table_size, name_size, data_size = struct.unpack_from("<IQQQ", data, 4)
    table_off = 32
    name_off = table_off + table_size
    data_off = _align_up(name_off + name_size)
    if data_off + data_size != len(data):
        raise VirError("tensor section: data_size does not match the section length")
    if any(data[name_off + name_size:data_off]):
        raise VirError("tensor section: non-zero alignment padding")
    names = data[name_off:name_off + name_size]
    tensors: list[TensorDesc] = []
    pos = table_off
    expected_data = 0
    last_end = 0
    seen_names: list[bytes] = []
    for i in range(count):
        if pos + 8 > name_off:
            raise VirError(f"tensor table truncated at entry {i}")
        name_rel, dtype_code, rank, storage_code, name_len = struct.unpack_from(
            "<IBBBB", data, pos)
        pos += 8
        if rank > 4:
            raise VirError(f"tensor {i}: rank {rank} > 4")
        if pos + 4 * rank + 16 > name_off:
            raise VirError(f"tensor table truncated at entry {i} (dims)")
        shape = tuple(struct.unpack_from(f"<{rank}I", data, pos)) if rank else ()
        pos += 4 * rank
        data_rel, nbytes = struct.unpack_from("<QQ", data, pos)
        pos += 16
        if dtype_code not in DTYPE_FROM_CODE:
            raise VirError(f"tensor {i}: bad dtype {dtype_code}")
        if storage_code not in (0, 1, 2, 3, 4):
            raise VirError(f"tensor {i}: bad storage {storage_code}")
        if name_len < 1 or name_rel + name_len > name_size:
            raise VirError(f"tensor {i}: name out of range")
        raw_name = names[name_rel:name_rel + name_len]
        if not all(0x21 <= b <= 0x7E for b in raw_name):
            raise VirError(f"tensor {i}: name is not printable ASCII")
        if seen_names and raw_name <= seen_names[-1]:
            raise VirError(f"tensor {i}: names are not strictly ascending")
        seen_names.append(raw_name)
        name = raw_name.decode("ascii")
        desc = TensorDesc(name, shape, DTYPE_FROM_CODE[dtype_code], Storage(storage_code))
        if desc.storage == Storage.ACTIVATION:
            if nbytes != 0 or data_rel != 0:
                raise VirError(f"tensor {i} ({name}): activation with stored bytes")
        else:
            if nbytes != desc.nbytes:
                raise VirError(f"tensor {i} ({name}): nbytes {nbytes} != {desc.count}"
                               f"*{DTYPE_SIZE[desc.dtype]}")
            if data_rel % ALIGN or data_rel != expected_data:
                raise VirError(f"tensor {i} ({name}): data offset {data_rel} is not the "
                               f"canonical {expected_data}")
            if data_rel + nbytes > data_size:
                raise VirError(f"tensor {i} ({name}): data past the data section")
            expected_data = _align_up(data_rel + nbytes)
            last_end = data_rel + nbytes
            raw = data[data_off + data_rel:data_off + data_rel + nbytes]
            if np is not None:
                desc.data = np.frombuffer(raw, dtype=DTYPE_NP[desc.dtype],
                                          count=desc.count).reshape(shape)
        tensors.append(desc)
    if pos != name_off:
        raise VirError("tensor table size mismatch")
    if data_size != last_end:
        raise VirError("tensor data section has trailing bytes")
    return tensors


def _pack_noise(spec: NoiseSpec) -> bytes:
    return struct.pack(_NOISE_FMT, spec.noise_version, spec.noise_kind, spec.sampler_kind, 0,
                       spec.n_samples, spec.gauss_tensor, spec.eps_tensor, spec.latent_tensor,
                       spec.ks_bytes, spec.cin_q30, spec.sigma_q16, spec.latent_shift,
                       spec.out_shift, spec.clamp_abs, spec.seed_len, spec.nonce_len,
                       spec.counter_start, spec.eps_q16, 0)


def _read_noise(data: bytes) -> NoiseSpec:
    if len(data) != NOISE_SIZE:
        raise VirError(f"noise section must be {NOISE_SIZE} bytes, got {len(data)}")
    fields = struct.unpack(_NOISE_FMT, data)
    spec = NoiseSpec()
    spec.noise_version, spec.noise_kind, spec.sampler_kind, r0 = fields[0:4]
    if spec.noise_version != FORMAT_VERSION:
        raise VirError(f"noise version {spec.noise_version} != {FORMAT_VERSION}")
    if r0:
        raise VirError("non-zero reserved byte in noise spec")
    if spec.noise_kind != 0:
        raise VirError(f"unsupported noise_kind {spec.noise_kind}")
    (spec.n_samples, spec.gauss_tensor, spec.eps_tensor, spec.latent_tensor,
     spec.ks_bytes, spec.cin_q30, spec.sigma_q16, spec.latent_shift, spec.out_shift,
     spec.clamp_abs, spec.seed_len, spec.nonce_len, spec.counter_start,
     spec.eps_q16, r2) = fields[4:]
    if r2:
        raise VirError("non-zero reserved word in noise spec")
    if spec.sampler_kind == 0 and spec.eps_q16:
        raise VirError("sampler_kind 0 must not set eps_q16")
    if spec.sampler_kind not in (0, 1):
        raise VirError(f"unsupported sampler_kind {spec.sampler_kind}")
    return spec


# ---------------------------------------------------------------------------
# ChaCha20 (RFC 8439, zero nonce, counter 0) -- the v2 noise source
# ---------------------------------------------------------------------------


def _rotl32(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF


def _quarter(s, a, b, c, d) -> None:
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF
    s[d] = _rotl32(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF
    s[b] = _rotl32(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF
    s[d] = _rotl32(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF
    s[b] = _rotl32(s[b] ^ s[c], 7)


def chacha20_keystream(key: bytes, nbytes: int, nonce: bytes = b"\0" * 12,
                       counter: int = 0) -> bytes:
    """Keystream of ChaCha20 with the RFC 8439 (96-bit nonce) layout.

    Identical to what the node produces with
    ``ChaCha20 rng{seed}; rng.Seek({0,0}, 0); rng.Keystream(out)`` and to
    ``sdint.chacha20_keystream(seed, n)``, which uses the 64/64 split with an
    all-zero 16-byte nonce: with a zero nonce and counter 0 the two layouts
    agree for the first 2**32 blocks.
    """
    if len(key) != 32:
        raise VirError("ChaCha20 key must be 32 bytes")
    if len(nonce) != 12:
        raise VirError("ChaCha20 nonce must be 12 bytes")
    const = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
    k = struct.unpack("<8I", key)
    n = struct.unpack("<3I", nonce)
    out = bytearray()
    block = 0
    while len(out) < nbytes:
        s = list(const) + list(k) + [counter + block & 0xFFFFFFFF] + list(n)
        x = list(s)
        for _ in range(10):
            _quarter(x, 0, 4, 8, 12)
            _quarter(x, 1, 5, 9, 13)
            _quarter(x, 2, 6, 10, 14)
            _quarter(x, 3, 7, 11, 15)
            _quarter(x, 0, 5, 10, 15)
            _quarter(x, 1, 6, 11, 12)
            _quarter(x, 2, 7, 8, 13)
            _quarter(x, 3, 4, 9, 14)
        out += struct.pack("<16I", *[(a + b) & 0xFFFFFFFF for a, b in zip(x, s)])
        block += 1
    return bytes(out[:nbytes])


@dataclass
class VirModel:
    header: VirHeader
    graph: Graph
    noise: NoiseSpec
    raw: bytes | None = None

    # -- serialisation -------------------------------------------------
    def to_bytes(self) -> bytes:
        g = self.graph
        g.validate(require_data=True)
        order = sorted(range(len(g.tensors)), key=lambda i: g.tensors[i].name.encode("ascii"))
        idmap = {old: new for new, old in enumerate(order)}
        graph_bytes = _pack_graph(g, idmap)
        tensor_bytes = _pack_tensors(g, order)
        h = replace(self.header)
        h.opset = OPSET
        h.graph_off = HEADER_SIZE
        h.graph_size = len(graph_bytes)
        h.tensors_off = _align_up(h.graph_off + h.graph_size)
        h.tensors_size = len(tensor_bytes)
        h.noise_off = _align_up(h.tensors_off + h.tensors_size)
        h.noise_size = NOISE_SIZE
        h.file_size = h.noise_off + NOISE_SIZE
        h.n_nodes = len(g.nodes)
        h.n_tensors = len(g.tensors)
        h.input_count = len(g.inputs)
        h.output_count = len(g.outputs)
        noise = replace(self.noise)
        for name in ("gauss_tensor", "eps_tensor", "latent_tensor"):
            setattr(noise, name, idmap[getattr(self.noise, name)])
        h.tok_vocab_tensor = idmap.get(self.header.tok_vocab_tensor, 0)
        h.tok_merges_tensor = idmap.get(self.header.tok_merges_tensor, 0)
        noise_bytes = _pack_noise(noise)
        out = bytearray(_pack_header(h))
        out += graph_bytes
        out += b"\0" * (h.tensors_off - len(out))
        out += tensor_bytes
        out += b"\0" * (h.noise_off - len(out))
        out += noise_bytes
        assert len(out) == h.file_size
        return bytes(out)

    def model_hash(self) -> str:
        data = self.raw if self.raw is not None else self.to_bytes()
        return model_hash(self.header.opset, data)

    # -- parsing -------------------------------------------------------
    @classmethod
    def parse(cls, data: bytes) -> "VirModel":
        if len(data) < HEADER_SIZE:
            raise VirError("file shorter than the header")
        h = _unpack_header(data)
        if h.graph_off != HEADER_SIZE:
            raise VirError("non-canonical graph offset")
        if h.tensors_off != _align_up(h.graph_off + h.graph_size):
            raise VirError("non-canonical tensor offset")
        if h.noise_off != _align_up(h.tensors_off + h.tensors_size):
            raise VirError("non-canonical noise offset")
        if h.noise_size != NOISE_SIZE or h.file_size != h.noise_off + NOISE_SIZE:
            raise VirError("file does not end after the noise section")
        if h.file_size > 4 << 30:
            raise VirError("file exceeds the 4 GiB admission limit")
        for start, length in ((h.graph_off + h.graph_size, h.tensors_off - h.graph_off - h.graph_size),
                              (h.tensors_off + h.tensors_size, h.noise_off - h.tensors_off - h.tensors_size)):
            if any(data[start:start + length]):
                raise VirError("non-zero bytes in section padding")
        graph_bytes = data[h.graph_off:h.graph_off + h.graph_size]
        tensors = _read_tensors(data[h.tensors_off:h.tensors_off + h.tensors_size])
        nodes, inputs, outputs = _read_graph(graph_bytes, tensors)
        g = Graph()
        g.tensors = tensors
        g.nodes = nodes
        g.inputs = inputs
        g.outputs = outputs
        g._by_name = {t.name: i for i, t in enumerate(tensors)}  # noqa: SLF001
        g.validate()
        noise = _read_noise(data[h.noise_off:h.noise_off + h.noise_size])
        model = cls(h, g, noise, data)
        model._check_header()
        return model

    def _check_header(self) -> None:
        h, g, n = self.header, self.graph, self.noise
        if h.file_kind != KIND_LATENT_1STEP:
            raise VirError(f"unsupported file_kind {h.file_kind}")
        if h.flags & 0xFE:
            raise VirError("reserved flag bits must be zero")
        if h.n_nodes != len(g.nodes):
            raise VirError("header node count does not match the graph")
        if h.n_tensors != len(g.tensors):
            raise VirError("header tensor count does not match the table")
        if h.input_count != len(g.inputs) or h.output_count != len(g.outputs):
            raise VirError("header io counts do not match the graph")
        if len(g.outputs) != 1:
            raise VirError("opset 1 expects exactly one graph output (eps)")
        for tid in (n.gauss_tensor, n.eps_tensor, n.latent_tensor,
                    h.tok_vocab_tensor, h.tok_merges_tensor):
            if tid >= len(g.tensors) and tid != 0:
                raise VirError(f"header/noise references missing tensor {tid}")
        latent = tuple(h.latent)
        if min(latent) < 1 or _prod(latent) > (1 << 26):
            raise VirError("bad latent shape")
        if n.n_samples != _prod(latent) or n.ks_bytes != 2 * n.n_samples:
            raise VirError("noise sample count does not match the latent shape")
        if n.seed_len != 32 or n.nonce_len != 12:
            raise VirError("ChaCha20 needs a 32-byte seed and a 96-bit zero nonce")
        gauss = g.tensors[n.gauss_tensor]
        if tuple(gauss.shape) != (65536,) or gauss.dtype != "i32":
            raise VirError("gauss table must be an i32[65536]")
        if tuple(g.tensors[n.eps_tensor].shape) != latent:
            raise VirError("eps tensor shape does not match the latent shape")
        if n.latent_tensor not in g.inputs:
            raise VirError("latent tensor must be a graph input")
        if tuple(h.out) != latent:
            raise VirError("opset 1: output shape must equal the latent shape")
        if h.output_bytes > 65536:
            raise VirError("output exceeds the 64 KiB admission limit")
        if not (10 <= n.latent_shift <= 30 and 0 <= n.out_shift <= 15):
            raise VirError("implausible sampler shifts")
        if n.clamp_abs < 1 or n.clamp_abs > 32767:
            raise VirError("bad output clamp")
        if h.out_shift != n.out_shift or h.clamp_abs != n.clamp_abs:
            raise VirError("header and noise spec disagree on the output contract")
        if h.tok_type == TOK_NONE:
            if h.flags & 1:
                raise VirError("tokenizer flag set without a tokenizer")
            if h.vocab_size or h.ctx_len or h.max_tokens:
                raise VirError("tokenizer metadata present but tok_type is none")
            if any(t.storage == Storage.BLOB for t in g.tensors):
                raise VirError("tokenizer blob without a tokenizer")
        elif h.tok_type == TOK_CLIP_BPE:
            if not h.flags & 1:
                raise VirError("CLIP tokenizer without the tokenizer flag")
            for tid in (h.tok_vocab_tensor, h.tok_merges_tensor):
                if tid >= len(g.tensors) or g.tensors[tid].storage != Storage.BLOB:
                    raise VirError("missing tokenizer blob tensor")
                if g.tensors[tid].dtype != "u8" or g.tensors[tid].rank != 1:
                    raise VirError("tokenizer blobs must be u8[1]-shaped byte strings")
            if not 1 <= h.vocab_size <= (1 << 20):
                raise VirError("bad vocab size")
            for name, tid in (("bos", h.bos_id), ("eos", h.eos_id), ("pad", h.pad_id)):
                if tid >= h.vocab_size:
                    raise VirError(f"{name} id {tid} >= vocab size")
            if not 3 <= h.ctx_len <= 1024:
                raise VirError("bad context length")
            if h.max_tokens + 2 > h.ctx_len:
                raise VirError("max_tokens does not fit the context window")
            if h.pad_policy not in (PAD_USE_PAD_ID, PAD_USE_EOS_ID, PAD_NONE):
                raise VirError(f"unknown padding policy {h.pad_policy}")
            if h.pad_policy == PAD_USE_EOS_ID and h.pad_id != h.eos_id:
                raise VirError("padding policy says EOS but pad_id differs")
            if h.pad_policy == PAD_NONE and h.pad_id != 0:
                raise VirError("padding policy says none but pad_id is set")
            ids_inputs = [t for t in g.inputs if tuple(g.tensors[t].shape) == (h.ctx_len,)]
            if not ids_inputs:
                raise VirError("tokenizer present but no [ctx_len] token-id input")
        else:
            raise VirError(f"unsupported tokenizer type {h.tok_type}")

    # -- inference -----------------------------------------------------
    def _value(self, tid: int) -> Any:
        t = self.graph.tensors[tid]
        if t.data is None:
            raise VirError(f"tensor {t.name!r} has no data")
        return t.data

    def context_ids(self, tokens: Sequence[int]) -> Any:
        h = self.header
        if h.tok_type == TOK_NONE:
            raise VirError("model has no tokenizer")
        toks = [int(t) for t in tokens]
        if len(toks) > h.max_tokens:
            raise VirError(f"prompt has {len(toks)} tokens, limit is {h.max_tokens}")
        for t in toks:
            if not 0 <= t < h.vocab_size or t in (h.bos_id, h.eos_id):
                raise VirError(f"token id {t} outside the prompt range")
        ids = [h.bos_id] + toks + [h.eos_id]
        if len(ids) > h.ctx_len:
            raise VirError("prompt does not fit the context window")
        if h.pad_policy == PAD_NONE:
            if len(ids) != h.ctx_len:
                raise VirError("padding policy is 'none' but the sequence is short")
        else:
            pad = h.pad_id if h.pad_policy == PAD_USE_PAD_ID else h.eos_id
            ids += [pad] * (h.ctx_len - len(ids))
        return np.asarray(ids, dtype=np.int32)

    def evaluate(self, feed: dict[int, Any], tracer: "TraceRecorder | None" = None) -> dict[int, Any]:
        values: dict[int, Any] = {}
        for tid in self.graph.inputs:
            if tid not in feed:
                raise VirError(f"graph input {self.graph.tensors[tid].name!r} not supplied")
            values[tid] = feed[tid]
        for index, node in enumerate(self.graph.nodes):
            ins = [values[t] if t in values else self._value(t) for t in node.inputs]
            outs = OP_FORWARD[node.op](node, ins)
            values[node.outputs[0]] = outs
            if tracer is not None:
                tracer.node(index, node, outs, len(self.graph.nodes) == index + 1)
        return values

    def noise_latent(self, seed: bytes) -> Any:
        n = self.noise
        if len(seed) != n.seed_len:
            raise VirError(f"seed must be {n.seed_len} bytes")
        ks = chacha20_keystream(seed, n.ks_bytes, b"\0" * 12, n.counter_start)
        u = np.frombuffer(ks, dtype="<u2").astype(np.int64)
        gauss = self._value(n.gauss_tensor)
        return gauss[u].reshape(tuple(self.header.latent))

    def run(self, seed: bytes, tokens: Sequence[int],
            tracer: "TraceRecorder | None" = None) -> bytes:
        """x0 payload bytes (int16le), identical to poi::Model::Run's output.

        With ``tracer`` the intermediate and final rows of this run are also
        committed to a Sampled-PoI trace; the return value is unchanged.
        """
        h, n = self.header, self.noise
        noise = self.noise_latent(seed)
        x = sat32(rshr(noise * np.int64(n.cin_q30), 30))
        feed = {n.latent_tensor: x}
        ids_tensor = None
        for tid in self.graph.inputs:
            if tid != n.latent_tensor:
                ids_tensor = tid
        if h.tok_type != TOK_NONE:
            if ids_tensor is None:
                raise VirError("tokenizer model without a token input")
            feed[ids_tensor] = self.context_ids(tokens)
        else:
            if ids_tensor is not None:
                raise VirError("model takes a token input but has no tokenizer")
        values = self.evaluate(feed, tracer)
        eps = values[n.eps_tensor]
        if n.sampler_kind == 1:
            x0 = sat32(rshr(noise * np.int64(n.sigma_q16) - eps * np.int64(n.eps_q16), n.latent_shift))
        else:
            x0 = sat32(rshr((noise - eps) * np.int64(n.sigma_q16), n.latent_shift))
        q = np.clip(rshr(x0, n.out_shift), -n.clamp_abs, n.clamp_abs)
        payload = q.astype("<i2").tobytes()
        if tracer is not None:
            tracer.finish(payload, self.header.latent)
        return payload

    def run_with_leaves(self, seed: bytes, tokens: Sequence[int]) -> tuple[bytes, list[bytes]]:
        """``x0`` and every leaf digest of every node output (Fork-M).

        Same arithmetic as :meth:`run`; outputs are digested as each node
        produces them and dropped when no later node needs them, so memory is
        that of one inference rather than of every activation at once.
        """
        h, n = self.header, self.noise
        noise = self.noise_latent(seed)
        x = sat32(rshr(noise * np.int64(n.cin_q30), 30))
        feed = {n.latent_tensor: x}
        ids_tensor = None
        for tid in self.graph.inputs:
            if tid != n.latent_tensor:
                ids_tensor = tid
        if h.tok_type != TOK_NONE:
            if ids_tensor is None:
                raise VirError("tokenizer model without a token input")
            feed[ids_tensor] = self.context_ids(tokens)
        elif ids_tensor is not None:
            raise VirError("model takes a token input but has no tokenizer")

        values: dict[int, Any] = dict(feed)
        refs: dict[int, int] = {}
        for node in self.graph.nodes:
            for t in node.inputs:
                refs[t] = refs.get(t, 0) + 1
        leaves: list[bytes] = []
        for index, node in enumerate(self.graph.nodes):
            if any(t not in values and self.graph.tensors[t].data is None for t in node.inputs):
                missing = [self.graph.tensors[t].name for t in node.inputs if t not in values]
                raise VirError(f"node {index}: inputs {missing} not available")
            ins = [values[t] if t in values else self._value(t) for t in node.inputs]
            outs = OP_FORWARD[node.op](node, ins)
            values[node.outputs[0]] = outs
            leaves.extend(node_leaves(index, outs))
            for t in node.inputs:
                refs[t] -= 1
                if refs[t] == 0:
                    values.pop(t, None)
        eps = values.get(n.eps_tensor)
        if eps is None:
            eps = self._value(n.eps_tensor)
        if n.sampler_kind == 1:
            x0 = sat32(rshr(noise * np.int64(n.sigma_q16) - eps * np.int64(n.eps_q16), n.latent_shift))
        else:
            x0 = sat32(rshr((noise - eps) * np.int64(n.sigma_q16), n.latent_shift))
        q = np.clip(rshr(x0, n.out_shift), -n.clamp_abs, n.clamp_abs)
        return q.astype("<i2").tobytes(), leaves

    # -- cost ----------------------------------------------------------
    def cost(self) -> "Cost":
        return estimate_cost(self.graph)


@dataclass
class Cost:
    macs: int = 0
    lookups: int = 0
    weights_bytes: int = 0
    activation_bytes: int = 0   # sum of every activation buffer (upper bound)
    peak_bytes: int = 0         # weighted + live activations at the peak


def estimate_cost(g: Graph) -> Cost:
    """Objective cost of one inference, computed from the graph (article §6).

    MACs are exact for GEMM/Conv2d/attention.  ``peak_bytes`` is the weight
    bytes plus the high-water mark of live activation buffers, computed from
    the use counts of the tensor table (a node's last use frees it), so it can
    be used for the admission limit without trusting the registrant.
    """
    c = Cost()
    refs: dict[int, int] = {}
    for node in g.nodes:
        for tid in node.inputs:
            refs[tid] = refs.get(tid, 0) + 1
    for node in g.nodes:
        ins = [g.tensors[t].shape for t in node.inputs]
        outs = [g.tensors[t].shape for t in node.outputs]
        if node.op is Op.GEMM:
            c.macs += _prod(outs[0]) * ins[0][1]
        elif node.op is Op.CONV2D:
            c.macs += _prod(outs[0]) * ins[1][1]
        elif node.op is Op.ATTENTION:
            tq, cdim = ins[0]
            c.macs += 2 * tq * ins[1][0] * cdim
            c.lookups += tq * ins[1][0] * int(node.attr(Attr.NUM_HEADS))
        elif node.op is Op.LUT_ACT:
            c.lookups += _prod(outs[0])
        else:
            c.macs += _prod(outs[0]) if outs else 0
    for t in g.tensors:
        size = t.count * 4
        if t.storage == Storage.ACTIVATION:
            c.activation_bytes += size
        elif t.storage == Storage.WEIGHT:
            c.weights_bytes += t.count * DTYPE_SIZE[t.dtype]
        elif t.storage in (Storage.PARAM, Storage.LUT):
            c.weights_bytes += size
        elif t.storage == Storage.BLOB:
            c.weights_bytes += t.nbytes
    live = 0
    peak = 0
    for node in g.nodes:
        for tid in node.inputs:
            refs[tid] -= 1
        for tid in node.outputs:
            live += g.tensors[tid].count * 4
        for tid in node.inputs:
            if refs[tid] == 0 and g.tensors[tid].storage == Storage.ACTIVATION:
                live -= g.tensors[tid].count * 4
        peak = max(peak, live)
    c.peak_bytes = peak + c.weights_bytes
    return c


# ---------------------------------------------------------------------------
# Sampled-PoI trace prototype (not part of the format and not consensus code)
#
# A trace commits to the rows a validator may sample: one leaf hash per check
# unit, and a Merkle root over the leaf hashes.  A miner keeps the unit rows
# off-chain and serves the requested units; the block commits only the 32-byte
# root.  The prototype is opt-in: with tracing off, run() is byte-for-byte the
# same as before.  See research/vir-trace-measurements.md for the numbers and
# the recommended fork parameters.
# ---------------------------------------------------------------------------

TRACE_LEAF_TAG = "POI/trace/leaf"    # leaf hash: row bytes of one check unit
TRACE_NODE_TAG = "POI/trace/node"    # inner node: left||right
TRACE_ROOT_TAG = "POI/trace/root"    # domain separator of the final root
TRACE_FORMAT = 1                     # prototype dump format version
POI_OUTPUT_TAG = "POI/out"           # hashPoI, see src/poi/poi.cpp


def poi_output_hash(seed: bytes, x0: bytes) -> bytes:
    """``TaggedHash("POI/out")(seed || x0)``: the current hashPoI commitment."""
    if len(seed) != 32:
        raise VirError("seed must be 32 bytes")
    return tagged_hash(POI_OUTPUT_TAG, seed + x0)


# ---------------------------------------------------------------------------
# PoI leaf commitments (Fork-M): src/poi/fraud.{h,cpp}
# ---------------------------------------------------------------------------

SUB_BYTES = 4096                     # sub-chunk inside a leaf
LEAF_SUBS = 64                       # sub-chunks per leaf
LEAF_BYTES = SUB_BYTES * LEAF_SUBS   # 256 KiB
LEAF_VALUES = LEAF_BYTES // 4        # int32 per leaf
LEAF_TAG = b"VELA/leaf/v1"
POI_LEAVES_TAG = "POI/leaves"        # R over every leaf digest
POI_OUTPUT_V5_TAG = "POI/out5"       # hashPoI from Fork-M on


def _be32(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def leaf_count(values: int) -> int:
    """Number of leaves of a tensor with ``values`` int32 elements (at least 1)."""
    return max(1, -(-values // LEAF_VALUES))


def leaf_digest(node: int, leaf: int, values) -> bytes:
    """``SHA256("VELA/leaf/v1" || be32(node) || be32(leaf) || sub_0 .. sub_63)``.

    Each sub-chunk is ``SHA256(be32(node) || be32(leaf*64 + s) || 4 KiB of
    int32le)``; a short final sub-chunk is hashed as it is.
    """
    v = np.ascontiguousarray(values, dtype="<i4").ravel()
    if v.size > LEAF_VALUES:
        raise VirError("leaf too large")
    raw = v.tobytes()
    nsub = max(1, -(-len(raw) // SUB_BYTES))
    out = hashlib.sha256(LEAF_TAG + _be32(node) + _be32(leaf))
    for s in range(nsub):
        a = s * SUB_BYTES
        chunk = raw[a:a + SUB_BYTES]
        out.update(hashlib.sha256(_be32(node) + _be32(leaf * LEAF_SUBS + s) + chunk).digest())
    return out.digest()


def node_leaves(node: int, output) -> list[bytes]:
    """All leaf digests of one node output (node index, leaves ascending)."""
    v = np.ascontiguousarray(output, dtype="<i4").ravel()
    return [leaf_digest(node, k, v[k * LEAF_VALUES:(k + 1) * LEAF_VALUES])
            for k in range(leaf_count(v.size))]


def leaves_commitment(leaves: Sequence[bytes]) -> bytes:
    """``R = TaggedHash("POI/leaves")(leaf digests, node order)``."""
    return tagged_hash(POI_LEAVES_TAG, b"".join(leaves))


def poi_output_hash_v5(seed: bytes, x0: bytes, commitment: bytes) -> bytes:
    """``TaggedHash("POI/out5")(seed || x0 || R)``: hashPoI from Fork-M on."""
    if len(seed) != 32 or len(commitment) != 32:
        raise VirError("seed and leaves commitment must be 32 bytes")
    return tagged_hash(POI_OUTPUT_V5_TAG, seed + x0 + commitment)


def trace_leaf(row: bytes) -> bytes:
    return tagged_hash(TRACE_LEAF_TAG, row)


def trace_inner(left: bytes, right: bytes) -> bytes:
    return tagged_hash(TRACE_NODE_TAG, left + right)


class _Merkle:
    """Merkle tree over check-unit row hashes, level-wise, last node duplicated.

    Leaf hashes use TRACE_LEAF_TAG, inner nodes TRACE_NODE_TAG and the root is
    wrapped in TRACE_ROOT_TAG, so a leaf hash can never be mistaken for a root
    or an inner node.  Proofs are the sibling list from the leaf up.
    """

    __slots__ = ("levels", "leaf_bytes", "unit_rows")

    def __init__(self) -> None:
        self.levels: list[list[bytes]] = [[]]
        self.leaf_bytes = 0
        self.unit_rows = 0

    def add_leaf(self, leaf: bytes, row_bytes: int) -> None:
        self.levels[0].append(leaf)
        self.leaf_bytes += row_bytes
        self.unit_rows += 1

    def build(self) -> bytes:
        if not self.levels[0]:
            return tagged_hash(TRACE_ROOT_TAG, b"")
        if len(self.levels) == 1 and len(self.levels[0]) > 1:
            level = list(self.levels[0])
            levels = [level]
            while len(level) > 1:
                if len(level) % 2:
                    level = level + [level[-1]]
                level = [trace_inner(level[i], level[i + 1]) for i in range(0, len(level), 2)]
                levels.append(level)
            self.levels = levels
        return tagged_hash(TRACE_ROOT_TAG, self.levels[-1][0])

    def proof(self, index: int) -> list[bytes]:
        self.build()
        path: list[bytes] = []
        i = index
        for level in self.levels[:-1]:
            sib = i ^ 1
            if sib >= len(level):
                sib = i
            path.append(level[sib])
            i //= 2
        return path


def merkle_root_from_leaves(leaves: Sequence[bytes]) -> bytes:
    m = _Merkle()
    for leaf in leaves:
        m.add_leaf(leaf, 0)
    return m.build()


class _TraceVariant:
    """One check-unit definition: a tree plus its counters."""

    __slots__ = ("name", "tree", "unit_rows", "leaf_bytes", "leaf_count", "hash_ns")

    def __init__(self, name: str) -> None:
        self.name = name
        self.tree = _Merkle()
        self.unit_rows = 0          # number of units (= leaves)
        self.leaf_bytes = 0         # unit row bytes the miner must keep
        self.leaf_count = 0
        self.hash_ns = 0

    def add(self, row: bytes) -> None:
        t0 = time.perf_counter_ns()
        leaf = trace_leaf(row)
        self.hash_ns += time.perf_counter_ns() - t0
        self.tree.add_leaf(leaf, len(row))
        self.leaf_bytes += len(row)
        self.leaf_count += 1

    def root(self) -> bytes:
        t0 = time.perf_counter_ns()
        root = self.tree.build()
        self.hash_ns += time.perf_counter_ns() - t0
        return root

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.leaf_count,
            "unit_row_bytes": self.leaf_bytes,
            "root_hex": self.root().hex(),
            "hash_ms": round(self.hash_ns / 1e6, 3),
            "root_bytes": 32,
        }


class TraceRecorder:
    """Collects one or more Sampled-PoI trace variants during a model run.

    Variants (all optional, all measured in the same run):

    ``final``  one unit per row of the final x0 payload (the consensus payload
               itself, so the units are exactly the bytes hashPoI commits to);
    ``nodes``  one unit per row of every intermediate node output;
    ``last``   one unit per row of the last node's output (the eps tensor that
               the final payload is derived from row by row);
    ``sample`` every ``sample_every``-th final payload row.

    Rows are serialized little-endian int32 for intermediates and exactly the
    payload bytes (int16le) for the final variant.  Nothing here is consensus
    critical: it is a measurement prototype.
    """

    def __init__(self, *, final: bool = True, nodes: bool = False,
                 last_node: bool = False, sample_every: int = 0, node_stride: int = 1,
                 keep_rows: bool = False) -> None:
        self.variants: dict[str, _TraceVariant] = {}
        if final:
            self.variants["final"] = _TraceVariant("final")
        if nodes:
            self.variants["nodes"] = _TraceVariant("nodes")
        if last_node:
            self.variants["last"] = _TraceVariant("last")
        if sample_every > 0:
            self.variants["sample"] = _TraceVariant("sample")
        self.sample_every = sample_every
        self.node_stride = node_stride
        self.keep_rows = keep_rows
        self.row_dump: dict[str, list[bytes]] = {name: [] for name in self.variants}
        self.node_units = 0
        self.node_ns = 0
        self.final_row_bytes = 0
        self.final_rows = 0

    # -- during the run ---------------------------------------------------
    def node(self, index: int, node: "Node", out: Any, is_last: bool = False) -> None:
        if "last" in self.variants and is_last:
            t0 = time.perf_counter_ns()
            for row in _rows_of(out):
                self.variants["last"].add(row)
                if self.keep_rows:
                    self.row_dump["last"].append(row)
            self.node_ns += time.perf_counter_ns() - t0
        if "nodes" not in self.variants or index % self.node_stride:
            return
        t0 = time.perf_counter_ns()
        for row in _rows_of(out):
            self.variants["nodes"].add(row)
            self.node_units += 1
            if self.keep_rows:
                self.row_dump["nodes"].append(row)
        self.node_ns += time.perf_counter_ns() - t0

    # -- after the payload is final --------------------------------------
    def finish(self, payload: bytes, latent: Sequence[int]) -> None:
        """Commit to the final payload rows (and the sampled subset)."""
        if not latent:
            raise VirError("latent shape is empty")
        row_bytes = int(latent[-1]) * 2
        nrows = 1
        for d in latent[:-1]:
            nrows *= int(d)
        if nrows * row_bytes != len(payload):
            raise VirError(
                f"payload {len(payload)} bytes does not match latent {tuple(latent)}")
        self.final_row_bytes = row_bytes
        self.final_rows = nrows
        t0 = time.perf_counter_ns()
        for i in range(nrows):
            row = payload[i * row_bytes:(i + 1) * row_bytes]
            if "final" in self.variants:
                self.variants["final"].add(row)
            if "sample" in self.variants and i % self.sample_every == 0:
                self.variants["sample"].add(row)
            if self.keep_rows:
                self.row_dump.setdefault("final", []).append(row)
        self.final_ns = time.perf_counter_ns() - t0

    # -- results ---------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        out = {
            "variants": {},
            "final_row_bytes": self.final_row_bytes,
            "final_rows": self.final_rows,
            "node_units": self.node_units,
            "node_walk_ms": round(self.node_ns / 1e6, 3),
        }
        for name, v in self.variants.items():
            out["variants"][name] = v.stats()
        return out

    def write(self, path: str) -> None:
        """Write a prototype trace dump: rows per unit, then the root."""
        import struct
        with open(path, "wb") as f:
            f.write(b"VIRT" + struct.pack("<H", TRACE_FORMAT))
            f.write(struct.pack("<H", len(self.variants)))
            for name, v in self.variants.items():
                blobs = self.row_dump.get(name, [])
                rows = [b"" for _ in range(v.leaf_count)] if not blobs else blobs
                f.write(struct.pack("<H", len(name)) + name.encode())
                f.write(struct.pack("<I", len(rows)))
                for row in rows:
                    f.write(struct.pack("<I", len(row)))
                    f.write(row)
                f.write(v.root())


def _rows_of(value: Any) -> list[bytes]:
    """Serialize an intermediate tensor as canonical little-endian int32 rows."""
    a = np.asarray(value)
    if a.dtype.kind == "f":
        a = np.rint(a)
    if a.dtype.kind == "b":
        a = a.astype(np.int64)
    if a.ndim == 0:
        a = a.reshape(1)
    i = a.astype(np.int64)
    i = np.clip(i, I32_MIN, I32_MAX).astype("<i4")
    if i.ndim == 1:
        return [i.tobytes()]
    rows = i.reshape(-1, i.shape[-1])
    return [rows[k].tobytes() for k in range(rows.shape[0])]


# ---------------------------------------------------------------------------
# CLIP BPE tokenizer (not consensus: it only decides which ids a miner commits)
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def pretokenize(text: str) -> list[bytes]:
    """ASCII-exact approximation of CLIP's pre-tokenizer (as in sdint.py)."""
    b = " ".join(text.lower().split()).encode()
    out: list[bytes] = []
    i = 0
    isl = lambda c: 97 <= c <= 122 or c >= 128  # noqa: E731
    isd = lambda c: 48 <= c <= 57  # noqa: E731
    iss = lambda c: c in b" \t\n\r\x0b\x0c"  # noqa: E731
    while i < len(b):
        c = b[i]
        hit = next((x for x in _CONTRACTIONS if b.startswith(x.encode(), i)), None)
        if hit:
            out.append(b[i:i + len(hit)])
            i += len(hit)
        elif isl(c):
            j = i
            while j < len(b) and isl(b[j]):
                j += 1
            out.append(b[i:j])
            i = j
        elif isd(c):
            out.append(b[i:i + 1])
            i += 1
        elif iss(c):
            i += 1
        else:
            j = i
            while j < len(b) and not (isl(b[j]) or isd(b[j]) or iss(b[j])):
                j += 1
            out.append(b[i:j])
            i = j
    return out


class ClipBpe:
    """The tokenizer implied by the two blobs in a .vir file."""

    def __init__(self, vocab_json: bytes, merges_txt: bytes, vocab_size: int = 49408):
        self.enc: dict[str, int] = json.loads(bytes(vocab_json).decode("utf-8"))
        self.ranks: dict[tuple[str, str], int] = {}
        text = bytes(merges_txt).decode("utf-8", "replace").strip().split("\n")[1:]
        for i, line in enumerate(text[:vocab_size - 256 - 2 + 1]):
            parts = line.split()
            if len(parts) == 2:
                self.ranks[(parts[0], parts[1])] = i
        self.b2u = _bytes_to_unicode()

    def bpe(self, word: str) -> list[str]:
        w = list(word[:-1]) + [word[-1] + "</w>"]
        while len(w) > 1:
            best = min(((self.ranks.get((w[i], w[i + 1]), 1 << 30), i)
                        for i in range(len(w) - 1)))
            if best[0] == 1 << 30:
                break
            a, b = w[best[1]], w[best[1] + 1]
            nw, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(a + b)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            w = nw
        return w

    def encode(self, text: str, max_tokens: int = 1 << 30) -> list[int]:
        ids: list[int] = []
        for tok in pretokenize(text):
            for sym in self.bpe("".join(self.b2u[c] for c in tok)):
                ident = self.enc.get(sym)
                if ident is not None:
                    ids.append(ident)
        return ids[:max_tokens]


def tokenizer(model: VirModel) -> ClipBpe:
    h = model.header
    if h.tok_type != TOK_CLIP_BPE:
        raise VirError("model has no CLIP tokenizer")
    vocab = model._value(h.tok_vocab_tensor)  # noqa: SLF001
    merges = model._value(h.tok_merges_tensor)  # noqa: SLF001
    return ClipBpe(np.asarray(vocab).tobytes(), np.asarray(merges).tobytes(), h.vocab_size)


# ---------------------------------------------------------------------------
# canonical lookup tables (exporter side; the file stores the result)
# ---------------------------------------------------------------------------

LUT_RANGE = 16
_TABLE_CACHE: dict[str, "np.ndarray"] = {}


def _xs(n: int, lo: float, step: float):
    return np.arange(n, dtype=np.float64) * step + lo


def table_sigmoid() -> "np.ndarray":
    """Q16 sigmoid over [-16, 16) in steps of 1/64, 2049 samples (v2 lut.sigmoid)."""
    xs = _xs(2 * LUT_RANGE * 64 + 1, -LUT_RANGE, 1.0 / 64)
    return np.round(65536.0 / (1.0 + np.exp(-xs))).astype(np.int32)


def table_phi() -> "np.ndarray":
    """Q16 standard normal CDF over [-16, 16), 2049 samples (v2 lut.phi)."""
    xs = _xs(2 * LUT_RANGE * 64 + 1, -LUT_RANGE, 1.0 / 64)
    return np.round(np.array([65536.0 * 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in xs])).astype(np.int32)


def table_exp() -> "np.ndarray":
    """Q30 exp(-x) over [0, 16) in steps of 1/64, 1025 samples (v2 lut.exp)."""
    return np.round(np.exp(-_xs(1024 + 1, 0.0, 1.0 / 64)) * 2.0 ** 30).astype(np.int32)


def table_gauss() -> "np.ndarray":
    """Q16 inverse normal at (i + 0.5)/65536, 65536 samples (v2 lut.gauss)."""
    from statistics import NormalDist

    nd = NormalDist()
    return np.round(np.array([nd.inv_cdf((i + 0.5) / 65536.0) for i in range(65536)]) * 65536.0).astype(np.int32)


def table_quick_sigmoid() -> "np.ndarray":
    """Q16 sigmoid(1.702*x) over [-16, 16): quick_gelu = x * this (SDXS text tower)."""
    xs = _xs(2 * LUT_RANGE * 64 + 1, -LUT_RANGE, 1.0 / 64)
    return np.round(65536.0 / (1.0 + np.exp(-1.702 * xs))).astype(np.int32)


def canonical_tables() -> dict[str, "np.ndarray"]:
    if not _TABLE_CACHE:
        _TABLE_CACHE.update({
            "lut.sigmoid": table_sigmoid(),
            "lut.phi": table_phi(),
            "lut.exp": table_exp(),
            "lut.gauss": table_gauss(),
            "lut.quick_sigmoid": table_quick_sigmoid(),
        })
    return _TABLE_CACHE


def q_scale(head_dim: int) -> int:
    """round(2**16 / sqrt(head_dim)); 8192 for 64, so v2's 19-shift is a special case."""
    return int(round(2.0 ** 16 / math.sqrt(head_dim)))


# ---------------------------------------------------------------------------
# graph builder: diffusers config -> opset-1 graph
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self, g: Graph | None = None, wgroup: int = 0, wdtype=None):
        self.g = g if g is not None else Graph()
        self.tables: dict[str, int] = {}
        #: weight-scale group size for every i8 GEMM/CONV2D (0 = per output row, v2)
        self.wgroup = int(wgroup)
        #: prefix -> "i8" | "i16"; None = every weight is i8 (the v2 form)
        self.wdtype = wdtype

    def _dtype(self, prefix: str) -> str:
        return self.wdtype(prefix) if self.wdtype else "i8"

    def _grp(self, k: int, dtype: str = "i8") -> int:
        # i16 weights are precise enough per row; a layer narrower than one
        # group keeps a single scale per row.
        if dtype != "i8":
            return 0
        return self.wgroup if self.wgroup and k > self.wgroup else 0

    # -- helpers -------------------------------------------------------
    def shape(self, tid: int) -> tuple[int, ...]:
        return tuple(self.g.tensors[tid].shape)

    def table(self, name: str) -> int:
        if name not in self.tables:
            data = canonical_tables()[name]
            self.tables[name] = self.g.add_tensor(
                TensorDesc(name, (int(data.shape[0]),), "i32", Storage.LUT, data))
        return self.tables[name]

    def param(self, name: str, shape: Sequence[int], data=None) -> int:
        return self.g.add_tensor(TensorDesc(name, tuple(int(d) for d in shape), "i32",
                                            Storage.PARAM, data))

    # -- ops -----------------------------------------------------------
    def gemm(self, x: int, prefix: str, n: int, k: int) -> int:
        dt = self._dtype(prefix)
        grp = self._grp(k, dt)
        w = self.g.weight(prefix, n, k, dtype=dt)
        m, s, b = self.g.channel_params(prefix, n, k=k, wgroup=grp)
        return self.g.emit(Op.GEMM, [x, w, m, s, b], (self.shape(x)[0], n), {Attr.WGROUP: grp})

    def conv(self, x: int, prefix: str, cout: int, cin: int, kh: int = 3, kw: int = 3,
             stride: int = 1, pad: int = 1) -> int:
        k = cin * kh * kw
        dt = self._dtype(prefix)
        grp = self._grp(k, dt)
        w = self.g.weight(prefix, cout, k, dtype=dt)
        m, s, b = self.g.channel_params(prefix, cout, k=k, wgroup=grp)
        _, h, wd = self.shape(x)
        ho = (h + 2 * pad - kh) // stride + 1
        wo = (wd + 2 * pad - kw) // stride + 1
        return self.g.emit(Op.CONV2D, [x, w, m, s, b], (cout, ho, wo),
                           {Attr.KH: kh, Attr.KW: kw, Attr.STRIDE: stride, Attr.PAD: pad,
                            Attr.WGROUP: grp})

    def _norm_params(self, prefix: str, c: int, den: int) -> tuple[int, int, int]:
        gamma = self.param(prefix + ".g", (c,))
        beta = self.param(prefix + ".beta", (c,))
        eps = self.param(prefix + ".eps", (1,), np.array([den], dtype=np.int32))
        return gamma, beta, eps

    def groupnorm(self, x: int, prefix: str, c: int, groups: int, den: int) -> int:
        gamma, beta, eps = self._norm_params(prefix, c, den)
        return self.g.emit(Op.GROUPNORM, [x, gamma, beta, eps], self.shape(x),
                           {Attr.GROUPS: groups})

    def layernorm(self, x: int, prefix: str, c: int, den: int) -> int:
        gamma, beta, eps = self._norm_params(prefix, c, den)
        return self.g.emit(Op.LAYERNORM, [x, gamma, beta, eps], self.shape(x))

    def act(self, x: int, table: str) -> int:
        return self.g.emit(Op.LUT_ACT, [x, self.table(table)], self.shape(x),
                           {Attr.LUT_LO: -(LUT_RANGE << BITS), Attr.LUT_STEP_BITS: 10,
                            Attr.LUT_EDGE_LO: 0, Attr.LUT_EDGE_HI: 65536,
                            Attr.POST_SHIFT: BITS})

    def attention(self, q: int, k: int, v: int, heads: int, dh: int, causal: bool = False) -> int:
        return self.g.emit(Op.ATTENTION, [q, k, v, self.table("lut.exp")],
                           (self.shape(q)[0], self.shape(q)[1]),
                           {Attr.NUM_HEADS: heads, Attr.HEAD_DIM: dh,
                            Attr.CAUSAL: int(bool(causal)), Attr.Q_SCALE: q_scale(dh)})

    def add(self, a: int, b: int) -> int:
        return self.g.emit(Op.ADD, [a, b], self.shape(a))

    def mul(self, a: int, b: int) -> int:
        return self.g.emit(Op.MUL, [a, b], self.shape(a), {Attr.SHIFT: BITS})

    def concat(self, a: int, b: int, axis: int = 0) -> int:
        sa, sb = self.shape(a), self.shape(b)
        out = list(sa)
        out[axis] += sb[axis]
        return self.g.emit(Op.CONCAT, [a, b], tuple(out), {Attr.AXIS: axis})

    def to_tokens(self, x: int) -> int:
        c, h, w = self.shape(x)
        return self.g.emit(Op.RESHAPE, [x], (h * w, c),
                           {Attr.PERM: (1, 2, 0), Attr.SHAPE: (h * w, c)})

    def to_channels(self, x: int, c: int, h: int, w: int) -> int:
        return self.g.emit(Op.RESHAPE, [x], (c, h, w),
                           {Attr.PERM: (1, 0), Attr.SHAPE: (c, h, w)})

    def upsample(self, x: int, factor: int = 2) -> int:
        c, h, w = self.shape(x)
        return self.g.emit(Op.UPSAMPLE, [x], (c, h * factor, w * factor),
                           {Attr.FACTOR: factor, Attr.MODE: 0})

    def scale(self, x: int, mult: int, shift: int) -> int:
        return self.g.emit(Op.SCALE, [x], self.shape(x), {Attr.MULT: mult, Attr.SHIFT: shift})


@dataclass
class UnetSpec:
    levels: int
    block_out: list[int]
    down_types: list[str]
    up_types: list[str]
    layers: int
    heads: list[int]
    head_dim: list[int]
    transformer_layers: list[int]
    only_cross: list[bool]
    cross_dim: int
    mid: bool
    groups: int
    sample_size: int
    in_ch: int
    out_ch: int
    norm_den: int
    ff_mult: int = 4


def unet_spec(cfg: dict) -> UnetSpec:
    """Resolve a diffusers UNet2DConditionModel config into explicit topology.

    ``attention_head_dim`` is *the number of heads* when ``num_attention_heads``
    is absent (diffusers backwards compatibility: ``num_attention_heads =
    num_attention_heads or attention_head_dim``), so the head dimension is
    ``block_out_channels[i] // heads[i]``.  v2's 5/10/20/20 with head dim 64 and
    SDXS's 8 heads with head dim 80/160 both follow from this rule.
    """
    bo = [int(c) for c in cfg["block_out_channels"]]
    n = len(bo)
    down = list(cfg["down_block_types"])
    up = list(cfg["up_block_types"])
    if len(down) != n or len(up) != n:
        raise VirError("block_out_channels and block types disagree")

    def expand(v):
        return [v] * n if not isinstance(v, (list, tuple)) else list(v)

    heads = cfg.get("num_attention_heads")
    if heads is None:
        heads = expand(cfg.get("attention_head_dim", 8))
    else:
        heads = expand(heads)
    heads = [int(h) for h in heads]
    head_dim = []
    for i in range(n):
        if bo[i] % heads[i]:
            raise VirError(f"level {i}: {bo[i]} channels not divisible by {heads[i]} heads")
        head_dim.append(bo[i] // heads[i])
    only = cfg.get("only_cross_attention", False)
    only = [bool(v) for v in only] if isinstance(only, (list, tuple)) else [bool(only)] * n
    tl = cfg.get("transformer_layers_per_block", 1)
    tl = [int(v) for v in tl] if isinstance(tl, (list, tuple)) else [int(tl)] * n
    layers = cfg.get("layers_per_block", 1)
    if isinstance(layers, (list, tuple)):
        raise VirError("layers_per_block as a list is not supported (TBD)")
    eps = float(cfg.get("norm_eps", 1e-5))
    return UnetSpec(
        levels=n, block_out=bo, down_types=down, up_types=up, layers=int(layers),
        heads=heads, head_dim=head_dim, transformer_layers=tl, only_cross=only,
        cross_dim=int(cfg["cross_attention_dim"]),
        mid=cfg.get("mid_block_type") not in (None, "None"),
        groups=int(cfg.get("norm_num_groups", 32)), sample_size=int(cfg.get("sample_size", 64)),
        in_ch=int(cfg.get("in_channels", 4)), out_ch=int(cfg.get("out_channels", 4)),
        norm_den=int(round(1.0 / eps)),
    )


@dataclass
class TextSpec:
    layers: int
    hidden: int
    heads: int
    inner: int
    vocab: int
    ctx_len: int
    max_tokens: int
    bos: int
    eos: int
    pad: int
    pad_policy: int
    act: str           # "gelu" (erf), "quick_gelu" or "silu"
    norm_den: int
    causal: bool


def text_spec(cfg: dict, causal: bool = False, pad_policy: int = PAD_USE_EOS_ID) -> TextSpec:
    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    vocab = int(cfg["vocab_size"])
    ctx = int(cfg.get("max_position_embeddings", 77))
    eps = float(cfg.get("layer_norm_eps", 1e-5))
    eos = 49407
    return TextSpec(
        layers=layers, hidden=hidden, heads=heads,
        inner=int(cfg.get("intermediate_size", 4 * hidden)), vocab=vocab,
        ctx_len=ctx, max_tokens=ctx - 2, bos=49406, eos=eos,
        pad=0 if pad_policy == PAD_USE_PAD_ID else eos,
        pad_policy=pad_policy, act=str(cfg.get("hidden_act", "quick_gelu")),
        norm_den=int(round(1.0 / eps)), causal=causal,
    )


TE_TABLE = {"gelu": "lut.phi", "quick_gelu": "lut.quick_sigmoid", "silu": "lut.sigmoid"}


def build_resnet(B: _Builder, x: int, prefix: str, cout: int, spec: UnetSpec) -> int:
    cin = B.shape(x)[0]
    h = B.groupnorm(x, prefix + ".norm1", cin, spec.groups, spec.norm_den)
    h = B.act(h, "lut.sigmoid")
    h = B.conv(h, prefix + ".conv1", cout, cin)
    h = B.groupnorm(h, prefix + ".norm2", cout, spec.groups, spec.norm_den)
    h = B.act(h, "lut.sigmoid")
    h = B.conv(h, prefix + ".conv2", cout, cout)
    if cin != cout:
        x = B.conv(x, prefix + ".conv_shortcut", cout, cin, kh=1, kw=1, pad=0)
    return B.add(x, h)


def build_ff(B: _Builder, n: int, prefix: str, c: int, inner: int, table: str) -> int:
    """GEGLU: a * gelu(g), with the two halves of ff.net.0.proj."""
    a = B.gemm(n, prefix + ".ff.net.0.proj.a", inner, c)
    g = B.gemm(n, prefix + ".ff.net.0.proj.g", inner, c)
    return B.mul(a, B.act(g, table))


def build_transformer(B: _Builder, x: int, prefix: str, spec: UnetSpec, level: int,
                      ctx: int, only_cross: bool, table: str) -> int:
    c, hh, ww = B.shape(x)
    heads, dh = spec.heads[level], spec.head_dim[level]
    x0 = x
    t = B.groupnorm(x, prefix + ".norm", c, spec.groups, spec.norm_den)
    t = B.to_tokens(t)
    t = B.gemm(t, prefix + ".proj_in", c, c)
    inner = spec.ff_mult * c
    for k in range(spec.transformer_layers[level]):
        b = f"{prefix}.transformer_blocks.{k}"
        if only_cross:
            n = B.layernorm(t, b + ".norm1", c, spec.norm_den)
            q = B.gemm(n, b + ".attn1.to_q", c, c)
            kk = B.gemm(ctx, b + ".attn1.to_k", c, spec.cross_dim)
            vv = B.gemm(ctx, b + ".attn1.to_v", c, spec.cross_dim)
            t = B.add(t, B.gemm(B.attention(q, kk, vv, heads, dh), b + ".attn1.to_out.0", c, c))
            n = B.layernorm(t, b + ".norm2", c, spec.norm_den)
            t = B.add(t, B.gemm(build_ff(B, n, b, c, inner, table), b + ".ff.net.2", c, inner))
        else:
            n = B.layernorm(t, b + ".norm1", c, spec.norm_den)
            q = B.gemm(n, b + ".attn1.to_q", c, c)
            kk = B.gemm(n, b + ".attn1.to_k", c, c)
            vv = B.gemm(n, b + ".attn1.to_v", c, c)
            t = B.add(t, B.gemm(B.attention(q, kk, vv, heads, dh), b + ".attn1.to_out.0", c, c))
            n = B.layernorm(t, b + ".norm2", c, spec.norm_den)
            q = B.gemm(n, b + ".attn2.to_q", c, c)
            kk = B.gemm(ctx, b + ".attn2.to_k", c, spec.cross_dim)
            vv = B.gemm(ctx, b + ".attn2.to_v", c, spec.cross_dim)
            t = B.add(t, B.gemm(B.attention(q, kk, vv, heads, dh), b + ".attn2.to_out.0", c, c))
            n = B.layernorm(t, b + ".norm3", c, spec.norm_den)
            t = B.add(t, B.gemm(build_ff(B, n, b, c, inner, table), b + ".ff.net.2", c, inner))
    t = B.gemm(t, prefix + ".proj_out", c, c)
    t = B.to_channels(t, c, hh, ww)
    return B.add(x0, t)


def build_unet(B: _Builder, spec: UnetSpec, x: int, ctx: int, table: str = "lut.phi") -> int:
    cur = B.conv(x, "unet.conv_in", spec.block_out[0], spec.in_ch)
    skips = [cur]
    for i in range(spec.levels):
        cout = spec.block_out[i]
        attn = "CrossAttn" in spec.down_types[i]
        for j in range(spec.layers):
            cur = build_resnet(B, cur, f"unet.down_blocks.{i}.resnets.{j}", cout, spec)
            if attn:
                cur = build_transformer(B, cur, f"unet.down_blocks.{i}.attentions.{j}",
                                        spec, i, ctx, spec.only_cross[i], table)
            skips.append(cur)
        if i < spec.levels - 1:
            cur = B.conv(cur, f"unet.down_blocks.{i}.downsamplers.0.conv", cout, cout, stride=2)
            skips.append(cur)
    if spec.mid:
        cur = build_resnet(B, cur, "unet.mid_block.resnets.0", spec.block_out[-1], spec)
        cur = build_transformer(B, cur, "unet.mid_block.attentions.0", spec,
                                spec.levels - 1, ctx, False, table)
        cur = build_resnet(B, cur, "unet.mid_block.resnets.1", spec.block_out[-1], spec)
    up_channels = list(reversed(spec.block_out))
    for i in range(spec.levels):
        cout = up_channels[i]
        level = spec.levels - 1 - i
        attn = "CrossAttn" in spec.up_types[i]
        for j in range(spec.layers + 1):
            cur = B.concat(cur, skips.pop())
            cur = build_resnet(B, cur, f"unet.up_blocks.{i}.resnets.{j}", cout, spec)
            if attn:
                cur = build_transformer(B, cur, f"unet.up_blocks.{i}.attentions.{j}",
                                        spec, level, ctx, spec.only_cross[level], table)
        if i < spec.levels - 1:
            cur = B.upsample(cur)
            cur = B.conv(cur, f"unet.up_blocks.{i}.upsamplers.0.conv", cout, cout)
    if skips:
        raise VirError(f"skip list not consumed: {len(skips)} left")
    cur = B.groupnorm(cur, "unet.conv_norm_out", spec.block_out[0], spec.groups, spec.norm_den)
    cur = B.act(cur, "lut.sigmoid")
    return B.conv(cur, "unet.conv_out", spec.out_ch, spec.block_out[0])


def build_text_encoder(B: _Builder, spec: TextSpec, ids: int) -> int:
    g = B.g
    w = g.weight("te.tok", spec.vocab, spec.hidden, dtype=B._dtype("te.tok"))
    m = B.param("te.tok.m", (spec.vocab,))
    s = B.param("te.tok.s", (spec.vocab,))
    pos = B.param("te.pos", (spec.ctx_len, spec.hidden))
    x = g.emit(Op.EMBED, [ids, w, m, s, pos], (spec.ctx_len, spec.hidden),
               {Attr.POST_SHIFT: BITS})
    table = TE_TABLE.get(spec.act, "lut.phi")
    dh = spec.hidden // spec.heads
    for layer in range(spec.layers):
        p = f"te.layers.{layer}"
        n = B.layernorm(x, p + ".layer_norm1", spec.hidden, spec.norm_den)
        q = B.gemm(n, p + ".self_attn.q_proj", spec.hidden, spec.hidden)
        k = B.gemm(n, p + ".self_attn.k_proj", spec.hidden, spec.hidden)
        v = B.gemm(n, p + ".self_attn.v_proj", spec.hidden, spec.hidden)
        a = B.attention(q, k, v, spec.heads, dh, spec.causal)
        x = B.add(x, B.gemm(a, p + ".self_attn.out_proj", spec.hidden, spec.hidden))
        n = B.layernorm(x, p + ".layer_norm2", spec.hidden, spec.norm_den)
        f1 = B.gemm(n, p + ".mlp.fc1", spec.inner, spec.hidden)
        x = B.add(x, B.gemm(B.act(f1, table), p + ".mlp.fc2", spec.hidden, spec.inner))
    return B.layernorm(x, "te.final_layer_norm", spec.hidden, spec.norm_den)


def sampler_constants(sched_cfg: dict) -> tuple[int, int]:
    """(cin Q30, sigma Q16) reconstructed from a scheduler config.

    sigma = sqrt((1 - alpha_bar) / alpha_bar) at the last training timestep,
    with ``scaled_linear`` betas (linear in sqrt(beta), as diffusers builds
    them) accumulated in float32 so the result matches the reference pipeline:
    for SD/SDXS (8.5e-4..1.2e-2, 1000 steps) that is 14.61465..., i.e. Q16
    957786 -- the value in the v2 model file.

    The exporter must take the two constants from the reference pipeline (or
    copy ``c.sig``/``c.cin`` when re-expressing v2): recomputing cin from an
    unquantised sigma in float64 differs by a few ulps from v2's 1071237033.
    """
    steps = int(sched_cfg.get("num_train_timesteps", 1000))
    b0 = float(sched_cfg.get("beta_start", 8.5e-4))
    b1 = float(sched_cfg.get("beta_end", 0.012))
    if sched_cfg.get("beta_schedule", "scaled_linear") == "scaled_linear":
        betas = (np.linspace(math.sqrt(b0), math.sqrt(b1), steps) ** 2).astype(np.float32)
    else:
        betas = np.linspace(b0, b1, steps, dtype=np.float32)
    alpha_bar = np.cumprod((1.0 - betas).astype(np.float32), dtype=np.float32)
    sigma = float(np.sqrt((1.0 - alpha_bar[-1]) / alpha_bar[-1]))
    cin = int(round(sigma / math.sqrt(sigma * sigma + 1.0) * 2.0 ** 30))
    return cin, int(round(sigma * 65536))


def build_model(unet_cfg: dict, te_cfg: dict, sched_cfg: dict | None = None,
                causal_text: bool = False, pad_policy: int = PAD_USE_EOS_ID,
                tokenizer_dir: str | None = None, latent: tuple[int, int, int] | None = None,
                ctx_tensor=None, wgroup: int = 0, wdtype=None) -> VirModel:
    """Build the graph of a text-to-image UNet model from diffusers configs.

    Weights are *not* exported here: every WEIGHT/PARAM tensor is registered
    with the right name, shape and dtype and left without data, so the same
    builder is used by the exporter (which fills them) and by the conformance
    tests (which only check topology, shapes and cost).

    ``wdtype`` maps a weight prefix to "i8" or "i16" (default: all i8);
    ``wgroup`` applies to the i8 weights only.
    """
    uspec = unet_spec(unet_cfg)
    tspec = text_spec(te_cfg, causal_text, pad_policy)
    g = Graph()
    B = _Builder(g, wgroup=wgroup, wdtype=wdtype)
    latent = latent or (uspec.in_ch, uspec.sample_size, uspec.sample_size)
    x = g.declare_input("latent", latent)
    if ctx_tensor is not None:
        ctx = ctx_tensor
    else:
        ids = g.declare_input("ids", (tspec.ctx_len,))
        ctx = build_text_encoder(B, tspec, ids)
    eps = build_unet(B, uspec, x, ctx)
    g.outputs = [eps]
    gauss = B.table("lut.gauss")
    header = VirHeader(
        latent=tuple(latent), out=tuple(latent), out_dtype=OUT_I16LE, clamp_abs=32767,
        out_shift=8, tok_type=TOK_CLIP_BPE if ctx_tensor is None else TOK_NONE,
        tok_vocab_tensor=0, tok_merges_tensor=0, vocab_size=tspec.vocab,
        bos_id=tspec.bos, eos_id=tspec.eos, pad_id=tspec.pad, pad_policy=tspec.pad_policy,
        ctx_len=tspec.ctx_len, max_tokens=tspec.max_tokens, te_hidden=tspec.hidden,
        te_layers=tspec.layers, flags=1 if ctx_tensor is None else 0,
        input_count=len(g.inputs), output_count=1,
    )
    cin, sig = sampler_constants(sched_cfg or {})
    noise = NoiseSpec(
        n_samples=_prod(latent), gauss_tensor=gauss, eps_tensor=eps, latent_tensor=x,
        cin_q30=cin, sigma_q16=sig, latent_shift=BITS, out_shift=8, clamp_abs=32767,
        ks_bytes=2 * _prod(latent),
    )
    model = VirModel(header, g, noise)
    if tokenizer_dir:
        add_tokenizer(model, tokenizer_dir)
    return model


def add_tokenizer(model: VirModel, tokenizer_dir: str) -> None:
    """Attach vocab.json/merges.txt as BLOB tensors and switch the header on."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")
    with open(vocab_path, "rb") as f:
        vocab = np.frombuffer(f.read(), dtype=np.uint8)
    with open(merges_path, "rb") as f:
        merges = np.frombuffer(f.read(), dtype=np.uint8)
    g = model.graph
    vid = g.add_tensor(TensorDesc("tok.vocab", (int(vocab.size),), "u8", Storage.BLOB, vocab))
    mid = g.add_tensor(TensorDesc("tok.merges", (int(merges.size),), "u8", Storage.BLOB, merges))
    h = model.header
    h.tok_type = TOK_CLIP_BPE
    h.flags |= 1
    h.tok_vocab_tensor = vid
    h.tok_merges_tensor = mid


def embed_context(B: _Builder, ctx: "np.ndarray", name: str = "ctx.const") -> int:
    """Store a fixed context matrix as a parameter tensor (used by the tests)."""
    return B.param(name, (int(ctx.shape[0]), int(ctx.shape[1])), ctx.astype(np.int32))


# ---------------------------------------------------------------------------
# tiny end-to-end model used by the self-test
# ---------------------------------------------------------------------------

DEMO_UNET = {
    "block_out_channels": [8, 16],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
    "layers_per_block": 1,
    "attention_head_dim": 2,
    "cross_attention_dim": 8,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_num_groups": 4,
    "norm_eps": 1e-5,
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 4,
    "only_cross_attention": False,
    "use_linear_projection": True,
}
DEMO_TE = {
    "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
    "intermediate_size": 16, "vocab_size": 32, "max_position_embeddings": 7,
    "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5,
}


def randomize_weights(model: VirModel, seed: int = 0) -> None:
    """Fill every data-less weight/param tensor with deterministic pseudo-weights."""
    rng = np.random.default_rng(seed)
    for t in model.graph.tensors:
        if t.data is not None:
            continue
        if t.storage == Storage.WEIGHT:
            if t.dtype == "i8":  # unchanged draw: keeps the checked-in demo vectors
                t.data = rng.integers(-127, 128, size=t.shape, dtype=np.int16).astype(np.int8)
            else:
                t.data = rng.integers(-32767, 32768, size=t.shape, dtype=np.int32).astype(np.int16)
        elif t.storage == Storage.PARAM:
            name = t.name
            if name.endswith(".m"):
                lo = 1 if len(t.shape) == 2 else 1 << 14  # group mantissas use [1, 2^15)
                t.data = rng.integers(lo, 1 << 15, size=t.shape).astype(np.int32)
            elif name.endswith(".s"):
                t.data = rng.integers(14, 20, size=t.shape).astype(np.int32)
            else:
                t.data = rng.integers(-(1 << 14), 1 << 14, size=t.shape).astype(np.int32)


def build_demo_model(seed: int = 0, wgroup: int = 0, wdtype=None) -> VirModel:
    rng = np.random.default_rng(seed + 1)
    ctx = rng.integers(-(1 << 15), 1 << 15, size=(7, 8)).astype(np.int32)
    g = Graph()
    B = _Builder(g, wgroup=wgroup, wdtype=wdtype)
    latent = g.declare_input("latent", (4, 8, 8))
    ctx_id = embed_context(B, ctx)
    eps = build_unet(B, unet_spec(DEMO_UNET), latent, ctx_id)
    g.outputs = [eps]
    gauss = B.table("lut.gauss")
    model = VirModel(
        VirHeader(latent=(4, 8, 8), out=(4, 8, 8), tok_type=TOK_NONE, flags=0,
                  input_count=1, output_count=1, ctx_len=0, max_tokens=0),
        g,
        NoiseSpec(n_samples=4 * 8 * 8, gauss_tensor=gauss, eps_tensor=eps,
                  latent_tensor=latent, cin_q30=1071237033,
                  sigma_q16=957786, ks_bytes=2 * 4 * 8 * 8),
    )
    randomize_weights(model, seed)
    return model


# ---------------------------------------------------------------------------
# reference implementations used only by the self-test (deliberately naive)
# ---------------------------------------------------------------------------


def _ref_row_exp(m: int) -> int:
    e = 0
    for k in range(15, 32):
        if m >= (1 << k):
            e += 1
    return e


def _ref_rshr(v: int, s: int) -> int:
    if s >= 0:
        p = min(s, 62)
        r = (v + ((1 << p) >> 1)) >> p
    else:
        n = min(-s, 31)
        lim = 1 << (62 - n)
        r = max(-lim, min(lim, v)) << n
    return max(I32_MIN, min(I32_MAX, r))


def _ref_quant(v: int, e: int) -> int:
    return max(-32767, min(32767, _ref_rshr(v, e)))


def _ref_gemm(x, w, m, s, b):
    t_n, k_n = x.shape
    n_n = w.shape[0]
    out = np.zeros((t_n, n_n), dtype=np.int64)
    for t in range(t_n):
        e = _ref_row_exp(max(abs(int(v)) for v in x[t]))
        q = [_ref_quant(int(v), e) for v in x[t]]
        for n in range(n_n):
            acc = sum(q[k] * int(w[n, k]) for k in range(k_n))
            y = _ref_rshr(acc * int(m[n]), int(s[n]) - e)
            out[t, n] = max(I32_MIN, min(I32_MAX, y + int(b[n])))
    return out


def _ref_gemm_grouped(x, w, m, s, b, grp):
    t_n, k_n = x.shape
    n_n = w.shape[0]
    out = np.zeros((t_n, n_n), dtype=np.int64)
    for t in range(t_n):
        e = _ref_row_exp(max(abs(int(v)) for v in x[t]))
        q = [_ref_quant(int(v), e) for v in x[t]]
        for n in range(n_n):
            acc = 0
            for k in range(k_n):
                acc += q[k] * int(w[n, k]) * int(m[n, k // grp])
            y = _ref_rshr(acc, int(s[n]) - e)
            out[t, n] = max(I32_MIN, min(I32_MAX, y + int(b[n])))
    return out


def _ref_conv(x, w, m, s, b, kh, kw, stride, pad):
    c_n, h_n, w_n = x.shape
    n_n = w.shape[0]
    ho = (h_n + 2 * pad - kh) // stride + 1
    wo = (w_n + 2 * pad - kw) // stride + 1
    out = np.zeros((n_n, ho, wo), dtype=np.int64)
    for oy in range(ho):
        for ox in range(wo):
            taps = []
            mx = 0
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = oy * stride + ky - pad, ox * stride + kx - pad
                    if 0 <= iy < h_n and 0 <= ix < w_n:
                        taps.append((ky, kx, iy, ix))
                        for c in range(c_n):
                            mx = max(mx, abs(int(x[c, iy, ix])))
            e = _ref_row_exp(mx)
            for n in range(n_n):
                acc = 0
                for c in range(c_n):
                    for ky, kx, iy, ix in taps:
                        acc += _ref_quant(int(x[c, iy, ix]), e) * int(w[n, (c * kh + ky) * kw + kx])
                y = _ref_rshr(acc * int(m[n]), int(s[n]) - e)
                out[n, oy, ox] = max(I32_MIN, min(I32_MAX, y + int(b[n])))
    return out


def _ref_norm_core(v, den: int):
    n = len(v)
    mean = (sum(v) + n // 2) // n
    d = [x - mean for x in v]
    e = max(0, _ref_row_exp(max(abs(x) for x in d)) - 5)
    var = sum(_ref_rshr(x, e) ** 2 for x in d) // n
    eps = ((1 << (32 - 2 * e)) + den // 2) // den
    sd = math.isqrt((var + eps) << 16)
    return [((x << (24 - e)) * 2 + sd) // (2 * sd) for x in d]


def _ref_lut(x: int, table, lo: int, step_bits: int, edge_lo: int, edge_hi: int) -> int:
    span = (len(table) - 1) << step_bits
    z = x - lo
    if z < 0:
        return edge_lo
    if z >= span:
        return edge_hi
    i = z >> step_bits
    f = z & ((1 << step_bits) - 1)
    return int(table[i]) + ((int(table[i + 1]) - int(table[i])) * f >> step_bits)


def _ref_attention(q, k, v, heads, dh, causal, exp_table, qs):
    tq = q.shape[0]
    tk = k.shape[0]
    c = q.shape[1]
    out = np.zeros((tq, c), dtype=np.int64)
    for h in range(heads):
        eq = [_ref_row_exp(max(abs(int(x)) for x in q[t, h * dh:(h + 1) * dh])) for t in range(tq)]
        ek = [_ref_row_exp(max(abs(int(x)) for x in k[t, h * dh:(h + 1) * dh])) for t in range(tk)]
        ev = [_ref_row_exp(max(abs(int(v[t, h * dh + d])) for t in range(tk))) for d in range(dh)]
        for i in range(tq):
            s = []
            for j in range(tk):
                if causal and j > i:
                    s.append(I32_MIN)
                    continue
                acc = 0
                for d in range(dh):
                    acc += _ref_quant(int(q[i, h * dh + d]), eq[i]) * _ref_quant(
                        int(k[j, h * dh + d]), ek[j])
                s.append(_ref_rshr(acc * qs, 32 - eq[i] - ek[j]))
            mx = max(s)
            e = [_ref_lut(mx - x, exp_table, 0, 10, 0, 0) for x in s]
            r = (1 << 62) // sum(e)
            p = [min((x * r + (1 << 46)) >> 47, 32767) for x in e]
            for d in range(dh):
                o = sum(p[j] * _ref_quant(int(v[j, h * dh + d]), ev[d]) for j in range(tk))
                out[i, h * dh + d] = _ref_rshr(o, 15 - ev[d])
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

#: regression pin for the tagged hash (see the structural checks as well)
_KAT_MSG = b"VIR format v0.1 reference test"
_KAT_HASH = "23838768ee71b500ac26fffb781361d9e0e84b5518daa15fd46924f81ca269fd"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def _tiny_params(rng, n):
    m = rng.integers(1 << 14, 1 << 15, size=n).astype(np.int32)
    s = rng.integers(14, 20, size=n).astype(np.int32)
    b = rng.integers(-(1 << 16), 1 << 16, size=n).astype(np.int32)
    return m, s, b


def t_primitives() -> str:
    for (v, s), want in (((5, 1), 3), ((-5, 1), -2), ((7, 2), 2), ((6, 2), 2),
                         ((0, 0), 0), ((-1, 0), -1), ((1 << 30, -1), I32_MAX),
                         ((-(1 << 30), -1), I32_MIN)):
        got = int(rshr(np.array(v), s))
        assert got == want, f"rshr({v},{s}) = {got}, want {want}"
    assert int(quant16(np.array(40000), 0)) == 32767
    assert int(quant16(np.array(-40000), 0)) == -32767
    assert int(quant16(np.array(1000), 2)) == 250
    assert int(floor_div(-7, 2)) == -4 and int(floor_div(7, 2)) == 3
    assert [int(x) for x in row_exp(np.array([0, 1, 32767, 32768, 65535, 1 << 20]))] == \
        [0, 0, 0, 1, 1, 6]
    for v in (0, 1, 2, 3, 4, 15, 16, 17, 10 ** 12, (1 << 40) - 1):
        assert int(isqrt(np.array(v))) == math.isqrt(v)
    return "rshr/quant16/floor_div/isqrt/row_exp"


def t_gemm() -> str:
    rng = np.random.default_rng(11)
    t_n, k_n, n_n = 5, 7, 4
    x = rng.integers(-(1 << 20), 1 << 20, size=(t_n, k_n)).astype(np.int64)
    w = rng.integers(-127, 128, size=(n_n, k_n)).astype(np.int8)
    m, s, b = _tiny_params(rng, n_n)
    node = Node(Op.GEMM, [], [], {})
    got = op_gemm(node, [x, w, m, s, b])
    want = _ref_gemm(x, w, m, s, b)
    assert np.array_equal(got, want), f"gemm mismatch\n{got}\n{want}"
    # i16 weights: same formula, larger |w|; s grows by 8 to keep the scale.
    w16 = rng.integers(-32767, 32768, size=(n_n, k_n)).astype(np.int16)
    s16 = (s + 8).astype(np.int32)
    got = op_gemm(node, [x, w16, m, s16, b])
    want = _ref_gemm(x, w16, m, s16, b)
    assert np.array_equal(got, want), "i16 gemm mismatch"
    return f"{t_n}x{k_n}x{n_n} exact (i8 and i16 weights)"


def t_gemm_grouped() -> str:
    rng = np.random.default_rng(31)
    t_n, k_n, n_n = 4, 70, 3  # K not a multiple of the group: the last group is short
    x = rng.integers(-(1 << 22), 1 << 22, size=(t_n, k_n)).astype(np.int64)
    w = rng.integers(-127, 128, size=(n_n, k_n)).astype(np.int8)
    b = rng.integers(-(1 << 16), 1 << 16, size=n_n).astype(np.int32)
    s = rng.integers(14, 20, size=n_n).astype(np.int32)
    for grp in (32, 7, 1):
        ng = -(-k_n // grp)
        m = rng.integers(1, 1 << 15, size=(n_n, ng)).astype(np.int32)
        got = op_gemm(Node(Op.GEMM, [], [], {Attr.WGROUP: grp}), [x, w, m, s, b])
        want = _ref_gemm_grouped(x, w, m, s, b, grp)
        assert np.array_equal(got, want), f"grouped gemm G={grp} mismatch"
    # A constant mantissa across the groups is exactly the per-row form.
    mrow = rng.integers(1 << 14, 1 << 15, size=n_n).astype(np.int32)
    mg = np.repeat(mrow[:, None], -(-k_n // 32), axis=1)
    g32 = op_gemm(Node(Op.GEMM, [], [], {Attr.WGROUP: 32}), [x, w, mg, s, b])
    g0 = op_gemm(Node(Op.GEMM, [], [], {Attr.WGROUP: 0}), [x, w, mrow, s, b])
    assert np.array_equal(g32, g0), "uniform group mantissas differ from the per-row form"
    # Grouped CONV2D through the same accumulate.
    xc = rng.integers(-(1 << 18), 1 << 18, size=(5, 4, 4)).astype(np.int64)
    wc = rng.integers(-127, 128, size=(2, 45)).astype(np.int8)
    mc = rng.integers(1, 1 << 15, size=(2, 2)).astype(np.int32)
    sc = rng.integers(14, 20, size=2).astype(np.int32)
    bc = rng.integers(-(1 << 16), 1 << 16, size=2).astype(np.int32)
    attrs = {Attr.KH: 3, Attr.KW: 3, Attr.STRIDE: 1, Attr.PAD: 1, Attr.WGROUP: 32}
    yc = op_conv2d(Node(Op.CONV2D, [], [], attrs), [xc, wc, mc, sc, bc])
    cols, e = _conv_cols(xc, 3, 3, 1, 1, 4, 4)
    want_c = np.zeros((16, 2), dtype=np.int64)
    for t in range(16):
        for n in range(2):
            acc = sum(int(cols[t, k]) * int(wc[n, k]) * int(mc[n, k // 32]) for k in range(45))
            want_c[t, n] = max(I32_MIN, min(I32_MAX, _ref_rshr(acc, int(sc[n]) - int(e.reshape(-1)[t])) + int(bc[n])))
    assert np.array_equal(yc, want_c.T.reshape(2, 4, 4)), "grouped conv mismatch"
    return "G=32/7/1 vs naive, uniform == per-row, conv G=32"


def t_conv() -> str:
    rng = np.random.default_rng(12)
    for (kh, kw, stride, pad) in ((3, 3, 1, 1), (3, 3, 2, 1), (1, 1, 1, 0)):
        c_n, h_n, w_n, n_n = 2, 5, 5, 3
        x = rng.integers(-(1 << 18), 1 << 18, size=(c_n, h_n, w_n)).astype(np.int64)
        w = rng.integers(-127, 128, size=(n_n, c_n * kh * kw)).astype(np.int8)
        m, s, b = _tiny_params(rng, n_n)
        node = Node(Op.CONV2D, [], [], {Attr.KH: kh, Attr.KW: kw, Attr.STRIDE: stride, Attr.PAD: pad})
        got = op_conv2d(node, [x, w, m, s, b])
        want = _ref_conv(x, w, m, s, b, kh, kw, stride, pad)
        assert np.array_equal(got, want), f"conv {kh}x{kw}/{stride}/{pad} mismatch"
    return "3x3 s1, 3x3 s2, 1x1"


def t_norm() -> str:
    rng = np.random.default_rng(13)
    den = 100000
    x = rng.integers(-(1 << 20), 1 << 20, size=(4, 2, 3)).astype(np.int64)
    gamma = rng.integers(-(1 << 15), 1 << 15, size=4).astype(np.int32)
    beta = rng.integers(-(1 << 15), 1 << 15, size=4).astype(np.int32)
    eps = np.array([den], dtype=np.int32)
    groups = 2
    node = Node(Op.GROUPNORM, [], [], {Attr.GROUPS: groups})
    got = op_groupnorm(node, [x, gamma, beta, eps])
    want = np.zeros_like(x)
    for gi in range(groups):
        chunk = x[gi * 2:(gi + 1) * 2].reshape(-1)
        y = _ref_norm_core([int(v) for v in chunk], den)
        y = np.array(y, dtype=np.int64).reshape(2, 6)
        for c in range(2):
            ci = gi * 2 + c
            for p in range(6):
                want[ci].reshape(-1)[p] = max(I32_MIN, min(I32_MAX,
                    _ref_rshr(int(y[c, p]) * int(gamma[ci]), BITS) + int(beta[ci])))
    assert np.array_equal(got, want), "groupnorm mismatch"
    x2 = rng.integers(-(1 << 20), 1 << 20, size=(3, 5)).astype(np.int64)
    g2 = rng.integers(-(1 << 15), 1 << 15, size=5).astype(np.int32)
    b2 = rng.integers(-(1 << 15), 1 << 15, size=5).astype(np.int32)
    got2 = op_layernorm(Node(Op.LAYERNORM, [], [], []), [x2, g2, b2, eps])
    want2 = np.zeros_like(x2)
    for r in range(3):
        y = _ref_norm_core([int(v) for v in x2[r]], den)
        for c in range(5):
            want2[r, c] = max(I32_MIN, min(I32_MAX,
                _ref_rshr(y[c] * int(g2[c]), BITS) + int(b2[c])))
    assert np.array_equal(got2, want2), "layernorm mismatch"
    return "GroupNorm(g=2) + LayerNorm"


def t_lut() -> str:
    table = canonical_tables()["lut.sigmoid"]
    attrs = {Attr.LUT_LO: -(16 << BITS), Attr.LUT_STEP_BITS: 10, Attr.LUT_EDGE_LO: 0,
             Attr.LUT_EDGE_HI: 65536, Attr.POST_SHIFT: BITS}
    node = Node(Op.LUT_ACT, [], [], attrs)
    xs = np.array([0, 1 << 16, -(1 << 16), 20 << 16, -20 << 16, 8 << 16], dtype=np.int64)
    y = op_lut_act(node, [xs, table])
    assert int(y[0]) == 0, y
    assert int(y[3]) == 20 << 16 and int(y[4]) == 0
    grid = np.arange(-(16 << 16), 16 << 16, 1 << 13, dtype=np.int64)
    want = []
    for v in grid:
        val = _ref_lut(int(v), table, -(16 << BITS), 10, 0, 65536)
        want.append(max(I32_MIN, min(I32_MAX, (int(v) * val + (1 << 15)) >> 16)))
    got = op_lut_act(node, [grid, table])
    assert [int(v) for v in got] == want, "LUT interpolation differs from the definition"
    # silu -> x for large |x|, -> 0 for very negative x (the table carries it)
    assert abs(int(got[0])) <= 1, int(got[0])
    assert abs(int(got[-1]) - int(grid[-1])) <= 2, (int(got[-1]), int(grid[-1]))
    return "sigmoid table: 4096-point exact match, domain edges"


def t_attention() -> str:
    rng = np.random.default_rng(14)
    tq = tk = 4
    heads, dh = 2, 2
    c = heads * dh
    q = rng.integers(-(1 << 18), 1 << 18, size=(tq, c)).astype(np.int64)
    k = rng.integers(-(1 << 18), 1 << 18, size=(tk, c)).astype(np.int64)
    v = rng.integers(-(1 << 18), 1 << 18, size=(tk, c)).astype(np.int64)
    table = canonical_tables()["lut.exp"]
    for causal in (False, True):
        node = Node(Op.ATTENTION, [], [], {Attr.NUM_HEADS: heads, Attr.HEAD_DIM: dh,
                                           Attr.CAUSAL: int(causal), Attr.Q_SCALE: q_scale(dh)})
        got = op_attention(node, [q, k, v, table])
        want = _ref_attention(q, k, v, heads, dh, causal, table, q_scale(dh))
        assert np.array_equal(got, want), f"attention causal={causal} mismatch\n{got}\n{want}"
    return "2 heads x 2 dim, causal + full"


def t_attention_shift() -> str:
    """For head_dim 64 the general score shift must collapse to v2's 19 - eq - ek."""
    rng = np.random.default_rng(15)
    s = rng.integers(-(1 << 33), 1 << 33, size=64).astype(np.int64)
    eq, ek = 3, 5
    a = rshr(s * np.int64(q_scale(64)), 32 - eq - ek)
    b = rshr(s, 19 - eq - ek)
    assert np.array_equal(a, b), "q_scale(64) does not reproduce the v2 shift"
    assert q_scale(64) == 8192 and q_scale(80) == 7327 and q_scale(160) == 5181
    return "19-eq-ek == rshr(S*8192, 32-eq-ek)"


def t_chacha() -> str:
    seed = bytes(range(32))
    ks = chacha20_keystream(seed, 2 * 4 * 8 * 8)
    assert len(ks) == 2 * 4 * 8 * 8
    assert chacha20_keystream(seed, 128) == ks[:128], "keystream is not prefix-stable"
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    except ImportError:
        return "prefix stable (cryptography not installed, cross-check skipped)"
    enc = Cipher(algorithms.ChaCha20(seed, b"\0" * 16), mode=None).encryptor()
    ref = enc.update(b"\0" * (2 * 4 * 8 * 8))
    assert ks == ref, "keystream differs from the independent ChaCha20 implementation"
    return "matches cryptography's ChaCha20 (zero nonce)"


def t_hash() -> str:
    empty = hashlib.sha256(b"").digest()
    assert tagged_hash("", b"") == hashlib.sha256(empty + empty).digest()
    msg = bytes([1]) + _KAT_MSG
    one = model_hash_digest(1, _KAT_MSG)
    stream = _HashStream("POI/model")
    stream.update(bytes([1]))
    stream.update(_KAT_MSG)
    assert stream.digest() == one
    d = model_hash_digest(1, _KAT_MSG)
    assert digest_to_display(d) == d[::-1].hex()
    assert model_hash(2, _KAT_MSG) != model_hash(1, _KAT_MSG)
    if _KAT_HASH != "TBD":
        assert digest_to_display(one) == _KAT_HASH, (digest_to_display(one), _KAT_HASH)
    return digest_to_display(one)[:16] + "..."


def t_hash_v2(path: str | None) -> str:
    path = path or os.path.join(_repo_root(), "models", "sd-turbo-int.poisd")
    if not os.path.exists(path):
        raise SkipTest(f"{path} not present")
    expect = "a7fd64b04c8bf36c1b74a629e32f093ce10e364787292f53fa1337557eb9776a"
    h = _HashStream("POI/model")
    h.update(bytes([2]))
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    got = digest_to_display(h.digest())
    assert got == expect, f"{got} != {expect} (SD_MODEL_HASH, vela/src/poi/sdmodel.h)"
    return "v2 .poisd reproduces SD_MODEL_HASH"


class SkipTest(Exception):
    pass


def t_roundtrip() -> str:
    m = build_demo_model(0)
    b1 = m.to_bytes()
    b2 = build_demo_model(0).to_bytes()
    assert b1 == b2, "two exports of the same graph differ"
    p = VirModel.parse(b1)
    assert p.to_bytes() == b1, "write(read(x)) != x"
    seed = bytes(range(32))
    r1 = m.run(seed, [])
    assert r1 == m.run(seed, []), "inference is not deterministic"
    assert p.run(seed, []) == r1, "parsed model does not reproduce the output"
    assert len(r1) == m.header.output_bytes
    assert p.model_hash() == m.model_hash()
    # Grouped weight scales survive the file format and change the identity.
    mg = build_demo_model(0, wgroup=4)
    assert any(int(n.attr(Attr.WGROUP) or 0) for n in mg.graph.nodes), "demo has no grouped layer"
    bg = mg.to_bytes()
    pg = VirModel.parse(bg)
    assert pg.to_bytes() == bg, "grouped: write(read(x)) != x"
    assert pg.run(seed, []) == mg.run(seed, []), "grouped: parsed model differs"
    assert mg.model_hash() != m.model_hash()
    # Two-coefficient sampler (DEIS one-step) survives the file and changes the output.
    m1 = build_demo_model(0)
    out0 = m1.run(seed, [])
    m1.noise.sampler_kind = 1
    m1.noise.sigma_q16 = 959617
    m1.noise.eps_q16 = 955468
    b1c = m1.to_bytes()
    p1 = VirModel.parse(b1c)
    assert (p1.noise.sampler_kind, p1.noise.sigma_q16, p1.noise.eps_q16) == (1, 959617, 955468)
    assert p1.to_bytes() == b1c
    assert p1.run(seed, []) == m1.run(seed, [])
    assert p1.run(seed, []) != out0, "sampler coefficients must change x0"
    # Mixed i8/i16 weights (i16 for attention, group-4 i8 elsewhere).
    mix = build_demo_model(0, wgroup=4, wdtype=lambda p: "i16" if ".attn" in p else "i8")
    dts = {t.dtype for t in mix.graph.tensors if t.storage == Storage.WEIGHT}
    assert dts == {"i8", "i16"}, dts
    bm = mix.to_bytes()
    pm = VirModel.parse(bm)
    assert pm.to_bytes() == bm and pm.run(seed, []) == mix.run(seed, [])
    return (f"{len(b1)} bytes, {len(m.graph.nodes)} nodes, {len(m.graph.tensors)} tensors; "
            f"wgroup=4 variant {len(bg)} bytes; sampler_kind=1 variant {len(b1c)} bytes; "
            f"i8/i16 mix {len(bm)} bytes")


def t_reject() -> str:
    base = build_demo_model(0).to_bytes()
    h = _unpack_header(base)
    cases = []
    bad = bytearray(base)
    bad[0] = ord("X")
    cases.append(("magic", bytes(bad)))
    bad = bytearray(base)
    bad[4] = 2
    cases.append(("opset", bytes(bad)))
    bad = bytearray(base)
    bad[16] ^= 1
    cases.append(("file_size", bytes(bad)))
    bad = bytearray(base)
    bad[146] = 1
    cases.append(("reserved", bytes(bad)))
    cases.append(("truncated", base[:-1]))
    pad = h.graph_off + h.graph_size
    if pad < h.tensors_off:
        bad = bytearray(base)
        bad[pad] = 1
        cases.append(("padding", bytes(bad)))
    bad = bytearray(base)
    bad[h.noise_off + 1] = 9
    cases.append(("noise_kind", bytes(bad)))
    for name, data in cases:
        try:
            VirModel.parse(data)
        except VirError:
            continue
        raise AssertionError(f"corrupted file ({name}) was accepted")
    return f"{len(cases)} corruptions rejected"


def t_builders() -> str:
    root = _repo_root()
    sdxs_dir = os.path.join(root, "models", "sdxs-512-dreamshaper")
    turbo_dir = os.path.join(root, "models", "sd-turbo")
    detail = []
    for name, mdir, want in (
        ("sdxs", sdxs_dir, dict(levels=3, mid=False, heads=[8, 8, 8], dim=[40, 80, 160],
                                cross=768, te=12, hidden=768)),
        ("sd-turbo", turbo_dir, dict(levels=4, mid=True, heads=[5, 10, 20, 20],
                                     dim=[64, 64, 64, 64], cross=1024, te=23, hidden=1024)),
    ):
        ucfg, tcfg, scfg = _load_configs(mdir)
        spec = unet_spec(ucfg)
        tspec = text_spec(tcfg)
        assert spec.levels == want["levels"], (name, spec.levels)
        assert spec.mid == want["mid"], (name, spec.mid)
        assert spec.heads == want["heads"], (name, spec.heads)
        assert spec.head_dim == want["dim"], (name, spec.head_dim)
        assert spec.cross_dim == want["cross"], (name, spec.cross_dim)
        assert tspec.layers == want["te"] and tspec.hidden == want["hidden"]
        m = build_model(ucfg, tcfg, scfg)
        m.graph.validate()
        cost = m.cost()
        assert cost.macs > 10 ** 9 and cost.weights_bytes > 0
        m2 = build_model(ucfg, tcfg, scfg)
        order = sorted(range(len(m.graph.tensors)),
                       key=lambda i: m.graph.tensors[i].name.encode("ascii"))
        idmap = {old: new for new, old in enumerate(order)}
        assert _pack_graph(m.graph, idmap) == _pack_graph(m2.graph, idmap), \
            "graph export is not deterministic"
        detail.append(f"{name}: {cost.macs / 1e9:.1f} GMAC, "
                      f"{cost.weights_bytes / 1e6:.0f} MB weights, {len(m.graph.nodes)} nodes")
        if name == "sdxs":
            mg = build_model(ucfg, tcfg, scfg, wgroup=32)
            mg.graph.validate()
            cg = mg.cost()
            assert cg.macs == cost.macs, "wgroup must not change the MAC count"
            detail.append(f"sdxs wgroup=32: {cg.weights_bytes / 1e6:.0f} MB "
                          f"(+{(cg.weights_bytes - cost.weights_bytes) / 2**20:.1f} MiB scales)")
    return "; ".join(detail)


def _load_configs(model_dir: str):
    def load(sub, fallback):
        path = os.path.join(model_dir, sub, "config.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return fallback

    ucfg = load("unet", SDXS_UNET if "sdxs" in model_dir else SDTURBO_UNET)
    tcfg = load("text_encoder", SDXS_TE if "sdxs" in model_dir else SDTURBO_TE)
    scfg = load("scheduler", {"num_train_timesteps": 1000, "beta_start": 8.5e-4,
                              "beta_end": 0.012, "prediction_type": "epsilon",
                              "timestep_spacing": "trailing", "steps_offset": 1})
    return ucfg, tcfg, scfg


def t_sampler_constants() -> str:
    cin, sig = sampler_constants({"num_train_timesteps": 1000, "beta_start": 8.5e-4,
                                  "beta_end": 0.012, "beta_schedule": "scaled_linear"})
    assert sig == 957786, f"sigma Q16 {sig}, expected 957786 (c.sig in the v2 file)"
    assert abs(cin - 1071237033) <= 8, f"cin {cin}, v2 stores 1071237033"
    return f"sigma={sig}/65536, cin={cin} (v2: 1071237033)"


def t_tokenizer() -> str:
    root = _repo_root()
    tdir = os.path.join(root, "models", "sdxs-512-dreamshaper", "tokenizer")
    if not os.path.exists(os.path.join(tdir, "vocab.json")):
        raise SkipTest("tokenizer files not present")
    with open(os.path.join(tdir, "vocab.json"), "rb") as f:
        vocab = f.read()
    with open(os.path.join(tdir, "merges.txt"), "rb") as f:
        merges = f.read()
    tb = ClipBpe(vocab, merges, 49408)
    ids = tb.encode("a lighthouse at dawn")
    assert ids and all(0 <= i < 49408 for i in ids), ids
    assert tb.encode("") == []
    # round-trip through a tokenizer attached to a model file
    m = build_demo_model(0)
    add_tokenizer(m, tdir)
    assert tokenizer(m).encode("a lighthouse at dawn") == ids
    return f"{len(ids)} ids: {ids[:6]}"


def t_leaves() -> str:
    """Leaf commitments against the C++ vector (src/test/poi_fraud_tests.cpp)."""
    m = build_demo_model(0)
    seed = b"\x01" * 32
    x0, leaves = m.run_with_leaves(seed, [])
    assert x0 == m.run(seed, []), "run_with_leaves must reproduce run's x0"
    assert len(leaves) == 180, f"{len(leaves)} leaves"
    assert leaves[0][::-1].hex() == "72213fbec133972a0d49348c42a89bf32a9d0e777915f4fa32869a09c96a7bd6", leaves[0][::-1].hex()
    assert leaves[-1][::-1].hex() == "1ca2b9e0d89204ea95dcddf30fddf4b2cd0aab789bd11130417905817d3cdfb6"
    root = leaves_commitment(leaves)
    assert root[::-1].hex() == "4ef5881b3663e5b61a6e32f0555351e64639c42168b44bfbcee9a4f4aeadb632", root[::-1].hex()
    h = poi_output_hash_v5(seed, x0, root)
    assert h[::-1].hex() == "f0caab2c8168e462554e51045d9ee1d9d2ee73d0be1e6ce06fdf0ebb381e7607", h[::-1].hex()
    # leaf_count and leaf_digest agree with node_leaves on a short tensor
    assert leaf_count(0) == 1 and leaf_count(LEAF_VALUES) == 1 and leaf_count(LEAF_VALUES + 1) == 2
    v = np.arange(10, dtype=np.int32)
    assert node_leaves(7, v) == [leaf_digest(7, 0, v)]
    return f"{len(leaves)} leaves, R={root[::-1].hex()[:16]}.."


def selftest(v2_model: str | None = None, verbose: bool = True) -> int:
    results: list[tuple[str, str, str]] = []

    def run(name, fn, *args):
        try:
            detail = fn(*args) or ""
            results.append(("PASS", name, str(detail)))
            if verbose:
                print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
        except SkipTest as exc:
            results.append(("SKIP", name, str(exc)))
            if verbose:
                print(f"  SKIP  {name}   [{exc}]")
        except Exception as exc:  # noqa: BLE001
            results.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    if verbose:
        print("VIR reference implementation -- self test")
        print(f"  format v0.1, opset {OPSET}, numpy {getattr(np, '__version__', 'none')}")
    run("integer primitives", t_primitives)
    run("GEMM vs naive", t_gemm)
    run("grouped weight scales (wgroup)", t_gemm_grouped)
    run("Conv2d vs naive", t_conv)
    run("GroupNorm/LayerNorm vs naive", t_norm)
    run("LUT activation", t_lut)
    run("attention vs naive", t_attention)
    run("attention q_scale identity", t_attention_shift)
    run("ChaCha20 keystream", t_chacha)
    run("tagged hash + ModelHash", t_hash)
    run("v2 model hash (SD_MODEL_HASH)", t_hash_v2, v2_model)
    run("sampler constants", t_sampler_constants)
    run("file round-trip + determinism", t_roundtrip)
    run("PoI leaf commitments", t_leaves)
    run("malformed file rejection", t_reject)
    run("SDXS / sd-turbo graph builders", t_builders)
    run("CLIP tokenizer", t_tokenizer)
    n_fail = sum(1 for st, _, _ in results if st == "FAIL")
    n_skip = sum(1 for st, _, _ in results if st == "SKIP")
    if verbose:
        print(f"  {len(results) - n_fail - n_skip} passed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# embedded fallbacks (used when the models/ directory is not available)
# ---------------------------------------------------------------------------

SDXS_UNET = {
    "block_out_channels": [320, 640, 1280],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "layers_per_block": 1, "attention_head_dim": 8, "num_attention_heads": None,
    "cross_attention_dim": 768, "mid_block_type": None,
    "norm_num_groups": 32, "norm_eps": 1e-5, "sample_size": 64,
    "in_channels": 4, "out_channels": 4, "only_cross_attention": [True, False, False],
    "use_linear_projection": False, "transformer_layers_per_block": 1,
}
SDXS_TE = {
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    "intermediate_size": 3072, "vocab_size": 49408, "max_position_embeddings": 77,
    "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5,
}
SDTURBO_UNET = {
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                         "CrossAttnDownBlock2D", "DownBlock2D"],
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                       "CrossAttnUpBlock2D"],
    "layers_per_block": 2, "attention_head_dim": [5, 10, 20, 20],
    "num_attention_heads": None, "cross_attention_dim": 1024,
    "mid_block_type": "UNetMidBlock2DCrossAttn", "norm_num_groups": 32, "norm_eps": 1e-5,
    "sample_size": 64, "in_channels": 4, "out_channels": 4,
    "only_cross_attention": False, "use_linear_projection": True,
    "transformer_layers_per_block": 1,
}
SDTURBO_TE = {
    "hidden_size": 1024, "num_hidden_layers": 23, "num_attention_heads": 16,
    "intermediate_size": 4096, "vocab_size": 49408, "max_position_embeddings": 77,
    "hidden_act": "gelu", "layer_norm_eps": 1e-5,
}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def check_module_names(model_dir: str) -> tuple[list[str], list[str]]:
    """Cross-check the graph's weight names against the diffusers modules.

    Returns (missing, extra): names the checkpoint has but the graph does not,
    and vice versa.  Requires diffusers + transformers; used by the exporter
    before a checkpoint is quantised, so a wrong topology shows up as a name
    mismatch instead of a bad model.
    """
    try:
        import torch.nn as nn
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextConfig
    except ImportError as exc:  # pragma: no cover
        raise SkipTest(f"diffusers/transformers not installed ({exc})") from None
    ucfg, tcfg, _ = _load_configs(model_dir)
    unet = UNet2DConditionModel.from_config(ucfg)
    te = CLIPTextModel(CLIPTextConfig(**tcfg))
    want: set[str] = set()
    for name, mod in unet.named_modules():
        if name.startswith("time_embedding") or name.endswith("time_emb_proj"):
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.GroupNorm, nn.LayerNorm)):
            want.add("unet." + name)
    for name, mod in te.named_modules():
        if not name.startswith("text_model."):
            continue
        key = name[len("text_model.encoder."):] if name.startswith("text_model.encoder.") \
            else name[len("text_model."):]
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            want.add("te." + key)
    split = {n for n in want if n.endswith(".ff.net.0.proj")}
    want -= split
    want |= {n + ".a" for n in split} | {n + ".g" for n in split}
    want -= {"te.embeddings.token_embedding", "te.embeddings.position_embedding"}
    want |= {"te.tok", "te.pos"}
    model = build_model(ucfg, tcfg, {})
    got: set[str] = set()
    for t in model.graph.tensors:
        if t.storage not in (Storage.WEIGHT, Storage.PARAM):
            continue
        name = t.name
        for suf in (".w", ".m", ".s", ".b", ".g", ".beta", ".eps"):
            if name.endswith(suf):
                name = name[:-len(suf)]
                break
        got.add(name)
    return sorted(want - got), sorted(got - want)


def _hash_file(path: str, opset: int) -> str:
    """ModelHash of a file, streamed (the v2 model is 1.2 GB)."""
    h = _HashStream("POI/model")
    h.update(bytes([opset & 0xFF]))
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            h.update(chunk)
    return digest_to_display(h.digest())


def _print_summary(model: VirModel) -> None:
    cost = model.cost()
    h = model.header
    print(f"nodes      {len(model.graph.nodes)}")
    print(f"tensors    {len(model.graph.tensors)}")
    print(f"latent     {h.latent}  output {h.output_bytes} bytes")
    print(f"macs       {cost.macs / 1e9:.2f} GMAC")
    print(f"weights    {cost.weights_bytes / 1e6:.1f} MB (stored)")
    print(f"activations{cost.activation_bytes / 1e6:9.1f} MB (sum over all buffers)")
    print(f"peak       {cost.peak_bytes / 1e6:.1f} MB (weights + live activations)")
    print(f"lookups    {cost.lookups / 1e6:.2f} M")
    for t in model.graph.tensors:
        if t.name in ("lut.sigmoid", "lut.phi", "lut.exp", "lut.gauss", "lut.quick_sigmoid"):
            print(f"table      {t.name}: {t.shape[0]} entries")


def _run_cli(args) -> int:
    """Handle ``--run``: one inference, optionally with a Sampled-PoI trace."""
    if not args.seed:
        print("--run needs --seed <64 hex chars>", file=sys.stderr)
        return 2
    try:
        seed = bytes.fromhex(args.seed)
    except ValueError:
        print("--seed is not valid hex", file=sys.stderr)
        return 2
    if len(seed) != 32:
        print("--seed must be 32 bytes (64 hex chars)", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    with open(args.run, "rb") as f:
        model = VirModel.parse(f.read())
    t_parse = time.perf_counter()

    if args.prompt is not None:
        if model.header.tok_type == TOK_NONE:
            print("model has no tokenizer; use --tokens", file=sys.stderr)
            return 2
        tokens: Sequence[int] = tokenizer(model).encode(args.prompt)
    elif args.tokens:
        tokens = [int(t) for t in args.tokens.replace(" ", "").split(",") if t]
    elif model.header.tok_type == TOK_NONE:
        tokens = []
    else:
        print("--run needs --prompt or --tokens", file=sys.stderr)
        return 2

    want_trace = bool(args.trace or args.trace_root or args.trace_nodes
                      or args.trace_last or args.trace_sample)
    tracer = None
    if want_trace:
        tracer = TraceRecorder(nodes=args.trace_nodes, last_node=args.trace_last,
                               sample_every=args.trace_sample,
                               keep_rows=bool(args.trace))
    t1 = time.perf_counter()
    payload = model.run(seed, tokens, tracer)
    t2 = time.perf_counter()

    result: dict[str, Any] = {
        "model": os.path.basename(args.run),
        "parse_s": round(t_parse - t0, 3),
        "run_s": round(t2 - t1, 3),
        "x0_bytes": len(payload),
        "x0_sha256": hashlib.sha256(payload).hexdigest(),
        "poi_output_hash": poi_output_hash(seed, payload).hex(),
        "tokens": list(tokens),
        "trace": tracer.stats() if tracer is not None else None,
    }
    if args.trace and tracer is not None:
        td0 = time.perf_counter()
        tracer.write(args.trace)
        result["trace_dump_s"] = round(time.perf_counter() - td0, 3)
        result["trace_dump_bytes"] = os.path.getsize(args.trace)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"model            {result['model']}")
        print(f"parse            {result['parse_s']} s")
        print(f"run              {result['run_s']} s")
        print(f"x0               {result['x0_bytes']} bytes  sha256 {result['x0_sha256'][:32]}...")
        print(f"hashPoI          {result['poi_output_hash']}")
        if tracer is not None:
            for name, st in result["trace"]["variants"].items():
                print(f"trace {name:6s}   units {st['units']:6d}  rows {st['unit_row_bytes']:9d} B"
                      f"  hash {st['hash_ms']:9.2f} ms  root {st['root_hex'][:16]}...")
            if result.get("trace_dump_bytes"):
                print(f"trace dump       {result['trace_dump_bytes']} bytes")
    if args.trace_root and tracer is not None:
        for name, st in result["trace"]["variants"].items():
            print(f"root {name} {st['root_hex']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VIR reference implementation (format v0.1)")
    ap.add_argument("--selftest", action="store_true", help="run the conformance self test")
    ap.add_argument("--v2-model", default=None, help="path to a v2 .poisd for the hash check")
    ap.add_argument("--hash", metavar="FILE", help="print TaggedHash('POI/model', u8(opset)||file)")
    ap.add_argument("--opset", type=int, default=OPSET, help="opset byte for --hash (default 1)")
    ap.add_argument("--v2-hash", metavar="FILE", help="ModelHash of a v2 .poisd file")
    ap.add_argument("--expect", metavar="HEX", help="fail unless the hash equals HEX")
    ap.add_argument("--build-sdxs", action="store_true", help="build the SDXS graph from configs")
    ap.add_argument("--build-sdturbo", action="store_true",
                    help="build the sd-turbo graph (v2 topology) from configs")
    ap.add_argument("--model-dir", default=None, help="model directory for --build-*")
    ap.add_argument("--dump", action="store_true", help="dump the node list")
    ap.add_argument("--check-names", action="store_true",
                    help="compare graph weight names with the diffusers module names")
    # --run: execute a model, optionally recording a Sampled-PoI trace
    ap.add_argument("--run", metavar="FILE", help="run a .vir model on one seed/prompt")
    ap.add_argument("--seed", metavar="HEX", help="32-byte hex seed for --run")
    ap.add_argument("--prompt", metavar="TEXT", help="prompt text for --run (uses the model tokenizer)")
    ap.add_argument("--tokens", metavar="CSV", help="token ids for --run, comma separated")
    ap.add_argument("--trace", metavar="FILE", help="write a prototype trace dump for --run")
    ap.add_argument("--trace-root", action="store_true", help="print the trace root(s) of --run")
    ap.add_argument("--trace-nodes", action="store_true",
                    help="also trace every intermediate node output (large)")
    ap.add_argument("--trace-last", action="store_true",
                    help="also trace the last node's rows (the eps tensor)")
    ap.add_argument("--trace-sample", type=int, default=0, metavar="N",
                    help="also trace every Nth final payload row")
    ap.add_argument("--json", action="store_true", help="print --run results as JSON")
    args = ap.parse_args(argv)

    if args.check_names:
        root = _repo_root()
        mdir = args.model_dir or os.path.join(root, "models", "sdxs-512-dreamshaper")
        try:
            missing, extra = check_module_names(mdir)
        except SkipTest as exc:
            print(f"skip: {exc}")
            return 0
        print(f"missing (in checkpoint, not in graph): {len(missing)}")
        for n in missing[:20]:
            print("   ", n)
        print(f"extra (in graph, not in checkpoint): {len(extra)}")
        for n in extra[:20]:
            print("   ", n)
        return 0 if not missing and not extra else 1

    if args.selftest:
        return selftest(args.v2_model)
    if args.v2_hash:
        got = _hash_file(args.v2_hash, 2)
        print(got)
        return 0 if (args.expect is None or args.expect == got) else 1
    if args.hash:
        opset = args.opset
        with open(args.hash, "rb") as f:
            head = f.read(5)
        if head[:4] == MAGIC:
            opset = head[4]
        got = _hash_file(args.hash, opset)
        print(got)
        return 0 if (args.expect is None or args.expect == got) else 1
    if args.build_sdxs or args.build_sdturbo:
        root = _repo_root()
        sub = "sdxs-512-dreamshaper" if args.build_sdxs else "sd-turbo"
        mdir = args.model_dir or os.path.join(root, "models", sub)
        ucfg, tcfg, scfg = _load_configs(mdir)
        model = build_model(ucfg, tcfg, scfg)
        model.graph.validate()
        print(f"model      {mdir}")
        _print_summary(model)
        if args.dump:
            for i, node in enumerate(model.graph.nodes):
                ins = ",".join(f"{t}:{model.graph.tensors[t].shape}" for t in node.inputs)
                out = ",".join(f"{model.graph.tensors[t].shape}" for t in node.outputs)
                print(f"{i:4d} {node.op.name:10s} [{ins}] -> {out}")
        return 0
    if args.run:
        return _run_cli(args)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())




