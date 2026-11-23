#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.

"""GPU VIR evaluator (PyTorch on CUDA), bit-exact against the node's interpreter.

All 12 operator families are implemented with integer torch ops; only the
multiplications that the CPU does as an int64 matmul use a float64 CUDA matmul
with an exact-range guard (every operand is bounded so that the products and
partial sums stay below 2^53, where float64 accumulation equals int64). The exact
integer helpers (rshr, sat32, row_exp, quant16, the lookup tables, the
interpolated exponential, normalisation and isqrt) mirror
src/poi/vir/interp.cpp through vir.py next to this file.

This is a library: contributors/poi-miner/vir_mine.py drives it. To check the
engine against the CPU reference on a small model, see vir_mine.py --selftest.

Requires PyTorch with CUDA. The leaf digests are produced by noderoot.py, which
needs CuPy (it runs the SHA-256 chunk kernel on the GPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Model tensors are read-only views of the file bytes; the engine only ever reads
# them, so torch's "not writable" warning is noise here.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

VIR_PY = Path(__file__).resolve().parent / "vir.py"
DEV = "cuda"
I32_MIN, I32_MAX = -2**31, 2**31 - 1
BITS = 16


def load_vir():
    sys.path.insert(0, str(VIR_PY.parent))
    import vir as _vir
    return _vir


vir = load_vir()
OP, ATTR = vir.Op, vir.Attr


def T(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=DEV)
    c = G64.get(id(x))
    if c is not None:
        return c
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(DEV)


_LAST_CUDA = [None]   # node-root hook: the CUDA tensor of the op output that N() just copied


def N(t):
    if isinstance(t, torch.Tensor) and t.is_cuda:
        _LAST_CUDA[0] = t
    return t.detach().cpu().numpy()


G64 = {}          # id(np.ndarray) -> cached int64 CUDA tensor (small constants)
W8 = {}           # id(np.ndarray) -> (weight int8 [N,K] CUDA, column-sum int64 [N])
E8 = {}           # id(np.ndarray) -> embed weight int8 [V,D] CUDA
EM = {}           # id(np.ndarray) -> embed per-row int64 tables {m,s,pos}


def prep_constants(model):
    """Upload small constant tensors once; keep big weights as int8 for int8 MM."""
    for node in model.graph.nodes:
        for k, tid in enumerate(node.inputs):
            t = model.graph.tensors[tid]
            if t.data is None:
                continue
            arr = t.data
            key = id(arr)
            if node.op in (OP.GEMM, OP.CONV2D) and k == 1:
                w = np.ascontiguousarray(arr, dtype=np.int8)
                w = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
                w_t = np.ascontiguousarray(w.T)           # [K,N] int8
                wt = torch.from_numpy(w_t).to(DEV)
                W8[key] = (wt, torch.from_numpy(
                    np.ascontiguousarray(w.sum(axis=1, dtype=np.int64))).to(DEV))
            elif node.op is OP.EMBED and k == 1:
                w = np.ascontiguousarray(arr, dtype=np.int8)
                E8[key] = torch.from_numpy(w).to(DEV)
            elif getattr(arr, "nbytes", 0) <= (8 << 20):
                if key not in G64:
                    G64[key] = torch.from_numpy(
                        np.ascontiguousarray(arr, dtype=np.int64)).to(DEV)


def int_mm_pad(a, b):
    """torch._int_mm with zero-padding of M/K/N to multiples of 8."""
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise RuntimeError(f"int_mm shape {tuple(a.shape)} x {tuple(b.shape)}")
    m8, k8, n8 = (-M) % 8, (-K) % 8, (-N) % 8
    if m8 or k8:
        a2 = torch.zeros((M + m8, K + k8), dtype=torch.int8, device=a.device)
        a2[:M, :K] = a
    else:
        a2 = a
    if k8 or n8:
        b2 = torch.zeros((K + k8, N + n8), dtype=torch.int8, device=b.device)
        b2[:K, :N] = b
    else:
        b2 = b
    return torch._int_mm(a2, b2)[:M, :N]


def mm16_w8(q, wcache):
    """Exact int64 A_int16 @ W_int8 via two int8 tensor-core matmuls.

    q = 256*hi + lo_s + 128 with hi, lo_s in int8 range; W is int8.
    acc = 256*(hi@W) + (lo_s@W) + 128*colsum(W).
    """
    w8, wsum = wcache          # w8 already [K,N] int8
    hi = (q >> 8).to(torch.int8)
    lo = ((q - (hi.to(torch.int64) << 8)) - 128).to(torch.int8)
    acc = (int_mm_pad(hi, w8).to(torch.int64) * 256
           + int_mm_pad(lo, w8).to(torch.int64)
           + 128 * wsum[None, :])
    return acc


def mm16_pair(a, b):
    """Exact int64 A_int16 @ B_int16^T via four int8 tensor-core matmuls."""
    ah = (a >> 8).to(torch.int8)
    al = ((a - (ah.to(torch.int64) << 8)) - 128).to(torch.int8)
    bh = (b >> 8).to(torch.int8)
    bl = ((b - (bh.to(torch.int64) << 8)) - 128).to(torch.int8)
    bht = bh.t().contiguous()
    blt = bl.t().contiguous()
    K = a.shape[1]
    acc = (torch._int_mm(ah, bht).to(torch.int64) * 65536
           + torch._int_mm(ah, blt).to(torch.int64) * 256
           + torch._int_mm(al, bht).to(torch.int64) * 256
           + torch._int_mm(al, blt).to(torch.int64))
    rs_ah = ah.to(torch.int64).sum(dim=1, keepdim=True)
    rs_al = al.to(torch.int64).sum(dim=1, keepdim=True)
    rs_bh = bh.to(torch.int64).sum(dim=1, keepdim=True).T
    rs_bl = bl.to(torch.int64).sum(dim=1, keepdim=True).T
    acc = (acc + 32768 * (rs_ah + rs_bh) + 128 * (rs_al + rs_bl)
           + 16384 * K)
    return acc


def mm64(a, b):
    """Exact int64 matmul via float64 CUDA (caller guarantees <2^53)."""
    A = T(a).to(torch.float64)
    B = T(b).to(torch.float64)
    C = A @ B
    if not bool(torch.isfinite(C).all()):
        raise RuntimeError("non-finite matmul")
    if C.numel() and float(C.abs().max()) >= 2**53:
        raise RuntimeError("matmul exceeds exact float64 integer range")
    return C.to(torch.int64)  # exact: all values are integers representable in float64


def mm64_any(a, b):
    """Batched/multi-dim exact integer matmul via float64 matmul."""
    A = a.to(torch.float64)
    B = b.to(torch.float64)
    C = torch.matmul(A, B)
    if C.numel() and float(C.abs().max()) >= 2**53:
        raise RuntimeError("batched matmul exceeds exact integer range")
    return C.to(torch.int64)


def sat32_t(v):
    return v.clamp(I32_MIN, I32_MAX)


def rshr_t(v, s):
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(int(s), dtype=torch.int64, device=v.device)
    else:
        s = s.to(device=v.device, dtype=torch.int64)
    if s.ndim == 0:
        sv = int(s.item())
        if sv >= 0:
            p = min(sv, 62)
            return sat32_t((v + torch.tensor((1 << p) >> 1, dtype=torch.int64,
                                            device=v.device)) >> p)
        n = min(-sv, 31)
        lim = torch.tensor(1 << (62 - n), dtype=torch.int64, device=v.device)
        return sat32_t(v.clamp(-lim, lim) << n)
    pos = s.clamp(0, 62)
    half = torch.where(pos > 0,
                       torch.tensor(1, dtype=torch.int64, device=v.device) << (pos - 1).clamp(min=0),
                       torch.tensor(0, dtype=torch.int64, device=v.device))
    r = (v + half) >> pos
    neg = (-s).clamp(0, 31)
    lim = torch.tensor(1, dtype=torch.int64, device=v.device) << (62 - neg)
    r = torch.where(s < 0, (r.clamp(-lim, lim)) << neg, r)
    return sat32_t(r)


def floor_div_t(a, b):
    return torch.floor_divide(a, b)


def row_exp_t(m):
    e = torch.zeros_like(m)
    for k in range(15, 32):
        e = e + (m >= (1 << k)).to(torch.int64)
    return e


def row_max_abs_t(x):
    return x.abs().max(dim=-1, keepdim=True).values


def quant16_t(v, e):
    if not isinstance(e, torch.Tensor):
        e = torch.tensor(int(e), dtype=torch.int64, device=v.device)
    return rshr_t(v, e).clamp(-32767, 32767)


def isqrt_t(v):
    if bool((v < 0).any()):
        raise RuntimeError("isqrt negative")
    f = torch.sqrt(v.to(torch.float64))
    r = torch.floor(f).to(torch.int64)
    r = torch.where(r * r > v, r - 1, r)
    r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def lut_value_t(x, table, lo, step_bits, edge_lo, edge_hi):
    tbl = T(table)
    n = tbl.numel()
    span = (n - 1) << step_bits
    mask = (1 << step_bits) - 1
    z = x - lo
    zc = z.clamp(0, span - 1)
    i = zc >> step_bits
    fr = zc & mask
    t0 = tbl[i]
    t1 = tbl[i + 1]
    v = t0 + (((t1 - t0) * fr) >> step_bits)
    v = torch.where(z < 0, torch.tensor(edge_lo, dtype=torch.int64, device=x.device), v)
    v = torch.where(z >= span, torch.tensor(edge_hi, dtype=torch.int64, device=x.device), v)
    return v


# ------------------------------------------------------------------ GPU ops
def op_gemm_gpu(node, ins):
    x, w, m, s, b = ins
    if int(node.attr(ATTR.WGROUP) or 0):
        return N(vir.op_gemm(node, ins))
    X = T(x)
    e = row_exp_t(row_max_abs_t(X))
    q = quant16_t(X, e)
    wc = W8.get(id(w))
    acc = mm16_w8(q, wc) if wc is not None else mm64(q, T(w).T)
    y = rshr_t(acc * T(m)[None, :], T(s)[None, :] - e)
    return N(sat32_t(y + T(b)[None, :]))


def op_conv2d_gpu(node, ins):
    x, w, m, s, b = ins
    kh, kw = int(node.attr(ATTR.KH)), int(node.attr(ATTR.KW))
    stride, pad = int(node.attr(ATTR.STRIDE)), int(node.attr(ATTR.PAD))
    if int(node.attr(ATTR.WGROUP) or 0):
        return N(vir.op_conv2d(node, ins))
    X = T(x)
    c, h, wd = X.shape
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    xf = X.to(torch.float64)[None]
    cols = F.unfold(xf, kernel_size=(kh, kw), padding=pad, stride=stride)[0].to(torch.int64).T  # [L, C*kh*kw]
    pm = X.abs().max(dim=0).values
    pmp = F.pad(pm.to(torch.float64)[None, None], (pad, pad, pad, pad))
    win = F.max_pool2d(pmp, kernel_size=(kh, kw), stride=stride)[0, 0].to(torch.int64)
    e = row_exp_t(win)
    e_flat = e.reshape(-1)
    q = quant16_t(cols, e_flat[:, None])
    wc = W8.get(id(w))
    acc = mm16_w8(q, wc) if wc is not None else mm64(q, T(w).T)
    y = rshr_t(acc * T(m)[None, :], T(s)[None, :] - e_flat[:, None])
    y = sat32_t(y + T(b)[None, :])
    return N(y.T.reshape(w.shape[0], ho, wo))


def op_attention_gpu(node, ins):
    q, k, v, exp_lut = ins
    heads = int(node.attr(ATTR.NUM_HEADS))
    dh = int(node.attr(ATTR.HEAD_DIM))
    causal = bool(int(node.attr(ATTR.CAUSAL)))
    qs = int(node.attr(ATTR.Q_SCALE))
    Q, K, V = T(q), T(k), T(v)
    tq, c = Q.shape
    tk = K.shape[0]
    # [H,T,D]
    Qh = Q.reshape(tq, heads, dh).permute(1, 0, 2).contiguous()
    Kh = K.reshape(tk, heads, dh).permute(1, 0, 2).contiguous()
    Vh = V.reshape(tk, heads, dh).permute(1, 0, 2).contiguous()
    eq = row_exp_t(row_max_abs_t(Qh))          # [H,T,1]
    ek = row_exp_t(row_max_abs_t(Kh))          # [H,Tk,1]
    s = mm64_any(quant16_t(Qh, eq), quant16_t(Kh, ek).transpose(1, 2)) * qs
    s = rshr_t(s, (32 - eq) - ek.transpose(1, 2))     # [H,T,Tk]
    if causal:
        tri = torch.triu(torch.ones((tq, tk), dtype=torch.bool, device=Q.device), 1)
        s = torch.where(tri.unsqueeze(0), torch.tensor(I32_MIN, dtype=torch.int64,
                                                       device=Q.device), s)
    mx = s.max(dim=2, keepdim=True).values
    z = mx - s
    e = lut_value_t(z, exp_lut, 0, 10, 0, 0)
    r = floor_div_t(torch.tensor(1 << 62, dtype=torch.int64, device=Q.device),
                    e.sum(dim=2, keepdim=True))
    p = torch.minimum((e * r + torch.tensor(1 << 46, dtype=torch.int64,
                                            device=Q.device)) >> 47,
                      torch.tensor(32767, dtype=torch.int64, device=Q.device))
    ev = row_exp_t(Vh.abs().max(dim=1, keepdim=True).values)   # [H,1,D]
    o = mm64_any(p, quant16_t(Vh, ev))                     # [H,T,D]
    out = rshr_t(o, 15 - ev)                               # [H,T,D]
    out = out.permute(1, 0, 2).reshape(tq, c)
    return N(out)


def norm_core_t(V, den):
    n = V.shape[1]
    nn = torch.tensor(n, dtype=torch.int64, device=V.device)
    mean = floor_div_t(V.sum(dim=1, keepdim=True) + n // 2, nn)
    d = V - mean
    five = torch.tensor(5, dtype=torch.int64, device=V.device)
    zero = torch.tensor(0, dtype=torch.int64, device=V.device)
    e = torch.maximum(row_exp_t(d.abs().max(dim=1, keepdim=True).values) - five, zero)
    dd = rshr_t(d, e)
    var = floor_div_t((dd * dd).sum(dim=1, keepdim=True), nn)
    one = torch.tensor(1, dtype=torch.int64, device=V.device)
    eps = floor_div_t((one << (32 - 2 * e)) + (den // 2),
                      torch.tensor(den, dtype=torch.int64, device=V.device))
    sd = isqrt_t((var + eps) << torch.tensor(BITS, dtype=torch.int64, device=V.device))
    two = torch.tensor(2, dtype=torch.int64, device=V.device)
    return floor_div_t(((d << (24 - e)) * two) + sd, two * sd)


def op_groupnorm_gpu(node, ins):
    x, gamma, beta, eps = ins
    X = T(x)
    groups = int(node.attr(ATTR.GROUPS))
    den = int(vir._i64(eps).reshape(-1)[0])
    c, h, w = X.shape
    Y = norm_core_t(X.reshape(groups, (c // groups) * h * w), den).reshape(c, h * w)
    Y = sat32_t(rshr_t(Y * T(gamma)[:, None], BITS) + T(beta)[:, None])
    return N(Y.reshape(c, h, w))


def op_layernorm_gpu(node, ins):
    x, gamma, beta, eps = ins
    den = int(vir._i64(eps).reshape(-1)[0])
    Y = norm_core_t(T(x), den)
    Y = sat32_t(rshr_t(Y * T(gamma)[None, :], BITS) + T(beta)[None, :])
    return N(Y)


def op_lut_act_gpu(node, ins):
    x, table = ins
    X = T(x)
    v = lut_value_t(X, table, int(node.attr(ATTR.LUT_LO)),
                    int(node.attr(ATTR.LUT_STEP_BITS)),
                    int(node.attr(ATTR.LUT_EDGE_LO)), int(node.attr(ATTR.LUT_EDGE_HI)))
    return N(sat32_t(rshr_t(X * v, int(node.attr(ATTR.POST_SHIFT)))))


def op_upsample_gpu(node, ins):
    X = T(ins[0])
    f = int(node.attr(ATTR.FACTOR))
    if int(node.attr(ATTR.MODE)) != 0:
        raise vir.NotSupported("UPSAMPLE mode")
    return N(X.repeat_interleave(f, dim=1).repeat_interleave(f, dim=2))


def op_add_gpu(node, ins):
    return N(sat32_t(T(ins[0]) + T(ins[1])))


def op_concat_gpu(node, ins):
    return N(torch.cat([T(ins[0]), T(ins[1])], dim=int(node.attr(ATTR.AXIS))))


def op_reshape_gpu(node, ins):
    X = T(ins[0])
    perm = tuple(int(d) for d in node.attr(ATTR.PERM))
    if perm:
        X = X.permute(perm)
    return N(X.reshape(tuple(int(d) for d in node.attr(ATTR.SHAPE))))


def op_scale_gpu(node, ins):
    X = T(ins[0])
    return N(sat32_t(rshr_t(X * int(node.attr(ATTR.MULT)), int(node.attr(ATTR.SHIFT)))))


def op_mul_gpu(node, ins):
    return N(sat32_t(rshr_t(T(ins[0]) * T(ins[1]), int(node.attr(ATTR.SHIFT)))))


def op_embed_gpu(node, ins):
    ids, w, m, s, pos = ins
    ID = T(ids).reshape(-1)
    we = E8.get(id(w))
    if we is not None:
        W = we.gather(0, ID[:, None].expand(-1, we.shape[1])).to(torch.int64)
    else:
        W = T(w)[ID]
    shift = (T(s)[ID] - int(node.attr(ATTR.POST_SHIFT)))[:, None]
    y = rshr_t(W * T(m)[ID][:, None], shift)
    return N(sat32_t(y + T(pos)))


GPU_OPS = {OP.GEMM: op_gemm_gpu, OP.CONV2D: op_conv2d_gpu,
           OP.GROUPNORM: op_groupnorm_gpu, OP.LAYERNORM: op_layernorm_gpu,
           OP.LUT_ACT: op_lut_act_gpu, OP.ATTENTION: op_attention_gpu,
           OP.UPSAMPLE: op_upsample_gpu, OP.ADD: op_add_gpu,
           OP.CONCAT: op_concat_gpu, OP.RESHAPE: op_reshape_gpu,
           OP.SCALE: op_scale_gpu, OP.MUL: op_mul_gpu, OP.EMBED: op_embed_gpu}


class TorchVM:
    def __init__(self, model_path, profile: bool = False):
        t0 = time.time()
        self.model = vir.VirModel.parse(Path(model_path).read_bytes())
        self.load_seconds = round(time.time() - t0, 2)
        prep_constants(self.model)
        self.profile = defaultdict(lambda: [0.0, 0])
        self.orig = dict(vir.OP_FORWARD)
        vir.OP_FORWARD.clear()
        vir.OP_FORWARD.update(GPU_OPS)
        if profile:
            for op, fn in list(vir.OP_FORWARD.items()):
                name = op.name

                def wrap(node, ins, _fn=fn, _name=name):
                    t0 = time.time()
                    out = _fn(node, ins)
                    self.profile[_name][0] += time.time() - t0
                    self.profile[_name][1] += 1
                    return out
                vir.OP_FORWARD[op] = wrap

    def evaluate(self, seed: bytes, tokens, want_values: bool = False, node_roots: bool = False,
                 root_timing: bool = False):
        m = self.model
        n = m.noise
        noise = m.noise_latent(seed)
        x = vir.sat32(vir.rshr(noise * np.int64(n.cin_q30), 30))
        feed = {n.latent_tensor: x}
        ids_tensor = None
        for tid in m.graph.inputs:
            if tid != n.latent_tensor:
                ids_tensor = tid
        feed[ids_tensor] = m.context_ids(tokens)
        tracer = None
        if node_roots:
            from noderoot import NodeRoots
            nr = NodeRoots(len(m.graph.nodes), timing=root_timing)

            class _Tracer:  # vir.VirModel.evaluate calls tracer.node(index, node, out, is_last)
                def node(self, index, node, out, is_last):
                    t = _LAST_CUDA[0]
                    if t is None or t.numel() != np.asarray(out).size:   # CPU fallback op
                        t = torch.from_numpy(np.ascontiguousarray(out, dtype=np.int64)).to(DEV)
                    nr.add(index, t)
                    _LAST_CUDA[0] = None
            tracer = _Tracer()
        t0 = time.time()
        values = m.evaluate(feed, tracer)
        eval_seconds = round(time.time() - t0, 2)
        roots = R = None
        if node_roots:
            roots = nr.finish()
            R = vir.tagged_hash("POI/roots", b"".join(roots))
        eps = values[n.eps_tensor]
        if n.sampler_kind == 1:
            x0 = vir.sat32(vir.rshr(noise * np.int64(n.sigma_q16) - eps * np.int64(n.eps_q16),
                                    n.latent_shift))
        else:
            x0 = vir.sat32(vir.rshr((noise - eps) * np.int64(n.sigma_q16), n.latent_shift))
        q = np.clip(vir.rshr(x0, n.out_shift), -n.clamp_abs, n.clamp_abs)
        res = {"payload": q.astype("<i2").tobytes(), "eval_seconds": eval_seconds,
               "values": values if want_values else None}
        if node_roots:
            res.update({"node_roots": roots, "R": R, "hash_seconds": round(nr.seconds, 4),
                        "hashed_bytes": nr.bytes})
        return res

    def restore(self):
        vir.OP_FORWARD.clear()
        vir.OP_FORWARD.update(self.orig)


def main(argv=None):
    """Self-check: run one inference on a model and print the output digest.

    python vir_torch.py --model sd-turbo-int.vir [--bench N] [--prompt "text"]
    """
    ap = argparse.ArgumentParser(description="GPU VIR evaluator (bit-exact)")
    ap.add_argument("--model", type=Path, required=True, help=".vir model file")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="repeat N times")
    ap.add_argument("--prompt", default="", help="prompt text (needs a tokenizer in the model)")
    ap.add_argument("--seed", default="", help="32-byte seed as hex (default: 32 zero bytes)")
    a = ap.parse_args(argv)
    seed = bytes.fromhex(a.seed) if a.seed else bytes(32)
    if len(seed) != 32:
        raise SystemExit("--seed must be 32 bytes of hex")
    vm = TorchVM(a.model)
    tokens = vm.model.tokenizer.encode(a.prompt) if a.prompt else []
    runs = max(1, a.bench)
    times = []
    for _ in range(runs):
        t0 = time.time()
        res = vm.evaluate(seed, tokens)
        times.append(round(time.time() - t0, 3))
        print(f"run: {times[-1]} s  eval={res['eval_seconds']} s  "
              f"x0 sha256 {hashlib.sha256(res['payload']).hexdigest()}")
    if runs > 1:
        print(json.dumps({"runs": runs, "min": min(times), "max": max(times),
                          "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
