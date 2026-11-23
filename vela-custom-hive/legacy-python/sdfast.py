#!/usr/bin/env python3
"""Fast batched CUDA engine for the PoI v2 integer sd-turbo model.

The arithmetic remains integer-exact and consensus-compatible with sdint.py.
On CUDA systems with Triton, the linear and attention GEMMs use signed INT8
Tensor Core dot products with exact INT16 decomposition. If Triton is absent,
unsupported, or fails its runtime arithmetic check, the original exact PyTorch
implementation is used as a fallback.
"""
import argparse, hashlib, logging, math, sys, time
import numpy as np
import torch
import sdint
from sdint import I32_MIN, I32_MAX, LUT_RANGE

F = torch.nn.functional
_LOG = logging.getLogger("vela-miner.sdfast")

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional; the exact PyTorch path remains available.
    triton = None
    tl = None


# Triton kernels use int8 MMA inputs and int32 accumulators. The input values
# are decomposed as x = 256*hi + lo + 128, where hi and lo are signed int8.
# The correction terms below reconstruct the exact original signed-int16 dot.
if triton is not None:
    @triton.jit
    def _mm_w_i8_kernel(A, W, WS, Y,
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        AS0: tl.constexpr, AS1: tl.constexpr,
                        WS0: tl.constexpr, WS1: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc_hi = tl.zeros((BM, BN), dtype=tl.int32)
        acc_lo = tl.zeros((BM, BN), dtype=tl.int32)

        for k0 in range(tl.cdiv(K, BK)):
            kk = k0 * BK + rk
            am = (rm[:, None] < M) & (kk[None, :] < K)
            wm = (kk[:, None] < K) & (rn[None, :] < N)
            av = tl.load(A + rm[:, None] * AS0 + kk[None, :] * AS1,
                         mask=am, other=0).to(tl.int64)
            wv = tl.load(W + kk[:, None] * WS0 + rn[None, :] * WS1,
                         mask=wm, other=0).to(tl.int8)
            ahi = (av >> 8).to(tl.int8)
            alo = (av - ((av >> 8) * 256) - 128).to(tl.int8)
            # A masked zero is padding, not a real zero-valued element: force
            # both decomposed pieces to zero outside the actual K/M extent.
            ahi = tl.where(am, ahi, 0).to(tl.int8)
            alo = tl.where(am, alo, 0).to(tl.int8)
            acc_hi = tl.dot(ahi, wv, acc_hi)
            acc_lo = tl.dot(alo, wv, acc_lo)

        wsum = tl.load(WS + rn, mask=rn < N, other=0).to(tl.int64)
        result = (acc_hi.to(tl.int64) << 8) + acc_lo.to(tl.int64) + 128 * wsum[None, :]
        tl.store(Y + rm[:, None] * N + rn[None, :], result,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


    @triton.jit
    def _bmm16_i8_kernel(A, B, Y,
                         G: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         AS0: tl.constexpr, AS1: tl.constexpr, AS2: tl.constexpr,
                         BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)

        acc_hh = tl.zeros((BM, BN), dtype=tl.int32)
        acc_hl = tl.zeros((BM, BN), dtype=tl.int32)
        acc_lh = tl.zeros((BM, BN), dtype=tl.int32)
        acc_ll = tl.zeros((BM, BN), dtype=tl.int32)
        row_hi = tl.zeros((BM,), dtype=tl.int32)
        row_lo = tl.zeros((BM,), dtype=tl.int32)
        col_hi = tl.zeros((BN,), dtype=tl.int32)
        col_lo = tl.zeros((BN,), dtype=tl.int32)

        for k0 in range(tl.cdiv(K, BK)):
            kk = k0 * BK + rk
            am = (rm[:, None] < M) & (kk[None, :] < K)
            bm = (kk[:, None] < K) & (rn[None, :] < N)
            av = tl.load(A + pid_g * AS0 + rm[:, None] * AS1 + kk[None, :] * AS2,
                         mask=am, other=0).to(tl.int64)
            bv = tl.load(B + pid_g * BS0 + kk[:, None] * BS1 + rn[None, :] * BS2,
                         mask=bm, other=0).to(tl.int64)

            ahi = (av >> 8).to(tl.int8)
            alo = (av - ((av >> 8) * 256) - 128).to(tl.int8)
            bhi = (bv >> 8).to(tl.int8)
            blo = (bv - ((bv >> 8) * 256) - 128).to(tl.int8)
            ahi = tl.where(am, ahi, 0).to(tl.int8)
            alo = tl.where(am, alo, 0).to(tl.int8)
            bhi = tl.where(bm, bhi, 0).to(tl.int8)
            blo = tl.where(bm, blo, 0).to(tl.int8)

            acc_hh = tl.dot(ahi, bhi, acc_hh)
            acc_hl = tl.dot(ahi, blo, acc_hl)
            acc_lh = tl.dot(alo, bhi, acc_lh)
            acc_ll = tl.dot(alo, blo, acc_ll)
            row_hi += tl.sum(ahi.to(tl.int32), axis=1)
            row_lo += tl.sum(alo.to(tl.int32), axis=1)
            col_hi += tl.sum(bhi.to(tl.int32), axis=0)
            col_lo += tl.sum(blo.to(tl.int32), axis=0)

        # Expand (256*ahi + alo + 128) @ (256*bhi + blo + 128).
        # INT64 is used only for recombination; the four INT8 dot accumulators
        # are safely within INT32 for K <= 8192.
        result = ((acc_hh.to(tl.int64) << 16)
                  + (acc_hl.to(tl.int64) << 8)
                  + (acc_lh.to(tl.int64) << 8)
                  + acc_ll.to(tl.int64))
        result += (row_hi.to(tl.int64) * 32768 + row_lo.to(tl.int64) * 128)[:, None]
        result += (col_hi.to(tl.int64) * 32768 + col_lo.to(tl.int64) * 128)[None, :]
        result += 16384 * K
        tl.store(Y + pid_g * M * N + rm[:, None] * N + rn[None, :], result,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


_TRITON_MMW_VERIFIED = set()
_TRITON_BMM_VERIFIED = set()
_TRITON_MMW_DISABLED = set()
_TRITON_BMM_DISABLED = set()


def sat32(v):
    return v.clamp(I32_MIN, I32_MAX)


def rshr_pos(v, s):
    """sdint.rshr for non-negative shifts (tensor or int), no host sync."""
    if isinstance(s, int):
        return sat32((v + ((1 << s) >> 1)) >> s) if s > 0 else sat32(v)
    return sat32((v + ((1 << s) >> 1)) >> s)


def rshr_any(v, s):
    """sdint.rshr for arbitrary-sign tensor shifts, branch-free."""
    pos = s.clamp(0, 62)
    r = (v + ((1 << pos) >> 1)) >> pos
    neg = (-s).clamp(0, 31)
    lim = torch.ones_like(neg) << (62 - neg)
    r = torch.where(s < 0, torch.minimum(torch.maximum(r, -lim), lim) << neg, r)
    return sat32(r)


def row_exp(x, dim=-1):
    """max(0, bitlen(max|x|) - 15) over `dim`, keepdim."""
    m = x.abs().amax(dim=dim, keepdim=True)
    e = torch.zeros_like(m)
    for k in range(15, 32):
        e += (m >= (1 << k)).long()
    return e


def quant16(x, e):
    return rshr_pos(x, e).clamp(-32767, 32767)


def floor_div(a, b):
    return torch.div(a, b, rounding_mode="floor")


def isqrt(v):
    s = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        s = torch.where(s * s > v, s - 1, s)
        s = torch.where((s + 1) * (s + 1) <= v, s + 1, s)
    return s


# ----------------------------------------------------------------------------- exact GEMMs

def _mm8(a8, b8):
    """int8 [M,K] @ int8 [K,N] -> int64 exact (PyTorch CUDA INT8 MMA)."""
    M, K = a8.shape
    N = b8.shape[1]
    Mp, Kp, Np = max(32, (M + 7) // 8 * 8), (K + 7) // 8 * 8, (N + 7) // 8 * 8
    if (Mp, Kp) != (M, K):
        a8 = F.pad(a8, (0, Kp - K, 0, Mp - M))
    if (Kp, Np) != (K, N):
        b8 = F.pad(b8, (0, Np - N, 0, Kp - K))
    return torch._int_mm(a8.contiguous(), b8.contiguous())[:M, :N].long()


def _split(a):
    return (a >> 8).to(torch.int8), ((a & 255) - 128).to(torch.int8)


def _mm_w_torch(a, w8, wsum):
    """Original exact fallback for int16-range @ int8 linear products."""
    ah, al = _split(a)
    return (_mm8(ah, w8) << 8) + _mm8(al, w8) + 128 * wsum[None, :]


def _mm_w_triton(a, w8, wsum):
    M, K = a.shape
    N = w8.shape[1]
    out = torch.empty((M, N), dtype=torch.int64, device=a.device)
    BM, BN, BK = 16, 32, 32
    _mm_w_i8_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, w8, wsum, out, M, N, K,
        a.stride(0), a.stride(1), w8.stride(0), w8.stride(1),
        BM, BN, BK, num_warps=4, num_stages=2)
    return out


def mm_w(a, w8, wsum):
    """Exact int16-range [M,K] @ int8 [K,N]; use fused INT8 MMA when safe."""
    if (triton is None or not a.is_cuda or a.device != w8.device
            or a.dtype != torch.int64 or w8.dtype != torch.int8
            or a.shape[1] > 65536):
        return _mm_w_torch(a, w8, wsum)

    device_key = str(a.device)
    if device_key in _TRITON_MMW_DISABLED:
        return _mm_w_torch(a, w8, wsum)
    try:
        out = _mm_w_triton(a, w8, wsum)
        if device_key not in _TRITON_MMW_VERIFIED:
            rows, cols = min(8, a.shape[0]), min(8, w8.shape[1])
            reference = _mm_w_torch(a[:rows, :], w8[:, :cols], wsum[:cols])
            if not torch.equal(out[:rows, :cols], reference):
                _TRITON_MMW_DISABLED.add(device_key)
                _LOG.error("Triton INT8 linear self-check failed on %s; using exact PyTorch fallback", device_key)
                return _mm_w_torch(a, w8, wsum)
            _TRITON_MMW_VERIFIED.add(device_key)
            _LOG.info("Triton INT8 linear self-check=PASS device=%s", device_key)
        return out
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise
        _TRITON_MMW_DISABLED.add(device_key)
        _LOG.warning("Triton INT8 linear unavailable on %s; using exact PyTorch fallback: %s", device_key, exc)
        return _mm_w_torch(a, w8, wsum)


def _bmm16_torch(a, b):
    """Exact FP64 reference/fallback: integer products remain exactly representable."""
    return torch.bmm(a.double(), b.double()).long()


def _bmm16_triton(a, b):
    G, M, K = a.shape
    N = b.shape[2]
    out = torch.empty((G, M, N), dtype=torch.int64, device=a.device)
    BM, BN, BK = 16, 32, 32
    _bmm16_i8_kernel[(G, triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, out, G, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        BM, BN, BK, num_warps=4, num_stages=2)
    return out


def bmm16(a, b):
    """Exact batched int16 @ int16; prefer exact signed-INT8 Tensor Core MMA.

    Inputs are |value| <= 32767. Decomposition is algebraically exact. INT32
    partial accumulators are safe for K <= 8192; larger shapes use the original
    FP64 exact implementation. Each CUDA device checks the first small output
    tile against the FP64 reference before enabling the Triton path.
    """
    assert a.shape[-1] <= (1 << 22)
    if (triton is None or not a.is_cuda or a.device != b.device
            or a.dtype != torch.int64 or b.dtype != torch.int64
            or a.shape[-1] > 8192):
        return _bmm16_torch(a, b)

    device_key = str(a.device)
    if device_key in _TRITON_BMM_DISABLED:
        return _bmm16_torch(a, b)
    try:
        out = _bmm16_triton(a, b)
        if device_key not in _TRITON_BMM_VERIFIED:
            rows, cols = min(4, a.shape[1]), min(4, b.shape[2])
            reference = _bmm16_torch(a[:1, :rows, :], b[:1, :, :cols])
            if not torch.equal(out[:1, :rows, :cols], reference):
                _TRITON_BMM_DISABLED.add(device_key)
                _LOG.error("Triton INT8 batched GEMM self-check failed on %s; using exact FP64 fallback", device_key)
                return _bmm16_torch(a, b)
            _TRITON_BMM_VERIFIED.add(device_key)
            _LOG.info("Triton INT8 batched GEMM self-check=PASS device=%s", device_key)
        return out
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise
        _TRITON_BMM_DISABLED.add(device_key)
        _LOG.warning("Triton INT8 batched GEMM unavailable on %s; using exact FP64 fallback: %s", device_key, exc)
        return _bmm16_torch(a, b)


# ----------------------------------------------------------------------------- engine

class Fast:
    def __init__(self, net):
        self.net = net
        self.t = net.t
        self.dev = net.dev
        self.wT, self.ws = {}, {}
        if triton is None:
            _LOG.warning("Triton runtime unavailable; exact PyTorch GEMM fallback active device=%s", self.dev)
        else:
            _LOG.info("Triton runtime detected version=%s device=%s", getattr(triton, "__version__", "unknown"), self.dev)

    def _w(self, name):
        if name not in self.wT:
            w = self.t[name + ".w"]
            self.wT[name] = w.t().contiguous()
            self.ws[name] = self.wT[name].long().sum(0)
        return self.wT[name], self.ws[name]

    # x: [..., K] -> [..., N]; rows are independent so batching is transparent
    def lin(self, x, name):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        w, ws = self._w(name)
        e = row_exp(x2)
        acc = mm_w(quant16(x2, e), w, ws)
        y = rshr_any(acc * self.t[name + ".m"][None, :], self.t[name + ".s"][None, :] - e)
        y = sat32(y + self.t[name + ".b"][None, :])
        return y.reshape(*shp[:-1], y.shape[-1])

    def conv(self, x, name, stride=1):
        """x [B,C,H,W]."""
        w = self.t[name + ".w"]
        B, C, H, W = x.shape
        if w.shape[1] == C:
            return self.lin(x.permute(0, 2, 3, 1), name).permute(0, 3, 1, 2).contiguous()
        xp = F.pad(x, (1, 1, 1, 1))
        Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
        cols = torch.stack([xp[:, :, ky:ky + stride * (Ho - 1) + 1:stride, kx:kx + stride * (Wo - 1) + 1:stride]
                            for ky in range(3) for kx in range(3)], 2)            # B,C,9,Ho,Wo
        cols = cols.reshape(B, C * 9, Ho * Wo).transpose(1, 2)                    # B,HW,C*9
        y = self.lin(cols, name)                                                  # B,HW,N
        return y.transpose(1, 2).reshape(B, -1, Ho, Wo)

    def _normcore(self, v, den):
        """v [..., n] -> normalised Q16 rows (sdint.Net.norm without affine)."""
        n = v.shape[-1]
        mean = floor_div(v.sum(-1, keepdim=True) + n // 2, n)
        d = v - mean
        e = (row_exp(d) - 5).clamp(min=0)
        dd = rshr_pos(d, e)
        var = floor_div((dd * dd).sum(-1, keepdim=True), n)
        eps = floor_div((torch.ones_like(e) << (32 - 2 * e)) + den // 2, den)
        sd = isqrt((var + eps) << 16)
        return floor_div((d << (24 - e)) * 2 + sd, 2 * sd)

    def gnorm(self, x, name, groups=32):
        """GroupNorm on x [B,C,H,W]."""
        g, b, den = self.t[name + ".g"], self.t[name + ".beta"], self.net.raw[name + ".eps"][0]
        B, C = x.shape[:2]
        y = self._normcore(x.reshape(B, groups, -1), int(den)).reshape(B, C, -1)
        y = sat32(rshr_pos(y * g[None, :, None], 16) + b[None, :, None])
        return y.reshape(x.shape)

    def lnorm(self, x, name):
        """LayerNorm over the last dim."""
        g, b, den = self.t[name + ".g"], self.t[name + ".beta"], self.net.raw[name + ".eps"][0]
        y = self._normcore(x, int(den))
        return sat32(rshr_pos(y * g, 16) + b)

    def lut(self, x, table):
        L = self.t[table]
        z = x + (LUT_RANGE << 16)
        zc = z.clamp(0, (2 * LUT_RANGE << 16) - 1)
        i, f = zc >> 10, zc & 1023
        v = L[i] + (((L[i + 1] - L[i]) * f) >> 10)
        v = torch.where(z < 0, torch.zeros_like(v), v)
        return torch.where(z >= (2 * LUT_RANGE << 16), torch.full_like(v, 65536), v)

    def silu(self, x):
        return sat32(rshr_pos(x * self.lut(x, "lut.sigmoid"), 16))

    def gelu(self, x):
        return sat32(rshr_pos(x * self.lut(x, "lut.phi"), 16))

    def attention(self, q, k, v, heads, causal=False, chunk=None):
        """q [B,Tq,C]; k,v [B,Tk,C] (or [1,Tk,C] shared) -> [B,Tq,C]. Head dim 64."""
        B, Tq, C = q.shape
        Tk = k.shape[1]
        dh = C // heads
        if chunk is None:   # bound the G x chunk x Tk score tensors (~5M elements)
            chunk = max(64, min(Tq, (5 << 20) // max(1, B * heads * Tk)))
        L = self.t["lut.exp"]
        if not hasattr(self, "_L32"):
            self._L32 = L.int()
            self._D32 = (L[1:] - L[:-1]).int()        # |L[i+1]-L[i]| < 2^25, *f (<2^10) fits int32? no -> int64 below
        L32 = self._L32
        qh = q.reshape(B, Tq, heads, dh).permute(0, 2, 1, 3)             # B,h,Tq,d
        kh = k.reshape(k.shape[0], Tk, heads, dh).permute(0, 2, 1, 3).expand(B, heads, Tk, dh)
        vh = v.reshape(v.shape[0], Tk, heads, dh).permute(0, 2, 1, 3).expand(B, heads, Tk, dh)
        G = B * heads
        qh, kh, vh = qh.reshape(G, Tq, dh), kh.reshape(G, Tk, dh), vh.reshape(G, Tk, dh)
        ek = row_exp(kh)                                                  # G,Tk,1
        kq = quant16(kh, ek).transpose(1, 2).contiguous()                 # G,d,Tk
        ev = row_exp(vh, dim=1)                                           # G,1,d (per channel)
        vq = quant16(vh, ev)                                              # G,Tk,d
        out = torch.empty(G, Tq, dh, dtype=torch.long, device=q.device)
        for c0 in range(0, Tq, chunk):
            c1 = min(Tq, c0 + chunk)
            qc = qh[:, c0:c1]
            eq = row_exp(qc)
            S = bmm16(quant16(qc, eq), kq)                                # G,c,Tk int64
            # s = rshr(S, 19-eq-ek) saturated to int32 (identical to sdint)
            s = rshr_any(S, 19 - eq - ek.transpose(1, 2)).int()           # values in int32 range
            del S
            if causal:
                mask = torch.ones(c1 - c0, Tk, dtype=torch.bool, device=q.device).triu(1 + c0)
                s = s.masked_fill(mask, I32_MIN)
            mx = s.amax(-1, keepdim=True)
            # z = mx - s can exceed int32 (mx - INT32_MIN); clamp in int64 only where needed
            z = mx.long() - s.long()
            del s
            big = z >= (16 << 16)
            zc = z.clamp(0, (16 << 16) - 1).int()
            del z
            i, f = (zc >> 10).long(), (zc & 1023)
            Li = L32[i]
            # (L[i+1]-L[i]) * f: |diff| < 2^30, f < 2^10 -> product < 2^40: needs int64
            E = Li.long() + (((L32[i + 1] - Li).long() * f) >> 10)
            del i, f, Li, zc
            E = E.masked_fill(big, 0)
            del big
            r = floor_div(torch.full_like(mx, 1 << 62, dtype=torch.long), E.sum(-1, keepdim=True))
            P = ((E * r + (1 << 46)) >> 47).clamp(max=32767)
            del E
            O = bmm16(P, vq)                                              # G,c,d
            out[:, c0:c1] = rshr_pos(O, 15 - ev)
        return out.reshape(B, heads, Tq, dh).permute(0, 2, 1, 3).reshape(B, Tq, C)

    # ------------------------------------------------------------------------- blocks

    def resnet(self, x, p):
        h = self.conv(self.silu(self.gnorm(x, p + ".norm1")), p + ".conv1")
        h = self.conv(self.silu(self.gnorm(h, p + ".norm2")), p + ".conv2")
        if p + ".conv_shortcut.w" in self.t:
            x = self.conv(x, p + ".conv_shortcut")
        return sat32(x + h)

    def transformer(self, x, p, kv, heads):
        B, C, H, W = x.shape
        h = self.gnorm(x, p + ".norm").reshape(B, C, H * W).transpose(1, 2)     # B,HW,C
        h = self.lin(h, p + ".proj_in")
        b = p + ".transformer_blocks.0"
        n = self.lnorm(h, b + ".norm1")
        h = sat32(h + self.lin(self.attention(self.lin(n, b + ".attn1.to_q"), self.lin(n, b + ".attn1.to_k"),
                                              self.lin(n, b + ".attn1.to_v"), heads), b + ".attn1.to_out.0"))
        n = self.lnorm(h, b + ".norm2")
        k, v = kv[b]
        h = sat32(h + self.lin(self.attention(self.lin(n, b + ".attn2.to_q"), k, v, heads), b + ".attn2.to_out.0"))
        n = self.lnorm(h, b + ".norm3")
        a = self.lin(n, b + ".ff.net.0.proj.a")
        g = self.lin(n, b + ".ff.net.0.proj.g")
        h = sat32(h + self.lin(sat32(rshr_pos(a * self.gelu(g), 16)), b + ".ff.net.2"))
        h = self.lin(h, p + ".proj_out")
        return sat32(x + h.transpose(1, 2).reshape(B, C, H, W))

    def cross_kv(self, ctx):
        """Cross-attention K/V of a [77,1024] context, shaped [1,77,C]."""
        kv = {}
        for name in self.t:
            if name.startswith("unet.") and name.endswith(".attn2.to_k.w"):
                b = name[:-len(".attn2.to_k.w")]
                kv[b] = (self.lin(ctx, b + ".attn2.to_k")[None], self.lin(ctx, b + ".attn2.to_v")[None])
        return kv

    def unet(self, x, kv):
        heads = {0: 5, 1: 10, 2: 20, 3: 20}
        h = self.conv(x, "unet.conv_in")
        skips = [h]
        for i in range(4):
            for j in range(2):
                h = self.resnet(h, f"unet.down_blocks.{i}.resnets.{j}")
                if i < 3:
                    h = self.transformer(h, f"unet.down_blocks.{i}.attentions.{j}", kv, heads[i])
                skips.append(h)
            if i < 3:
                h = self.conv(h, f"unet.down_blocks.{i}.downsamplers.0.conv", stride=2)
                skips.append(h)
        h = self.resnet(h, "unet.mid_block.resnets.0")
        h = self.transformer(h, "unet.mid_block.attentions.0", kv, 20)
        h = self.resnet(h, "unet.mid_block.resnets.1")
        for i in range(4):
            for j in range(3):
                h = torch.cat([h, skips.pop()], 1)
                h = self.resnet(h, f"unet.up_blocks.{i}.resnets.{j}")
                if i > 0:
                    h = self.transformer(h, f"unet.up_blocks.{i}.attentions.{j}", kv, heads[3 - i])
            if i < 3:
                h = h.repeat_interleave(2, 2).repeat_interleave(2, 3)
                h = self.conv(h, f"unet.up_blocks.{i}.upsamplers.0.conv")
        h = self.conv(self.silu(self.gnorm(h, "unet.conv_norm_out")), "unet.conv_out")
        return h

    # ------------------------------------------------------------------------- consensus entry

    def run_batch(self, seeds, kv):
        """seeds: list of 32-byte seeds -> list of 32768-byte x0 payloads."""
        n = sdint.noise_batch(self.net, seeds)                                    # B,4,64,64; one H2D transfer
        cin, sig = int(self.net.raw["c.cin"][0]), int(self.net.raw["c.sig"][0])
        eps = self.unet(rshr_pos(n * cin, 30), kv)
        x0 = rshr_pos((n - eps) * sig, 16)
        q = rshr_pos(x0, 8).clamp(-32767, 32767).to(torch.int16).cpu().numpy().astype("<i2")
        return [q[b].tobytes() for b in range(len(seeds))]


def check(model, batch):
    net, mh = sdint.load(model, "cuda")
    fast = Fast(net)
    toks = [320, 736, 3240]
    ctx = net.text(torch.tensor(sdint.context_ids(toks), device="cuda"))
    kv = fast.cross_kv(ctx)
    seeds = [hashlib.sha256(bytes([i]) + b"x").digest() for i in range(batch)]
    seeds[0] = hashlib.sha256(b"x").digest()
    torch.cuda.synchronize(); t = time.time()
    outs = fast.run_batch(seeds, kv)
    torch.cuda.synchronize(); dt = time.time() - t
    ok = hashlib.sha256(outs[0]).hexdigest() == "bef682cc9651fe8d42f075e98b907432f86a0b9e9a40e772fef99fc424f5c518"
    print(f"batch {batch}: {dt:.2f}s ({dt / batch:.2f}s per nonce), vector match: {ok}")
    ref = sdint.run(net, seeds[-1], toks, ctx)
    print(f"last sample == sdint reference: {ref == outs[-1]}")
    return ok and ref == outs[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="MODEL")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.check, a.batch) else 1)
