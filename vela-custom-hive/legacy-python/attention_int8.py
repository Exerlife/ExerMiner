#!/usr/bin/env python3
"""Exact batched INT16 x INT16 -> INT64 GEMM using INT8 Tensor Cores via Triton.

The split is exact:
    x = 256*x_hi + x_lo + 128, with x_hi/x_lo in signed INT8.
Four INT8 dot products plus row/column corrections reconstruct the exact
integer product. The kernel accumulates each INT8 dot in INT32 and combines
those terms in INT64. For Vela v2 Attention, K is at most 4096, safely inside
that INT32 per-term accumulation bound.
"""
from __future__ import annotations

import torch

_TRITON_IMPORT_ERROR = None
try:
    import triton
    import triton.language as tl
except Exception as exc:  # optional until the experimental mode is requested
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


if triton is not None:
    @triton.jit
    def _bmm16_int8_kernel(
        A, B, C,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        ASG: tl.constexpr, ASM: tl.constexpr, ASK: tl.constexpr,
        BSG: tl.constexpr, BSK: tl.constexpr, BSN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """One CTA computes one (batch, M tile, N tile); batch is parallel."""
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        pg = tl.program_id(2)
        om = pm * BM + tl.arange(0, BM)
        on = pn * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)

        acc_hh = tl.zeros((BM, BN), dtype=tl.int32)
        acc_hl = tl.zeros((BM, BN), dtype=tl.int32)
        acc_lh = tl.zeros((BM, BN), dtype=tl.int32)
        acc_ll = tl.zeros((BM, BN), dtype=tl.int32)
        row_h = tl.zeros((BM,), dtype=tl.int32)
        row_l = tl.zeros((BM,), dtype=tl.int32)
        col_h = tl.zeros((BN,), dtype=tl.int32)
        col_l = tl.zeros((BN,), dtype=tl.int32)

        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + ok
            a_mask = (om[:, None] < M) & (kk[None, :] < K)
            b_mask = (kk[:, None] < K) & (on[None, :] < N)
            av = tl.load(
                A + pg * ASG + om[:, None] * ASM + kk[None, :] * ASK,
                mask=a_mask, other=0,
            ).to(tl.int32)
            bv = tl.load(
                B + pg * BSG + kk[:, None] * BSK + on[None, :] * BSN,
                mask=b_mask, other=0,
            ).to(tl.int32)

            # Signed high byte and biased signed low byte; x = 256*hi + lo + 128.
            ah = (av >> 8).to(tl.int8)
            al = ((av & 255) - 128).to(tl.int8)
            bh = (bv >> 8).to(tl.int8)
            bl = ((bv & 255) - 128).to(tl.int8)

            acc_hh = tl.dot(ah, bh, acc_hh, out_dtype=tl.int32)
            acc_hl = tl.dot(ah, bl, acc_hl, out_dtype=tl.int32)
            acc_lh = tl.dot(al, bh, acc_lh, out_dtype=tl.int32)
            acc_ll = tl.dot(al, bl, acc_ll, out_dtype=tl.int32)

            row_h += tl.sum(ah.to(tl.int32), axis=1)
            row_l += tl.sum(al.to(tl.int32), axis=1)
            col_h += tl.sum(bh.to(tl.int32), axis=0)
            col_l += tl.sum(bl.to(tl.int32), axis=0)

        # K is padded by masked zero inputs to a multiple of BK. A mathematical
        # zero splits to (hi=0, lo=-128), so the constant must use K_PAD.
        k_pad = tl.cdiv(K, BK) * BK
        result = (
            acc_hh.to(tl.int64) * 65536
            + acc_hl.to(tl.int64) * 256
            + acc_lh.to(tl.int64) * 256
            + acc_ll.to(tl.int64)
            + 32768 * (row_h.to(tl.int64)[:, None] + col_h.to(tl.int64)[None, :])
            + 128 * (row_l.to(tl.int64)[:, None] + col_l.to(tl.int64)[None, :])
            + 16384 * k_pad
        )
        c_ptrs = C + pg * (M * N) + om[:, None] * N + on[None, :]
        c_mask = (om[:, None] < M) & (on[None, :] < N)
        tl.store(c_ptrs, result, mask=c_mask)


def is_available() -> bool:
    return triton is not None


def require_supported_device(device) -> None:
    if triton is None:
        detail = f" ({_TRITON_IMPORT_ERROR})" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(
            "--attention-matmul int8 requires the Triton package matching the installed PyTorch" + detail
        )
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("INT8 Attention requires a CUDA device")
    capability = torch.cuda.get_device_capability(dev)
    if capability < (7, 5):
        raise RuntimeError(
            f"INT8 Tensor-Core Attention requires compute capability >= 7.5; got {capability}"
        )


def bmm16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact ``[G,M,K] @ [G,K,N]`` for signed values in [-32767, 32767]."""
    require_supported_device(a.device)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"bmm16 expects rank-3 inputs, got {a.ndim}D and {b.ndim}D")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError(f"incompatible batched GEMM shapes {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.device != b.device:
        raise ValueError("bmm16 inputs must be on the same CUDA device")
    if a.dtype not in (torch.int16, torch.int32, torch.int64) or b.dtype not in (
        torch.int16, torch.int32, torch.int64
    ):
        raise TypeError("bmm16 inputs must have an integer dtype")

    g, m, k = map(int, a.shape)
    n = int(b.shape[2])
    if g == 0 or m == 0 or n == 0:
        return torch.empty((g, m, n), dtype=torch.int64, device=a.device)
    if k <= 0:
        raise ValueError("bmm16 reduction dimension K must be positive")

    bm, bn, bk = 16, 32, 32
    k_pad = triton.cdiv(k, bk) * bk
    # Every signed INT8 dot term is bounded by 128*128*K_PAD. Keep its INT32
    # accumulator provably below INT32_MAX, including the final padded tile.
    if k_pad > 131040:
        raise ValueError(f"K={k} is too large for the exact INT8 accumulator bound")

    out = torch.empty((g, m, n), dtype=torch.int64, device=a.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), g)
    _bmm16_int8_kernel[grid](
        a, b, out,
        m, n, k,
        *map(int, a.stride()), *map(int, b.stride()),
        bm, bn, bk,
        num_warps=4,
        num_stages=2,
    )
    return out
