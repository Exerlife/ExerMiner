#!/usr/bin/env python3
"""Exactness tests for the experimental Vela v2 INT8 Attention GEMM."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import attention_int8


def decomposition_reference(a, b, block_k=32):
    """CPU/GPU PyTorch integer reference for the four-product identity."""
    g, m, k = a.shape
    n = b.shape[2]
    k_pad = ((k + block_k - 1) // block_k) * block_k
    if k_pad != k:
        a = torch.nn.functional.pad(a, (0, k_pad - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, k_pad - k))
    ah = a >> 8
    al = (a & 255) - 128
    bh = b >> 8
    bl = (b & 255) - 128
    row_h = ah.sum(dim=2, keepdim=True)
    row_l = al.sum(dim=2, keepdim=True)
    col_h = bh.sum(dim=1).unsqueeze(1)
    col_l = bl.sum(dim=1).unsqueeze(1)
    return (
        (ah @ bh) * 65536
        + (ah @ bl) * 256
        + (al @ bh) * 256
        + (al @ bl)
        + 32768 * (row_h + col_h)
        + 128 * (row_l + col_l)
        + 16384 * k_pad
    )


class ExactInt8AttentionTests(unittest.TestCase):
    def test_decomposition_matches_integer_matmul_with_signed_edges_and_padding(self):
        torch.manual_seed(7341)
        for g, m, k, n in ((1, 3, 1, 7), (2, 17, 77, 23), (3, 9, 64, 11)):
            a = torch.randint(-32767, 32768, (g, m, k), dtype=torch.int64)
            b = torch.randint(-32767, 32768, (g, k, n), dtype=torch.int64)
            a.reshape(-1)[0] = -32767
            a.reshape(-1)[-1] = 32767
            b.reshape(-1)[0] = 32767
            b.reshape(-1)[-1] = -32767
            self.assertTrue(torch.equal(decomposition_reference(a, b), torch.bmm(a, b)))

    @unittest.skipUnless(torch.cuda.is_available() and attention_int8.is_available(),
                         "requires CUDA + Triton; run on the RTX 3070 rig")
    def test_triton_kernel_matches_fp64_exact_reference(self):
        torch.manual_seed(991)
        device = "cuda"
        for g, m, k, n in ((2, 19, 64, 23), (3, 33, 77, 17), (2, 64, 4096, 64)):
            a = torch.randint(-32767, 32768, (g, m, k), dtype=torch.int64, device=device)
            b = torch.randint(-32767, 32768, (g, k, n), dtype=torch.int64, device=device)
            a.reshape(-1)[0] = -32767
            a.reshape(-1)[-1] = 32767
            b.reshape(-1)[0] = 32767
            b.reshape(-1)[-1] = -32767
            got = attention_int8.bmm16(a, b)
            expected = torch.bmm(a.double(), b.double()).long()
            self.assertTrue(torch.equal(got, expected), (g, m, k, n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
