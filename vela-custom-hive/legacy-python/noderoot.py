#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.

"""Node output commitments computed while the output is still on the GPU.

Two things are produced from the same digest kernel, both over the int32le bytes
of a node's output split into 4 KiB (4096-byte) chunks:

    chunk_j  = SHA256(be32 node || be32 j || chunk j)                       (level 1)
    leaf_ik  = SHA256("VELA/leaf/v1" || be32 i || be32 k || 64 sub-chunks)  (leaf spec)
    root_i   = SHA256("VELA/noderoot/v1" || chunk_0 || ... )                (legacy root)
    R        = TaggedHash("POI/roots", root_0 || ...)                       (legacy R)

The block payload commits to the *leaf* digests (see doc/fork-m.md); the node
roots are the earlier, coarser instantiation of the same idea and are kept for
the fraud-proof tooling. Both levels run on the GPU: level 2 is one thread per
node over its chunk digests, so a caller can drop the activation as soon as the
digests exist.

Usage (inside a vir_torch evaluation): NodeRoots.add(i, cuda_int64_tensor) for
every node output before it is copied to the host, then finish() or
finish_leaves() at the end. Requires CuPy; the kernel is checked against hashlib
by vir_mine.py --selftest.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

if os.name == "nt":
    # CuPy loads against the CUDA runtime shipped with the torch wheels.
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
try:
    import cupy as cp  # noqa: E402
except ImportError as exc:  # a miner without CuPy can still run the CPU reference
    raise SystemExit("noderoot.py needs CuPy for the GPU digest kernel: "
                     "pip install cupy-cuda12x (or use the CPU reference path)") from exc

CHUNK = 4096

_SHA = r'''
__constant__ unsigned int K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ __forceinline__ unsigned int rotr(unsigned int x,int n){return (x>>n)|(x<<(32-n));}
__device__ __forceinline__ unsigned int bswap(unsigned int x){return __byte_perm(x,0,0x0123);}
__device__ void compress(unsigned int* s,const unsigned int* w16){
  unsigned int w[64];
  #pragma unroll
  for(int i=0;i<16;i++)w[i]=w16[i];
  #pragma unroll
  for(int i=16;i<64;i++){unsigned int s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
    unsigned int s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
  unsigned int a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
  #pragma unroll
  for(int i=0;i<64;i++){unsigned int S1=rotr(e,6)^rotr(e,11)^rotr(e,25);unsigned int ch=(e&f)^(~e&g);
    unsigned int t1=h+S1+ch+K[i]+w[i];unsigned int S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
    unsigned int mj=(a&b)^(a&c)^(b&c);unsigned int t2=S0+mj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
  s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;}
__device__ void sha_init(unsigned int* s){
  s[0]=0x6a09e667;s[1]=0xbb67ae85;s[2]=0x3c6ef372;s[3]=0xa54ff53a;
  s[4]=0x510e527f;s[5]=0x9b05688c;s[6]=0x1f83d9ab;s[7]=0x5be0cd19;}

// Level 1: chunk j of one node's int32 buffer (words = little-endian int32 values).
extern "C" __global__ void chunk_sha(const unsigned int* data, unsigned int nwords,
                                     unsigned int node, unsigned int* out, unsigned int nchunks){
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
  if (j >= nchunks) return;
  const unsigned int cw = 1024;                       // 4 KiB / 4
  unsigned int w0 = j*cw;
  unsigned int words = (nwords - w0 < cw) ? (nwords - w0) : cw;
  const unsigned int* p = data + w0;
  unsigned int s[8]; sha_init(s);
  unsigned int total = words + 2;
  unsigned long long bits = (unsigned long long)total*32ULL;
  unsigned int blk[16]; unsigned int idx=0;
  while (idx+16 <= total){
    #pragma unroll
    for(int k=0;k<16;k++){unsigned int q=idx+k; blk[k]=(q==0)?node:((q==1)?j:bswap(p[q-2]));}
    compress(s,blk); idx+=16;}
  unsigned int rem=total-idx;
  for(int k=0;k<16;k++)blk[k]=0;
  for(unsigned int k=0;k<rem;k++){unsigned int q=idx+k; blk[k]=(q==0)?node:((q==1)?j:bswap(p[q-2]));}
  blk[rem]=0x80000000u;
  if(rem>=14){compress(s,blk);for(int k=0;k<16;k++)blk[k]=0;}
  blk[14]=(unsigned int)(bits>>32); blk[15]=(unsigned int)bits;
  compress(s,blk);
  #pragma unroll
  for(int k=0;k<8;k++) out[8*j+k]=s[k];
}

// Level 2: one thread per node: SHA256("VELA/noderoot/v1" || its chunk digests).
// prefix is 16 bytes = 4 words; digests are big-endian words already.
extern "C" __global__ void node_root(const unsigned int* dig, const unsigned int* start,
                                     const unsigned int* count, unsigned int nnodes,
                                     unsigned int* out){
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= nnodes) return;
  const unsigned int pre[4]={0x56454c41u,0x2f6e6f64u,0x65726f6fu,0x742f7631u}; // "VELA/noderoot/v1"
  const unsigned int* d = dig + 8ull*start[i];
  unsigned int nw = 8*count[i];
  unsigned int total = nw + 4;
  unsigned long long bits=(unsigned long long)total*32ULL;
  unsigned int s[8]; sha_init(s);
  unsigned int blk[16]; unsigned int idx=0;
  while(idx+16<=total){
    for(int k=0;k<16;k++){unsigned int q=idx+k; blk[k]=(q<4)?pre[q]:d[q-4];}
    compress(s,blk); idx+=16;}
  unsigned int rem=total-idx;
  for(int k=0;k<16;k++)blk[k]=0;
  for(unsigned int k=0;k<rem;k++){unsigned int q=idx+k; blk[k]=(q<4)?pre[q]:d[q-4];}
  blk[rem]=0x80000000u;
  if(rem>=14){compress(s,blk);for(int k=0;k<16;k++)blk[k]=0;}
  blk[14]=(unsigned int)(bits>>32); blk[15]=(unsigned int)bits;
  compress(s,blk);
  for(int k=0;k<8;k++) out[8*i+k]=s[k];
}
'''

_mod = cp.RawModule(code=_SHA)
_chunk = _mod.get_function("chunk_sha")
_root = _mod.get_function("node_root")


LEAF_SUBS = 64          # 256 KiB leaves (poi-fraud-proof-spec.md §2)


def cpu_leaves(node: int, int32le: bytes, leaf_subs: int = LEAF_SUBS) -> list[bytes]:
    """Reference leaf digests (spec §3.1), hashlib."""
    nsub = max(1, (len(int32le) + CHUNK - 1) // CHUNK)
    subs = [hashlib.sha256(node.to_bytes(4, "big") + j.to_bytes(4, "big") +
                           int32le[j * CHUNK:(j + 1) * CHUNK]).digest() for j in range(nsub)]
    nleaf = max(1, (nsub + leaf_subs - 1) // leaf_subs)
    return [hashlib.sha256(b"VELA/leaf/v1" + node.to_bytes(4, "big") + k.to_bytes(4, "big") +
                           b"".join(subs[k * leaf_subs:(k + 1) * leaf_subs])).digest()
            for k in range(nleaf)]


def leaves_from_subs(node: int, sub_words: np.ndarray, leaf_subs: int = LEAF_SUBS) -> list[bytes]:
    """Leaf digests from GPU sub-chunk digests (uint32 words, big-endian per digest)."""
    subs = sub_words.reshape(-1, 8).astype(">u4").tobytes()
    nsub = len(subs) // 32
    nleaf = max(1, (nsub + leaf_subs - 1) // leaf_subs)
    return [hashlib.sha256(b"VELA/leaf/v1" + node.to_bytes(4, "big") + k.to_bytes(4, "big") +
                           subs[k * leaf_subs * 32:(k + 1) * leaf_subs * 32]).digest()
            for k in range(nleaf)]


def cpu_root(node: int, int32le: bytes) -> bytes:
    """Reference implementation of root_i (hashlib)."""
    digs = []
    for j in range(max(1, (len(int32le) + CHUNK - 1) // CHUNK)):
        digs.append(hashlib.sha256(node.to_bytes(4, "big") + j.to_bytes(4, "big") +
                                    int32le[j * CHUNK:(j + 1) * CHUNK]).digest())
    return hashlib.sha256(b"VELA/noderoot/v1" + b"".join(digs)).digest()


class NodeRoots:
    """Accumulates chunk digests per node on the GPU; finish() -> 903 roots + R."""

    def __init__(self, n_nodes: int, timing: bool = False):
        """timing=True measures each node's hash with a device sync (serialises the
        hash with the next op); timing=False is the mining mode: asynchronous."""
        self.n = n_nodes
        self.parts: list[cp.ndarray | None] = [None] * n_nodes
        self.keep: list[torch.Tensor] = []   # keep int32 copies alive until finish()
        self.timing = timing
        self.seconds = 0.0
        self.bytes = 0

    def add(self, i: int, t: torch.Tensor):
        """t: CUDA integer tensor of node i's output (values fit int32)."""
        if self.timing:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        w = t.reshape(-1).to(torch.int32).contiguous()          # int32le canonical bytes
        nwords = w.numel()
        nch = max(1, (nwords * 4 + CHUNK - 1) // CHUNK)
        out = cp.empty(8 * nch, dtype=cp.uint32)
        src = cp.from_dlpack(w.view(torch.uint32) if hasattr(torch, "uint32") else w)
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            _chunk(((nch + 127) // 128,), (128,), (src, np.uint32(nwords), np.uint32(i), out, np.uint32(nch)))
        self.parts[i] = out
        self.bytes += nwords * 4
        if self.timing:
            ev1.record()
            torch.cuda.synchronize()
            self.seconds += ev0.elapsed_time(ev1) / 1000.0
        else:
            # The kernel runs on the same stream as the next op, so `w` only needs to
            # outlive this enqueue; PyTorch's caching allocator will not reuse its block
            # for a later op on the same stream before the kernel has consumed it.
            self.keep = [w]

    def finish(self):
        missing = [i for i, p in enumerate(self.parts) if p is None]
        if missing:
            raise RuntimeError(f"no output recorded for nodes {missing[:5]}")
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        counts = np.array([p.size // 8 for p in self.parts], np.uint32)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint32)
        dig = cp.concatenate(self.parts)
        out = cp.empty(8 * self.n, dtype=cp.uint32)
        _root(((self.n + 127) // 128,), (128,),
              (dig, cp.asarray(starts), cp.asarray(counts), np.uint32(self.n), out))
        roots_words = out.get().reshape(self.n, 8)
        t1.record()
        torch.cuda.synchronize()
        self.seconds += t0.elapsed_time(t1) / 1000.0
        self.keep.clear()
        roots = [r.astype(">u4").tobytes() for r in roots_words]
        return roots

    def finish_leaves(self):
        """Spec §3.1 leaf digests for every node (sub-chunk digests from the GPU)."""
        missing = [i for i, p in enumerate(self.parts) if p is None]
        if missing:
            raise RuntimeError(f"no output recorded for nodes {missing[:5]}")
        torch.cuda.synchronize()
        out = []
        for i, p in enumerate(self.parts):
            out.extend(leaves_from_subs(i, p.get()))
        self.keep.clear()
        return out
