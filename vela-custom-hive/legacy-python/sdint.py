#!/usr/bin/env python3
"""PoI v2: bit-exact integer sd-turbo (W8A16). This file IS the specification.

Number format
  * activations: int32 "Q16" (real = v / 65536), saturated to int32 after every op
  * weights: int8 per output channel, real = w * m * 2^-s  (m in [2^14, 2^15))
  * GEMM inputs: per row, power-of-two scale: e = max(0, bitlen(max|x|) - 15),
    q = clamp(rshr(x, e), +-32767)  (16-bit), acc exact in int64
  * rshr(v, s) = (v + 2^(s-1)) >> s for s > 0 (arithmetic shift, rounds half up),
    v for s == 0, saturating v << -s for s < 0
  * every non-linearity is a table lookup (tables are part of the model file)

Pipeline (one Euler step, t = 999, sigma = SIG):
  tokens -> CLIP text encoder (23 layers) -> ctx[77,1024]
  n[4,64,64]   = GAUSS_LUT[uint16le(ChaCha20(seed))]           (Q16)
  eps          = UNet(rshr(n * CIN, 30), ctx)                  (Q16)
  x0           = rshr((n - eps) * SIG, 16)                     (Q16, latent space)
  output bytes = int16le(clamp(rshr(x0, 8), +-32767))          (Q8, 32768 bytes)

Model file "POISDv1\\0": u32 count, then sorted entries
  u16 namelen | name | u8 dtype (0 i8, 1 i16, 2 i32, 3 u8) | u8 ndim | u32 dims | LE data
model_hash = TaggedHash("POI/model")(u8 version=2 || whole file).
"""
import hashlib, math, struct, sys
import numpy as np
import torch

VERSION = 2
MAGIC = b"POISDv1\0"
DT = {0: np.int8, 1: np.int16, 2: np.int32, 3: np.uint8}
DT_CODE = {np.dtype(np.int8): 0, np.dtype(np.int16): 1, np.dtype(np.int32): 2, np.dtype(np.uint8): 3}
I32_MAX, I32_MIN = 2**31 - 1, -2**31
LUT_RANGE = 16       # sigmoid / phi tables cover [-16, 16) in steps of 1/64
EXP_STEPS = 1024     # exp table covers [0, 16) in steps of 1/64, Q30
BOS, EOS, PAD, CTX_LEN, MAX_TOKENS, VOCAB = 49406, 49407, 0, 77, 75, 49408
LATENT = (4, 64, 64)
OUT_SIZE = 2 * 4 * 64 * 64


# ----------------------------------------------------------------------------- integer primitives

def sat32(v):
    return v.clamp(I32_MIN, I32_MAX)


def rshr(v, s):
    """Rounding arithmetic right shift by tensor/int s; negative s = saturating left shift."""
    v = v.long()
    if isinstance(s, int) and 0 <= s <= 62:
        return sat32((v + ((1 << s) >> 1)) >> s)
    s = torch.as_tensor(s, dtype=torch.long, device=v.device)
    if int(s.min()) >= 0 and int(s.max()) <= 62:      # fast path, identical result
        return sat32((v + ((1 << s) >> 1)) >> s)
    pos = s.clamp(0, 62)
    half = torch.where(pos > 0, torch.ones_like(pos) << (pos - 1).clamp(min=0), torch.zeros_like(pos))
    r = (v + half) >> pos
    neg = (-s).clamp(0, 31)
    lim = (torch.ones_like(neg) << (62 - neg))
    r = torch.where(s < 0, r.clamp(-lim, lim) << neg, r)
    return sat32(r)


def row_exp(x):
    """e = max(0, bitlen(max|x| over last dim) - 15), shape [..., 1]."""
    m = x.abs().amax(dim=-1, keepdim=True)
    e = torch.zeros_like(m)
    for k in range(15, 32):
        e += (m >= (1 << k)).long()
    return e


def quant16(x, e):
    return rshr(x, e).clamp(-32767, 32767)


def floor_div(a, b):
    return torch.div(a, b, rounding_mode="floor")


def isqrt(v):
    s = torch.sqrt(v.double()).floor().long()
    for _ in range(2):
        s = torch.where(s * s > v, s - 1, s)
        s = torch.where((s + 1) * (s + 1) <= v, s + 1, s)
    return s


# ----------------------------------------------------------------------------- exact GEMMs

def _pad_mm(a8, b8):
    """int8 [M,K] @ int8 [K,N] -> int64, exact (CUDA _int_mm with padding)."""
    M, K = a8.shape
    N = b8.shape[1]
    Mp, Kp, Np = max(32, (M + 7) // 8 * 8), (K + 7) // 8 * 8, (N + 7) // 8 * 8
    if (Mp, Kp) != (M, K):
        a8 = torch.nn.functional.pad(a8, (0, Kp - K, 0, Mp - M))
    if (Kp, Np) != (K, N):
        b8 = torch.nn.functional.pad(b8, (0, Np - N, 0, Kp - K))
    return torch._int_mm(a8.contiguous(), b8.contiguous())[:M, :N].long()


def _split(a):
    """a in [-32767, 32767] -> (hi, lo) int8 with a = 256*hi + lo + 128."""
    hi = (a >> 8).to(torch.int8)
    lo = ((a & 255) - 128).to(torch.int8)
    return hi, lo


def mm(a, b):
    """Exact int64 product of integer matrices a [M,K] (|a| <= 32767) and b [K,N] (|b| <= 32767)."""
    if a.device.type != "cuda":
        return (a.double() @ b.double()).round().long()
    ah, al = _split(a.long())
    if b.dtype == torch.int8:
        return (_pad_mm(ah, b) << 8) + _pad_mm(al, b) + 128 * b.long().sum(0, keepdim=True)
    bh, bl = _split(b.long())
    def m8(x8):  # int8 @ b16
        return (_pad_mm(x8, bh) << 8) + _pad_mm(x8, bl) + 128 * x8.long().sum(1, keepdim=True)
    return (m8(ah) << 8) + m8(al) + 128 * b.long().sum(0, keepdim=True)


# ----------------------------------------------------------------------------- layers

class Net:
    def __init__(self, tensors, device):
        self.raw = tensors
        self.t = {k: torch.from_numpy(v.astype(np.int64) if v.dtype != np.int8 else v.copy()).to(device)
                  for k, v in tensors.items()}
        self.dev = device
        self.wT = {}

    def lin(self, x, name):
        """x [R,K] Q16 -> [R,N] Q16."""
        w = self.t[name + ".w"]
        if name not in self.wT:
            self.wT[name] = w.t().contiguous()
        e = row_exp(x)
        acc = mm(quant16(x, e), self.wT[name])
        y = rshr(acc * self.t[name + ".m"][None, :], self.t[name + ".s"][None, :] - e)
        return sat32(y + self.t[name + ".b"][None, :])

    def conv(self, x, name, stride=1):
        """x [C,H,W] Q16, 3x3 (pad 1) or 1x1 conv."""
        w = self.t[name + ".w"]
        C, H, W = x.shape
        k = int(round(math.sqrt(w.shape[1] // C)))
        if k == 1:
            return self.lin(x.reshape(C, H * W).t(), name).t().reshape(-1, H, W)
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
        Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
        cols = torch.stack([xp[:, ky:ky + stride * (Ho - 1) + 1:stride, kx:kx + stride * (Wo - 1) + 1:stride]
                            for ky in range(3) for kx in range(3)], 1)          # C,9,Ho,Wo
        cols = cols.reshape(C * 9, Ho * Wo).t()
        return self.lin(cols, name).t().reshape(-1, Ho, Wo)

    def norm(self, x, name, groups):
        """GroupNorm (x [C,HW], groups) or LayerNorm (x [R,C], groups=0 -> per row)."""
        g, b, den = self.t[name + ".g"], self.t[name + ".beta"], int(self.t[name + ".eps"][0])
        if groups:
            C = x.shape[0]
            v = x.reshape(groups, -1)
        else:
            v = x
        n = v.shape[1]
        mean = floor_div(v.sum(1, keepdim=True) + n // 2, n)
        d = v - mean
        e = (row_exp(d) - 5).clamp(min=0)                       # |d >> k| < 2^20
        dd = rshr(d, e)
        var = floor_div((dd * dd).sum(1, keepdim=True), n)
        eps = floor_div((torch.ones_like(e) << (32 - 2 * e)) + den // 2, den)
        sd = isqrt((var + eps) << 16)                           # real sd * 2^(24-k)
        y = floor_div((d << (24 - e)) * 2 + sd, 2 * sd)         # Q16
        if groups:
            y = y.reshape(C, -1)
            return sat32(rshr(y * g[:, None], 16) + b[:, None])
        return sat32(rshr(y * g[None, :], 16) + b[None, :])

    def lut(self, x, table):
        """Interpolated table over [-16, 16): returns Q16 sigmoid/phi of x."""
        L = self.t[table]
        z = x.long() + (LUT_RANGE << 16)
        zc = z.clamp(0, (2 * LUT_RANGE << 16) - 1)
        i, f = zc >> 10, zc & 1023
        v = L[i] + (((L[i + 1] - L[i]) * f) >> 10)
        v = torch.where(z < 0, torch.zeros_like(v), v)
        return torch.where(z >= (2 * LUT_RANGE << 16), torch.full_like(v, 65536), v)

    def silu(self, x):
        return sat32(rshr(x * self.lut(x, "lut.sigmoid"), 16))

    def gelu(self, x):
        return sat32(rshr(x * self.lut(x, "lut.phi"), 16))

    def attention(self, q, k, v, heads, causal=False):
        """q [Tq,C], k/v [Tk,C] Q16 -> [Tq,C] Q16; head dim 64, scale 1/8."""
        Tq, C = q.shape
        dh = C // heads
        assert dh == 64
        L = self.t["lut.exp"]
        out = []
        for h in range(heads):
            qh, kh, vh = q[:, h * dh:(h + 1) * dh], k[:, h * dh:(h + 1) * dh], v[:, h * dh:(h + 1) * dh]
            eq, ek = row_exp(qh), row_exp(kh)
            S = mm(quant16(qh, eq), quant16(kh, ek).t())
            s = rshr(S, 19 - eq - ek.t())
            if causal:
                mask = torch.ones(Tq, kh.shape[0], dtype=torch.bool, device=q.device).triu(1)
                s = s.masked_fill(mask, I32_MIN)
            mx = s.amax(1, keepdim=True)
            z = mx - s
            zc = z.clamp(0, (16 << 16) - 1)
            i, f = zc >> 10, zc & 1023
            E = L[i] + (((L[i + 1] - L[i]) * f) >> 10)
            E = torch.where(z >= (16 << 16), torch.zeros_like(E), E)
            r = floor_div(torch.full_like(mx, 1 << 62), E.sum(1, keepdim=True))
            P = ((E * r + (1 << 46)) >> 47).clamp(max=32767)
            ev = row_exp(vh.t()).t()                             # per channel [1, dh]
            O = mm(P, quant16(vh, ev))
            out.append(rshr(O, 15 - ev))
        return torch.cat(out, 1)

    # ------------------------------------------------------------------------- blocks

    def resnet(self, x, p):
        C, H, W = x.shape
        h = self.silu(self.norm(x.reshape(C, -1), p + ".norm1", 32)).reshape(C, H, W)
        h = self.conv(h, p + ".conv1")
        Co = h.shape[0]
        h = self.silu(self.norm(h.reshape(Co, -1), p + ".norm2", 32)).reshape(Co, H, W)
        h = self.conv(h, p + ".conv2")
        if p + ".conv_shortcut.w" in self.t:
            x = self.conv(x, p + ".conv_shortcut")
        return sat32(x + h)

    def transformer(self, x, p, ctx_kv, heads):
        C, H, W = x.shape
        h = self.norm(x.reshape(C, -1), p + ".norm", 32).t()     # tokens [HW, C]
        h = self.lin(h, p + ".proj_in")
        b = p + ".transformer_blocks.0"
        n = self.norm(h, b + ".norm1", 0)
        h = sat32(h + self.lin(self.attention(self.lin(n, b + ".attn1.to_q"), self.lin(n, b + ".attn1.to_k"),
                                              self.lin(n, b + ".attn1.to_v"), heads), b + ".attn1.to_out.0"))
        n = self.norm(h, b + ".norm2", 0)
        k, v = ctx_kv[b]
        h = sat32(h + self.lin(self.attention(self.lin(n, b + ".attn2.to_q"), k, v, heads), b + ".attn2.to_out.0"))
        n = self.norm(h, b + ".norm3", 0)
        a = self.lin(n, b + ".ff.net.0.proj.a")
        g = self.lin(n, b + ".ff.net.0.proj.g")
        h = sat32(h + self.lin(sat32(rshr(a * self.gelu(g), 16)), b + ".ff.net.2"))
        h = self.lin(h, p + ".proj_out")
        return sat32(x + h.t().reshape(C, H, W))

    def unet(self, x, ctx):
        heads = {0: 5, 1: 10, 2: 20, 3: 20}
        kv = {}
        for name in self.t:
            if name.startswith("unet.") and name.endswith(".attn2.to_k.w"):
                b = name[:-len(".attn2.to_k.w")]
                kv[b] = (self.lin(ctx, b + ".attn2.to_k"), self.lin(ctx, b + ".attn2.to_v"))
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
                h = torch.cat([h, skips.pop()], 0)
                h = self.resnet(h, f"unet.up_blocks.{i}.resnets.{j}")
                if i > 0:
                    h = self.transformer(h, f"unet.up_blocks.{i}.attentions.{j}", kv, heads[3 - i])
            if i < 3:
                h = h.repeat_interleave(2, 1).repeat_interleave(2, 2)
                h = self.conv(h, f"unet.up_blocks.{i}.upsamplers.0.conv")
        C, H, W = h.shape
        h = self.silu(self.norm(h.reshape(C, -1), "unet.conv_norm_out", 32)).reshape(C, H, W)
        return self.conv(h, "unet.conv_out")

    def text(self, ids):
        q = self.t["te.tok.w"][ids].long()
        x = rshr(q * self.t["te.tok.m"][ids][:, None], (self.t["te.tok.s"][ids] - 16)[:, None])
        x = sat32(x + self.t["te.pos"])
        for l in range(23):
            p = f"te.layers.{l}"
            n = self.norm(x, p + ".layer_norm1", 0)
            a = self.attention(self.lin(n, p + ".self_attn.q_proj"), self.lin(n, p + ".self_attn.k_proj"),
                               self.lin(n, p + ".self_attn.v_proj"), 16, causal=True)
            x = sat32(x + self.lin(a, p + ".self_attn.out_proj"))
            n = self.norm(x, p + ".layer_norm2", 0)
            x = sat32(x + self.lin(self.gelu(self.lin(n, p + ".mlp.fc1")), p + ".mlp.fc2"))
        return self.norm(x, "te.final_layer_norm", 0)


# ----------------------------------------------------------------------------- CLIP BPE tokenizer (not consensus)

def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def pretokenize(text):
    """ASCII-exact approximation of CLIP's pre-tokenizer (non-ASCII bytes count as letters)."""
    b = " ".join(text.lower().split()).encode()
    out, i = [], 0
    isl = lambda c: 97 <= c <= 122 or c >= 128
    isd = lambda c: 48 <= c <= 57
    iss = lambda c: c in b" \t\n\r\x0b\x0c"
    while i < len(b):
        c = b[i]
        hit = next((x for x in _CONTRACTIONS if b.startswith(x.encode(), i)), None)
        if hit:
            out.append(b[i:i + len(hit)]); i += len(hit)
        elif isl(c):
            j = i
            while j < len(b) and isl(b[j]): j += 1
            out.append(b[i:j]); i = j
        elif isd(c):
            out.append(b[i:i + 1]); i += 1
        elif iss(c):
            i += 1
        else:
            j = i
            while j < len(b) and not (isl(b[j]) or isd(b[j]) or iss(b[j])): j += 1
            out.append(b[i:j]); i = j
    return out


class Tokenizer:
    def __init__(self, vocab_json, merges_txt):
        import json
        self.enc = json.loads(bytes(vocab_json).decode())
        self.dec = {v: k for k, v in self.enc.items()}
        lines = bytes(merges_txt).decode().strip().split("\n")[1:49152 - 256 - 2 + 1]
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines)}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def bpe(self, word):
        w = list(word[:-1]) + [word[-1] + "</w>"]
        while len(w) > 1:
            best = min(((self.ranks.get((w[i], w[i + 1]), 1 << 30), i) for i in range(len(w) - 1)))
            if best[0] == 1 << 30:
                break
            a, b = w[best[1]], w[best[1] + 1]
            nw, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(a + b); i += 2
                else:
                    nw.append(w[i]); i += 1
            w = nw
        return w

    def encode(self, text):
        ids = []
        for tok in pretokenize(text):
            for sym in self.bpe("".join(self.b2u[c] for c in tok)):
                ids.append(self.enc[sym])
        return ids[:MAX_TOKENS]

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            sym = self.dec.get(i, "")
            end = sym.endswith("</w>")
            out += bytes(self.u2b[c] for c in (sym[:-4] if end else sym) if c in self.u2b)
            if end:
                out += b" "
        return out.decode(errors="replace").strip()


def tokenizer(net_or_tensors):
    t = net_or_tensors.raw if hasattr(net_or_tensors, "raw") else net_or_tensors
    return Tokenizer(t["tok.vocab"], t["tok.merges"])


# ----------------------------------------------------------------------------- consensus entry points

def chacha20_keystream(key, nbytes):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(key, b"\0" * 16), mode=None).encryptor()
    return enc.update(b"\0" * nbytes)


def context_ids(tokens):
    assert len(tokens) <= MAX_TOKENS and all(0 <= t < BOS for t in tokens)
    ids = [BOS] + list(tokens) + [EOS]
    return ids + [PAD] * (CTX_LEN - len(ids))


def noise(net, seed):
    u = np.frombuffer(chacha20_keystream(seed, 2 * 4 * 64 * 64), dtype="<u2").astype(np.int64)
    return net.t["lut.gauss"][torch.from_numpy(u).to(net.dev)].reshape(LATENT)


def noise_batch(net, seeds):
    """Generate a batch of exact ChaCha20 noise with one host-to-device transfer."""
    if not seeds:
        return torch.empty((0, *LATENT), dtype=net.t["lut.gauss"].dtype, device=net.dev)
    expected_bytes = 2 * 4 * 64 * 64
    rows = [np.frombuffer(chacha20_keystream(seed, expected_bytes), dtype="<u2") for seed in seeds]
    # uint16 -> int64 indexing is required by the LUT; stack once, then transfer once.
    indices = np.stack(rows, axis=0).astype(np.int64, copy=False)
    index_tensor = torch.from_numpy(indices).to(net.dev)
    return net.t["lut.gauss"][index_tensor].reshape(len(seeds), *LATENT)


def run(net, seed, tokens, ctx=None):
    """seed: 32 bytes, tokens: list[int]. Returns the 32768-byte x0 payload."""
    if ctx is None:
        ctx = net.text(torch.tensor(context_ids(tokens), device=net.dev))
    n = noise(net, seed)
    eps = net.unet(rshr(n * int(net.t["c.cin"][0]), 30), ctx)
    x0 = rshr((n - eps) * int(net.t["c.sig"][0]), 16)
    q = rshr(x0, 8).clamp(-32767, 32767)
    return q.cpu().numpy().astype("<i2").tobytes()


def x0_to_latent(payload):
    return torch.from_numpy(np.frombuffer(payload, dtype="<i2").astype(np.float32) / 256.0).reshape(1, *LATENT)


# ----------------------------------------------------------------------------- model file

def write_model(path, tensors):
    h = hashlib.sha256()
    with open(path, "wb") as f:
        def w(b):
            f.write(b); h.update(b)
        w(MAGIC + struct.pack("<I", len(tensors)))
        for name in sorted(tensors):
            a = np.ascontiguousarray(tensors[name])
            nb = name.encode()
            w(struct.pack("<HB", len(nb), 0)[:2] + nb)
            w(struct.pack("<BB", DT_CODE[a.dtype], a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape))
            w(a.astype(a.dtype.newbyteorder("<")).tobytes())
    return h.hexdigest()


def read_model(path):
    data = open(path, "rb").read()
    assert data[:8] == MAGIC
    (count,) = struct.unpack_from("<I", data, 8)
    pos, out = 12, {}
    for _ in range(count):
        (nl,) = struct.unpack_from("<H", data, pos); pos += 2
        name = data[pos:pos + nl].decode(); pos += nl
        dt, nd = data[pos], data[pos + 1]; pos += 2
        dims = struct.unpack_from(f"<{nd}I", data, pos); pos += 4 * nd
        cnt = int(np.prod(dims)) if nd else 1
        a = np.frombuffer(data, dtype=np.dtype(DT[dt]).newbyteorder("<"), count=cnt, offset=pos).reshape(dims)
        pos += a.nbytes
        out[name] = a
    assert pos == len(data)
    mh = hashlib.sha256(hashlib.sha256(b"POI/model").digest() * 2 + bytes([VERSION]) + data).digest()[::-1]
    return out, mh.hex()


def load(path, device="cuda"):
    tensors, mh = read_model(path)
    return Net(tensors, device), mh


# ----------------------------------------------------------------------------- exporter

def _enc_weight(w):
    """float [N,K] -> (int8 w, int32 m, int32 s) with real ~= w * m * 2^-s."""
    w = w.double()
    mx = w.abs().amax(1).clamp_min(1e-30)
    sw = mx / 127
    s = (14 - torch.floor(torch.log2(sw))).long()
    m = torch.round(sw * torch.exp2(s.double())).long()
    over = m >= (1 << 15)
    s = torch.where(over, s - 1, s)
    m = torch.round(sw * torch.exp2(s.double())).long().clamp(1 << 14, (1 << 15) - 1)
    q = torch.round(w / (m.double() * torch.exp2(-s.double()))[:, None]).clamp(-127, 127)
    return q.to(torch.int8).numpy(), m.to(torch.int32).numpy(), s.to(torch.int32).numpy()


def _q16(x):
    return torch.round(x.double() * 65536).clamp(I32_MIN, I32_MAX).to(torch.int32).numpy()


def export(model_dir, out_path):
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel
    from statistics import NormalDist
    nd = NormalDist()
    T = {}
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", variant="fp16", torch_dtype=torch.float32).eval()
    te = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", variant="fp16", dtype=torch.float32).eval()
    with torch.no_grad():
        tt = torch.tensor([999.0])
        temb = unet.time_embedding(unet.time_proj(tt))
    tadd = {}
    for name, mod in unet.named_modules():
        if getattr(mod, "time_emb_proj", None) is not None:
            with torch.no_grad():
                tadd[name + ".conv1"] = mod.time_emb_proj(torch.nn.functional.silu(temb))[0]

    def add_lin(prefix, weight, bias, extra=None):
        w = weight.reshape(weight.shape[0], -1)
        T[prefix + ".w"], T[prefix + ".m"], T[prefix + ".s"] = _enc_weight(w)
        b = bias.double() if bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
        if extra is not None:
            b = b + extra.double()
        T[prefix + ".b"] = _q16(b)

    def add_norm(prefix, mod):
        T[prefix + ".g"] = _q16(mod.weight.data)
        T[prefix + ".beta"] = _q16(mod.bias.data)
        T[prefix + ".eps"] = np.array([round(1 / mod.eps)], dtype=np.int32)

    for comp, root, model in (("unet", "", unet), ("te", "text_model.", te)):
        for name, mod in model.named_modules():
            if comp == "te":
                if not name.startswith("text_model."):
                    continue
                key = "te." + name[len("text_model."):].replace("encoder.layers", "layers")
            else:
                key = "unet." + name
            if isinstance(mod, torch.nn.Linear) and name.endswith("ff.net.0.proj"):
                half = mod.out_features // 2
                add_lin(key + ".a", mod.weight.data[:half], mod.bias.data[:half])
                add_lin(key + ".g", mod.weight.data[half:], mod.bias.data[half:])
            elif isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                if comp == "unet" and name.startswith("time_embedding") or name.endswith("time_emb_proj"):
                    continue
                add_lin(key, mod.weight.data, mod.bias.data if mod.bias is not None else None, tadd.get(name))
            elif isinstance(mod, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
                add_norm(key, mod)
    emb = te.text_model.embeddings
    T["te.tok.w"], T["te.tok.m"], T["te.tok.s"] = _enc_weight(emb.token_embedding.weight.data)
    T["te.pos"] = _q16(emb.position_embedding.weight.data[:CTX_LEN])
    xs = np.arange(2 * LUT_RANGE * 64 + 1, dtype=np.float64) / 64 - LUT_RANGE
    T["lut.sigmoid"] = np.round(65536 / (1 + np.exp(-xs))).astype(np.int32)
    T["lut.phi"] = np.round(np.array([65536 * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in xs])).astype(np.int32)
    T["lut.exp"] = np.round(np.exp(-np.arange(EXP_STEPS + 1) / 64.0) * 2**30).astype(np.int32)
    T["lut.gauss"] = np.round(np.array([nd.inv_cdf((i + 0.5) / 65536) for i in range(65536)]) * 65536).astype(np.int32)
    sig = 14.614646911621094
    import os
    T["tok.vocab"] = np.frombuffer(open(os.path.join(model_dir, "tokenizer", "vocab.json"), "rb").read(), dtype=np.uint8)
    T["tok.merges"] = np.frombuffer(open(os.path.join(model_dir, "tokenizer", "merges.txt"), "rb").read(), dtype=np.uint8)
    T["c.sig"] = np.array([round(sig * 65536)], dtype=np.int32)
    T["c.cin"] = np.array([round(sig / math.sqrt(sig * sig + 1) * 2**30)], dtype=np.int32)
    return write_model(out_path, T), len(T)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        sha, n = export(sys.argv[2], sys.argv[3])
        print(f"wrote {sys.argv[3]}: {n} tensors, sha256 {sha}")
    else:
        print("usage: sdint.py export <sd-turbo dir> <out.poisd>")
