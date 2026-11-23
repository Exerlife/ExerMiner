#!/usr/bin/env python3
# Copyright (c) 2026 The Vela developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""Render the Proof-of-Inference latent x0 of a block as a PNG.

The toy model's latent is 4x16x16 int8. A real latent-diffusion model would
decode x0 with its VAE; here a fixed linear "latent -> RGB" projection (the
same idea as the SD/Sana latent preview matrices) plus nearest-neighbour
upscaling is used. Rendering is NOT consensus-critical.

PoI v2 blocks (integer sd-turbo, x0 = 4x64x64 int16 Q8 latent) are decoded
with the sd-turbo VAE into a real 512x512 image (needs torch + diffusers and
the sd-turbo directory, --vae).

Usage:
  render_poi.py --rpc http://u:p@127.0.0.1:18743 <blockhash|height> [-o out.png] [--scale 16]
      [--vae <sd-turbo dir>] [--no-verify]
  render_poi.py --hex <x0 hex> -o out.png
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poi_miner import RPC  # noqa: E402

# latent channel -> RGB preview matrix (rows: channels)
LATENT_RGB = np.array([
    [0.35, 0.23, 0.19],
    [0.12, 0.30, 0.12],
    [-0.10, 0.07, 0.36],
    [-0.20, -0.18, -0.27],
])


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render(x0, shape, scale):
    c, h, w = shape
    lat = np.frombuffer(x0, dtype=np.int8).astype(np.float64).reshape(c, h, w) / 128.0
    m = LATENT_RGB[:c] if c <= 4 else np.vstack([LATENT_RGB] * ((c + 3) // 4))[:c]
    rgb = np.einsum("chw,ck->hwk", lat, m)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (rgb * 255 + 0.5).astype(np.uint8)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def render_vae(x0, shape, vae_dir):
    import torch
    from diffusers import AutoencoderKL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev)
    lat = torch.from_numpy(np.frombuffer(x0, dtype="<i2").astype(np.float32) / 256.0).reshape(1, *shape).to(dev)
    with torch.no_grad():
        img = vae.decode(lat / vae.config.scaling_factor).sample
    return ((img[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", nargs="?", help="block hash or height")
    ap.add_argument("--rpc", default="http://u:p@127.0.0.1:18743")
    ap.add_argument("--hex", help="render a raw x0 hex string (shape 4x16x16)")
    ap.add_argument("-o", "--out", help="output PNG (default poi_<height>.png)")
    ap.add_argument("--scale", type=int, default=16)
    ap.add_argument("--vae", default=os.environ.get("POI_SD_DIR"), help="sd-turbo directory (v2 blocks)")
    ap.add_argument("--no-verify", action="store_true", help="do not ask the node to recompute the inference")
    args = ap.parse_args()
    version = 1

    if args.hex:
        x0, shape, name = bytes.fromhex(args.hex), (4, 16, 16), "x0"
    else:
        if not args.block:
            ap.error("block hash/height or --hex required")
        rpc = RPC(args.rpc)
        h = rpc("getblockhash", int(args.block)) if args.block.isdigit() and len(args.block) < 64 else args.block
        info = rpc("getblockpoi", h, not args.no_verify)
        version = info["version"]
        x0, shape, name = bytes.fromhex(info["x0"]), tuple(info["shape"]), str(info["height"])
        print(f"block {info['height']} {info['hash'][:16]}..  model v{info['version']} {info['model']}")
        print(f"  prompt : {info['prompt']!r}")
        print(f"  seed   : {info['seed']}")
        print(f"  hashPoI: {info['hashpoi']}  commitment_valid={info['commitment_valid']}  "
              f"recomputed_match={info.get('recomputed_match')} ({info.get('recompute_ms', 0):.2f} ms)")
    out = args.out or f"poi_{name}.png"
    if version >= 2:
        if not args.vae:
            sys.exit("v2 block: pass --vae <sd-turbo dir> (or set POI_SD_DIR) to decode the latent")
        write_png(out, render_vae(x0, shape, args.vae))
    else:
        write_png(out, render(x0, shape, args.scale))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()