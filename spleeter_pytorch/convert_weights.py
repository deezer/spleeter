"""Convert Spleeter TF1 checkpoint → PyTorch .pt files (4-stem or 5-stem).

Usage:
    python convert_weights.py --checkpoint /path/to/4stems/model \\
                               --output-dir ./models4
    python convert_weights.py --checkpoint /path/to/5stems/model \\
                               --output-dir ./models5 --stems 5

Instrument order in the checkpoints:
    4stems (from spleeter/resources/4stems.json): vocals, drums, bass, other
    5stems (from spleeter/resources/5stems.json): vocals, piano, drums, bass, other

Each instrument's U-Net occupies a consecutive block of TF variable indices:
    4stems:
      instrument 0 (vocals): conv2d_0..6,   conv2d_transpose_0..5,  bn_0..11
      instrument 1 (drums):  conv2d_7..13,  conv2d_transpose_6..11, bn_12..23
      instrument 2 (bass):   conv2d_14..20, conv2d_transpose_12..17, bn_24..35
      instrument 3 (other):  conv2d_21..27, conv2d_transpose_18..23, bn_36..47
    5stems (same offsets, one more instrument appended):
      instrument 0 (vocals): conv2d_0..6,   conv2d_transpose_0..5,  bn_0..11
      instrument 1 (piano):  conv2d_7..13,  conv2d_transpose_6..11, bn_12..23
      instrument 2 (drums):  conv2d_14..20, conv2d_transpose_12..17, bn_24..35
      instrument 3 (bass):   conv2d_21..27, conv2d_transpose_18..23, bn_36..47
      instrument 4 (other):  conv2d_28..34, conv2d_transpose_24..29, bn_48..59
"""

import argparse
import os
import sys

import torch

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet
from checkpoint_reader import read_checkpoint, load_tensor


INSTRUMENTS_4 = ['vocals', 'drums', 'bass', 'other']
INSTRUMENTS_5 = ['vocals', 'piano', 'drums', 'bass', 'other']


def build_state_dict(var_info: dict, data_path: str, stem_idx: int) -> dict:
    """Build a PyTorch state dict for one instrument's UNet.

    Args:
        var_info: dict from read_checkpoint (name → (offset, size, dtype))
        data_path: path to .data-00000-of-00001 file
        stem_idx: 0=vocals, 1=drums, 2=bass, 3=other
    """
    def get(name: str) -> torch.Tensor:
        entry = var_info.get(name)
        if entry is None:
            raise KeyError(f"Variable '{name}' not found in checkpoint")
        off, sz, _ = entry
        return load_tensor(data_path, off, sz)

    def conv(tf_name: str) -> torch.Tensor:
        # TF Conv2D kernel: (H, W, in_ch, out_ch) → PyTorch (out_ch, in_ch, H, W)
        return get(tf_name).reshape(-1).view(0, 0, 0, 0)  # placeholder; real reshape below

    def kernel(tf_name: str, shape_hwio) -> torch.Tensor:
        """Load a TF Conv2D/ConvTranspose2D kernel and permute to PyTorch layout."""
        t = get(tf_name)
        H, W, I, O = shape_hwio
        t = t.reshape(H, W, I, O)
        return t.permute(3, 2, 0, 1).contiguous()  # HWIO → OIHW

    def tkernel(tf_name: str, shape_hwoi) -> torch.Tensor:
        """Load a TF ConvTranspose2D kernel: (H, W, out_ch, in_ch) → PyTorch (in_ch, out_ch, H, W)."""
        t = get(tf_name)
        H, W, O, I = shape_hwoi
        t = t.reshape(H, W, O, I)
        return t.permute(3, 2, 0, 1).contiguous()  # HWOI → IOIHW

    co = stem_idx * 7        # conv2d offset  (0, 7, 14, 21)
    to = stem_idx * 6        # transpose offset (0, 6, 12, 18)
    bo = stem_idx * 12       # batch-norm offset

    def cname(i):  return f"conv2d_{i}/kernel" if i else "conv2d/kernel"
    def cbname(i): return f"conv2d_{i}/bias"   if i else "conv2d/bias"
    def tnname(i): return f"conv2d_transpose_{i}/kernel" if i else "conv2d_transpose/kernel"
    def tbname(i): return f"conv2d_transpose_{i}/bias"   if i else "conv2d_transpose/bias"
    def bnname(i, attr): return f"batch_normalization_{i}/{attr}" if i else f"batch_normalization/{attr}"

    # Encoder conv shapes: (H, W, in, out)
    enc_shapes = [
        (5, 5,   2,  16),
        (5, 5,  16,  32),
        (5, 5,  32,  64),
        (5, 5,  64, 128),
        (5, 5, 128, 256),
        (5, 5, 256, 512),
    ]
    # Decoder conv-transpose shapes: (H, W, out, in)
    # PyTorch ConvTranspose2d(in, out) has weight (in, out, H, W)
    # TF has (H, W, out, in) where out=smaller side after transpose
    dec_shapes = [
        (5, 5, 256, 512),
        (5, 5, 128, 512),
        (5, 5,  64, 256),
        (5, 5,  32, 128),
        (5, 5,  16,  64),
        (5, 5,   1,  32),
    ]

    sd = {}

    # Encoder
    enc_py_names = ['conv', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    enc_bn_py    = ['bn',   'bn1',   'bn2',   'bn3',   'bn4',   None]
    # TF bn indices for encoder (skipping conv5's BN since PyTorch UNet doesn't use it)
    enc_bn_tf    = [bo+0,   bo+1,    bo+2,    bo+3,    bo+4,    None]

    for i, (py_name, shape) in enumerate(zip(enc_py_names, enc_shapes)):
        tf_k = cname(co + i)
        tf_b = cbname(co + i)
        sd[f"{py_name}.weight"] = kernel(tf_k, shape)
        sd[f"{py_name}.bias"]   = get(tf_b)

        bn_py = enc_bn_py[i]
        bn_tf = enc_bn_tf[i]
        if bn_py is not None:
            bn_i = bn_tf
            sd[f"{bn_py}.weight"]       = get(bnname(bn_i, 'gamma'))
            sd[f"{bn_py}.bias"]         = get(bnname(bn_i, 'beta'))
            sd[f"{bn_py}.running_mean"] = get(bnname(bn_i, 'moving_mean'))
            sd[f"{bn_py}.running_var"]  = get(bnname(bn_i, 'moving_variance'))

    # Decoder conv-transpose (up1–up6) and their BNs
    dec_py_names = ['up1', 'up2', 'up3', 'up4', 'up5', 'up6']
    dec_bn_py    = ['bn5', 'bn6', 'bn7', 'bn8', 'bn9', 'bn10']
    # TF BN indices for decoder: bn_6, bn_7, ... bn_11 for stem 0 (and +12 per stem)
    dec_bn_tf_base = bo + 6  # stem0=6, stem1=18, stem2=30, stem3=42

    for i, (py_name, shape) in enumerate(zip(dec_py_names, dec_shapes)):
        tf_k = tnname(to + i)
        tf_b = tbname(to + i)
        sd[f"{py_name}.weight"] = tkernel(tf_k, shape)
        sd[f"{py_name}.bias"]   = get(tf_b)

        bn_py = dec_bn_py[i]
        bn_i  = dec_bn_tf_base + i
        sd[f"{bn_py}.weight"]       = get(bnname(bn_i, 'gamma'))
        sd[f"{bn_py}.bias"]         = get(bnname(bn_i, 'beta'))
        sd[f"{bn_py}.running_mean"] = get(bnname(bn_i, 'moving_mean'))
        sd[f"{bn_py}.running_var"]  = get(bnname(bn_i, 'moving_variance'))

    # up7: regular Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)
    # TF stores as conv2d_{co+6}/kernel with shape (4, 4, 1, 2)
    up7_tf = cname(co + 6)
    sd['up7.weight'] = kernel(up7_tf, (4, 4, 1, 2))
    sd['up7.bias']   = get(cbname(co + 6))

    return sd


def convert(checkpoint_prefix: str, output_dir: str, n_stems: int = 5):
    """Convert a Spleeter TF1 checkpoint to .pt files.

    Args:
        checkpoint_prefix: path without extension (e.g. /path/to/model)
        output_dir: directory to write <instrument>.pt files
        n_stems: 4 or 5
    """
    instruments = INSTRUMENTS_5 if n_stems == 5 else INSTRUMENTS_4

    index_path = checkpoint_prefix + '.index'
    data_path  = checkpoint_prefix + '.data-00000-of-00001'

    print(f"[Spleeter] Reading checkpoint index: {index_path}")
    var_info = read_checkpoint(index_path)
    print(f"[Spleeter] Found {len(var_info)} variables ({n_stems}-stem)")

    os.makedirs(output_dir, exist_ok=True)

    for stem_idx, name in enumerate(instruments):
        print(f"[Spleeter] Converting stem {stem_idx}: {name}")
        sd = build_state_dict(var_info, data_path, stem_idx)

        model = UNet()
        model.load_state_dict(sd, strict=True)
        model.eval()

        out_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(model.state_dict(), out_path)
        print(f"[Spleeter]   → {out_path}")

    print("[Spleeter] Conversion complete.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True,
                    help='Path prefix to TF checkpoint (e.g. /path/to/model)')
    ap.add_argument('--output-dir', required=True,
                    help='Directory to write <instrument>.pt files')
    ap.add_argument('--stems', type=int, default=5, choices=[4, 5],
                    help='Number of stems (4 or 5, default: 5)')
    args = ap.parse_args()
    convert(args.checkpoint, args.output_dir, n_stems=args.stems)
