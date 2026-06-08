"""Read a TF1 checkpoint (.index + .data) without TensorFlow.

Parses the LevelDB SSTable .index file to extract variable names and their
offsets/sizes in the .data file, then reads the raw float32 tensors and
returns them as numpy arrays or PyTorch tensors.

Only supports float32 (dtype=1) tensors (all Spleeter weights are float32).
"""

import struct
import numpy as np
import torch


def _read_varint(data: bytes, pos: int):
    result, shift = 0, 0
    while pos < len(data):
        b = data[pos]; pos += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_bundle_entry(value: bytes):
    """Decode a BundleEntryProto to extract (dtype, offset, size)."""
    pos = 0
    dtype = 1    # default: float32
    offset = 0   # default: start of file
    size = None
    while pos < len(value):
        tag, pos = _read_varint(value, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            v, pos = _read_varint(value, pos)
            if field_num == 1: dtype = v
            if field_num == 4: offset = v
            if field_num == 5: size = v
        elif wire_type == 2:
            l, pos = _read_varint(value, pos)
            pos += l
        elif wire_type == 5:
            pos += 4
        elif wire_type == 1:
            pos += 8
        else:
            break
    return dtype, offset, size


def _parse_block(block_data: bytes):
    """Parse an SSTable data block: sequence of (key, value) pairs."""
    n = len(block_data)
    num_restarts = struct.unpack_from('<I', block_data, n - 4)[0]
    content_end = n - 4 - num_restarts * 4

    entries = []
    pos = 0
    last_key = b''
    while pos < content_end:
        shared,   pos = _read_varint(block_data, pos)
        unshared, pos = _read_varint(block_data, pos)
        vlen,     pos = _read_varint(block_data, pos)
        if pos + unshared + vlen > content_end + 1:
            break
        key = last_key[:shared] + block_data[pos: pos + unshared]
        pos += unshared
        value = block_data[pos: pos + vlen]
        pos += vlen
        last_key = key
        entries.append((key, value))
    return entries


def read_checkpoint(index_path: str) -> dict:
    """Return a dict mapping variable_name → (offset_in_data, size_bytes, dtype)."""
    with open(index_path, 'rb') as f:
        raw = f.read()

    footer = raw[-48:]
    pos = 0
    _, pos   = _read_varint(footer, pos)  # metaindex offset (skip)
    _, pos   = _read_varint(footer, pos)  # metaindex size   (skip)
    idx_off, pos = _read_varint(footer, pos)
    idx_sz,  _   = _read_varint(footer, pos)

    # Index block → find all data block locations
    idx_block = raw[idx_off: idx_off + idx_sz]
    idx_entries = _parse_block(idx_block)

    # Parse data blocks
    var_info = {}
    for _, handle_bytes in idx_entries:
        blk_off, p2 = _read_varint(handle_bytes, 0)
        blk_sz,  _  = _read_varint(handle_bytes, p2)
        blk = raw[blk_off: blk_off + blk_sz]
        for name_bytes, val in _parse_block(blk):
            name = name_bytes.decode('utf-8', errors='replace')
            dtype, off, sz = _parse_bundle_entry(val)
            var_info[name] = (off, sz, dtype)
    return var_info


def load_tensor(data_path: str, offset: int, size: int) -> torch.Tensor:
    """Load a float32 tensor from the .data file at the given offset."""
    with open(data_path, 'rb') as f:
        f.seek(offset)
        raw = f.read(size)
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return torch.from_numpy(arr)
