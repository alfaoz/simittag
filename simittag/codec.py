"""
Payload <-> cell grid.

A cell grid is an int array of shape (RING_COUNT, SECTOR_COUNT), values 0/1.
  ring 0            = sync pattern (spec.SYNC), used for rotation alignment
  rings 1..RING-1   = RS(9,5) codeword over GF(256)

Encode:  payload bytes -> +CRC8 -> RS encode -> bits -> data rings; sync ring set.
Decode:  data-ring bits -> bytes -> RS decode (with optional erasures) -> CRC check.

Data-cell linear order is sector-major: k = sector*(data_rings) + (ring-1),
packed MSB-first into bytes. Generator and detector both use this module so
the ordering is defined in exactly one place.
"""
from __future__ import annotations

import numpy as np

from .spec import MarkerSpec, DEFAULT
from . import gf256


def _bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | (bits[i + j] & 1)
        out.append(b)
    return bytes(out)


def _bytes_to_bits(data):
    bits = []
    for b in data:
        for j in range(7, -1, -1):
            bits.append((b >> j) & 1)
    return bits


def _cell_order(spec):
    """Yield (k, ring, sector) for the data cells in codeword-bit order.
    Data rings are every ring except the sync ring (spec.data_rings)."""
    rings = spec.data_rings
    dr = len(rings)
    for s in range(spec.SECTOR_COUNT):
        for rd in range(dr):
            yield s * dr + rd, rings[rd], s


def encode(payload: bytes, spec: MarkerSpec = DEFAULT) -> np.ndarray:
    if len(payload) != spec.payload_bytes:
        raise ValueError(f"payload must be {spec.payload_bytes} bytes, got {len(payload)}")
    data = bytes(payload) + bytes([gf256.crc8(payload)])  # RS_K data bytes
    code = gf256.rs_encode(data, spec.RS_NSYM)             # RS_K + RS_NSYM bytes
    bits = _bytes_to_bits(code)

    grid = np.zeros((spec.RING_COUNT, spec.SECTOR_COUNT), dtype=np.int8)
    if spec.HAS_SYNC:
        grid[spec.SYNC_RING, :] = np.asarray(spec.SYNC, dtype=np.int8)
    for k, ring, sector in _cell_order(spec):
        grid[ring, sector] = bits[k]
    return grid


def find_rotation(ring0_bits, spec: MarkerSpec = DEFAULT) -> int:
    """
    Best sector rotation aligning observed sync ring to spec.SYNC, via circular
    cross-correlation (one-shot, not brute force over the ECC). Returns the shift
    s such that np.roll(observed, -s) matches SYNC.
    """
    obs = np.where(np.asarray(ring0_bits) > 0, 1, -1)
    ref = np.where(np.asarray(spec.SYNC) > 0, 1, -1)
    n = len(ref)
    scores = [int(np.dot(ref, np.roll(obs, -s))) for s in range(n)]
    return int(np.argmax(scores)), scores


def _decode_aligned(grid, spec, erasure_grid=None):
    """Decode a rotation-aligned grid (sector 0 in place) -> payload bytes or None."""
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    bits = [0] * nbits
    cell_unreliable = [False] * nbits
    for k, ring, sector in _cell_order(spec):
        bits[k] = int(grid[ring, sector])
        if erasure_grid is not None:
            cell_unreliable[k] = bool(erasure_grid[ring, sector])

    code = _bits_to_bytes(bits)
    erase_pos = None
    if erasure_grid is not None:
        erase_pos = sorted({k // 8 for k in range(nbits) if cell_unreliable[k]})
        # Cap erasures at NSYM-1, never NSYM. With NSYM erasures the codeword is fully
        # determined by the erasure positions and RS ALWAYS "succeeds" (e.g. it fills
        # to the all-zeros codeword, whose CRC is 0 -> a valid-looking ID 0). Keeping
        # one syndrome of margin means the surviving cells must actually be consistent,
        # which is what stops a wrong-variant grid from decoding to a phantom ID.
        cap = max(0, spec.RS_NSYM - 1)
        if len(erase_pos) > cap:
            erase_pos = erase_pos[:cap]

    try:
        data, _ = gf256.rs_decode(code, spec.RS_NSYM, erase_pos=erase_pos)
    except Exception:
        return None

    payload, crc = data[:spec.payload_bytes], data[spec.payload_bytes]
    if gf256.crc8(payload) != crc:
        return None
    return payload


def _byte_reliability(conf_grid, spec):
    """Per-codeword-byte reliability = weakest cell confidence inside the byte."""
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    rel = np.full(nbits // 8, np.inf)
    for k, ring, sector in _cell_order(spec):
        c = float(conf_grid[ring, sector])
        if c < rel[k // 8]:
            rel[k // 8] = c
    return rel


def _decode_aligned_ranked(grid, spec, conf_grid, conf_erasure):
    """
    Decode a rotation-aligned grid using confidence-RANKED erasures.

    The boolean-mask path erases every byte containing a cell under the
    confidence threshold and, when that exceeds the NSYM-1 cap, keeps the
    LOWEST-INDEXED bytes -- an arbitrary subset. Here the erased set is always
    the WEAKEST bytes under the threshold, so the cap discards the most
    reliable erasure candidates instead of whichever happened to sort last.
    Attempt count is identical to the legacy path (one RS decode).
    """
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    bits = [0] * nbits
    for k, ring, sector in _cell_order(spec):
        bits[k] = int(grid[ring, sector])
    code = _bits_to_bytes(bits)

    rel = _byte_reliability(conf_grid, spec)
    order = np.argsort(rel, kind="stable")          # weakest byte first
    cap = max(0, spec.RS_NSYM - 1)
    erase_pos = sorted(int(i) for i in order[:cap] if rel[i] < conf_erasure)

    try:
        data, _ = gf256.rs_decode(code, spec.RS_NSYM,
                                  erase_pos=erase_pos or None)
    except Exception:
        return None
    payload, crc = data[:spec.payload_bytes], data[spec.payload_bytes]
    if gf256.crc8(payload) != crc:
        return None
    return payload


def decode(grid: np.ndarray, spec: MarkerSpec = DEFAULT, erasure_grid=None,
           conf_grid=None, conf_erasure=0.25):
    """
    Decode a cell grid -> payload bytes, or None if it fails CRC/RS.
    erasure_grid: optional bool array (RING_COUNT, SECTOR_COUNT); True = unreliable
    cell. A codeword byte is marked an RS erasure if any of its cells is unreliable.
    conf_grid: optional float array (RING_COUNT, SECTOR_COUNT) of per-cell
    confidences; when given, ranked-erasure decoding is used instead of
    erasure_grid (see _decode_aligned_ranked).

    Rotation: HAS_SYNC variants lock rotation in one shot via the sync ring; no-sync
    variants (T) brute-force all SECTOR_COUNT shifts, RS+CRC arbitrating.
    Returns (payload_bytes, rotation_applied) or (None, rotation_applied).
    """
    grid = np.asarray(grid)
    eg = np.asarray(erasure_grid) if erasure_grid is not None else None
    cg = np.asarray(conf_grid) if conf_grid is not None else None
    if spec.HAS_SYNC:
        shift0, _ = find_rotation(grid[spec.SYNC_RING], spec)
        shifts = [shift0]
    else:
        shifts = range(spec.SECTOR_COUNT)
    for shift in shifts:
        aligned = np.roll(grid, -shift, axis=1)
        if cg is not None:
            pb = _decode_aligned_ranked(aligned, spec,
                                        np.roll(cg, -shift, axis=1), conf_erasure)
        else:
            eg_s = np.roll(eg, -shift, axis=1) if eg is not None else None
            pb = _decode_aligned(aligned, spec, eg_s)
        if pb is not None:
            return pb, shift
    return None, (shifts[0] if spec.HAS_SYNC else 0)


if __name__ == "__main__":
    import random
    from .spec import VARIANTS
    random.seed(1)
    N = 2000
    print(f"codec self-test over {N} trials/variant:")
    for name, sp in VARIANTS.items():
        ok = ok_rot = ok_err = 0
        max_err = sp.RS_NSYM // 2  # guaranteed-correctable byte errors
        for _ in range(N):
            payload = bytes(random.randint(0, 255) for _ in range(sp.payload_bytes))
            grid = encode(payload, sp)
            ok += (decode(grid, sp)[0] == payload)
            r = random.randint(0, sp.SECTOR_COUNT - 1)
            ok_rot += (decode(np.roll(grid, r, axis=1), sp)[0] == payload)
            # flip up to max_err codeword *bytes* worth of cells (worst case for RS):
            g = np.roll(grid, r, axis=1).copy()
            for _ in range(random.randint(0, max_err)):
                ring = random.randint(sp.first_data_ring, sp.RING_COUNT - 1)
                sec = random.randint(0, sp.SECTOR_COUNT - 1)
                g[ring, sec] ^= 1
            ok_err += (decode(g, sp)[0] == payload)
        print(f"  {name}: clean {ok}/{N}  +rot {ok_rot}/{N}  +flips {ok_err}/{N}")
