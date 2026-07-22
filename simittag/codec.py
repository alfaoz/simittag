"""
Payload <-> cell grid.

A cell grid is an int array of shape (RING_COUNT, SECTOR_COUNT), values 0/1.
  ring 0            = sync pattern (spec.SYNC), used for rotation alignment
  rings 1..RING-1   = RS codeword over GF(2^SYMBOL_BITS)
                      (bytes + CRC8 for the v1 variants; nibbles + CRC4 for
                      the small-grid v2 variants -- see spec.SYMBOL_BITS)

Encode:  payload bytes -> +CRC -> RS encode -> bits -> data rings; sync ring set.
Decode:  data-ring bits -> symbols -> RS decode (with optional erasures) -> CRC.

Data-cell linear order is sector-major: k = sector*(data_rings) + (ring-1),
packed MSB-first into symbols. Generator and detector both use this module so
the ordering is defined in exactly one place.
"""
from __future__ import annotations

import numpy as np

from .spec import MarkerSpec, DEFAULT
from . import gf256, gf16


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


def _bits_to_syms(bits, sb):
    """Pack bits MSB-first into symbols of sb bits (sb=8 -> byte values)."""
    out = []
    for i in range(0, len(bits), sb):
        v = 0
        for j in range(sb):
            v = (v << 1) | (bits[i + j] & 1)
        out.append(v)
    return out


def _syms_to_bits(syms, sb):
    bits = []
    for v in syms:
        for j in range(sb - 1, -1, -1):
            bits.append((v >> j) & 1)
    return bits


def _payload_to_syms(payload, spec):
    """Canonical payload bytes -> data symbols. Payloads are big-endian byte
    strings carrying payload_bits; the high pad bits (nibble variants whose
    bit count is not a byte multiple) must be zero."""
    if len(payload) != spec.payload_bytes:
        raise ValueError(
            f"payload must be {spec.payload_bytes} bytes, got {len(payload)}")
    value = int.from_bytes(payload, "big")
    if value >> spec.payload_bits:
        raise ValueError(
            f"payload exceeds {spec.payload_bits} bits for {spec.NAME}")
    sb = spec.SYMBOL_BITS
    nsyms = spec.payload_bits // sb
    return [(value >> (sb * (nsyms - 1 - i))) & ((1 << sb) - 1)
            for i in range(nsyms)]


def _syms_to_payload(syms, spec):
    """Data symbols -> canonical payload bytes (big-endian, zero pad bits)."""
    sb = spec.SYMBOL_BITS
    value = 0
    for v in syms:
        value = (value << sb) | v
    return value.to_bytes(spec.payload_bytes, "big")


def _rs(spec):
    """The symbol-field module for this spec (duck-typed: gf16 and gf256
    share rs_encode/rs_decode signatures; CRC width differs)."""
    return gf16 if spec.SYMBOL_BITS == 4 else gf256


def _crc(data_syms, spec):
    if spec.SYMBOL_BITS == 4:
        return gf16.crc4(data_syms)
    return gf256.crc8(bytes(data_syms))


def _cell_order(spec):
    """Yield (k, ring, sector) for the data cells in codeword-bit order.
    Data rings are every ring except the sync ring (spec.data_rings)."""
    rings = spec.data_rings
    dr = len(rings)
    for s in range(spec.SECTOR_COUNT):
        for rd in range(dr):
            yield s * dr + rd, rings[rd], s


def encode(payload: bytes, spec: MarkerSpec = DEFAULT) -> np.ndarray:
    syms = _payload_to_syms(payload, spec)                 # RS_K - CRC symbols
    data = syms + [_crc(syms, spec)]                       # RS_K data symbols
    code = list(_rs(spec).rs_encode(data, spec.RS_NSYM))   # RS_K + RS_NSYM
    bits = _syms_to_bits(code, spec.SYMBOL_BITS)

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
    sb = spec.SYMBOL_BITS
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    bits = [0] * nbits
    cell_unreliable = [False] * nbits
    for k, ring, sector in _cell_order(spec):
        bits[k] = int(grid[ring, sector])
        if erasure_grid is not None:
            cell_unreliable[k] = bool(erasure_grid[ring, sector])

    code = _bits_to_syms(bits, sb)
    erase_pos = None
    if erasure_grid is not None:
        erase_pos = sorted({k // sb for k in range(nbits) if cell_unreliable[k]})
        # Cap erasures at NSYM-1, never NSYM. With NSYM erasures the codeword is fully
        # determined by the erasure positions and RS ALWAYS "succeeds" (e.g. it fills
        # to the all-zeros codeword, whose CRC is 0 -> a valid-looking ID 0). Keeping
        # one syndrome of margin means the surviving cells must actually be consistent,
        # which is what stops a wrong-variant grid from decoding to a phantom ID.
        cap = max(0, spec.RS_NSYM - 1)
        if len(erase_pos) > cap:
            erase_pos = erase_pos[:cap]

    try:
        data, _ = _rs(spec).rs_decode(code, spec.RS_NSYM, erase_pos=erase_pos,
                                      max_errors=spec.MAX_ERRORS)
    except Exception:
        return None

    nds = spec.RS_K - spec.CRC_BYTES
    payload_syms, crc = list(data[:nds]), data[nds]
    if _crc(payload_syms, spec) != crc:
        return None
    return _syms_to_payload(payload_syms, spec)


def _sym_reliability(conf_grid, spec):
    """Per-codeword-symbol reliability = weakest cell confidence in the symbol."""
    sb = spec.SYMBOL_BITS
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    rel = np.full(nbits // sb, np.inf)
    for k, ring, sector in _cell_order(spec):
        c = float(conf_grid[ring, sector])
        if c < rel[k // sb]:
            rel[k // sb] = c
    return rel


def _decode_aligned_ranked(grid, spec, conf_grid, conf_erasure):
    """
    Decode a rotation-aligned grid using confidence-RANKED erasures.

    The boolean-mask path erases every symbol containing a cell under the
    confidence threshold and, when that exceeds the NSYM-1 cap, keeps the
    LOWEST-INDEXED symbols -- an arbitrary subset. Here the erased set is
    always the WEAKEST symbols under the threshold, so the cap discards the
    most reliable erasure candidates instead of whichever happened to sort
    last. Attempt count is identical to the legacy path (one RS decode).
    """
    sb = spec.SYMBOL_BITS
    nbits = spec.data_ring_count * spec.SECTOR_COUNT
    bits = [0] * nbits
    for k, ring, sector in _cell_order(spec):
        bits[k] = int(grid[ring, sector])
    code = _bits_to_syms(bits, sb)

    rel = _sym_reliability(conf_grid, spec)
    order = np.argsort(rel, kind="stable")          # weakest symbol first
    cap = max(0, spec.RS_NSYM - 1)
    erase_pos = sorted(int(i) for i in order[:cap] if rel[i] < conf_erasure)

    try:
        data, _ = _rs(spec).rs_decode(code, spec.RS_NSYM,
                                      erase_pos=erase_pos or None,
                                      max_errors=spec.MAX_ERRORS)
    except Exception:
        return None
    nds = spec.RS_K - spec.CRC_BYTES
    payload_syms, crc = list(data[:nds]), data[nds]
    if _crc(payload_syms, spec) != crc:
        return None
    return _syms_to_payload(payload_syms, spec)


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
        max_err = sp.RS_NSYM // 2  # guaranteed-correctable symbol errors
        for _ in range(N):
            payload = random.getrandbits(sp.payload_bits).to_bytes(
                sp.payload_bytes, "big")
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
        assert ok == N and ok_rot == N and ok_err == N, f"codec self-test FAILED for {name}"
