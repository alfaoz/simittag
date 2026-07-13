"""
Render a Simittag marker to a numpy array / PNG.

Geometry and cell ordering come from simittag.spec / simittag.codec, so the image
is guaranteed consistent with what the detector expects.

Convention (shared with detector): pixel (x=col, y=row); marker centered;
dx=x-cx, dy=y-cy; theta = atan2(dy, dx) mod 2*pi; sector = floor(theta/step);
radius normalized so the outer ring's outer edge = 1.0.
"""
from __future__ import annotations

import argparse
import numpy as np

from simittag.spec import MarkerSpec, DEFAULT, VARIANTS
from simittag import codec, payload as _payload


def render(payload: bytes, spec: MarkerSpec = DEFAULT, size: int = 1024,
           supersample: int = 4, margin: float = 0.12) -> np.ndarray:
    """
    Return a (size, size) uint8 grayscale image (255=white, 0=black).
    margin: white quiet-zone fraction added around the marker (radius 1.0).
    """
    grid = codec.encode(payload, spec)
    S = size * supersample
    # radius 1.0 maps to (1 - margin) of the half-extent, leaving a quiet border
    cx = cy = (S - 1) / 2.0
    R_px = (S / 2.0) * (1.0 - margin)

    ys, xs = np.mgrid[0:S, 0:S]
    dx = (xs - cx) / R_px
    dy = (ys - cy) / R_px
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    img = np.ones((S, S), dtype=np.float32)  # white
    step = 2 * np.pi / spec.SECTOR_COUNT
    ring_w = (spec.R_DATA_OUT - spec.R_DATA_IN) / spec.RING_COUNT

    # outer ring
    img[(r >= spec.R_RING_IN) & (r <= 1.0)] = 0.0
    # bullseye
    img[r <= spec.R_BULLSEYE] = 0.0
    # data cells
    data_mask = (r >= spec.R_DATA_IN) & (r < spec.R_DATA_OUT)
    ring_idx = np.clip(((r - spec.R_DATA_IN) / ring_w).astype(int), 0, spec.RING_COUNT - 1)
    sec_idx = np.clip((theta / step).astype(int), 0, spec.SECTOR_COUNT - 1)
    cell_val = grid[ring_idx, sec_idx]  # 1 => black
    img[data_mask & (cell_val == 1)] = 0.0
    # beyond 1.0 stays white (quiet zone)

    # downsample (anti-alias)
    img = img.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_png(arr: np.ndarray, path: str):
    from PIL import Image
    Image.fromarray(arr, mode="L").save(path)


def _parse_payload(s: str, n: int) -> bytes:
    if s.startswith("0x"):
        v = int(s, 16)
        return v.to_bytes(n, "big")
    b = s.encode()
    if len(b) > n:
        raise ValueError(f"payload string too long ({len(b)} > {n} bytes)")
    return b.ljust(n, b"\x00")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render a Simittag marker")
    ap.add_argument("--variant", default="M", choices=list(VARIANTS),
                    help="T=tracking 3x16, M=balanced 4x24, D=data 5x36")
    ap.add_argument("--id", type=lambda s: int(s, 0), default=None,
                    help="ID-mode integer (e.g. 0x1234); fills the payload")
    ap.add_argument("--payload", default=None,
                    help="raw payload: hex (0x...) or text (fills payload bytes)")
    ap.add_argument("--out", default="marker.png")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    sp = VARIANTS[args.variant]
    if args.payload is not None:
        payload = _parse_payload(args.payload, sp.payload_bytes)
        desc = f"payload={payload.hex()}"
    else:
        # default to an ID so every variant has a sensible no-arg render
        idval = args.id if args.id is not None else 0x1234
        payload = _payload.encode_id(idval, sp)
        desc = f"id=0x{idval:x}"
    arr = render(payload, sp, size=args.size)
    save_png(arr, args.out)
    # confirm it decodes from its own grid (sanity, not validation)
    back, _ = codec.decode(codec.encode(payload, sp), sp)
    print(f"wrote {args.out} ({args.size}x{args.size}) variant={sp.NAME} {desc} "
          f"self-decode={'OK' if back == payload else 'FAIL'}")
