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
           supersample: int = 4, margin: float = 0.12,
           tile_rows: int = 64, inverted: bool = False) -> np.ndarray:
    """
    Return a (size, size) uint8 grayscale image (255=white, 0=black).
    margin: white quiet-zone fraction added around the marker (radius 1.0).
    inverted: white marker foreground on a black background.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if supersample <= 0:
        raise ValueError("supersample must be positive")
    if not (0.0 <= margin < 1.0):
        raise ValueError("margin must be in [0, 1)")
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")

    grid = codec.encode(payload, spec)
    S = size * supersample
    # radius 1.0 maps to (1 - margin) of the half-extent, leaving a quiet border
    cx = cy = (S - 1) / 2.0
    R_px = (S / 2.0) * (1.0 - margin)

    # Work in horizontal tiles. The original full-frame implementation kept
    # several SxS float64 arrays alive together and peaked around 1.4 GB RSS at
    # the default 1024x / 4x supersampling. A binary supersample is only SxS
    # bytes; temporary coordinate arrays are bounded by tile_rows x S.
    img = np.full((S, S), 255, dtype=np.uint8)
    dx = (np.arange(S, dtype=np.float64)[None, :] - cx) / R_px
    step = 2 * np.pi / spec.SECTOR_COUNT
    ring_w = (spec.R_DATA_OUT - spec.R_DATA_IN) / spec.RING_COUNT

    for y0 in range(0, S, tile_rows):
        y1 = min(S, y0 + tile_rows)
        dy = (np.arange(y0, y1, dtype=np.float64)[:, None] - cy) / R_px
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        tile = img[y0:y1]

        # outer ring and bullseye
        tile[(r >= spec.R_RING_IN) & (r <= 1.0)] = 0
        tile[r <= spec.R_BULLSEYE] = 0

        # data cells
        data_mask = (r >= spec.R_DATA_IN) & (r < spec.R_DATA_OUT)
        ring_idx = np.clip(((r - spec.R_DATA_IN) / ring_w).astype(int),
                           0, spec.RING_COUNT - 1)
        sec_idx = np.clip((theta / step).astype(int), 0, spec.SECTOR_COUNT - 1)
        cell_val = grid[ring_idx, sec_idx]  # 1 => black
        tile[data_mask & (cell_val == 1)] = 0

    # downsample (anti-alias)
    result = img.reshape(size, supersample, size, supersample).mean(axis=(1, 3)).astype(np.uint8)
    return 255 - result if inverted else result


def save_png(arr: np.ndarray, path: str):
    # OpenCV is already a detector dependency; using it here avoids making
    # Pillow an undeclared generation-only dependency.
    import cv2
    if not cv2.imwrite(path, arr):
        raise OSError(f"could not write PNG: {path}")


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
    ap.add_argument("--inverted", action="store_true",
                    help="render white foreground on a black background")
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
    arr = render(payload, sp, size=args.size, inverted=args.inverted)
    save_png(arr, args.out)
    # confirm it decodes from its own grid (sanity, not validation)
    back, _ = codec.decode(codec.encode(payload, sp), sp)
    print(f"wrote {args.out} ({args.size}x{args.size}) variant={sp.NAME} {desc} "
          f"self-decode={'OK' if back == payload else 'FAIL'}")
