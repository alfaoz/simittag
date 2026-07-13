#!/usr/bin/env python3
"""
Simittag app: encode a payload to a marker, or decode an image.

  python app.py encode --id 12345 --out marker.png
  python app.py encode --raw "hi" --out marker.png
  python app.py decode image.png
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

from simittag.spec import DEFAULT
from simittag import payload, detect
from marker.generate import render as render_marker, save_png


def cmd_encode(a):
    sp = DEFAULT
    if a.id is not None:
        pl = payload.encode_id(a.id, sp)
    elif a.raw is not None:
        data = bytes.fromhex(a.raw[2:]) if a.raw.startswith("0x") else a.raw.encode()
        pl = payload.encode_raw(data, sp)
    else:
        raise SystemExit("give --id or --raw")
    arr = render_marker(pl, sp, size=a.size)
    save_png(arr, a.out)
    print(f"wrote {a.out}  payload={pl.hex()}  decode={payload.decode(pl, sp)}")


def cmd_decode(a):
    gray = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"cannot read {a.image}")
    res = detect.detect(gray, DEFAULT)
    if not res:
        print("no marker decoded"); return
    for r in res:
        print(f"  {r['mode']}={r['value']}  payload={r['payload'].hex()}  "
              f"center=({r['center'][0]:.0f},{r['center'][1]:.0f})  "
              f"tilt~{r['tilt_deg_approx']:.0f}deg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--id", type=int)
    e.add_argument("--raw"); e.add_argument("--out", default="marker.png")
    e.add_argument("--size", type=int, default=1024); e.set_defaults(fn=cmd_encode)
    d = sub.add_parser("decode"); d.add_argument("image"); d.set_defaults(fn=cmd_decode)
    args = ap.parse_args()
    args.fn(args)
