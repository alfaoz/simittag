#!/usr/bin/env python3
"""
Simittag command line: encode a payload to a marker, decode an image,
or solve camera intrinsics from calibration-sheet photos.

  simittag encode --id 12345 --out marker.png
  simittag encode --raw "hi" --out marker.png
  simittag decode image.png
  simittag calibrate img1.png img2.png ... --out intrinsics.json
  simittag decode image.png --intrinsics intrinsics.json
"""
from __future__ import annotations
import argparse
import cv2

from simittag.spec import VARIANTS, ALIASES, LEGACY_NAMES
from simittag import payload, detect
from simittag import board as board_mod
from simittag.calibrate import CameraIntrinsics, calibrate_images
from marker.generate import render as render_marker, save_png


def cmd_encode(a):
    sp = VARIANTS[a.variant]
    if a.id is not None:
        pl = payload.encode_id(a.id, sp)
    elif a.raw is not None:
        data = bytes.fromhex(a.raw[2:]) if a.raw.startswith("0x") else a.raw.encode()
        pl = payload.encode_raw(data, sp)
    else:
        raise SystemExit("give --id or --raw")
    arr = render_marker(pl, sp, size=a.size, inverted=a.inverted)
    save_png(arr, a.out)
    print(f"wrote {a.out}  payload={pl.hex()}  decode={payload.decode(pl, sp)}")


def cmd_decode(a):
    gray = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"cannot read {a.image}")
    versions = None if a.variant == "auto" else a.variant
    K = dist = None
    if a.intrinsics:
        intr = CameraIntrinsics.load(a.intrinsics)
        K, dist = intr.K, intr.dist_array
    res = detect.detect(gray, versions=versions, K=K, dist=dist)
    if not res:
        print("no marker decoded")
        return
    for r in res:
        polarity = "white-on-black" if r["inverted"] else "black-on-white"
        print(f"  {r['variant']} ({r['alias']}) {r['mode']}={r['value']}  "
              f"center=({r['center'][0]:.0f},{r['center'][1]:.0f})  "
              f"tilt={r['tilt_deg']:.1f}deg  {polarity}")


def cmd_calibrate(a):
    board = board_mod.load_board(a.board) if a.board else None
    intr = calibrate_images(a.images, board=board)
    intr.save(a.out)
    print(f"wrote {a.out}")
    print(f"  fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"  dist={['%.4f' % v for v in intr.dist]}")
    print(f"  {intr.views} views, reprojection rms {intr.rms_px:.3f} px")


def main():
    ap = argparse.ArgumentParser(prog="simittag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # accepted spellings: canonical (sim96c32), alias (s16m), deprecated (M)
    variant_names = (*VARIANTS.keys(), *ALIASES, *LEGACY_NAMES)
    variant_help = ("marker variant: " +
                    ", ".join(f"{n} ({sp.ALIAS})" for n, sp in VARIANTS.items()) +
                    "; T/M/D are deprecated aliases")
    e = sub.add_parser("encode")
    e.add_argument("--variant", choices=variant_names, default="sim96c32",
                   metavar="VARIANT", help=variant_help)
    source = e.add_mutually_exclusive_group(required=True)
    source.add_argument("--id", type=lambda s: int(s, 0))
    source.add_argument("--raw")
    e.add_argument("--out", default="marker.png")
    e.add_argument("--size", type=int, default=1024)
    e.add_argument("--inverted", action="store_true")
    e.set_defaults(fn=cmd_encode)
    d = sub.add_parser("decode")
    d.add_argument("image")
    d.add_argument("--variant", choices=("auto", *variant_names), default="auto",
                   metavar="VARIANT", help="auto (default) or " + variant_help)
    d.add_argument("--intrinsics", help="intrinsics.json from `calibrate` "
                   "(replaces the default 60-degree-FOV guess)")
    d.set_defaults(fn=cmd_decode)
    c = sub.add_parser("calibrate",
                       help="solve camera intrinsics from calibration-sheet photos")
    c.add_argument("images", nargs="+")
    c.add_argument("--board", help="board JSON from the studio (default: "
                   "reconstruct from the sheet's descriptor tag)")
    c.add_argument("--out", default="intrinsics.json")
    c.set_defaults(fn=cmd_calibrate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
