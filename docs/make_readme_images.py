#!/usr/bin/env python3
"""
Regenerate the README images into docs/images/.

  python docs/make_readme_images.py

variants.png   the three variants side by side
anatomy.png    one tag with the rings labeled
detection.png  detect_markers() output drawn on a pinned test frame
tag_size.png   where tag size is measured, and the quiet zone
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simittag import payload
from simittag.spec import T_SPEC, M_SPEC, D_SPEC
from simittag import detect
from marker.generate import render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
ROOT = os.path.dirname(HERE)
MARGIN = 0.12          # render() default quiet-zone fraction
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (60, 60, 60)

os.makedirs(OUT, exist_ok=True)


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print("wrote", path, img.shape[1], "x", img.shape[0])


def tag_bgr(spec, value, size):
    pl = payload.encode_id(value, spec)
    g = render(pl, spec, size=size)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def text_center(img, s, cx, y, scale, color, thick=1):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    cv2.putText(img, s, (int(cx - w / 2), int(y)), FONT, scale, color, thick,
                cv2.LINE_AA)


# ---------------------------------------------------------------- variants
def make_variants():
    size, gap, label_h = 420, 46, 86
    tags = [(T_SPEC, 0x2A, "T", "3x16, 1 byte"),
            (M_SPEC, 0x1234, "M", "4x24, 4 bytes"),
            (D_SPEC, 0x53494D, "D", "5x36, 11 bytes")]
    W = 3 * size + 4 * gap
    H = size + label_h + 2 * gap
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i, (sp, val, name, sub) in enumerate(tags):
        x = gap + i * (size + gap)
        canvas[gap:gap + size, x:x + size] = tag_bgr(sp, val, size)
        cx = x + size / 2
        text_center(canvas, name, cx, gap + size + 40, 1.1, (0, 0, 0), 2)
        text_center(canvas, sub, cx, gap + size + 72, 0.62, INK, 1)
    save("variants.png", canvas)


# ---------------------------------------------------------------- anatomy
def make_anatomy():
    ts = 640
    W, H = 1120, ts + 40
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[20:20 + ts, 20:20 + ts] = tag_bgr(M_SPEC, 0x1234, ts)
    c = np.array([20 + ts / 2, 20 + ts / 2])
    R = (ts / 2) * (1 - MARGIN)  # radius 1.0 in pixels

    sp = M_SPEC
    sync_out = sp.R_DATA_IN + (sp.R_DATA_OUT - sp.R_DATA_IN) / sp.RING_COUNT
    red = (0, 0, 200)
    # label, band [r_in, r_out], marker angle (deg, y-down)
    marks = [("outer ring", (sp.R_RING_IN, 1.0), -58),
             ("quiet ring", (sp.R_DATA_OUT, sp.R_RING_IN), -30),
             ("data rings", (sync_out, sp.R_DATA_OUT), -4),
             ("sync ring", (sp.R_DATA_IN, sync_out), 18),
             ("bullseye", (0.0, sp.R_BULLSEYE), 78)]
    tx = 20 + ts + 60
    rows = np.linspace(80, H - 80, len(marks))
    for (label, (r0, r1), ang), ty in zip(marks, rows):
        a = np.deg2rad(ang)
        u = np.array([np.cos(a), np.sin(a)])          # radial direction
        n = np.array([-np.sin(a), np.cos(a)])         # perpendicular
        p0, p1 = (c + R * r0 * u), (c + R * r1 * u)
        cv2.line(canvas, tuple(p0.astype(int)), tuple(p1.astype(int)), red, 2,
                 cv2.LINE_AA)
        for p in (p0, p1):                            # end ticks
            t0, t1 = (p - 7 * n).astype(int), (p + 7 * n).astype(int)
            cv2.line(canvas, tuple(t0), tuple(t1), red, 2, cv2.LINE_AA)
        elbow = (tx - 24, int(ty) - 7)
        lead = (p1 + 6 * u).astype(int)
        cv2.line(canvas, tuple(lead), elbow, INK, 1, cv2.LINE_AA)
        cv2.line(canvas, elbow, (tx - 8, elbow[1]), INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, int(ty)), FONT, 0.75, (0, 0, 0), 1,
                    cv2.LINE_AA)
    save("anatomy.png", canvas)


# ---------------------------------------------------------------- detection
def make_detection():
    entries = json.load(open(os.path.join(ROOT, "fixtures", "frames.json")))["entries"]
    e = next(x for x in entries if "multitag_mixed6" in x["file"])
    gray = cv2.imread(os.path.join(ROOT, "fixtures", e["file"]), 0)
    K = np.array(e["K"]).reshape(3, 3)
    dets = detect.detect_markers(gray, K=K)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def project(X):
        x = K @ X
        return (int(round(x[0] / x[2])), int(round(x[1] / x[2])))

    for d in dets:
        cx, cy = d["center"]
        ax, ay = d["axes"]
        cv2.ellipse(img, (int(cx), int(cy)), (int(ax / 2), int(ay / 2)),
                    d["angle"], 0, 360, (0, 200, 0), 2, cv2.LINE_AA)
        R, t = np.array(d["R"]), np.array(d["t"])
        o = project(t)
        for axis, color in ((0, (60, 60, 230)), (1, (60, 200, 60)),
                            (2, (230, 130, 40))):
            v = np.zeros(3)
            v[axis] = 0.6
            cv2.line(img, o, project(R @ v + t), color, 3, cv2.LINE_AA)
        if d["decoded"]:
            label = f'{d["variant"]} 0x{d["value"]:X}' if d["mode"] == "ID" \
                else f'{d["variant"]} {d["value"]}'
            org = (int(cx - ax / 2), int(cy - ay / 2) - 12)
            cv2.putText(img, label, org, FONT, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, label, org, FONT, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
    save("detection.png", img)


# ---------------------------------------------------------------- tag size
def make_tag_size():
    ts = 560
    W, H = 880, ts + 100
    canvas = np.full((H, W, 3), 255, np.uint8)
    x0, y0 = (W - ts) // 2, 24
    canvas[y0:y0 + ts, x0:x0 + ts] = tag_bgr(M_SPEC, 0x1234, ts)
    cx, cy = x0 + ts / 2, y0 + ts / 2
    R = (ts / 2) * (1 - MARGIN)
    red = (0, 0, 200)

    a, b = (int(cx - R), int(cy)), (int(cx + R), int(cy))
    cv2.arrowedLine(canvas, a, b, red, 2, cv2.LINE_AA, tipLength=0.025)
    cv2.arrowedLine(canvas, b, a, red, 2, cv2.LINE_AA, tipLength=0.025)
    text_center(canvas, "tag size", cx, y0 + ts + 48, 0.85, red, 2)

    qa, qb = (int(cx), int(cy - R)), (int(cx), y0)
    cv2.arrowedLine(canvas, qa, qb, INK, 1, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(canvas, qb, qa, INK, 1, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(canvas, "quiet zone", (int(cx) + 12, y0 + 26), FONT, 0.6, INK,
                1, cv2.LINE_AA)
    save("tag_size.png", canvas)


if __name__ == "__main__":
    make_variants()
    make_anatomy()
    make_detection()
    make_tag_size()
