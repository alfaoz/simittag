"""
Vector (SVG) renderer for Simittag markers, for clean printing at an exact physical
size. Geometry matches marker/generate.py (the raster renderer) cell-for-cell, which
in turn matches the detector's sampling convention (simittag.spec / simittag.codec):

  angle:  theta = atan2(y, x) in screen coords (y DOWN), sector s spans [s*step,(s+1)*step)
  radius: normalized so the outer ring's outer edge = 1.0
  ring r spans [R_DATA_IN + r*ring_w, R_DATA_IN + (r+1)*ring_w), grid[r,s]==1 => black

Painter's order (no even-odd needed): white bg -> black outer disk -> white disk at
R_RING_IN (carves the ring) -> black bullseye -> black data cells.
"""
from __future__ import annotations
import math

from simittag.spec import MarkerSpec, DEFAULT
from simittag import codec


def _pt(cx, cy, rho, ang, R):
    return (cx + R * rho * math.cos(ang), cy + R * rho * math.sin(ang))


def _sector_path(cx, cy, ri, ro, a0, a1, R):
    """Annular-sector path (one data cell), radii normalized, angles in radians."""
    x0o, y0o = _pt(cx, cy, ro, a0, R)
    x1o, y1o = _pt(cx, cy, ro, a1, R)
    x1i, y1i = _pt(cx, cy, ri, a1, R)
    x0i, y0i = _pt(cx, cy, ri, a0, R)
    Ro, Ri = ro * R, ri * R
    return (f"M{x0o:.3f},{y0o:.3f} A{Ro:.3f},{Ro:.3f} 0 0 1 {x1o:.3f},{y1o:.3f} "
            f"L{x1i:.3f},{y1i:.3f} A{Ri:.3f},{Ri:.3f} 0 0 0 {x0i:.3f},{y0i:.3f} Z")


def marker_svg_body(grid, spec: MarkerSpec, cx, cy, R):
    """SVG element string for one marker centered at (cx,cy) with outer radius R (units)."""
    step = 2 * math.pi / spec.SECTOR_COUNT
    ring_w = (spec.R_DATA_OUT - spec.R_DATA_IN) / spec.RING_COUNT
    out = []
    # black outer disk, then white disk to leave the outer ring annulus
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{R:.3f}" fill="#000"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{R*spec.R_RING_IN:.3f}" fill="#fff"/>')
    # bullseye
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{R*spec.R_BULLSEYE:.3f}" fill="#000"/>')
    # data cells
    paths = []
    for ring in range(spec.RING_COUNT):
        ri = spec.R_DATA_IN + ring * ring_w
        ro = ri + ring_w
        for s in range(spec.SECTOR_COUNT):
            if grid[ring, s] == 1:
                paths.append(_sector_path(cx, cy, ri, ro, s * step, (s + 1) * step, R))
    if paths:
        out.append(f'<path d="{" ".join(paths)}" fill="#000"/>')
    return "\n".join(out)


def marker_svg(payload: bytes, spec: MarkerSpec = DEFAULT, size_mm: float = 40.0,
               margin: float = 0.12, label: str = None) -> str:
    """
    Standalone SVG (sized in mm) for one marker. `size_mm` is the marker diameter
    (the outer ring's outer edge); a white quiet zone of `margin` is added around it.
    Optional `label` text is drawn under the quiet zone.
    """
    grid = codec.encode(payload, spec)
    R = size_mm / 2.0
    pad = R * margin
    W = size_mm + 2 * pad
    cx = cy = W / 2.0
    label_h = max(3.0, size_mm * 0.10) if label else 0.0
    H = W + label_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}mm" height="{H:.2f}mm" '
        f'viewBox="0 0 {W:.3f} {H:.3f}">',
        f'<rect x="0" y="0" width="{W:.3f}" height="{H:.3f}" fill="#fff"/>',
        marker_svg_body(grid, spec, cx, cy, R),
    ]
    if label:
        fs = label_h * 0.62
        parts.append(
            f'<text x="{cx:.3f}" y="{W + label_h*0.72:.3f}" font-family="monospace" '
            f'font-size="{fs:.2f}" text-anchor="middle" fill="#000">{_esc(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---- printable sheet ----
PAGES_MM = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}


def sheet_svg(items, page="A4", size_mm=40.0, margin_mm=12.0, gap_mm=8.0,
              cut_marks=True):
    """
    Lay markers out on a printable page (SVG sized in mm). `items` = list of
    (payload_bytes, spec, label). Auto-flows into a grid; returns one SVG string.
    Print at 100% scale (no fit-to-page) for the physical size to be exact.
    """
    pw, ph = PAGES_MM.get(page, PAGES_MM["A4"])
    cell = size_mm * 1.12  # marker + its 12% quiet zone
    label_h = max(3.0, size_mm * 0.10)
    cell_h = cell + label_h + gap_mm
    cell_w = cell + gap_mm
    cols = max(1, int((pw - 2 * margin_mm + gap_mm) // cell_w))
    rows = max(1, int((ph - 2 * margin_mm + gap_mm) // cell_h))
    per_page = cols * rows

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw}mm" height="{ph}mm" '
        f'viewBox="0 0 {pw} {ph}">',
        f'<rect x="0" y="0" width="{pw}" height="{ph}" fill="#fff"/>',
    ]
    # only the first page (browsers print one SVG page; multi-page is a future nicety)
    for i, (payload, spec, label) in enumerate(items[:per_page]):
        r, c = divmod(i, cols)
        x0 = margin_mm + c * cell_w
        y0 = margin_mm + r * cell_h
        grid = codec.encode(payload, spec)
        R = size_mm / 2.0
        cx = x0 + cell / 2.0
        cy = y0 + cell / 2.0
        if cut_marks:
            parts.append(_cut_marks(x0, y0, cell, cell))
        parts.append(f'<g>{marker_svg_body(grid, spec, cx, cy, R)}</g>')
        fs = label_h * 0.6
        parts.append(
            f'<text x="{cx:.3f}" y="{y0 + cell + label_h*0.7:.3f}" '
            f'font-family="monospace" font-size="{fs:.2f}" text-anchor="middle" '
            f'fill="#000">{_esc(label)}</text>')
    extra = max(0, len(items) - per_page)
    if extra:
        parts.append(
            f'<text x="{pw/2:.1f}" y="{ph-4:.1f}" font-family="monospace" '
            f'font-size="3.5" text-anchor="middle" fill="#888">'
            f'+{extra} more (reduce size or count to fit one page)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _cut_marks(x, y, w, h, m=2.0):
    L = []
    for (px, py, dx, dy) in [(x, y, 1, 0), (x, y, 0, 1),
                             (x + w, y, -1, 0), (x + w, y, 0, 1),
                             (x, y + h, 1, 0), (x, y + h, 0, -1),
                             (x + w, y + h, -1, 0), (x + w, y + h, 0, -1)]:
        L.append(f'<line x1="{px:.2f}" y1="{py:.2f}" x2="{px+dx*m:.2f}" '
                 f'y2="{py+dy*m:.2f}" stroke="#bbb" stroke-width="0.2"/>')
    return "".join(L)


if __name__ == "__main__":
    from simittag import payload as _p
    svg = marker_svg(_p.encode_id(0x1234, DEFAULT), DEFAULT, size_mm=40,
                     label="SIMITTAG-s16m 0x1234")
    open("/tmp/marker.svg", "w").write(svg)
    print("wrote /tmp/marker.svg", len(svg), "bytes")
