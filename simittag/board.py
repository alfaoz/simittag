"""
Calibration board definitions.

A board is a set of markers at known planar positions (mm), used to solve for
camera intrinsics (see simittag.calibrate). Board coordinates: origin at the
center of tag 0 (top-left), x right, y down — matching image conventions.

Boards are self-describing: each printed sheet carries one D-variant tag whose
RAW payload encodes the board parameters, so the calibrator can configure
itself from the sheet alone. The studio (web/) also writes a JSON sidecar with
explicit per-tag positions; when available it is the preferred source of truth
(it also locates the descriptor tag itself, adding one more point per view).

Descriptor RAW payload (8 bytes, version 1):

  [0] version (1)
  [1] family  (1 = grid, 2 = multiscale)
  [2:4] pitch  in 0.1 mm, big-endian   (grid: cell pitch; multiscale: frame step)
  [4:6] tag diameter in 0.1 mm, big-endian
  [6] rows    (grid: rows; multiscale: side tags per column)
  [7] cols    (grid: cols; multiscale: tags per top/bottom row)

Layout algorithms per family are frozen here and mirrored in web/js/calib.js;
a descriptor fully determines every tag position.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

DESCRIPTOR_VERSION = 1
FAMILY_GRID = 1
FAMILY_MULTISCALE = 2
FAMILY_NAMES = {FAMILY_GRID: "grid", FAMILY_MULTISCALE: "multiscale"}

# multiscale layout constants (frozen, v1): the center anchor is an M tag with
# this id, ANCHOR_RATIO times the perimeter tag diameter; side tags are spaced
# SIDE_STEP_RATIO times the top/bottom step.
MULTISCALE_ANCHOR_ID = 500
ANCHOR_RATIO = 3.5
SIDE_STEP_RATIO = 1.6


@dataclass
class BoardTag:
    """One marker at a known board position (center, mm)."""
    variant: str          # "T" | "M" | "D"
    mode: str             # "ID" | "RAW"
    value: object         # int for ID, bytes for RAW
    x_mm: float
    y_mm: float
    diameter_mm: float


@dataclass
class Board:
    family: int
    pitch_mm: float
    diameter_mm: float
    rows: int
    cols: int
    tags: list = field(default_factory=list)   # [BoardTag]

    def point_for(self, variant, mode, value):
        """Board (x, y) for a detection, or None if it is not on this board."""
        for t in self.tags:
            if t.variant == variant and t.mode == mode and t.value == value:
                return (t.x_mm, t.y_mm)
        return None

    @property
    def descriptor_raw(self) -> bytes:
        return pack_descriptor(self.family, self.pitch_mm, self.diameter_mm,
                               self.rows, self.cols)


def pack_descriptor(family, pitch_mm, diameter_mm, rows, cols) -> bytes:
    p, d = round(pitch_mm * 10), round(diameter_mm * 10)
    if not (0 < p < 65536 and 0 < d < 65536 and 0 < rows < 256 and 0 < cols < 256):
        raise ValueError("board parameters out of descriptor range")
    return bytes([DESCRIPTOR_VERSION, family, p >> 8, p & 0xFF,
                  d >> 8, d & 0xFF, rows, cols])


def unpack_descriptor(raw: bytes) -> dict:
    if len(raw) < 8 or raw[0] != DESCRIPTOR_VERSION:
        raise ValueError(f"not a v{DESCRIPTOR_VERSION} board descriptor")
    family = raw[1]
    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown board family {family}")
    return {"family": family,
            "pitch_mm": ((raw[2] << 8) | raw[3]) / 10.0,
            "diameter_mm": ((raw[4] << 8) | raw[5]) / 10.0,
            "rows": raw[6], "cols": raw[7]}


def grid_board(pitch_mm, diameter_mm, rows, cols) -> Board:
    """Uniform rows x cols grid of ID tags, row-major ids from 0.
    Variant T while ids fit in one byte, M beyond."""
    variant = "T" if rows * cols <= 256 else "M"
    b = Board(FAMILY_GRID, pitch_mm, diameter_mm, rows, cols)
    for i in range(rows * cols):
        b.tags.append(BoardTag(variant, "ID", i,
                               (i % cols) * pitch_mm, (i // cols) * pitch_mm,
                               diameter_mm))
    return b


def multiscale_board(step_mm, diameter_mm, side_rows, top_cols) -> Board:
    """Perimeter frame of small T tags + one large M anchor in the center.
    Top and bottom rows have `top_cols` tags at `step_mm` pitch; each side has
    `side_rows` tags between them at SIDE_STEP_RATIO * step_mm pitch."""
    b = Board(FAMILY_MULTISCALE, step_mm, diameter_mm, side_rows, top_cols)
    w = (top_cols - 1) * step_mm
    sstep = step_mm * SIDE_STEP_RATIO
    h = (side_rows + 1) * sstep
    i = 0
    for c in range(top_cols):                        # top + bottom rows
        b.tags.append(BoardTag("T", "ID", i, c * step_mm, 0.0, diameter_mm)); i += 1
        b.tags.append(BoardTag("T", "ID", i, c * step_mm, h, diameter_mm)); i += 1
    for r in range(1, side_rows + 1):                # left + right columns
        b.tags.append(BoardTag("T", "ID", i, 0.0, r * sstep, diameter_mm)); i += 1
        b.tags.append(BoardTag("T", "ID", i, w, r * sstep, diameter_mm)); i += 1
    b.tags.append(BoardTag("M", "ID", MULTISCALE_ANCHOR_ID,
                           w / 2.0, h / 2.0, diameter_mm * ANCHOR_RATIO))
    return b


def board_from_descriptor(raw: bytes) -> Board:
    d = unpack_descriptor(raw)
    if d["family"] == FAMILY_GRID:
        return grid_board(d["pitch_mm"], d["diameter_mm"], d["rows"], d["cols"])
    return multiscale_board(d["pitch_mm"], d["diameter_mm"], d["rows"], d["cols"])


def load_board(path) -> Board:
    """Load a studio JSON sidecar (explicit tag positions, preferred)."""
    j = json.load(open(path))
    if j.get("simittag_board") != 1:
        raise ValueError(f"{path} is not a simittag board file")
    family = {v: k for k, v in FAMILY_NAMES.items()}[j["family"]]
    b = Board(family, j["pitch_mm"], j["diameter_mm"], j["rows"], j["cols"])
    for t in j["tags"]:
        value = t["value"]
        if t["mode"] == "RAW":
            value = bytes.fromhex(value)
        b.tags.append(BoardTag(t["variant"], t["mode"], value,
                               t["x_mm"], t["y_mm"], t["diameter_mm"]))
    return b


def find_board(detections) -> Board | None:
    """Reconstruct the board from a descriptor tag among detections, if any."""
    for r in detections:
        if r["variant"] == "D" and r["mode"] == "RAW":
            try:
                return board_from_descriptor(bytes(r["value"]))
            except ValueError:
                continue
    return None
