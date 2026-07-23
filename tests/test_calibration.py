"""Calibration: board descriptors, geometry, and synthetic end-to-end solve."""
import numpy as np
import cv2
import pytest

from simittag import board as board_mod
from simittag import payload
from simittag.spec import VARIANTS
from simittag.calibrate import calibrate, CameraIntrinsics
from marker.generate import render


def test_descriptor_roundtrip():
    # v1 (sim48c8 boards): byte-frozen 8-byte form, no variant byte
    raw = board_mod.pack_descriptor(board_mod.FAMILY_GRID, 30.0, 22.0, 8, 6,
                                    variant="sim48c8")
    assert len(raw) == 8 and raw[0] == 1
    d = board_mod.unpack_descriptor(raw)
    assert d == {"family": 1, "pitch_mm": 30.0, "diameter_mm": 22.0,
                 "rows": 8, "cols": 6, "variant": "sim48c8"}
    # v2 (default s4k boards): trailing variant code
    raw2 = board_mod.pack_descriptor(board_mod.FAMILY_GRID, 30.0, 22.0, 8, 6)
    assert len(raw2) == 9 and raw2[0] == 2 and raw2[1:8] == raw[1:8]
    d2 = board_mod.unpack_descriptor(raw2)
    assert d2["variant"] == "sim48c12"
    assert {k: d2[k] for k in d if k != "variant"} == \
        {k: d[k] for k in d if k != "variant"}
    # both descriptor forms must fit an sdata RAW body (max 9 bytes)
    for r in (raw, raw2):
        pl = payload.encode_raw(r, VARIANTS["D"])
        mode, value = payload.decode(pl, VARIANTS["D"])
        assert (mode, value) == ("RAW", r)


def test_board_from_descriptor_variants():
    for variant in ("sim48c8", "sim48c12", "sim48c16"):
        b = board_mod.grid_board(30.0, 22.0, 4, 4, variant=variant)
        b2 = board_mod.board_from_descriptor(b.descriptor_raw)
        assert b2.tags[0].variant == variant


def test_grid_board_geometry():
    b = board_mod.grid_board(30.0, 22.0, 8, 6)
    assert len(b.tags) == 48 and b.tags[0].variant == "sim48c12"  # s4k default
    assert b.point_for("s4k", "ID", 0) == (0.0, 0.0)      # alias accepted
    assert b.point_for("sim48c12", "ID", 7) == (30.0, 30.0)   # row 1, col 1
    assert b.point_for("s4k", "ID", 47) == (150.0, 210.0)     # row 7, col 5
    assert b.point_for("s4k", "ID", 48) is None
    # explicit legacy variant still generates s256 boards (deprecated letter)
    legacy = board_mod.grid_board(30.0, 22.0, 8, 6, variant="T")
    assert legacy.tags[0].variant == "sim48c8"
    assert legacy.point_for("T", "ID", 7) == (30.0, 30.0)
    # beyond the s4k ID space (4096) auto-switches to variant s16m
    big = board_mod.grid_board(30.0, 22.0, 65, 65)
    assert big.tags[0].variant == "sim96c32"


def test_multiscale_board_geometry():
    b = board_mod.multiscale_board(27.0, 20.0, 4, 7)
    # 7 top + 7 bottom + 4 left + 4 right + anchor
    assert len(b.tags) == 23
    assert b.tags[0].variant == "sim48c12"    # s4k perimeter by default
    anchor = b.point_for("M", "ID", board_mod.MULTISCALE_ANCHOR_ID)
    w = 6 * 27.0
    h = 5 * 27.0 * board_mod.SIDE_STEP_RATIO
    assert anchor == (w / 2, h / 2)
    assert b.tags[-1].diameter_mm == 20.0 * board_mod.ANCHOR_RATIO


def test_js_board_json_parity():
    """The web generator's sidecar must agree with board.py tag-for-tag.
    These sidecars predate the s4k default; they are the pinned LEGACY
    loader coverage, so the reference boards pin variant sim48c8."""
    import pathlib
    fixtures = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
    for path, make in [
        ("board_grid_a4.json",
         lambda: board_mod.grid_board(30.0, 22.0, 8, 6, variant="sim48c8")),
        ("board_multiscale_a4.json",
         lambda: board_mod.multiscale_board(27.0, 20.0, 4, 7,
                                            variant="sim48c8")),
    ]:
        loaded = board_mod.load_board(fixtures / path)
        ref = make()
        got = {(t.variant, t.mode, t.value): (t.x_mm, t.y_mm, t.diameter_mm)
               for t in loaded.tags if t.mode == "ID"}
        want = {(t.variant, t.mode, t.value): (t.x_mm, t.y_mm, t.diameter_mm)
                for t in ref.tags}
        assert got == want, path
        # the sidecar also locates the descriptor tag, with the right payload
        descs = [t for t in loaded.tags if t.mode == "RAW"]
        assert len(descs) == 1 and descs[0].value == ref.descriptor_raw


def test_board_from_descriptor_matches_generator():
    b = board_mod.grid_board(30.0, 22.0, 8, 6)
    b2 = board_mod.board_from_descriptor(b.descriptor_raw)
    assert [(t.variant, t.mode, t.value, t.x_mm, t.y_mm) for t in b.tags] == \
           [(t.variant, t.mode, t.value, t.x_mm, t.y_mm) for t in b2.tags]


# ---------------------------------------------------------------------------
# synthetic end-to-end: render a board, image it with a known camera from
# several poses, and require calibrate() to recover the intrinsics.
# ---------------------------------------------------------------------------

PX_PER_MM = 4.0


def _board_raster(board):
    """White canvas with each tag rendered at its board position."""
    xs = [t.x_mm for t in board.tags]
    ys = [t.y_mm for t in board.tags]
    rmax = max(t.diameter_mm for t in board.tags) / 2 * 1.4
    x0, y0 = min(xs) - rmax, min(ys) - rmax
    w = int(round((max(xs) - min(xs) + 2 * rmax) * PX_PER_MM))
    h = int(round((max(ys) - min(ys) + 2 * rmax) * PX_PER_MM))
    img = np.full((h, w), 255, np.uint8)
    for t in board.tags:
        if t.mode == "RAW":
            pl = payload.encode_raw(t.value, VARIANTS[t.variant])
        else:
            pl = payload.encode_id(t.value, VARIANTS[t.variant])
        size = int(round(t.diameter_mm * 1.24 * PX_PER_MM))  # incl. 12% margin
        tile = render(pl, VARIANTS[t.variant], size=size, supersample=2)
        cx = (t.x_mm - x0) * PX_PER_MM
        cy = (t.y_mm - y0) * PX_PER_MM
        px, py = int(round(cx - size / 2)), int(round(cy - size / 2))
        img[py:py + size, px:px + size] = np.minimum(
            img[py:py + size, px:px + size], tile)
    return img, x0, y0


def _with_descriptor(board):
    """Append the sheet's sdata descriptor tag, like a studio sheet carries."""
    xs = [t.x_mm for t in board.tags]
    ys = [t.y_mm for t in board.tags]
    board.tags.append(board_mod.BoardTag(
        "sim180c88", "RAW", board.descriptor_raw,
        min(xs), max(ys) + 55.0, 36.0))
    return board


def _view(img, x0, y0, K, rvec, tvec, out_size):
    """Project the board plane (mm coords) through a pinhole camera."""
    R, _ = cv2.Rodrigues(rvec)
    # plane (X, Y, 0) -> camera: H = K [r1 r2 t]
    H_mm = K @ np.column_stack((R[:, 0], R[:, 1], tvec))
    # board raster px -> mm
    S = np.array([[1 / PX_PER_MM, 0, x0], [0, 1 / PX_PER_MM, y0], [0, 0, 1]])
    H = H_mm @ S
    return cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_synthetic_calibration_recovers_intrinsics():
    board = board_mod.grid_board(50.0, 40.0, 4, 3)
    img, x0, y0 = _board_raster(board)
    K_true = np.array([[900.0, 0, 640.0], [0, 900.0, 400.0], [0, 0, 1]])
    out = (1280, 800)
    views = []
    poses = [(0.00, 0.00, 0.0), (0.25, 0.10, 0.1), (-0.22, 0.18, -0.1),
             (0.15, -0.25, 0.2), (-0.10, -0.20, -0.2), (0.30, 0.25, 0.0)]
    for rx, ry, rz in poses:
        rvec = np.array([rx, ry, rz])
        tvec = np.array([-95.0, -110.0, 420.0])   # board roughly centered
        views.append(_view(img, x0, y0, K_true, rvec, tvec, out))
    intr = calibrate(views, board=board)
    assert intr.views >= 5
    assert abs(intr.fx - 900) / 900 < 0.02
    assert abs(intr.fy - 900) / 900 < 0.02
    assert abs(intr.cx - 640) < 25
    assert abs(intr.cy - 400) < 25
    assert intr.rms_px < 1.5
    # round-trips through JSON, matching the AprilTag-style field contract
    intr.save("/tmp/simittag-test-intrinsics.json")
    back = CameraIntrinsics.load("/tmp/simittag-test-intrinsics.json")
    assert back.fx == intr.fx and back.K[0, 2] == intr.cx


def _views_of(board):
    img, x0, y0 = _board_raster(board)
    K_true = np.array([[900.0, 0, 640.0], [0, 900.0, 400.0], [0, 0, 1]])
    out = (1280, 800)
    poses = [(0.00, 0.00, 0.0), (0.25, 0.10, 0.1), (-0.22, 0.18, -0.1),
             (0.15, -0.25, 0.2), (-0.10, -0.20, -0.2), (0.30, 0.25, 0.0)]
    return [_view(img, x0, y0, K_true, np.array([rx, ry, rz]),
                  np.array([-95.0, -110.0, 420.0]), out)
            for rx, ry, rz in poses]


def test_calibration_descriptor_self_config_s4k():
    # A generated (s4k-default) sheet with its v2 descriptor tag must
    # calibrate from photos alone: default BOARD_VERSIONS decodes the grid,
    # find_board reconstructs the layout from the descriptor.
    board = _with_descriptor(board_mod.grid_board(50.0, 40.0, 4, 3))
    assert board.tags[0].variant == "sim48c12"
    intr = calibrate(_views_of(board))
    assert intr.views >= 5
    assert abs(intr.fx - 900) / 900 < 0.02
    assert intr.rms_px < 1.5


def test_calibration_descriptor_self_config_legacy_s256():
    # A legacy sheet (s256 tags, v1 descriptor) under the s4k-default
    # BOARD_VERSIONS: the first pass decodes only the sdata descriptor; the
    # v1 semantics reconstruct an s256 board and trigger the re-detection
    # with the board's own variant set. The sheet must still calibrate.
    board = _with_descriptor(
        board_mod.grid_board(50.0, 40.0, 4, 3, variant="sim48c8"))
    assert board.tags[0].variant == "sim48c8"
    assert board.descriptor_raw[0] == 1
    intr = calibrate(_views_of(board))
    assert intr.views >= 5
    assert abs(intr.fx - 900) / 900 < 0.02
    assert intr.rms_px < 1.5


def test_s4k_board_sidecar_fixture():
    # The committed s4k sidecar (studio format, canonical names) must load
    # and agree with board.py's s4k default generator tag-for-tag.
    import pathlib
    fixtures = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
    loaded = board_mod.load_board(fixtures / "board_grid_s4k_a4.json")
    ref = board_mod.grid_board(30.0, 22.0, 8, 6)
    got = {(t.variant, t.mode, t.value): (t.x_mm, t.y_mm, t.diameter_mm)
           for t in loaded.tags if t.mode == "ID"}
    want = {(t.variant, t.mode, t.value): (t.x_mm, t.y_mm, t.diameter_mm)
            for t in ref.tags}
    assert got == want
    descs = [t for t in loaded.tags if t.mode == "RAW"]
    assert len(descs) == 1 and descs[0].value == ref.descriptor_raw
    assert descs[0].value[0] == 2      # v2 descriptor carrying sim48c12


def test_calibrate_rejects_too_few_views():
    board = board_mod.grid_board(50.0, 40.0, 4, 3)
    img, x0, y0 = _board_raster(board)
    K_true = np.array([[900.0, 0, 640.0], [0, 900.0, 400.0], [0, 0, 1]])
    v = _view(img, x0, y0, K_true, np.zeros(3), np.array([-95.0, -110.0, 420.0]),
              (1280, 800))
    with pytest.raises(ValueError):
        calibrate([v, v], board=board)
