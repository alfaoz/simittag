"""Independent regression tests for public workflows and calibrated accept gates."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from marker.generate import render
from simittag import codec, detect, payload, pose
from simittag.spec import VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def _rot_xyz(tilt_deg, pan_deg, roll_deg):
    """Marker->camera rotation, the sim rig's convention (Rz @ Ry @ Rx)."""
    tx, ty, tz = np.radians([tilt_deg, pan_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)],
                   [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0],
                   [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0],
                   [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _gt_tilt(R):
    normal = R @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))


def _fov_K(out, fov_deg):
    f = (out / 2) / np.tan(np.radians(fov_deg) / 2)
    c = (out - 1) / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])


def _render_pose(spec, value, x, y, z, tilt, pan, roll, K, out, noise_seed):
    """Project a marker at an arbitrary 6-DoF pose (the sim rig's model: the
    marker image spans a 1x1 world square) with light degradation."""
    marker = render(payload.encode_id(value, spec), spec, size=512)
    R = _rot_xyz(tilt, pan, roll)
    t = np.array([x, y, z], dtype=np.float64)
    size = marker.shape[0]
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    world = np.float32([[u / size - 0.5, v / size - 0.5, 0] for u, v in src])
    cam = (R @ world.T).T + t
    proj = (K @ cam.T).T
    dst = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(marker, H, (out, out), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255).astype(np.float32)
    rng = np.random.default_rng(noise_seed)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), R


def _unsorted_mirror_tilt(gray, K, spec):
    """Emulate the pre-run-4 headless path (R3.13): conic-order Hs, no
    bullseye origin_err sort, first decode wins. Used to keep the pose-mirror
    pins live: the fixture must still discriminate sorted from unsorted."""
    for (ei, _inner, _alt, small), g, _inv in detect._candidate_views(gray, K):
        if small:
            continue
        geom0 = (ei[0], ei[1], ei[2])
        geom1 = detect._refine_ellipse(g, geom0)
        Hs = pose.pose_homographies(geom1, K)
        if geom1 is not geom0 and max(geom1[1]) < 100:
            Hs = Hs + pose.pose_homographies(geom0, K)
        decoded, H = detect._try_decode_spec(g, Hs, spec, 0.25)
        if decoded is not None:
            return pose.tilt_from_H(H, K)
    return None


def _naive_render(payload_bytes, spec, size, supersample, margin):
    """Small, direct definition used to verify the tiled production renderer."""
    grid = codec.encode(payload_bytes, spec)
    sample_size = size * supersample
    center = (sample_size - 1) / 2.0
    radius_px = (sample_size / 2.0) * (1.0 - margin)
    ys, xs = np.mgrid[0:sample_size, 0:sample_size]
    dx = (xs - center) / radius_px
    dy = (ys - center) / radius_px
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    image = np.ones((sample_size, sample_size), dtype=np.float32)
    step = 2 * np.pi / spec.SECTOR_COUNT
    ring_width = (spec.R_DATA_OUT - spec.R_DATA_IN) / spec.RING_COUNT
    image[(radius >= spec.R_RING_IN) & (radius <= 1.0)] = 0.0
    image[radius <= spec.R_BULLSEYE] = 0.0
    data = (radius >= spec.R_DATA_IN) & (radius < spec.R_DATA_OUT)
    rings = np.clip(((radius - spec.R_DATA_IN) / ring_width).astype(int),
                    0, spec.RING_COUNT - 1)
    sectors = np.clip((theta / step).astype(int), 0, spec.SECTOR_COUNT - 1)
    image[data & (grid[rings, sectors] == 1)] = 0.0
    image = image.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return (image * 255).astype(np.uint8)


def _radial_clutter_frames(indices=(83, 99)):
    """Two deterministic near-marker negatives that scored 0.6503/0.6536."""
    wanted = set(indices)
    rng = np.random.default_rng(731)
    for frame_index in range(max(wanted) + 1):
        image = np.clip(rng.normal(220, 8, (360, 480)), 0, 255).astype(np.uint8)
        for _ in range(8):
            center = (int(rng.integers(45, 435)), int(rng.integers(45, 315)))
            outer = int(rng.integers(14, 48))
            colors = rng.choice([0, 255], 5)
            for fraction, color in zip((1, .82, .58, .38, .2), colors):
                cv2.circle(image, center, max(1, int(outer * fraction)), int(color), -1)
        for _ in range(8):
            p1 = (int(rng.integers(0, 480)), int(rng.integers(0, 360)))
            p2 = (int(rng.integers(0, 480)), int(rng.integers(0, 360)))
            cv2.line(image, p1, p2, int(rng.integers(0, 256)),
                     int(rng.integers(1, 7)))
        if frame_index in wanted:
            yield frame_index, image


class DetectorRegressionTests(unittest.TestCase):
    def test_outer_ring_occlusion_recovers_from_bullseye(self):
        image = cv2.imread(str(ROOT / "fixtures/frames/multitag_occl.png"),
                           cv2.IMREAD_GRAYSCALE)
        for inverted, frame in ((False, image), (True, 255 - image)):
            with self.subTest(inverted=inverted):
                results = detect.detect(frame)
                decoded = sorted((r["variant"], r["mode"], r["value"])
                                 for r in results)
                self.assertEqual(decoded, [
                    ("sim96c32", "ID", 170), ("sim96c32", "ID", 187),
                    ("sim96c32", "ID", 204), ("sim96c32", "ID", 221),
                ])
                self.assertTrue(all(r["alias"] == "s16m" for r in results))
                self.assertTrue(all(r["inverted"] == inverted for r in results))

    def test_inverted_tags_preserve_range_fixtures(self):
        cases = (
            ("range_T_z11_t25.png", "sim48c8", "ID", 42),
            ("range_M_z8_t25.png", "sim96c32", "ID", 0xABCDEF),
            ("range_D_z7_t00.png", "sim180c88", "GEO",
             (48.85837000000001, 2.2944809999999904, 330)),
        )
        for filename, variant, mode, value in cases:
            with self.subTest(variant=variant):
                image = cv2.imread(str(ROOT / "fixtures/frames" / filename),
                                   cv2.IMREAD_GRAYSCALE)
                # s256 left the default auto set in run 3; select explicitly
                results = detect.detect(255 - image, versions=variant)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertEqual((result["variant"], result["mode"], result["value"]),
                                 (variant, mode, value))
                self.assertTrue(result["inverted"])

    def test_mixed_polarity_scene(self):
        spec = VARIANTS["M"]
        size = 280
        canvas = np.full((320, 600), 127, dtype=np.uint8)
        normal = render(payload.encode_id(111, spec), spec, size=size)
        inverted = 255 - render(payload.encode_id(222, spec), spec, size=size)
        canvas[20:300, 10:290] = normal
        canvas[20:300, 310:590] = inverted
        results = detect.detect(canvas)
        decoded = sorted((r["value"], r["inverted"]) for r in results)
        self.assertEqual(decoded, [(111, False), (222, True)])

    def test_bullseye_fallback_is_not_angle_specific(self):
        size = 360
        margin = 0.10
        value = 0xABC123
        spec = VARIANTS["M"]
        base = render(payload.encode_id(value, spec), spec, size=size,
                      margin=margin)
        ys, xs = np.mgrid[:size, :size]
        center = (size - 1) / 2.0
        radius = size / 2.0 * (1.0 - margin)

        # Remove a cap from four sides. This opens the outer-ring contour but
        # stops well before the bullseye; the fallback must recover the known
        # outer scale from the surviving central conic in every orientation.
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                radians = np.radians(angle)
                along = ((xs - center) * np.cos(radians)
                         + (ys - center) * np.sin(radians)) / radius
                image = base.copy()
                image[along < -0.65] = 255
                image = cv2.GaussianBlur(image, (0, 0), 0.7)
                # versions="M" exercises the deprecated-letter input mapping
                decoded = [(r["variant"], r["mode"], r["value"])
                           for r in detect.detect(image, versions="M")]
                self.assertEqual(decoded, [("sim96c32", "ID", value)])

    def test_s64k_deconv_only_fixture(self):
        # Floor-region nibble-variant tags that decode ONLY through the
        # Wiener ISI retry. Pins the deconv path on the fixture bytes.
        cases = (("isiretry_S64K_z12.png", "s64k", "sim48c16", 0xBEEF),
                 ("isiretry_S4K_z11.png", "s4k", "sim48c12", 0xABC))
        for filename, pin, variant, value in cases:
            with self.subTest(frame=filename):
                image = cv2.imread(str(ROOT / "fixtures/frames" / filename),
                                   cv2.IMREAD_GRAYSCALE)
                results = detect.detect(image, versions=pin)
                self.assertEqual([(r["variant"], r["value"]) for r in results],
                                 [(variant, value)])
                old = detect.DECONV_SMALL
                try:
                    detect.DECONV_SMALL = False
                    self.assertEqual(detect.detect(image, versions=pin), [])
                finally:
                    detect.DECONV_SMALL = old

    def test_same_grid_variants_cross_reject(self):
        # s256 and s64k share the 3x16 grid; disambiguation rests on sync +
        # codec + verify only. A tag of one pinned to the other's decoder
        # must never decode (the field-safety property for printed tags).
        cases = (("cross_T_as_S64K.png", "s64k"),
                 ("cross_S64K_as_T.png", "s256"),
                 ("cross_T_as_S4K.png", "s4k"),
                 ("cross_S4K_as_T.png", "s256"),
                 ("cross_S64K_as_S4K.png", "s4k"),
                 ("cross_S4K_as_S64K.png", "s64k"))
        for filename, wrong_variant in cases:
            with self.subTest(frame=filename):
                image = cv2.imread(str(ROOT / "fixtures/frames" / filename),
                                   cv2.IMREAD_GRAYSCALE)
                self.assertEqual(detect.detect(image, versions=wrong_variant),
                                 [])

    def test_s64k_verify_floor_rejects_cross_leak(self):
        # The one cross-decode found in ~43k matrix/stress trials: an INVERTED
        # s256 tag at 40px whose deconvolved, misregistered view self-
        # consistently decodes as s64k id 21701 with verify corr 0.759 --
        # inside the clutter-calibrated margin but below sim48c16's raised
        # per-variant floor (spec.VERIFY_MIN = 0.78). The frame must not
        # decode, AND the RS+CRC survivor must still appear in the tightened
        # band, proving this pin exercises the per-variant gate (not vacuous).
        image = cv2.imread(str(ROOT / "tests/data/crossleak_invT_px40.png"),
                           cv2.IMREAD_GRAYSCALE)
        K = detect.default_K(1080, 1080)
        K[0, 0] = K[1, 1] = (1920 / 2) / np.tan(np.radians(60.0) / 2)
        detect._VERIFY_LOG = log = []
        try:
            results = detect.detect(image, K=K, versions="s64k")
        finally:
            detect._VERIFY_LOG = None
        self.assertEqual(results, [])
        survivors = [c for c, v, _ in log if v == "sim48c16"]
        self.assertTrue(any(detect.VERIFY_MIN <= c < 0.78 for c in survivors),
                        f"survivor no longer in the tightened band: {survivors}")

    def test_s256_integrity_config_rejects_wrong_id(self):
        # A 19px s256 tag (true ID 15) that the pre-run-3 config wrongly
        # decoded as ID 3 (erasure-path codeword collision below the reliable
        # floor). The shipped config (CONF_ERASURE=0.0 + VERIFY_MIN=0.76)
        # must reject it; restoring the old knobs must reproduce the wrong
        # accept, proving the pin exercises the config (not vacuous).
        import dataclasses
        from simittag import spec as specmod
        image = cv2.imread(str(ROOT / "tests/data/wrongid_T_px19.png"),
                           cv2.IMREAD_GRAYSCALE)
        K = detect.default_K(1080, 1080)
        K[0, 0] = K[1, 1] = (1920 / 2) / np.tan(np.radians(60.0) / 2)
        self.assertEqual(detect.detect(image, K=K, versions="s256"), [])
        legacy = dataclasses.replace(specmod.T_SPEC,
                                     CONF_ERASURE=None, VERIFY_MIN=None)
        dict.__setitem__(specmod.VARIANTS, "sim48c8", legacy)
        try:
            wrong = [r["value"] for r in
                     detect.detect(image, K=K, versions="s256")]
        finally:
            dict.__setitem__(specmod.VARIANTS, "sim48c8", specmod.T_SPEC)
        self.assertEqual(wrong, [3])

    def test_s256_noeras_recall_fixture(self):
        # The fixture frame that decodes ONLY under the shipped s256 config:
        # with ranked erasures re-enabled (legacy), RS(4,2) forfeits its
        # blind-correction budget and the decode fails.
        import dataclasses
        from simittag import spec as specmod
        image = cv2.imread(str(ROOT / "fixtures/frames/noeras_T_z13.png"),
                           cv2.IMREAD_GRAYSCALE)
        K = detect.default_K(1080, 1080)
        K[0, 0] = K[1, 1] = (1920 / 2) / np.tan(np.radians(60.0) / 2)
        results = detect.detect(image, K=K, versions="s256")
        self.assertEqual([(r["variant"], r["value"]) for r in results],
                         [("sim48c8", 0x2A)])
        legacy = dataclasses.replace(specmod.T_SPEC,
                                     CONF_ERASURE=None, VERIFY_MIN=None)
        dict.__setitem__(specmod.VARIANTS, "sim48c8", legacy)
        try:
            self.assertEqual(detect.detect(image, K=K, versions="s256"), [])
        finally:
            dict.__setitem__(specmod.VARIANTS, "sim48c8", specmod.T_SPEC)

    def test_offaxis_pose_mirror_fixture(self):
        # The R3.13 pin: an off-center healthy tag whose tilt+pan flip the
        # conic eigen-order. Headless detect() must pick the correct mirror
        # (bullseye origin_err sort, like detect_markers always did); the
        # unsorted emulation must still pick the WRONG one, proving the frame
        # discriminates and the pin stays live.
        image = cv2.imread(str(ROOT / "fixtures/frames/offaxis_M_pose.png"),
                           cv2.IMREAD_GRAYSCALE)
        K = _fov_K(1280, 55)
        gt = _gt_tilt(_rot_xyz(8.6, 11.2, 60.6))
        results = detect.detect(image, K=K, versions="sim96c32")
        self.assertEqual([(r["variant"], r["value"]) for r in results],
                         [("sim96c32", 0xBEE5)])
        self.assertLess(abs(results[0]["tilt_deg"] - gt), 1.0)
        markers = detect.detect_markers(image, K=K, versions="sim96c32")
        self.assertLess(abs(markers[0]["tilt_deg"] - gt), 1.0)
        wrong = _unsorted_mirror_tilt(image, K, VARIANTS["sim96c32"])
        self.assertIsNotNone(wrong)
        self.assertGreater(abs(wrong - gt), 5.0,
                           "unsorted emulation no longer flips the mirror; "
                           "the fixture has gone vacuous")

    def test_offaxis_pose_regression_slice(self):
        # mass1200-style slice: 24 deterministic off-axis poses biased into
        # the mirror-flip region (low tilt, strong pan, corner positions).
        # Pre-fix headless detect() takes the wrong mirror on ~2/24 of these
        # (p95 tilt error ~8 deg); the sorted path must stay sub-degree.
        spec = VARIANTS["sim96c32"]
        K = _fov_K(960, 55)
        rng = np.random.default_rng(41)
        errors = []
        for i in range(24):
            x = float(rng.choice([-0.85, 0.85]))
            y = float(rng.choice([-0.85, 0.0, 0.85]))
            tilt = float(rng.uniform(6, 22))
            pan = float(rng.choice([-1, 1]) * rng.uniform(7, 12))
            roll = float(rng.uniform(0, 360))
            gray, R = _render_pose(spec, 0x1000 + i, x, y, 3.6,
                                   tilt, pan, roll, K, 960, 100 + i)
            dets = detect.detect(gray, K=K, versions="sim96c32")
            self.assertEqual(len(dets), 1, f"pose {i} did not decode")
            self.assertEqual(dets[0]["value"], 0x1000 + i)
            errors.append(abs(dets[0]["tilt_deg"] - _gt_tilt(R)))
        errors = np.array(errors)
        self.assertLess(float(np.percentile(errors, 95)), 1.0,
                        f"off-axis tilt p95 {np.percentile(errors, 95):.2f} "
                        f"deg (mirror sort regressed?)")
        self.assertLess(float(np.median(errors)), 0.2)

    def test_radial_clutter_does_not_decode(self):
        for frame_index, image in _radial_clutter_frames():
            with self.subTest(frame=frame_index):
                self.assertEqual(detect.detect(image), [])

    def test_tiled_renderer_matches_direct_definition(self):
        cases = (("T", 31, 1, .12, 42),
                 ("M", 63, 3, .07, 0xABCDEF),
                 ("D", 79, 2, .20, 123456789))
        for name, size, supersample, margin, value in cases:
            with self.subTest(variant=name):
                spec = VARIANTS[name]
                data = payload.encode_id(value, spec)
                expected = _naive_render(data, spec, size, supersample, margin)
                actual = render(data, spec, size, supersample, margin)
                self.assertTrue(np.array_equal(actual, expected))
                actual_inverted = render(data, spec, size, supersample, margin,
                                         inverted=True)
                self.assertTrue(np.array_equal(actual_inverted, 255 - expected))

    def test_documented_app_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for inverted in (False, True):
                with self.subTest(inverted=inverted):
                    marker = Path(directory) / f"tag-{inverted}.png"
                    command = [sys.executable, "app.py", "encode",
                               "--variant", "s256",  # alias input accepted
                               "--id", "0x2a", "--size", "256", "--out", str(marker)]
                    if inverted:
                        command.append("--inverted")
                    encoded = subprocess.run(
                        command, cwd=ROOT, check=True, capture_output=True, text=True)
                    self.assertIn("decode=('ID', 42)", encoded.stdout)
                    # s256 is decoded by explicit selection (not in the
                    # default auto set since the s4k default swap)
                    decoded = subprocess.run(
                        [sys.executable, "app.py", "decode", str(marker),
                         "--variant", "s256"],
                        cwd=ROOT, check=True, capture_output=True, text=True)
                    self.assertIn("sim48c8 (s256) ID=42", decoded.stdout)
                    polarity = "white-on-black" if inverted else "black-on-white"
                    self.assertIn(polarity, decoded.stdout)

    def test_malformed_semantics_remain_verified_bytes(self):
        spec = VARIANTS["M"]
        malformed_geo = bytes.fromhex("01000000")
        with self.assertRaisesRegex(ValueError, "GEO body is truncated"):
            payload.decode(malformed_geo, spec)
        image = render(malformed_geo, spec, size=256)
        results = detect.detect(image)
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0]["mode"], results[0]["value"]),
                         ("UNKNOWN", malformed_geo))

    def test_future_payload_version_is_not_misparsed(self):
        spec = VARIANTS["M"]
        future_id = bytes.fromhex("1000002a")
        with self.assertRaisesRegex(ValueError, "version 1"):
            payload.decode(future_id, spec)


if __name__ == "__main__":
    unittest.main()
