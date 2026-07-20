"""
Detector v0.1 (Python prototype).

Pipeline: adaptive threshold -> contour tree -> fit outer ellipse -> locate
concentric bullseye -> CONIC perspective transform (pose.py, ported from Cantag's
TransformEllipseFull) -> sample data rings through the homography -> soft grayscale
cell sampling -> codec decode (sync rotation + RS/CRC).

This replaces the earlier affine sampler, which could not handle tilt because
concentric circles do not project to concentric ellipses. The conic transform
samples through true perspective. The circle's inherent ambiguities are resolved by
try-and-check: the two conic solutions x a fine in-plane rotation search, with the
CRC selecting the winner.
"""
from __future__ import annotations

import numpy as np
import cv2

from .spec import resolve_specs
from . import codec, payload, pose

# Sync-ring gate: a decode attempt must correlate at least this fraction of
# SECTOR_COUNT against the variant's sync pattern before RS is tried. Pre-filter
# only -- RS + CRC remain the actual accept gate.
SYNC_MIN = 0.70
# Wiener-deconvolution retry for small candidates: at the range floor cells are
# ~2px and defocus/JPEG smear bleeds neighboring cells into each other
# (inter-symbol interference); deconvolving the patch before sampling recovers
# otherwise-lost decodes. Only runs when the normal attempts all failed and the
# ellipse is small, so it costs nothing on healthy frames. Decode-verified
# (sync + RS + CRC), so it adds decode power, not false accepts. Measured
# floors (A4 tag, 1080p, blur 1.0 noise 3 jpeg 85): T 9->12m, M 6->8.5m,
# D 5->7m; decode floors T~24px M~34px D~42px outer diameter.
DECONV_SMALL = True
DECONV_MAX_PX = 80        # only retry candidates smaller than this (major axis)
DECONV_SIGMAS = (1.0, 1.6)  # assumed PSF sigmas (image px) to try
# NOTE: no upsampling before the FFT -- measured STRICTLY BETTER than cubic
# 2x/3x upsample variants (up=3 cost D@7.5m 29->19/30; cubic overshoot is
# amplified by the Wiener filter), and native-res sampling keeps the Rust port
# free of interpolation parity concerns.
# Decode-verify gate: minimum full-grid grayscale correlation (from
# _refine_phase) between the image and the DECODED codeword's re-encoded
# pattern -- a matched filter of the image against what was decoded. Rejects
# wrong-value RS+CRC collisions that no sync gate can catch (the sync ring is
# genuinely present on a true tag at the range floor). Calibrated on measured
# distributions (sim/exp_verify*.py): right-value corr min 0.807 over 341
# accepts; manufactured wrong-value accepts max 0.517 over 75 -- 0.65 sits in
# the middle of an empty 0.29-wide gap.
VERIFY_MIN = 0.65
_VERIFY_LOG = None        # experiment hook: (corr, variant, decoded) tuples


def _deconv_patch(gray, geom, sigma, lam=0.01):
    """
    Crop the candidate (1.5x its ellipse) and Wiener-deconvolve a Gaussian PSF
    of the given sigma (image px). Returns (patch, T) with T the image->patch
    homography (a pure translation), or (None, None) when the crop degenerates.
    """
    (cx, cy), (MA, ma), _ = geom
    r = 0.75 * max(MA, ma)
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1 = int(min(gray.shape[1], cx + r)); y1 = int(min(gray.shape[0], cy + r))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None, None
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    # Pad edge-replicate to the next power of two: a plain radix-2 FFT then
    # suffices (the Rust port carries no FFT dependency), and replication keeps
    # the border smooth so the pad adds no ringing of its own.
    ph, pw = patch.shape
    fh = 1 << (ph - 1).bit_length()
    fw = 1 << (pw - 1).bit_length()
    padded = np.pad(patch, ((0, fh - ph), (0, fw - pw)), mode="edge")
    F = np.fft.rfft2(padded)
    fy = np.fft.fftfreq(fh)[:, None]
    fx = np.fft.rfftfreq(fw)[None, :]
    G = np.exp(-2.0 * np.pi ** 2 * sigma * sigma * (fx * fx + fy * fy))
    out = np.fft.irfft2(F * G / (G * G + lam), s=padded.shape)[:ph, :pw]
    out = np.clip(out, 0, 255).astype(np.uint8)
    T = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
    return out, T


def default_K(width, height, fov_deg=60.0):
    f = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, (width-1)/2.0], [0, f, (height-1)/2.0], [0, 0, 1.0]])


_UNDIST_CACHE = {}


def _undistort(gray, K, dist):
    """
    Full-frame lens-distortion correction (maps cached per K/dist/shape). The whole
    pipeline -- conic pose above all -- assumes a PINHOLE camera: under radial
    distortion an off-center circle does not project to an ellipse, and the conic
    transform silently returns a biased pose (worst exactly in Simittag's niche:
    wide FOV, marker near the frame edge). One cv2.remap makes the pinhole model
    true again; the same K stays valid for pose. Point-wise undistortion (contours
    only) would be cheaper -- that's a Rust-port optimization, not prototype work.
    """
    dist = np.asarray(dist, np.float64).ravel()
    if not dist.any():
        return gray
    key = (gray.shape, K.tobytes(), dist.tobytes())
    maps = _UNDIST_CACHE.get(key)
    if maps is None:
        if len(_UNDIST_CACHE) > 8:          # bound the cache; keys are per-camera
            _UNDIST_CACHE.clear()
        h, w = gray.shape
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
        _UNDIST_CACHE[key] = maps = (m1, m2)
    return cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)


def _sharpen(gray, amount=0.6, sigma=1.0):
    """
    Unsharp mask to counter camera defocus -- the dominant range limiter. At long
    range a marker's cells are only ~3-4px and blur bleeds them together; subtracting
    a blurred copy restores per-cell contrast (AprilTag does the same to its sampled
    bit grid). Measured to push T's reliable range ~6m -> ~8m at 1080p. amount=0 = off.
    """
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def _project(H, X, Y):
    """Project canonical marker coords (arrays X,Y) through H -> pixel xs, ys."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    P = H @ np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    w = P[2]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return (P[0] / w).reshape(X.shape), (P[1] / w).reshape(X.shape)


def _sample_many(img, xs, ys):
    """Vectorized bilinear sample of a float image. Returns (values, valid_mask)."""
    h, w = img.shape
    finite = np.isfinite(xs) & np.isfinite(ys)
    xsf = np.where(finite, xs, -1.0); ysf = np.where(finite, ys, -1.0)
    x0 = np.floor(xsf).astype(np.int64); y0 = np.floor(ysf).astype(np.int64)
    valid = finite & (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2); y0c = np.clip(y0, 0, h - 2)
    fx = xsf - x0c; fy = ysf - y0c
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy) +
         img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return v, valid


def _find_marker_ellipses(gray):
    """
    Find outer-ring ellipse candidates using the contour NESTING TREE (Cantag's
    approach), NOT ellipse-center distance. Under perspective, concentric circles
    project to NON-concentric ellipses (centers separate by tens of px at tilt), so
    a center-distance concentricity test wrongly rejects the true outer ring. The
    nesting tree encodes containment topologically, which is perspective-invariant.
    """
    # Adaptive-threshold block: big enough to span local lighting gradients, but
    # CAPPED -- a block of min(shape)//8 (=161px at 1280) made adaptiveThreshold ~92%
    # of total runtime. ~51px handles gradients fine and is ~5x faster; it only needs
    # to localize ring EDGES (the ellipse fit + sampling run at full res regardless).
    blk = max(11, min(51, (min(gray.shape) // 8) | 1) | 1)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, blk, 7)
    # Despeckle: on a noisy frame the adaptive threshold turns sensor grain in flat
    # regions into tens of THOUSANDS of 1-2px blobs, and findContours then dominates
    # runtime (~50ms). A 3x3 median erases isolated noise while leaving marker rings
    # (>=3px thick, hundreds of px long) intact -> contour count drops ~100x.
    thr = cv2.medianBlur(thr, 3)
    cnts, hier = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]  # [next, prev, first_child, parent]

    ell = [None] * len(cnts)
    for i, c in enumerate(cnts):
        # Relaxed size floor (was len<12 / area<120). At long range the outer ring is
        # still a clean ellipse well below the old floor -- the gate, not the optics,
        # was capping range. Roundness (below) is the real discriminator.
        if len(c) < 6 or cv2.contourArea(c) < 25:
            continue
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
        a, b = MA / 2, ma / 2
        if a < 1 or b < 1:
            continue
        # tilt-invariant roundness: transform contour points into the fitted
        # ellipse's canonical frame; for a true (even steeply tilted) ellipse every
        # point has normalized radius ~1. A SQUARE (a marker's white backing/quad)
        # has corners at r~1.4 and edges caving in -> large residual -> rejected.
        # (area-ratio does NOT catch squares: a square scores ~0.06 there.)
        pts = c.reshape(-1, 2).astype(np.float64)
        th = np.radians(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = (dx * np.cos(th) + dy * np.sin(th)) / a
        v = (-dx * np.sin(th) + dy * np.cos(th)) / b
        rnorm = np.sqrt(u * u + v * v)
        # Roundness residual: how well contour points lie on the fitted ellipse.
        # Measured on real browser frames: true marker rings score ~0.005-0.008
        # (even small/far/antialiased), white backing SQUARES score ~0.056-0.099.
        # 0.03 cleanly separates them -> squares never become candidates.
        if np.mean(np.abs(rnorm - 1.0)) > 0.03:
            continue
        ell[i] = ((cx, cy), (MA, ma), ang, max(MA, ma) / 2, cv2.contourArea(c))

    def descend(i):
        out, stack = [], [hier[i][2]]
        while stack:
            j = stack.pop()
            while j != -1:
                out.append(j)
                if hier[j][2] != -1:
                    stack.append(hier[j][2])
                j = hier[j][0]
        return out

    def has_round_ancestor(i):
        # A single marker's ring stack (outer-ring outer edge -> its inner edge ->
        # bullseye) is NESTED in the contour tree, so each tag emits several round
        # ellipses. Suppress any candidate that sits inside a larger round one: keep
        # only the OUTERMOST ring per tag. This is topological (perspective-invariant)
        # and is what lets the candidate cap count distinct TAGS, not redundant rings
        # of the few biggest tags -- without it a dense scene (>~8 tags) starves.
        p = hier[i][3]
        while p != -1:
            if ell[p] is not None and ell[p][3] >= 8:
                return True
            p = hier[p][3]
        return False

    cands = []
    for i, ei in enumerate(ell):
        if ei is None or ei[3] < 8:           # was 15px; small markers are valid
            continue
        if has_round_ancestor(i):             # inner ring of an already-kept tag
            continue
        # AprilTag lesson: do NOT require a separately-contoured concentric child. At
        # range the bullseye is only a few px and never gets its own clean contour, so
        # the old "must have a round child" rule threw away perfectly round outer rings
        # (measured: outer ellipse stays round to ~40px while children vanish at ~58px).
        # We accept any round, sized outer ellipse and let the DECODE (sync+CRC) verify
        # it's really a marker. A child, when present, still gives the bullseye centre
        # for the 2-fold pose disambiguation; otherwise we fall back to the ellipse
        # centre (the decode folds in rotation anyway).
        kids = [ell[j] for j in descend(i) if ell[j] is not None]
        inner = min(kids, key=lambda k: k[3]) if kids else None
        inner_geom = (inner[0], inner[1], inner[2]) if inner is not None else None
        # ALT geometry: the largest suppressed round child clearly SMALLER than
        # this contour (<= 0.9r; the tag's own ring inner edge sits at 0.86r).
        # When the outer contour is NOT the tag -- a white circular sticker or
        # label AROUND the tag is the real-world case -- decode at the outer
        # ellipse fails, and the alt gives the decoder a second, correctly
        # scaled geometry to try. Verified by decode, so no false-accept risk.
        alts = [k for k in kids if k[3] <= 0.9 * ei[3]]
        alt = max(alts, key=lambda k: k[3]) if alts else None
        alt_geom = (alt[0], alt[1], alt[2]) if alt is not None else None
        cands.append((ei[:4], inner_geom, alt_geom))
    cands.sort(key=lambda e: -e[0][3])  # largest outer edge first
    pruned = []
    for e in cands:
        eo = e[0]
        if all(not (abs(eo[3]-p[0][3])/p[0][3] < 0.18 and
                    np.hypot(eo[0][0]-p[0][0][0], eo[0][1]-p[0][0][1]) < 0.25*p[0][3])
               for p in pruned):
            pruned.append(e)
    # Cap so a pathological frame can't blow up the decode budget. With nested inner
    # rings now suppressed each candidate is ~one distinct tag, so this is effectively
    # a max-simultaneous-tags limit; 48 supports dense multi-tag scenes while bounding
    # worst-case cost (each candidate costs specs x scales x phase decode attempts).
    return pruned[:48]


def _refine_ellipse(gray, geom, n_rays=128):
    """
    Sub-pixel refinement of the outer-edge ellipse. The coarse fit comes from
    cv2.fitEllipse on ADAPTIVE-THRESHOLD contour pixels, which carries a systematic
    radial bias: the contour traces whole pixels inside the true edge, and the
    threshold's -C offset shifts the level set further. A radial bias doesn't hurt
    tilt much but corrupts SCALE -> recovered depth. Here we march along the coarse
    ellipse's outward normals, localize the dark->bright transition at the peak of
    the grayscale derivative (parabolic sub-pixel), and refit on those points.
    Selecting the max POSITIVE (outward-brightening) slope automatically rejects the
    ring's INNER edge (bright->dark going outward) when the window overlaps it.
    """
    (cx, cy), (MA, ma), ang = geom
    a, b = MA / 2.0, ma / 2.0
    th = np.radians(ang)
    c, s = np.cos(th), np.sin(th)
    t = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    ex = cx + a * ct * c - b * st * s          # points on the coarse ellipse
    ey = cy + a * ct * s + b * st * c
    nx = ct / a * c - st / b * s               # outward normal = grad of implicit form
    ny = ct / a * s + st / b * c
    nn = np.hypot(nx, ny)
    nx, ny = nx / nn, ny / nn
    # Window: wide enough to catch the true edge past the coarse fit's ~1px bias,
    # narrow enough to stay near THIS edge on a small far marker (ring is 0.14*r wide).
    w = float(np.clip(0.25 * min(a, b), 1.5, 3.0))
    NS = 13
    offs = np.linspace(-w, w, NS)
    xs = ex[:, None] + offs[None, :] * nx[:, None]
    ys = ey[:, None] + offs[None, :] * ny[:, None]
    vals, valid = _sample_many(gray, xs, ys)
    d = vals[:, 1:] - vals[:, :-1]             # derivative at midpoints (n_rays, NS-1)
    step = offs[1] - offs[0]
    i = np.argmax(d, axis=1)
    rows = np.arange(n_rays)
    dmax = d[rows, i]
    ok = (valid.all(axis=1) & (i > 0) & (i < NS - 2) & (dmax > 4.0))
    if ok.sum() < max(12, n_rays // 4):
        return geom
    # parabolic sub-sample peak of the derivative
    y0 = d[rows, np.maximum(i - 1, 0)]
    y2 = d[rows, np.minimum(i + 1, NS - 2)]
    denom = y0 - 2 * dmax + y2
    with np.errstate(invalid="ignore", divide="ignore"):   # NaN rows are gated by `ok`
        delta = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.clip(np.nan_to_num(delta), -1.0, 1.0)
    pos = offs[i] + 0.5 * step + delta * step  # midpoint offset + sub-sample shift
    px = (ex + pos * nx)[ok]
    py = (ey + pos * ny)[ok]
    (rcx, rcy), (rMA, rma), rang = cv2.fitEllipse(
        np.stack([px, py], axis=1).astype(np.float32))
    # sanity: refinement removes sub-pixel bias; a big jump means it latched onto
    # something else (clutter, neighbor edge) -> keep the coarse fit.
    if (np.hypot(rcx - cx, rcy - cy) > 0.05 * max(a, b) + 1.0
            or not (0.8 < rMA / MA < 1.25) or not (0.8 < rma / ma < 1.25)):
        return geom
    return ((rcx, rcy), (rMA, rma), rang)


_SAMPLE_CACHE = {}


def _cell_sample_points(spec):
    """
    Precompute canonical (X,Y) sample coords for every cell, cached per spec.
    Shape (RING_COUNT, SECTOR_COUNT, NSUB) for a 3x3 subgrid within each cell.
    Returns (X, Y, rho_q) where rho_q is the white quiet-ring radius.
    """
    key = id(spec)
    hit = _SAMPLE_CACHE.get(key)
    if hit is not None:
        return hit
    _, ring_c, _ = spec.ring_radii()
    step = 2 * np.pi / spec.SECTOR_COUNT
    dR = ring_c[1] - ring_c[0]
    drs = np.array([-0.25, 0.0, 0.25])
    dps = np.array([-0.3 * step, 0.0, 0.3 * step])
    DR, DP = np.meshgrid(drs, dps, indexing="ij")
    sub = DR.size
    X = np.empty((spec.RING_COUNT, spec.SECTOR_COUNT, sub))
    Y = np.empty_like(X)
    for ring in range(spec.RING_COUNT):
        rho0 = ring_c[ring]
        for s in range(spec.SECTOR_COUNT):
            phi0 = (s + 0.5) * step
            rho = (rho0 + DR * dR).ravel()
            phi = (phi0 + DP).ravel()
            X[ring, s] = rho * np.cos(phi)
            Y[ring, s] = rho * np.sin(phi)
    rho_q = (spec.R_BULLSEYE + spec.R_DATA_IN) / 2
    _SAMPLE_CACHE[key] = (X, Y, rho_q)
    return _SAMPLE_CACHE[key]


def _build_grid(gray, H, spec):
    X, Y, rho_q = _cell_sample_points(spec)
    # black/white reference: bullseye center vs the white quiet ring
    refx, refy = _project(H, np.array([0.0, rho_q]), np.array([0.0, 0.0]))
    refv, refok = _sample_many(gray, refx, refy)
    if not refok.all() or abs(refv[1] - refv[0]) < 20:
        return None, None
    black, white = float(refv[0]), float(refv[1])
    mid, span = 0.5 * (black + white), abs(white - black) / 2
    xs, ys = _project(H, X, Y)            # (rings, sectors, sub)
    vals, valid = _sample_many(gray, xs, ys)
    cnt = valid.sum(axis=2)
    if (cnt == 0).any():
        return None, None
    m = np.where(valid, vals, 0.0).sum(axis=2) / cnt   # mean of valid subs
    grid = (m < mid).astype(np.int8)
    conf = np.minimum(1.0, np.abs(m - mid) / span).astype(np.float32)
    return grid, conf


def _rotate_H(H, dphi):
    """Compose H with an in-plane rotation of the marker (about its normal)."""
    c, s = np.cos(dphi), np.sin(dphi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return H @ Rz


_REFINE_CACHE = {}


def _refine_sample_points(spec):
    """
    Denser-ANGULAR sample pattern used only by the phase refine (cached per spec).
    The decode grid's 3 angular subsamples per cell make the theta-correlation a
    STAIRCASE (samples cross cell edges at discrete thetas), and parabola-fitting a
    staircase under noise is exactly the axis wobble measured in jitter_test. Seven
    angular positions per cell keep the curve smooth across the +-half-cell window.
    """
    key = id(spec)
    hit = _REFINE_CACHE.get(key)
    if hit is not None:
        return hit
    _, ring_c, _ = spec.ring_radii()
    step = 2 * np.pi / spec.SECTOR_COUNT
    dR = ring_c[1] - ring_c[0]
    drs = np.array([-0.25, 0.0, 0.25])
    # angular positions per cell scale with sector width: T's 22.5deg sectors need
    # more coverage than D's 10deg for a comparably smooth correlation curve.
    n_ang = int(np.clip(np.degrees(step) / 2.0, 7, 13)) | 1
    dps = np.linspace(-0.38, 0.38, n_ang) * step
    DR, DP = np.meshgrid(drs, dps, indexing="ij")
    X = np.empty((spec.RING_COUNT, spec.SECTOR_COUNT, DR.size))
    Y = np.empty_like(X)
    for ring in range(spec.RING_COUNT):
        for sct in range(spec.SECTOR_COUNT):
            rho = (ring_c[ring] + DR * dR).ravel()
            phi = ((sct + 0.5) * step + DP).ravel()
            X[ring, sct] = rho * np.cos(phi)
            Y[ring, sct] = rho * np.sin(phi)
    _REFINE_CACHE[key] = (X, Y)
    return _REFINE_CACHE[key]


def _refine_phase(gray, Hbase, spec, theta0, ref_grid):
    """
    Continuously refine the in-plane rotation near theta0 by peak-fitting the
    grayscale correlation of ALL cells against the KNOWN decoded pattern.

    The decode search only resolves phi0 to step/6, and RS error-correction happily
    fixes the cell flips a misaligned phase causes -- so the first phi0 bin that
    decodes can be off by several degrees (measured: 3-5deg median rotation error at
    tilt, a stable BIAS, not jitter; the conic contributes no in-plane rotation of
    its own, the decoded theta is the only source). After a successful decode the
    full true grid is known (re-encoded from the payload), which gives RING_COUNT x
    more correlation signal than the sync ring alone and works for sync-less T too.
    The peak is geometry-exact at any tilt (sampling goes through H), so this runs
    UNGATED -- unlike the old sync-only refine, which was gated to near-fronto
    because its single-ring noise (~1deg) exceeded the quantization error it fixed.
    Window is half a cell: theta0 is within step/12 by construction, and the full
    grid's autocorrelation peaks sharply. Bit 1 = a DARK cell -> reference inverted.
    """
    step = 2 * np.pi / spec.SECTOR_COUNT
    X, Y = _refine_sample_points(spec)           # (rings, sectors, sub)
    ref = np.where(np.asarray(ref_grid).ravel() > 0, -1.0, 1.0)

    def corr(ths):
        c, s = np.cos(ths), np.sin(ths)
        Xr = X[None] * c[:, None, None, None] - Y[None] * s[:, None, None, None]
        Yr = X[None] * s[:, None, None, None] + Y[None] * c[:, None, None, None]
        xs, ys = _project(Hbase, Xr, Yr)         # one shot for all rotations
        vals, valid = _sample_many(gray, xs, ys)
        cnt = valid.sum(axis=3)
        if (cnt == 0).any():
            return None
        m = (np.where(valid, vals, 0.0).sum(axis=3) / cnt).reshape(len(ths), -1)
        m = m - m.mean(axis=1, keepdims=True)
        nm = np.linalg.norm(m, axis=1) * np.linalg.norm(ref)
        return np.where(nm > 1e-6, (m @ ref) / np.maximum(nm, 1e-9), -2.0)

    # Two-stage scan: coarse over the +-half-cell window, then a fine pass around
    # the coarse peak, with a least-squares quadratic over the WHOLE fine window
    # (not a 3-point parabola: the curve is smooth but noisy, and the wide fit
    # averages the noise down -- the dominant residual axis-jitter source).
    ths1 = theta0 + np.linspace(-0.5 * step, 0.5 * step, 13)
    cs1 = corr(ths1)
    if cs1 is None:
        return theta0, -1.0
    t_pk = ths1[int(np.argmax(cs1))]
    ths2 = t_pk + np.linspace(-step / 10, step / 10, 13)
    cs2 = corr(ths2)
    if cs2 is None:
        return t_pk, float(np.max(cs1))
    vc = float(np.max(cs2))
    xs_f = ths2 - t_pk
    A = np.stack([xs_f * xs_f, xs_f, np.ones_like(xs_f)], axis=1)
    (a2, a1, _), *_ = np.linalg.lstsq(A, cs2, rcond=None)
    if a2 < -1e-9:
        v = float(t_pk - a1 / (2 * a2))
        if ths2[0] <= v <= ths2[-1]:
            return v, vc
    return float(ths2[int(np.argmax(cs2))]), vc


def _try_decode_spec(gray, Hs, spec, conf_erasure):
    """
    Attempt a Simittag decode of ONE variant across the 2-fold pose solutions, a few
    radial scales, and the sub-cell phase. Returns (decoded, chosen_H) or (None,None).

    HAS_SYNC variants gate on the sync ring (CRC8 alone leaks across the search);
    no-sync variants (T) rely on RS minimum-distance + CRC, which is enough for the
    tiny grid (brute-force rotation lives in codec.decode).
    """
    step = 2 * np.pi / spec.SECTOR_COUNT
    for H in Hs:
        for scale in (1.0, 1.06, 1.12, 0.94):
            Hs_ = H @ np.diag([scale, scale, 1.0])
            for phi0 in np.linspace(0, step, 6, endpoint=False):
                grid, conf = _build_grid(gray, _rotate_H(Hs_, phi0), spec)
                if grid is None:
                    continue
                if spec.HAS_SYNC:
                    _, scores = codec.find_rotation(grid[spec.SYNC_RING], spec)
                    if max(scores) < SYNC_MIN * spec.SECTOR_COUNT:
                        continue
                pb, sh = codec.decode(grid, spec, conf_grid=conf,
                                      conf_erasure=conf_erasure)
                if pb is not None:
                    # Fold the decoded in-plane rotation (phase phi0 + sector shift)
                    # into the pose so the recovered X axis points at sector 0 (else
                    # near-circular ellipses' conic eigenvectors spin), then refine it
                    # to a continuous, unbiased angle by correlating the full known
                    # grid (see _refine_phase; the coarse bin can be ~5deg off).
                    theta = phi0 + sh * step
                    refined, vcorr = _refine_phase(gray, Hs_, spec, theta,
                                                   codec.encode(pb, spec))
                    try:
                        decoded = payload.decode(pb, spec)
                    except Exception:
                        decoded = ("?", pb)
                    if _VERIFY_LOG is not None:
                        _VERIFY_LOG.append((vcorr, spec.NAME, decoded))
                    # Decode-verify gate: the refine correlation IS a matched
                    # filter of the image against the decoded codeword's full
                    # grid. A wrong-value accept (RS+CRC collision on marginal
                    # bits) disagrees with the image in ~half its cells and
                    # collapses this correlation, so thresholding it rejects
                    # wrong IDs that no sync gate can catch (the sync ring is
                    # genuinely present on a true tag at the range floor).
                    if vcorr < VERIFY_MIN:
                        continue
                    d = (refined - theta + np.pi) % (2 * np.pi) - np.pi  # no wrap
                    theta = theta + d
                    chosen_H = _rotate_H(Hs_, theta)
                    return (spec.NAME, decoded), chosen_H
    return None, None


def detect_markers(gray, spec=None, K=None, conf_erasure=0.25, versions=None,
                   pose_only=True, dist=None):
    """
    Detect ALL concentric-ring markers and recover 6-DoF pose from pixels for each,
    whether or not they Simittag-decode. Used by the interactive 3D demo (the
    bias-free conic pose is identical for any concentric circle; only DECODE differs).

    versions: None -> auto-detect among T/M/D (each variant's sync ring rejects
    wrong-variant grids, CRC confirms); a name or list (e.g. "M", ["M","D"]) -> only
    those (faster). `spec` is a back-compat alias for a single pinned variant.

    pose_only: False -> return ONLY decoded Simittags, dropping the "pose only" boxes
    on undecoded nested-ring candidates. Skips their pose recovery; note the decode
    ATTEMPTS on every round candidate still run -- that's how we know it isn't a
    Simittag -- so this declutters and trims per-candidate cost, it doesn't remove
    the search.

    dist: OpenCV lens-distortion coefficients (k1,k2,p1,p2[,k3]) for K's camera;
    the frame is undistorted once up front (see _undistort). None/zeros = pinhole.
    All returned pixel coords (center/axes) are in UNDISTORTED image space.

    Returns list of dicts: center, axes, angle, R, t (camera frame), tilt_deg, decoded,
    and if it Simittag-decodes: variant, mode, value.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if K is None:
        K = default_K(gray.shape[1], gray.shape[0])
    if dist is not None:
        gray = _undistort(gray, K, dist)
    gray = _sharpen(gray)
    if versions is None and spec is not None:
        versions = spec.NAME
    specs = resolve_specs(versions)
    out = []
    for ei, inner, alt in _find_marker_ellipses(gray):
        geom0 = (ei[0], ei[1], ei[2])
        geom1 = _refine_ellipse(gray, geom0)
        (cx, cy), (MA, ma), ang = geom1
        Hs = pose.pose_homographies(geom1, K)
        # At the range floor (marker ~decode-floor px) the discrete decode search is
        # marginal against the sync gate, and sub-0.1px geometry differences decide it
        # (measured: even the GT ellipse only scores 0.83). Refined-first preserves
        # pose accuracy; the coarse fit appended after is a decode fallback that
        # restores the frames where only IT clears the gate (and vice versa). Only
        # small ellipses need it (floors are 42-73px), and skipping it for big ones
        # keeps the wrong-variant failure scan in auto-detect at its old cost.
        Hs_coarse = (pose.pose_homographies(geom0, K)
                     if geom1 is not geom0 and max(MA, ma) < 100 else [])
        if not Hs:
            continue
        # Disambiguate the 2-fold pose ambiguity geometrically using the concentric
        # bullseye: its center is the marker-plane origin, so the correct H maps the
        # DETECTED inner-ring center back closest to (0,0). Ratio-free, per-frame.
        if inner is not None:
            icx, icy = inner[0]
            def _origin_err(H):
                p = np.linalg.inv(H) @ np.array([icx, icy, 1.0])
                return np.hypot(p[0]/p[2], p[1]/p[2])
            Hs = sorted(Hs, key=_origin_err)
            Hs_coarse = sorted(Hs_coarse, key=_origin_err)
        Hs = Hs + Hs_coarse
        # Try each candidate variant; its sync ring rejects wrong-variant grids
        # cheaply (before RS), and CRC is the final arbiter.
        decoded, chosen_H = None, Hs[0]
        for sp in specs:
            decoded, H = _try_decode_spec(gray, Hs, sp, conf_erasure)
            if decoded is not None:
                chosen_H = H
                break
        # Sticker fallback: when the outermost round contour is NOT the tag (a
        # circular white label AROUND the tag is the real-world case), decoding
        # at its scale fails -- retry at the largest suppressed round child,
        # which IS the tag's outer ring. Decode-verified, so no false accepts.
        if decoded is None and alt is not None:
            alt1 = _refine_ellipse(gray, alt)
            Hs_alt = pose.pose_homographies(alt1, K)
            if inner is not None:
                Hs_alt = sorted(Hs_alt, key=_origin_err)
            for sp in specs:
                decoded, H = _try_decode_spec(gray, Hs_alt, sp, conf_erasure)
                if decoded is not None:
                    chosen_H = H
                    geom1 = alt1                # report the TAG's geometry
                    (cx, cy), (MA, ma), ang = geom1
                    break
        # ISI retry: small candidate, all attempts failed -> deconvolve the patch
        # and search again (see DECONV_SMALL above).
        if decoded is None and DECONV_SMALL and max(MA, ma) < DECONV_MAX_PX:
            for sg in DECONV_SIGMAS:
                patch, T = _deconv_patch(gray, geom1, sg)
                if patch is None:
                    break
                Hp = [T @ Hh for Hh in Hs]
                for sp in specs:
                    decoded, H = _try_decode_spec(patch, Hp, sp, conf_erasure)
                    if decoded is not None:
                        chosen_H = np.linalg.inv(T) @ H
                        break
                if decoded is not None:
                    break
        # Relaxed gates let lone round contours through; show a POSE-ONLY box only when
        # there's a concentric child (real nested-ring marker). A decoded marker is
        # always kept. This keeps the range win without spraying boxes on stray circles.
        if decoded is None and (inner is None or not pose_only):
            continue
        R, t = pose.decompose_H(chosen_H, K)
        rec = {"center": (cx, cy), "axes": (MA, ma), "angle": ang,
               "R": R, "t": t, "tilt_deg": pose.tilt_from_H(chosen_H, K),
               "decoded": decoded is not None}
        if decoded:
            rec["variant"], (rec["mode"], rec["value"]) = decoded
        out.append(rec)
    # de-dup overlapping detections (same physical marker fit at 2 edges/scales):
    # keep the larger-radius one within a center-proximity cluster.
    out.sort(key=lambda r: -max(r["axes"]))
    kept = []
    for r in out:
        cx0, cy0 = r["center"]; R0 = max(r["axes"]) / 2
        if all(np.hypot(cx0 - k["center"][0], cy0 - k["center"][1]) > 0.4 * R0
               for k in kept):
            kept.append(r)
    return kept


def detect(gray, spec=None, K=None, conf_erasure=0.25, versions=None, dist=None):
    """
    Headless decode-only detector for validation harnesses. Returns
    only successfully decoded markers, with mode/value/variant/H. `spec` pins a single
    variant (back-compat); `versions` selects the auto/specific set. `dist` as in
    detect_markers.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if K is None:
        K = default_K(gray.shape[1], gray.shape[0])
    if dist is not None:
        gray = _undistort(gray, K, dist)
    gray = _sharpen(gray)
    if versions is None and spec is not None:
        versions = spec.NAME
    specs = resolve_specs(versions)
    results = []
    for ei, _inner, alt in _find_marker_ellipses(gray):
        geom0 = (ei[0], ei[1], ei[2])
        geom1 = _refine_ellipse(gray, geom0)
        (cx, cy), (MA, ma), ang = geom1
        Hs = pose.pose_homographies(geom1, K)
        if geom1 is not geom0 and max(MA, ma) < 100:
            # coarse-fit decode fallback near the range floor (see detect_markers)
            Hs = Hs + pose.pose_homographies(geom0, K)
        if not Hs:
            continue
        hit = None
        for sp in specs:
            decoded, H = _try_decode_spec(gray, Hs, sp, conf_erasure)
            if decoded is not None:
                hit = (decoded, H, geom1)
                break
        if hit is None and alt is not None:
            # circular-sticker fallback (see detect_markers)
            alt1 = _refine_ellipse(gray, alt)
            for sp in specs:
                decoded, H = _try_decode_spec(gray, pose.pose_homographies(alt1, K),
                                              sp, conf_erasure)
                if decoded is not None:
                    hit = (decoded, H, alt1)
                    break
        # ISI retry: deconvolve small failed candidates (see detect_markers)
        if hit is None and DECONV_SMALL and max(MA, ma) < DECONV_MAX_PX:
            for sg in DECONV_SIGMAS:
                patch, T = _deconv_patch(gray, geom1, sg)
                if patch is None:
                    break
                Hp = [T @ Hh for Hh in Hs]
                for sp in specs:
                    decoded, H = _try_decode_spec(patch, Hp, sp, conf_erasure)
                    if decoded is not None:
                        hit = (decoded, np.linalg.inv(T) @ H, geom1)
                        break
                if hit is not None:
                    break
        if hit is None:
            continue
        decoded, H, g = hit
        variant, (mode, value) = decoded
        results.append({
            "variant": variant, "mode": mode, "value": value,
            "center": g[0], "axes": g[1], "angle": g[2],
            "tilt_deg": pose.tilt_from_H(H, K), "H": H,
        })
    return results
