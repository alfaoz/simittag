"""
Conic -> perspective transform, ported from Cantag's TransformEllipseFull.cc.

A circle viewed under perspective projects to an ellipse (a conic). Given the conic
(in normalized camera coords) and that it IS a circle, eigendecomposition of the
conic matrix recovers the supporting-plane transform -- the bias-free pose method.
It yields the inherent 2-fold ambiguity (two solutions); the data decode picks one.

We use this to build a HOMOGRAPHY mapping marker-plane points (unit circle, radius
1.0 = the fitted ellipse) to image pixels, so we can sample the data rings through
true perspective -- which an affine model cannot do (concentric circles do not
project to concentric ellipses).

Reference: src/algorithms/TransformEllipseFull.cc (Andrew Rice, Cantag).
"""
from __future__ import annotations
import numpy as np


def ellipse_to_conic(cx, cy, MA, ma, angle_deg):
    """cv2.fitEllipse geometric form -> 3x3 conic matrix C with [x y 1] C [x y 1]^T = 0."""
    a, b = MA / 2.0, ma / 2.0
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = R @ np.diag([1.0 / a**2, 1.0 / b**2]) @ R.T
    c = np.array([cx, cy])
    C = np.zeros((3, 3))
    C[:2, :2] = M
    C[:2, 2] = -M @ c
    C[2, :2] = -M @ c
    C[2, 2] = c @ M @ c - 1.0
    return C


def transforms_from_conic(C, bullseye_size=1.0):
    """
    Port of TransformEllipseFull: conic (normalized coords) -> two 4x4 transforms
    mapping the unit-circle plane (z=0) into camera coords.
    """
    w, V = np.linalg.eigh(C)          # symmetric -> real eigenpairs (ascending)
    if np.sum(w < 0) > 1:             # equation defined up to scale; want 1 negative
        w, V = -w, V
    idx = np.argsort(w)[::-1]         # descending: l1 >= l2 >= l3
    w, V = w[idx], V[:, idx]
    if V[2, 2] < 0:                   # normal points toward camera
        V[:, 2] = -V[:, 2]
    j = int(np.argmax(np.abs(V[:, 1])))
    if V[j, 1] < 0:                   # eigenvector signs are solver-arbitrary; pin
        V[:, 1] = -V[:, 1]            # col 1 so any implementation (LAPACK, the Rust
                                      # port's 3x3 solver) lands on the SAME H rather
                                      # than one rotated 180 deg in-plane
    if np.linalg.det(V) < 0:          # no reflection
        V[:, 0] = -V[:, 0]
    l1, l2, l3 = w

    denom = l3 - l1
    pmcos = np.sqrt(max(0.0, (l3 - l2) / denom))
    pmsin = np.sqrt(max(0.0, (l2 - l1) / denom))
    tx = np.sqrt(max(0.0, (l2 - l1) * (l3 - l2))) / l2
    scale = np.sqrt(max(0.0, -l1 * l3 / (l2 * l2))) / bullseye_size

    R1 = np.eye(4); R1[:3, :3] = V
    out = []
    for sgn in (+1.0, -1.0):
        r2 = np.eye(4)
        r2[0, 0] = pmcos; r2[0, 2] = -sgn * pmsin
        r2[2, 0] = sgn * pmsin; r2[2, 2] = pmcos
        trans = np.eye(4)
        trans[0, 3] = sgn * tx / scale
        trans[2, 3] = 1.0 / scale
        out.append(R1 @ r2 @ trans)
    return out


def pose_homographies(ellipse_geom, K):
    """
    ellipse_geom = ((cx,cy),(MA,ma),angle) from cv2.fitEllipse (pixel coords).
    Return up to 2 homographies H (3x3) mapping marker-plane (X,Y,1) -> image pixels,
    where the unit circle (X^2+Y^2=1) is the fitted ellipse.
    """
    (cx, cy), (MA, ma), ang = ellipse_geom
    C_pix = ellipse_to_conic(cx, cy, MA, ma, ang)
    C_norm = K.T @ C_pix @ K                  # to normalized camera coords
    Hs = []
    for T in transforms_from_conic(C_norm, bullseye_size=1.0):
        # marker point (X,Y,0,1): camera coords use T columns 0,1,3 (z=0 drops col 2)
        H_norm = T[:3][:, [0, 1, 3]]
        H_pix = K @ H_norm
        if abs(H_pix[2, 2]) > 1e-12:
            H_pix = H_pix / H_pix[2, 2]
        Hs.append(H_pix)
    return Hs


def apply_H(H, X, Y):
    p = H @ np.array([X, Y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def decompose_H(H, K):
    """
    Homography (marker-plane unit circle -> image px) -> (R, t) of the marker in
    CAMERA coordinates, with the marker's z axis = its surface normal. Standard
    K^-1 H decomposition with orthonormalization. Scale is in 'unit-circle radii'
    (the fitted outer ellipse = radius 1), i.e. metric up to marker size.
    """
    Kinv = np.linalg.inv(K)
    L = Kinv @ H
    l1 = np.linalg.norm(L[:, 0])
    l2 = np.linalg.norm(L[:, 1])
    lam = 2.0 / (l1 + l2)
    L = L * lam
    r1 = L[:, 0]; r2 = L[:, 1]; t = L[:, 2]
    r1 /= (np.linalg.norm(r1) or 1.0)
    r2 = r2 - r1 * (r1 @ r2)
    r2 /= (np.linalg.norm(r2) or 1.0)
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    if t[2] < 0:                 # marker must be in front of camera
        R[:, :2] *= -1; t = -t
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
    return R, t


def tilt_from_H(H, K):
    """Approximate tilt angle (deg) of the marker plane normal vs camera axis."""
    Kinv = np.linalg.inv(K)
    h1, h2 = Kinv @ H[:, 0], Kinv @ H[:, 1]
    n1, n2 = h1 / np.linalg.norm(h1), h2 / np.linalg.norm(h2)
    normal = np.cross(n1, n2)
    normal /= np.linalg.norm(normal)
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
