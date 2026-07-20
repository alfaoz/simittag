"""
Pose of a circle from its perspective projection.

A circle viewed by a pinhole camera projects to a conic. The rays through the
image conic form a cone in normalized camera coordinates, and asking which
planes cut that cone in a circle recovers the circle's supporting plane, up to
the inherent 2-fold ambiguity (two mirror planes; the data decode picks one).
This is the classical bias-free method for circular-feature pose; see
Y. C. Shiu and S. Ahmad, "3D location of circular and spherical features by
monocular model-based vision," IEEE SMC 1989, and K. Kanatani and W. Liu,
"3D interpretation of conics and orthogonality," CVGIP 1993. Circular fiducial
systems have used it since Rice, Beresford and Harle's Cantag (PerCom 2006).

Derivation used here: scale the conic matrix so its eigenvalues satisfy
l1 >= l2 > 0 > l3. In the eigenbasis the cone is elliptical, fattest along v1
and narrowest along v3. A plane cuts it in a circle when tilted about v2 by
the angle theta with cos(theta) = sqrt((l2-l3)/(l1-l3)) and sin(theta) =
sqrt((l1-l2)/(l1-l3)); the two tilt directions are the two solutions. For the
circle of unit radius, the center sits at inverse distance rho =
sqrt(-l1*l3)/l2 along the tilted axis, displaced u = sqrt((l1-l2)(l2-l3))/l2
within the plane. The homography columns are then the plane's in-plane basis
and the center, so marker-plane points (X, Y, 1) map through it to rays, and
through K to pixels. Sampling the data rings through this homography is true
perspective, which an affine model cannot provide: concentric circles do not
project to concentric ellipses.
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


def circle_poses_from_conic(C):
    """
    Conic (normalized camera coords) -> the two 3x3 homographies mapping the
    unit circle's plane into rays. Columns: plane x basis, plane y basis,
    circle center. Derivation in the module docstring.
    """
    w, V = np.linalg.eigh(C)          # symmetric -> real eigenpairs (ascending)
    if np.sum(w < 0) > 1:             # conic defined up to scale; fix signature +,+,-
        w, V = -w, V
    idx = np.argsort(w)[::-1]         # descending: l1 >= l2 > 0 > l3
    w, V = w[idx], V[:, idx]
    if V[2, 2] < 0:                   # plane normal points toward the camera
        V[:, 2] = -V[:, 2]
    j = int(np.argmax(np.abs(V[:, 1])))
    if V[j, 1] < 0:                   # eigenvector signs are solver-arbitrary; pin
        V[:, 1] = -V[:, 1]            # col 1 so any implementation (LAPACK, the Rust
                                      # port's 3x3 solver) lands on the SAME H rather
                                      # than one rotated 180 deg in-plane
    if np.linalg.det(V) < 0:          # right-handed basis
        V[:, 0] = -V[:, 0]
    l1, l2, l3 = w

    span = l3 - l1
    ct = np.sqrt(max(0.0, (l3 - l2) / span))            # cos(tilt) in the eigenbasis
    st = np.sqrt(max(0.0, (l2 - l1) / span))            # sin(tilt)
    u = np.sqrt(max(0.0, (l2 - l1) * (l3 - l2))) / l2   # center's in-plane offset
    rho = np.sqrt(max(0.0, -l1 * l3 / (l2 * l2)))       # inverse center distance
    v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]

    out = []
    for s in (+1.0, -1.0):
        b1 = ct * v1 + s * st * v3        # plane x basis
        axis = -s * st * v1 + ct * v3     # tilted axis through the center
        center = (s * u * b1 + axis) / rho
        out.append(np.column_stack([b1, v2, center]))
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
    for H_norm in circle_poses_from_conic(C_norm):
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
