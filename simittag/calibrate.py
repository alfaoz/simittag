"""
Camera calibration from simittag boards (see simittag.board).

The intrinsics contract mirrors AprilTag's apriltag_detection_info_t: fx, fy,
cx, cy in pixels (aprilrobotics/apriltag, apriltag_pose.h) — plus the OpenCV
distortion vector, which AprilTag leaves to the caller. detect.detect() takes
the same parameters as `K=` / `dist=`; feeding it a saved CameraIntrinsics
replaces the default 60-degree-FOV guess with measured values, which is what
turns the reported poses from approximate into metric.

Usage:
    intr = calibrate_images(["a.png", "b.png", ...])       # board from sheet
    intr.save("intrinsics.json")
    ...
    intr = CameraIntrinsics.load("intrinsics.json")
    detect.detect(gray, K=intr.K, dist=intr.dist_array)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

import numpy as np
import cv2

from . import detect as _detect
from .spec import normalize_variant

# Calibration pins ITS OWN variant set rather than following the detector's
# default auto set, so detector-default changes cannot silently break board
# workflows. The default matches boards the studio generates today: s4k grid
# or perimeter tags, an s16m multiscale anchor, an sdata descriptor. Sheets
# of any other variant still calibrate: a provided board file narrows the
# set to the board's own variants, and a descriptor-only sheet triggers a
# re-detection with the descriptor's variant set (see calibrate()).
BOARD_VERSIONS = ("sim48c12", "sim96c32", "sim180c88")
from . import board as _board

MIN_POINTS_PER_VIEW = 6
MIN_VIEWS = 4


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics, AprilTag field convention (pixels) + distortion."""
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list = field(default_factory=list)   # OpenCV k1 k2 p1 p2 k3
    width: int = 0
    height: int = 0
    rms_px: float = 0.0                        # calibration reprojection RMS
    views: int = 0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def dist_array(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64)

    def save(self, path):
        json.dump({"simittag_intrinsics": 1, **self.__dict__},
                  open(path, "w"), indent=2)

    @classmethod
    def load(cls, path) -> "CameraIntrinsics":
        j = json.load(open(path))
        if j.pop("simittag_intrinsics", None) != 1:
            raise ValueError(f"{path} is not a simittag intrinsics file")
        return cls(**j)


def _view_points(detections, board):
    """Match one image's detections against the board -> (obj Nx3, img Nx2)."""
    obj, img = [], []
    for r in detections:
        value = r["value"]
        if r["mode"] == "RAW":
            value = bytes(value)
        pt = board.point_for(r["variant"], r["mode"], value)
        if pt is not None:
            obj.append([pt[0], pt[1], 0.0])
            img.append(r["center"])
    return (np.array(obj, dtype=np.float32),
            np.array(img, dtype=np.float32))


def _board_versions(board):
    """The board's own variants (+ sdata for the descriptor tag)."""
    return sorted({normalize_variant(t.variant)
                   for t in board.tags} | {"sim180c88"})


def calibrate(images, board=None, versions=None) -> CameraIntrinsics:
    """
    Solve intrinsics (Zhang's method via cv2.calibrateCamera) from grayscale
    images of a simittag calibration board. If `board` is None it is
    reconstructed from the descriptor tag found on the sheet; when that
    descriptor names variants outside the initial detection set (a legacy
    s256 sheet under the s4k default, or vice versa), the images are
    re-detected with the board's own set, so any sheet self-configures.
    """
    if versions is not None:
        view_versions = versions
    elif board is not None:
        view_versions = _board_versions(board)
    else:
        view_versions = list(BOARD_VERSIONS)
    size = None
    for gray in images:
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            raise ValueError("all calibration images must share one resolution")
    dets_per_view = [_detect.detect(gray, versions=view_versions)
                     for gray in images]
    if board is None:
        for dets in dets_per_view:
            board = _board.find_board(dets)
            if board is not None:
                break
    if board is None:
        raise ValueError("no board descriptor found — pass board= or use a "
                         "sheet with a descriptor tag")
    if versions is None:
        needed = _board_versions(board)
        if not set(needed) <= set(view_versions):
            # descriptor-driven self-configuration: the sheet carries variants
            # the initial set did not decode
            dets_per_view = [_detect.detect(gray, versions=needed)
                             for gray in images]
    obj_pts, img_pts = [], []
    for dets in dets_per_view:
        obj, img = _view_points(dets, board)
        if len(obj) >= MIN_POINTS_PER_VIEW:
            obj_pts.append(obj)
            img_pts.append(img)
    if len(obj_pts) < MIN_VIEWS:
        raise ValueError(f"only {len(obj_pts)} usable view(s) "
                         f"(>= {MIN_POINTS_PER_VIEW} board tags each); "
                         f"need at least {MIN_VIEWS}")
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None)
    return CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=[float(v) for v in dist.ravel()],
        width=size[0], height=size[1],
        rms_px=float(rms), views=len(obj_pts))


def calibrate_images(paths, board=None, versions=None) -> CameraIntrinsics:
    """calibrate() over image files."""
    images = []
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read {p}")
        images.append(gray)
    return calibrate(images, board=board, versions=versions)
