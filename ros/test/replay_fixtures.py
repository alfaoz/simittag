#!/usr/bin/env python3
"""Fixture replay integration test for simittag_ros.

Publishes frame PNGs from the main simittag repo's fixtures/ (the
compatibility contract) together with their CameraInfo, and asserts that the
node reports exactly the expected payloads with matching poses.

Run inside a sourced ROS 2 environment with the node already running, e.g.:

    ros2 run simittag_ros simittag_node --ros-args \
        -p tag_diameter_m:=2.0 -p publish_tf:=false &
    python3 test/replay_fixtures.py --fixtures ../simittag/fixtures

tag_diameter_m=2.0 makes the tag radius 1.0, so published metric poses are
numerically equal to the fixture translations (which are in tag-radius
units).

Frame correlation is by header.stamp: each published frame gets a unique
stamp and the assertion reads only detections carrying that stamp, so
negatives (frames that must NOT decode) are exact rather than timing-based.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy)
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3DArray

TILT_TOL_DEG = 0.05
TRANS_TOL = 2e-3  # tag-radius units; fixtures store t to ~1e-6


def expected_label(det: dict):
    """Rebuild the node's class_id string from a fixture detection entry."""
    if not det.get("decoded"):
        return None
    variant, mode, value = det["variant"], det["mode"], det["value"]
    if mode == "ID":
        body = "id:0x%x" % int(value["int"])
    elif mode == "GEO":
        lat, lon, alt = value["list"]
        body = "geo:%.6f,%.6f,%+d" % (lat, lon, int(alt))
    elif mode == "TAGGED":
        ns, ident = value["list"]
        body = "tag:%d:0x%x" % (int(ns), int(ident))
    elif mode == "RAW":
        body = "raw:" + value["hex"].lower()
    else:
        return None
    return "%s:%s" % (variant, body)


class Replayer(Node):
    def __init__(self, image_topic, info_topic, det_topic):
        super().__init__("simittag_replay")
        # RELIABLE publisher against the node's best-effort subscription:
        # compatible per DDS QoS rules, and retransmission keeps large
        # multi-fragment frames from being silently dropped on loopback.
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_img = self.create_publisher(Image, image_topic, pub_qos)
        self.pub_info = self.create_publisher(CameraInfo, info_topic, pub_qos)
        self.results = {}  # (sec, nsec) -> Detection3DArray
        self.create_subscription(
            Detection3DArray, det_topic, self._on_det, 10
        )

    def _on_det(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.results[key] = msg

    def publish_frame(self, png_path, k, seq, dist=None):
        img = PilImage.open(png_path).convert("L")
        w, h = img.size
        stamp_key = (seq, 42)

        info = CameraInfo()
        info.header.stamp.sec, info.header.stamp.nanosec = stamp_key
        info.header.frame_id = "camera_optical"
        info.width, info.height = w, h
        info.k = [float(v) for v in k]
        info.d = [float(v) for v in (dist or [])]
        info.distortion_model = "plumb_bob"
        self.pub_info.publish(info)

        msg = Image()
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_key
        msg.header.frame_id = "camera_optical"
        msg.height, msg.width = h, w
        msg.encoding = "mono8"
        msg.step = w
        msg.data = img.tobytes()
        self.pub_img.publish(msg)
        return stamp_key

    def wait_for(self, key, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if key in self.results:
                return self.results[key]
        return None


def check_frame(entry, msg, failures):
    name = entry["file"]
    exp = [l for l in (expected_label(d) for d in entry["detections"]) if l]
    got = {}
    if msg is not None:
        for det in msg.detections:
            for res in det.results:
                got[res.hypothesis.class_id] = res

    if sorted(exp) != sorted(got):
        failures.append(
            "%s: expected payloads %s, got %s" % (name, sorted(exp), sorted(got.keys()))
        )
        return

    # pose check, decodable frames only
    for det in entry["detections"]:
        label = expected_label(det)
        if not label:
            continue
        res = got[label]
        t = det["t"]
        p = res.pose.pose.position
        err = math.dist((p.x, p.y, p.z), (t[0], t[1], t[2]))
        if err > TRANS_TOL:
            failures.append(
                "%s [%s]: translation off by %.2e (tol %.0e)" % (name, label, err, TRANS_TOL)
            )
        cov = res.pose.covariance
        if not (cov[0] > 0 and cov[14] > 0 and cov[35] > 0):
            failures.append("%s [%s]: covariance diagonal not populated" % (name, label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera_info")
    ap.add_argument("--det-topic", default="/simittag/detections")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    fixtures = Path(args.fixtures)
    frames = json.loads((fixtures / "frames.json").read_text())["entries"]
    # Pinned-variant entries encode expectations for a pinned detector; the
    # node under test runs auto, so they are out of contract here.
    frames = [e for e in frames if not e.get("versions")]
    if not frames:
        print("no fixture frames matched", file=sys.stderr)
        return 2

    rclpy.init()
    node = Replayer(args.image_topic, args.info_topic, args.det_topic)

    # let discovery settle so the first frame is not dropped
    deadline = time.time() + 10.0
    while (node.pub_img.get_subscription_count() == 0
           and time.time() < deadline):
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.pub_img.get_subscription_count() == 0:
        print("node never subscribed to the image topic", file=sys.stderr)
        return 2
    time.sleep(1.0)

    failures = []
    tested = 0
    for seq, entry in enumerate(frames, start=1):
        png = fixtures / entry["file"]
        # A camera stream repeats frames; a dropped large frame is normal
        # sensor-QoS behavior, so republish (same stamp -> exact assertion)
        # until the node's reply for this stamp arrives.
        msg = None
        for _ in range(5):
            key = node.publish_frame(png, entry["K"], seq, entry.get("dist"))
            msg = node.wait_for(key, args.timeout)
            if msg is not None:
                break
        if msg is None:
            failures.append(
                "%s: no detection message after 5 attempts" % entry["file"]
            )
            continue
        check_frame(entry, msg, failures)
        tested += 1

    node.destroy_node()
    rclpy.shutdown()

    print("replayed %d fixture frames" % tested)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print("all fixture expectations matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
