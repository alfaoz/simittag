#!/usr/bin/env python3
"""Replay a directory of PNG frames (e.g. extracted from real footage)
through a running simittag node and summarize what decoded.

No CameraInfo is published on purpose: this also exercises the node's
FOV-guess path. Decode is calibration-robust; pose from a guess is
approximate, which is fine for a decode gate.

    python3 test/replay_video.py --frames /path/to/pngs \
        --expect M:id:0xabcdef T:id:0x2a
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray


class Player(Node):
    def __init__(self, image_topic, det_topic):
        super().__init__("simittag_video_replay")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(Image, image_topic, qos)
        self.results = {}
        self.create_subscription(Detection3DArray, det_topic, self._on_det, 50)

    def _on_det(self, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.results[key] = msg

    def send(self, path, seq):
        img = PilImage.open(path).convert("L")
        w, h = img.size
        msg = Image()
        msg.header.stamp.sec, msg.header.stamp.nanosec = seq, 7
        msg.header.frame_id = "camera_optical"
        msg.height, msg.width, msg.encoding, msg.step = h, w, "mono8", w
        msg.data = img.tobytes()
        self.pub.publish(msg)
        return (seq, 7)

    def wait(self, key, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if key in self.results:
                return self.results[key]
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--det-topic", default="/simittag/detections")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--expect", nargs="*", default=None,
                    help="if given, any other decoded payload is a failure")
    args = ap.parse_args()

    pngs = sorted(Path(args.frames).glob("*.png"))
    if not pngs:
        print("no frames found", file=sys.stderr)
        return 2

    rclpy.init()
    node = Player(args.image_topic, args.det_topic)
    deadline = time.time() + 10.0
    while node.pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.pub.get_subscription_count() == 0:
        print("node never subscribed", file=sys.stderr)
        return 2
    time.sleep(1.0)

    seen = Counter()
    frames_with = 0
    missed_replies = 0
    for seq, png in enumerate(pngs, start=1):
        msg = None
        for _ in range(3):
            key = node.send(png, seq)
            msg = node.wait(key, args.timeout)
            if msg is not None:
                break
        if msg is None:
            missed_replies += 1
            continue
        labels = [r.hypothesis.class_id for d in msg.detections for r in d.results]
        if labels:
            frames_with += 1
        seen.update(labels)

    node.destroy_node()
    rclpy.shutdown()

    total = len(pngs)
    print(f"frames: {total}, replies missed: {missed_replies}, "
          f"frames with >=1 decode: {frames_with} ({100.0*frames_with/total:.0f}%)")
    for label, cnt in seen.most_common():
        print(f"  {label}: {cnt} frames")

    if args.expect is not None:
        bad = [l for l in seen if l not in set(args.expect)]
        if bad:
            print(f"FALSE ACCEPTS: {bad}")
            return 1
        print("no unexpected payloads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
