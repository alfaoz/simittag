simittag_ros
============

ROS 2 node for the simittag circular fiducial system (the `ros/` package of the [main repository](https://github.com/alfaoz/simittag)). Subscribes an image stream plus CameraInfo, publishes verified tag payloads and metric 6-DoF poses with covariance.

Simittag tags carry a data payload (IDs, coordinates, raw bytes) and yield the full camera pose from a single tag. Every reported payload has passed Reed-Solomon decoding, a CRC, and a matched-filter verification against the re-encoded pattern; there is no temporal filtering. See the main repository for the tag format, accuracy tables, and the honest comparison with AprilTag and DataMatrix.

Status: early. Targets ROS 2 Jazzy on Ubuntu 24.04, built on [r2r](https://github.com/sequenceplanner/r2r).

Quickstart
----------

1. Print a tag from the [studio](https://simittag.simitrobotics.com) (variant s16m is the default recommendation) and note the printed diameter, or measure the outer edge of the outer black ring.
2. Build:

```
sudo apt install ros-jazzy-vision-msgs ros-jazzy-tf2-msgs clang libclang-dev
# rust via rustup.rs if you do not have it
cd ~/ws/src && git clone https://github.com/alfaoz/simittag
cd ~/ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select simittag_ros
```

3. Run against your camera driver:

```
source install/setup.bash
ros2 run simittag_ros simittag_node --ros-args \
  -p image_topic:=/camera/image_raw \
  -p camera_info_topic:=/camera/camera_info \
  -p tag_diameter_m:=0.16
ros2 topic echo /simittag/detections
```

Or use the launch file with `config/params.yaml`:

```
ros2 launch simittag_ros simittag.launch.py
```

Interface
---------

Subscribes:

- `image_topic` (sensor_msgs/Image, sensor-data QoS, depth 1): mono8, rgb8, or bgr8. Frames are dropped rather than queued when the node falls behind.
- `camera_info_topic` (sensor_msgs/CameraInfo): K and plumb-bob distortion are used directly; a rectified stream with D zeroed also works. Without CameraInfo the node warns once and uses a 60-degree-FOV guess; decoding is unaffected, pose is approximate.

Publishes:

- `/simittag/detections` (vision_msgs/Detection3DArray), one message per processed frame, empty when nothing verified. Per detection:
  - `results[0].hypothesis.class_id`: the payload, e.g. `sim96c32:id:0x2a`, `sim180c88:geo:48.858370,2.294481,+330`, `sim96c32:tag:12:0x1f4`, `sim180c88:raw:c0ffee`
  - `results[0].hypothesis.score`: decode-verify correlation (0 to 1)
  - `results[0].pose`: PoseWithCovariance in the camera optical frame, meters. The covariance diagonal is an empirical model fitted on the reference accuracy harness (see `src/covariance.rs`), so the output can be fused (robot_localization and friends) without hand-tuned noise.
- `/tf` (when `publish_tf`): `<optical frame> -> simittag/<payload>` per detection.
- `/simittag/debug_image` (when `debug_image`): input with rings and axes drawn.

Frame conventions: the input image header frame is assumed to be a REP-103 camera optical frame. The published tag frame is x right, y up, z out of the tag toward the viewer (the apriltag_ros convention, so migration is a topic remap).

Parameters
----------

| name | default | |
|---|---|---|
| `tag_diameter_m` | required | outer edge of the outer black ring, meters, as printed |
| `variant` | `auto` | a canonical name (`sim96c32`), an alias (`s16m`, including the experimental `s64k`), or `auto`; pinning is faster and refuses other variants. Deprecated `T`/`M`/`D` still accepted |
| `image_topic` | `/camera/image_raw` | |
| `camera_info_topic` | `/camera/camera_info` | |
| `detections_topic` | `/simittag/detections` | |
| `publish_tf` | `true` | |
| `pose_only` | `false` | also emit undecoded round candidates (no payload, no TF) |
| `debug_image` | `false` | |
| `detect_width` | `0` | 0 = native; e.g. 640 to downscale before detecting |
| `frame_id` | `""` | override the outgoing frame id |

All tags in view are reported every frame; mixed variants and both tag polarities are handled. One tag is enough for a full pose; the 2-fold circle-pose ambiguity is resolved per frame geometrically and by decode verification.

Performance
-----------

The detector is simittag-core with rayon: about 9 ms for a 1280 px six-tag frame on an M-series laptop core, comfortably inside 30 Hz. On weak boards set `detect_width` (range scales linearly with resolution; see the main README's range tables).

Testing
-------

CI builds the package in a `ros:jazzy` container and replays the repository's fixture frames (the cross-implementation compatibility contract) through the running node, asserting exact payloads and poses. Run locally from the repository root:

```
ros2 run simittag_ros simittag_node --ros-args -p tag_diameter_m:=2.0 &
python3 ros/test/replay_fixtures.py --fixtures fixtures
```

`test/replay_video.py` replays any directory of extracted camera frames and flags unexpected payloads, for validating against your own footage.

License
-------

BSD-2-Clause, same as simittag.
