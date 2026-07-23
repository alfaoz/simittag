> [!IMPORTANT]
> Simittag is early and experimental. Do not use it for critical infrastructure.

Simittag
========
[![ci](https://github.com/alfaoz/simittag/actions/workflows/ci.yml/badge.svg)](https://github.com/alfaoz/simittag/actions/workflows/ci.yml)

Simittag is a circular visual fiducial system. Each tag carries a data payload and yields the full 6-DoF pose of the camera from a single frame. This repository contains the Python reference implementation and a dependency-free Rust port of the detector, which also builds to WebAssembly. Tags come in five variants that share one ring layout; we recommend s16m unless you have a specific reason to choose otherwise.

<img src="docs/images/variants.png" alt="The three default Simittag variants" width="760">

Install
=======
From PyPI:

```
pip install simittag
```

Prebuilt Rust CLI binaries for Linux (x86_64 and aarch64), macOS (Apple silicon), and Windows are attached to each GitHub release.

From source:

```
git clone https://github.com/alfaoz/simittag.git
cd simittag
pip install numpy opencv-python
cd rust && cargo build --release
```

For the WebAssembly build, see `rust/build-wasm.sh`. For ROS 2 (Jazzy), see the [`ros/`](ros/) package in this repository.

Usage
=====
Generate a tag:

```
python -m marker.generate --variant s16m --id 0x1234 --out tag.png
```

Detect in Python:

```python
import cv2
from simittag import detect
from simittag.spec import DEFAULT

gray = cv2.imread("frame.png", cv2.IMREAD_GRAYSCALE)
results = detect.detect(gray, DEFAULT)

for r in results:
    print(r["variant"], r["mode"], r["value"], r["center"],
          r["tilt_deg"], r["inverted"])
```

Detect with the Rust CLI:

```
./rust/target/release/simittag detect frame.png
```

This prints one JSON line per decoded tag, with the payload, the pose, and the recovered ellipse. Two optional arguments pin the variant and the assumed horizontal field of view: `simittag detect frame.png s16m 78`.

Tags may be black on white or inverted white on black, and both polarities can appear in the same frame; results include an `inverted` boolean. Leave a white quiet zone around the outer ring when printing. Tag size is measured across the outer edge of the black outer ring.

<img src="docs/images/tag_size.png" alt="Where tag size is measured" width="520">

Variants
========
The five variants share the same ring layout, detection, and pose estimation. They differ only in the data grid.

| Variant | Alias | Grid | Payload | Distinct IDs | Corrects |
|:-:|:-:|:-:|:-:|:-:|:-:|
| sim48c12 | s4k | 3×16 | 12-bit ID | 4,096 | 2 errors / 3 erasures |
| sim48c8 | s256 | 3×16 | 1 byte | 256 | 1 error |
| sim48c16 | s64k | 3×16 | 16-bit ID | 65,536 | 1 error / 2 erasures |
| sim96c32 | s16m | 4×24 | 4 bytes | 16.7 M | 2 errors / 3 erasures |
| sim180c88 | sdata | 5×36 | 11 bytes | 2⁸⁸ | 3 errors / 5 erasures |

Every variant has two interchangeable names: a canonical technical name (`sim<cells>c<payload bits>`) and a short alias. Both are accepted wherever a variant is selected, and detections report both. The T, M, and D names from earlier releases are still accepted as input but are deprecated. Printed tags are unaffected by naming, since tags carry sync patterns, not names.

The detector identifies the default set (s4k, s16m, sdata) automatically. s256 and s64k are decoded when selected explicitly, pinned alone or in any explicit set. Pinning to a single variant is always the fastest configuration. Deploy at most one 3×16 variant per physical environment and configure the detector to match.

Choosing a variant:

1. For maximum detection distance or motion blur, use a 3×16 tracking tag: s4k by default, or s256 (selected explicitly) where its niche applies, as described below.
2. If an ID is not enough and you need text, namespaces, or coordinates, use sdata.
3. Otherwise use s16m.

s256 is the maximum-range choice, and measurably the most tolerant of steep tilt (beyond 25 degrees) and hard shadow edges. Prefer it, selected explicitly, for steep-view or harsh-lighting deployments that can accept a rare sub-floor wrong ID. Do not treat it as generally better under damage: at 30% occlusion or 64 px partial cover, s4k is the robust one. Its single payload byte carries the weakest code of the family, and below its reliable decode floor a small fraction of reads can return a wrong ID. In our measurements every wrong read happened on tags smaller than about 20 px, where fewer than three quarters of frames decode at all; there, roughly 0.5% of trials (under 1% of successful decodes) returned a wrong ID, and we measured zero wrong reads at 21 px and above in 3,000 trials. s4k, s16m, and sdata have not produced a wrong read in any of our measurements.

The 3×16 tags hold a bare unsigned ID and nothing else. s16m and sdata start with a one-byte header that selects a payload mode. `ID` holds an unsigned integer and is the default. `RAW` holds opaque bytes or short text. `TAGGED` holds a namespace byte and an ID, so independent deployments do not collide. `GEO` holds latitude, longitude, and altitude; it fits only in an sdata tag, and one detection then tells the camera where it is in the world. Payloads with an unknown mode decode as verified raw bytes, never misparsed. There is deliberately no URL mode.

Pose
====
Every decoded detection includes the tag's pose. Translation is expressed in units of the tag's outer-ring radius; multiply by the physical radius in meters to get metric translation. The camera frame has its origin at the camera center, with z pointing out of the lens, x to the right, and y down; this matches the ROS optical frame convention (REP-103). An ellipse admits two pose interpretations, the circular counterpart of the planar ambiguity of square tags; the detector evaluates both and keeps the one confirmed by the decoded data grid. Median accuracy on degraded synthetic frames with s16m at tilts from 0 to 70 degrees is 0.01 to 0.03 degrees of tilt error, 0.07 degrees of rotation error, 0.04% depth error, and about 0.6 px of center reprojection error. The pose math assumes a pinhole camera; pass distortion coefficients (`dist=` in Python, CameraInfo in ROS) to correct wide lenses. Frame conventions, the near-floor tilt bias, and the distortion measurements are in [docs/robustness.md](docs/robustness.md).

Calibration
===========
Metric pose needs camera intrinsics. The Python package solves them from photos of a printed calibration board:

```
simittag calibrate img1.png img2.png ... --out intrinsics.json
simittag decode photo.png --intrinsics intrinsics.json
```

Print a calibration sheet from the [studio](https://simittag.simitrobotics.com) and photograph it from varied positions and tilts; the solver needs at least 4 usable views with at least 6 board tags visible in each. Boards are self-describing: each sheet carries a descriptor tag encoding the layout and the tag variant, so the calibrator configures itself from the photos alone. The command reports fx, fy, cx, cy, the OpenCV distortion vector, and the reprojection RMS, interchangeable with any standard OpenCV calibration. Calibration is Python-only by design: the Rust and WebAssembly detectors consume intrinsics but do not produce them, and the ROS 2 node takes them from CameraInfo.

Performance
===========
Timings for a 1280x1280 frame containing six tags from the default variant set, auto-detection on, measured on an Apple M4 Pro:

| Detector | Time |
|---|--:|
| Rust native (rayon) | ~5 ms |
| WASM, single-thread + SIMD | ~23 ms |
| Python reference (OpenCV, 14 threads) | ~65 ms |

On the same machine, AprilTag (pupil-apriltags) measures 5.3 ms and ArUco (OpenCV) 3.0 ms on an equivalent six-tag frame. Prefer the single-threaded WASM build when variants beyond the default set are enabled; the multi-threaded build measures ~19 ms on the default set but loses to single-threaded on failure-heavy frames.

Range
=====
Measured with an A4-printed tag (175 mm outer diameter) on a 60-degree-HFOV camera at 15 degrees of tilt, under mild defocus, sensor noise, and JPEG compression. Range is the farthest distance with at least 90% decode. Every system below is measured on the identical rig at the identical threshold, so external vendor guidance that assumes conservative thresholds will read lower.

| System | Decode floor (px) | Range at 1280 px (m) | Range at 1920 px (m) |
|---|--:|--:|--:|
| Simittag s4k | ~23 | 8.4 | 12.6 |
| Simittag s256 | ~22 | 8.9 | 13.3 |
| Simittag s64k | ~24 | 8.1 | 12.2 |
| Simittag s16m | ~29 | 6.7 | 10.0 |
| Simittag sdata | ~34 | 5.6 | 8.5 |
| AprilTag 36h11 | ~20 | 9.6 | 14.4 |
| ArUco 6x6 | ~19 | 10.0 | 15.0 |

The decode floor is the smallest outer-ring diameter, in image pixels, that still meets the 90% threshold. Range scales linearly with print size and with camera resolution.

Small or degraded candidates that fail to decode are retried with deconvolution against defocus and motion blur, rethresholding against hard shadows, and a bullseye-geometry fallback under occlusion. Under a straight-edge occluder, a 96 px s4k tag decodes through 30% area occlusion in 57 of 60 trials; on the same frames AprilTag and ArUco stop detecting at 5% occlusion. Across 600 procedurally generated clutter frames the detector produced zero false positives with all five variants enabled, and across about 280,000 adversarial trials between the same-grid variants it produced zero cross-decodes. Mechanisms and full tables are in [docs/robustness.md](docs/robustness.md).

Implementation Notes
====================
Use the Rust detector in production. The Python package defines the correct behavior of the format and the detector, and exists for reference, experimentation, and regenerating the test fixtures. The fixtures in `fixtures/` hold the two implementations together bit-for-bit; `./check.sh` runs the full contract, and CI runs the same script on every push.

License
=======
Simittag is licensed under the [BSD 2-Clause License](LICENSE). The Rust ellipse-fitting routine is derived from OpenCV; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

The tag format is free for anyone to implement.
