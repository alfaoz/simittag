Simittag
========
Simittag is a circular visual fiducial system. Each tag carries a data payload and can be used to estimate the full 6-DoF pose of the camera. This repository contains the Python reference implementation and a Rust port of the detector. The Rust port has no dependencies and also builds to WebAssembly.

Tags come in three variants. We recommend the M variant unless you have a specific reason to choose otherwise.

Table of Contents
=================
- [Overview](#overview)
- [Choosing a Variant](#choosing-a-variant)
- [Install](#install)
- [Usage](#usage)
  - [Generating Tags](#generating-tags)
  - [Getting Started with the Detector](#getting-started-with-the-detector)
    - [Python](#python)
    - [Rust](#rust)
    - [WebAssembly](#webassembly)
  - [Payload Modes](#payload-modes)
  - [Pose Estimation](#pose-estimation)
  - [Lens Distortion](#lens-distortion)
- [Performance](#performance)
- [Comparison with Other Fiducial Systems](#comparison-with-other-fiducial-systems)
- [Implementation Notes](#implementation-notes)
- [License](#license)

Overview
========
A Simittag is made of concentric rings. From the center out these are a black bullseye disk, a quiet ring, several rings of data cells, another quiet ring, and a black outer ring. The detector finds the outer ring and fits an ellipse to it. The pose is recovered from that ellipse. The data cells are then sampled in grayscale and decoded with Reed-Solomon error correction. Cells the sampler is not confident about are passed to the decoder as erasures rather than guesses.

The detector never returns unverified data. Every payload is checked against a CRC before it is reported, and a tag that fails the check is simply not reported. There is also no temporal filtering. Every frame is detected on its own.

The innermost data ring holds a fixed synchronization pattern. The tag's rotation is recovered by correlating against this pattern once. The same mechanism lets the detector identify the variant automatically, because the wrong variant's pattern fails to correlate.

Choosing a Variant
==================
The three variants share the same ring layout. Detection and pose estimation are identical for all of them. They differ only in the data grid.

| Variant | Grid | Payload | Distinct IDs | Corrects |
|:-:|:-:|:-:|:-:|:-:|
| T | 3×16 | 1 byte | 256 | 1 error / 1 erasure |
| M | 4×24 | 4 bytes | 16.7 M | 2 errors / 3 erasures |
| D | 5×36 | 11 bytes | 2⁸⁸ | 3 errors / 5 erasures |

Some heuristics for choosing:
1. If you need maximum detection distance, or expect motion blur, use T. Its cells are the largest, so they survive the most degradation.
2. If an ID is not enough and you need text, namespaces, or coordinates, use D.
3. Otherwise use M.

The detector identifies the variant automatically. You can also pin it to a single variant, which is faster.

Install
=======
The Python reference implementation requires NumPy and OpenCV:

```
git clone https://github.com/alfaoz/simittag.git
cd simittag
pip install numpy opencv-python
```

The Rust port has no dependencies:

```
cd rust
cargo build --release
```

This builds the `simittag` command-line tool at `rust/target/release/simittag`.

For the WebAssembly build, see [WebAssembly](#webassembly) below.

Usage
=====

## Generating Tags

```
python -m marker.generate --variant M --id 0x1234 --out tag.png
```

There is also a small command-line app:

```
python app.py encode --id 12345 --out tag.png
python app.py encode --raw "hi" --out tag.png
python app.py decode photo.png
```

The SVG generator in `marker/svg.py` produces the same geometry as the raster generator. It is important to leave a white quiet zone around the outer ring when printing, preferably square, but not necessary for high performance.

## Getting Started with the Detector

### Python

```python
import cv2
from simittag import detect
from simittag.spec import DEFAULT

gray = cv2.imread("frame.png", cv2.IMREAD_GRAYSCALE)
results = detect.detect(gray, DEFAULT)

for r in results:
    print(r["mode"], r["value"], r["center"], r["tilt_deg_approx"])
```

Decoding works without camera calibration. For a metrically correct pose, pass your camera intrinsics as a 3×3 matrix with `K=`.

### Rust

```
./rust/target/release/simittag detect frame.png
```

You can also use the `simittag-core` crate as a library. The command-line tool additionally provides a benchmark mode, a serve mode that reads raw grayscale frames from stdin, and the parity gates described under [Implementation Notes](#implementation-notes).

### WebAssembly

```
rust/build-wasm.sh
```

This builds two modules into `rust/dist/`. The `wasm/` module is single-threaded and uses SIMD. The `wasm-mt/` module is multi-threaded. Both expose a `detect` function that takes a grayscale buffer and the camera intrinsics and returns JSON.

The threaded build has unusual requirements. It needs nightly Rust, a rebuilt standard library, explicit shared-memory linker flags, and a cross-origin-isolated page to run on. All of this is documented in the build script. Read its comments before changing anything.

## Payload Modes

A T tag holds a single raw byte and nothing else. M and D tags start with a one-byte header that selects a mode:

* `ID` holds an unsigned integer. This is the default.
* `RAW` holds opaque bytes or short text.
* `TAGGED` holds a namespace byte and an ID, so independent deployments do not collide.
* `GEO` holds latitude, longitude, and altitude. It fits only in a D tag. A GEO tag knows its own position, so one detection tells the camera where it is in the world.

Payloads with an unknown mode decode as verified raw bytes. They are never misparsed. There is deliberately no URL mode.

## Pose Estimation

Every decoded detection includes the tag's pose. The translation is expressed in units of the tag's outer-ring radius. Multiply by the physical radius in meters to get metric translation. Tag size is measured across the outer edge of the black outer ring.

The camera frame has its origin at the camera center. The z-axis points out of the lens, x is to the right in the image, and y is down. The tag frame is centered on the tag. From the viewer's perspective, x is to the right, y is down, and z points into the tag surface.

An ellipse admits two pose interpretations. This is the circular counterpart of the planar pose ambiguity that square tags have. The detector evaluates both interpretations and picks the one confirmed by the decoded data grid. The two solutions converge as the tag becomes fronto-parallel, so the ambiguity is harmless exactly where it is hardest to distinguish.

Median pose accuracy on realistically degraded synthetic frames, variant M, tilts from 0 to 70 degrees: 0.01 to 0.03 degrees of tilt error, 0.07 degrees of full rotation error, 0.04% depth error, and about 0.6 px of center reprojection error.

## Lens Distortion

The pose math assumes a pinhole camera. Under radial distortion an off-center circle does not project to an ellipse, and the pose becomes biased. The effect is worst with wide lenses and tags near the edge of the frame. Pass your distortion coefficients to correct for it:

```python
detect.detect(gray, DEFAULT, K=K, dist=(k1, k2, p1, p2, k3))
```

The frame is undistorted once, with cached maps. With a typical webcam lens and the tag near the frame edge, the uncorrected detector loses 20% of its decodes and misreads rotation by 9.5 degrees. The corrected detector matches the pinhole control.

Performance
===========
Timings for a 1280x1280 frame containing six tags, variant auto-detection on, measured on a modern ARM processor:

| Detector | Time |
|---|---:|
| Rust native (rayon) | ~9 ms |
| WASM, threaded (8 workers) | ~15 ms |
| WASM, single-thread + SIMD | ~35 ms |
| Python reference (OpenCV, 14 threads) | ~65 ms |

> Detection range is not properly tested as of 13-Jul-2026

Comparison with Other Fiducial Systems
======================================
We did some head-to-head testing with identical print size, camera, and image degradation.

AprilTag out-ranges Simittag by roughly 2x.

DataMatrix and QR codes store more bytes in the same area, because squares tile and rings do not. They provide no pose.

Implementation Notes
====================
Its advised that the Rust detector is used production. The Python package defines the correct behavior of the format and the detector, and exists for reference, experimentation, and regenerating the test fixtures.

License
=======
TBD.
