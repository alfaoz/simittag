Simittag
========
Simittag is a circular visual fiducial system. Each tag carries a data payload (1–11 bytes, Reed-Solomon protected) and provides full 6-DoF pose estimation from a single detection. This repository contains the Python reference implementation and a dependency-free Rust port of the detector, which also builds to WebAssembly.

A detection either returns a CRC-verified payload or nothing. There is no unverified output and no temporal filtering: every frame stands on its own.

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
- [Support](#support)

Overview
========
A Simittag consists of concentric rings, read from the center out: a black bullseye disk, a quiet ring, several rings of data sectors, a quiet ring, and a black outer ring. The detector locates the outer ring with an adaptive threshold and contour analysis, fits an ellipse to it with sub-pixel refinement, recovers the pose from the conic, and samples the data grid in grayscale. Weak cells are passed to the Reed-Solomon decoder as erasures rather than hard guesses, and a CRC8 rejects false positives.

The innermost data ring is a fixed sync sequence. Rotation is recovered by a single circular cross-correlation against it, so no search over the error-correction code is needed, and the detector can identify the variant automatically: a wrong variant's sync fails to correlate, and the CRC confirms the winner.

Choosing a Variant
==================
All three variants share the same radial layout, so detection and pose estimation are identical. They differ only in the data grid.

| Variant | Grid | Payload | Distinct IDs | Corrects |
|:-:|:-:|:-:|:-:|:-:|
| T | 3×16 | 1 byte | 256 | 1 error / 1 erasure |
| M | 4×24 | 4 bytes | 16.7 M | 2 errors / 3 erasures |
| D | 5×36 | 11 bytes | 2⁸⁸ | 3 errors / 5 erasures |

Some heuristics:
1. If you need maximum detection distance or expect motion blur, use T. Fewer, larger cells survive the most degradation.
2. If you need more than an ID — text, namespaces, or coordinates — use D.
3. Otherwise use M.

You can pin the detector to a single variant for speed, or let it auto-detect.

Install
=======
Python (reference implementation, requires `numpy` and `opencv-python`):

```
git clone https://github.com/alfaoz/simittag.git
cd simittag
pip install numpy opencv-python
```

Rust (no external dependencies):

```
cd rust
cargo build --release
```

This produces the `simittag` CLI at `rust/target/release/simittag`. The core library (`simittag-core`) has an optional `parallel` feature (rayon) which the release build enables.

WebAssembly: see [WebAssembly](#webassembly) below.

Usage
=====

## Generating Tags

```
python -m marker.generate --variant M --id 0x1234 --out tag.png
```

or via the small CLI app:

```
python app.py encode --id 12345 --out tag.png
python app.py encode --raw "hi" --out tag.png
```

`marker/svg.py` generates SVG output that matches the raster generator cell-for-cell, for printing at exact physical scale. Print tags with a white quiet zone around the outer ring; the detector needs the ring's outer edge to contrast cleanly against its surroundings.

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

Pass `K=` (a 3×3 intrinsics matrix) for metrically correct pose; decoding works without it.

### Rust

```
./rust/target/release/simittag detect frame.png
```

or use `simittag-core` as a library. The CLI also provides `bench`, a `serve` mode that accepts raw grayscale frames on stdin for harness integration, and the parity gates described under [Implementation Notes](#implementation-notes).

### WebAssembly

```
rust/build-wasm.sh
```

builds two variants into `rust/dist/`: `wasm/` (single-threaded, SIMD) and `wasm-mt/` (threaded via wasm-bindgen-rayon). The module exposes `detect(gray, w, h, fx, fy, cx, cy, versions, poseOnly)` returning JSON.

The threaded build requires nightly Rust with `build-std`, explicit shared-memory linker arguments, and a cross-origin-isolated page (COOP `same-origin`, COEP `credentialless`) to run on. These requirements are non-obvious and all documented in the build script; read its comments before modifying it.

## Payload Modes

T tags are headerless: the byte is a raw ID. M and D payloads start with a one-byte header (version and mode):

* `ID` — a big-endian unsigned integer. The default.
* `RAW` — opaque bytes or short text.
* `TAGGED` — a namespace byte plus an ID, so independent deployments do not collide.
* `GEO` (D only) — latitude, longitude, and altitude packed into 10 bytes. A GEO tag knows its own position, so a single detection gives the camera its absolute position in the world.

Unknown future modes decode as verified raw bytes, never a misparse. There is deliberately no URL mode.

## Pose Estimation

Every decoded detection includes the pose recovered from the conic. The translation is in units of the marker's outer-ring radius: multiply by the physical radius in meters to get metric translation. The tag size is measured across the outer edge of the black outer ring.

Coordinate system: the camera frame has its origin at the camera center, z pointing out of the lens, x right and y down in the image. The tag frame is centered on the tag; from the viewer's perspective, x is to the right, y is down, and z points into the tag surface.

A note on ambiguity: an ellipse alone admits two pose interpretations (the circular equivalent of planar pose ambiguity). The detector evaluates both candidate homographies and selects using the decoded data grid and reprojection consistency. The two solutions converge as the tag approaches fronto-parallel, where the ambiguity is harmless by construction.

Accuracy, measured against ground truth on realistically degraded renders (variant M, tilts 0–70°, median): 0.01–0.03° tilt error, 0.07° full rotation error, 0.04% depth error, ~0.6 px center reprojection.

## Lens Distortion

The conic pose assumes a pinhole camera. Under radial distortion an off-center circle is not an ellipse, and the pose is silently biased — worst with wide lenses and tags near the frame edge. Pass distortion coefficients to correct for this:

```python
detect.detect(gray, DEFAULT, K=K, dist=(k1, k2, p1, p2, k3))
```

The frame is undistorted once with cached maps. Measured with a typical webcam lens (k1 = −0.25) and the tag near the edge: uncorrected loses 20% of decodes and reads rotation 9.5° wrong; corrected is indistinguishable from the pinhole control.

Performance
===========
On an M-series laptop, 1280×1280 frame containing six tags, auto-detect:

| Detector | Time |
|---|---:|
| Rust native (rayon) | ~9 ms |
| WASM, threaded (8 workers) | ~15 ms |
| WASM, single-thread + SIMD | ~35 ms |
| Python reference (OpenCV, 14 threads) | ~65 ms |

Detection range, A4-printed tag (175 mm ring), 1080p camera, 60° lens, mild blur/noise, ≥90% decode rate: T ~7 m, M ~6 m, D ~5 m. Range scales linearly with camera resolution and print size.

Motion blur degrades gracefully to a cliff: a ~180 px T tag decodes through 20 px of smear, D through 9 px. Verified on a simulated 6 m/s conveyor with a 0.2 ms strobe.

Comparison with Other Fiducial Systems
======================================
Measured head-to-head under identical print size, camera, and degradation:

* **AprilTag** (tag36h11) out-ranges Simittag-T by roughly 2.2× (~14.5 m vs ~6.5 m at 1080p). This is architectural: square corners and a square bit grid are more pixel-efficient than a ring. If you only need long-range tracking, use AprilTag.
* **DataMatrix / QR** pack more bytes per area — squares tile, rings do not — but provide no pose.

Simittag's niche is the combination: a useful payload and full 6-DoF pose from one circular mark, readable at steep angles.

Implementation Notes
====================
The Python package is the reference; the Rust port is what ships. Every OpenCV/NumPy operation the detector uses is re-implemented by hand in Rust and verified against golden fixtures (`fixtures/`) — the reference implementation's exact outputs. Rust never calls Python. The parity gates, runnable via the CLI:

| Gate | Command | Status |
|---|---|---|
| Spec and codec vectors | `parity-spec`, `parity-codec` | bit-exact |
| 10,000 randomized codec cases | `cross-gen` | identical decisions |
| Conic pose geometry | `parity-geometry` | agrees to 1e-9 |
| Imaging stages (blur, threshold, contours, fitEllipse) | `parity-stages` | bitwise |
| Candidate sets, all frames | `parity-candidates` | 124/124 within 0.1 px |
| Full detector decisions and pose | `parity-detect` | 126/126, pose diff < 1e-4 |

Port details that turned out to matter: OpenCV's 8-bit Gaussian blur is a fixed-point code path; `fitEllipse` is ported line-for-line from OpenCV 4.13, since all tuned thresholds were calibrated against that exact fit; undistortion reproduces the quantized CV_16SC2 remap bit-exactly; contour extraction is Suzuki–Abe with the full hierarchy, because the detector walks the nesting tree.

Support
=======
Please open an issue on this repository for questions.
