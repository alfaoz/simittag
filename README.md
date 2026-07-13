# simittag

circular fiducial marker. one tag gives you a decoded payload plus the full 6-dof camera pose.

## how it works

concentric rings, center out: black bullseye, quiet ring, a few rings of data sectors, quiet ring, black outer ring. the detector finds the outer ring, fits an ellipse to it, gets the pose from the conic, then samples the data cells in grayscale and runs reed-solomon. you either get a crc-verified payload or nothing. never a maybe.

## where this sits, honestly

- apriltag out-ranges this by about 2.2x. same print size, same camera, same degradation, real head-to-head. square corners are just more pixel-efficient than a ring. if all you need is tracking, use apriltag.
- datamatrix packs more bytes into the same area. squares tile, rings don't.

so why bother: neither of those gives you data *and* pose from one mark. datamatrix has no pose at all, apriltag carries a handful of bits. simittag does both, and reads fine at steep angles.

vs the code cantag actually shipped in 2005: real RS over GF(256) with erasure support plus a crc (cantag had parity bits), grayscale soft sampling instead of one hard pixel per cell, sub-pixel ring edges, adaptive thresholding.

## variants

same rings, same detection, same pose math. only the data grid changes.

| variant | grid | payload | ids | corrects |
|:-:|:-:|:-:|:-:|:-:|
| **T** | 3×16 | 1 B | 256 | 1 err / 1 eras |
| **M** | 4×24 | 4 B | 16.7 M | 2 err / 3 eras |
| **D** | 5×36 | 11 B | a lot | 3 err / 5 eras |

T when you want range, D when you want bytes, M otherwise. the innermost data ring is a sync sequence, so rotation comes from one circular correlation, and the detector can tell the variants apart on its own (the wrong variant's sync doesn't correlate, and the crc settles it).

payload modes: plain id, raw bytes, tagged (namespace + id so two deployments don't collide), and geo — lat/lon/alt packed into 10 bytes, D only. a geo tag knows where it is, so a single detection tells the camera where *it* is, in the world. there is no url mode on purpose.

## numbers

all measured on degraded renders (blur, noise, lighting, lens distortion, motion) against ground truth, then checked on a real webcam. no temporal filtering anywhere. if a frame fails, it fails.

- pose, variant M, tilts 0–70°: tilt error 0.01–0.03°, full rotation 0.07°, depth 0.04%, center ~0.6 px.
- range, A4 print, 1080p, 60° lens: T ~7 m, M ~6 m, D ~5 m. scales linearly with resolution and print size. (apriltag: ~14.5 m. see above.)
- motion blur: a ~180 px T tag survives 20 px of smear, D about 9. holds up on a simulated 6 m/s conveyor with a 0.2 ms strobe.
- lens distortion: uncorrected k1=−0.25 near the frame edge loses 20% of decodes and reads rotation 9.5° wrong. pass `dist=` to `detect()` and it's back to pinhole numbers.

## python and rust

the python package (`simittag/`, `marker/`) is the reference. the rust port (`rust/`) is the one you ship — no opencv, no dependencies, every cv2/numpy call reimplemented by hand and checked against golden fixtures in `fixtures/`. the fixtures are the python implementation's exact outputs; rust has to reproduce them and never calls python.

current gates: codec bit-exact, 10k randomized codec cases with identical decisions, pose geometry to 1e-9, imaging stages bitwise, 124/124 candidate sets within 0.1 px, 126/126 frames with identical decode decisions and pose diff under 1e-4.

things that turned out to matter for the port: cv2's 8-bit gaussian blur is a fixed-point path, fitEllipse had to be ported line-for-line from opencv 4.13, undistort emulates the quantized CV_16SC2 remap bit-exact, and findContours is suzuki–abe with the full hierarchy because the nesting tree is what the frontend walks.

speed, 1280² frame with six tags: ~9 ms native (rayon), ~15 ms threaded wasm, ~35 ms single-thread wasm. the 14-thread opencv/python reference needs 65 ms on the same frame.

## use

```bash
python -m marker.generate --variant M --id 0x1234 --out tag.png
python app.py decode photo.png
```

rust:

```bash
cd rust && cargo build --release
./target/release/simittag detect frame.png
./target/release/simittag parity-spec ../fixtures/spec.json   # and the other gates
```

wasm builds (both single-thread and threaded) land in `rust/dist/`:

```bash
rust/build-wasm.sh
```

the threaded build needs nightly, build-std, explicit shared-memory link args, and a cross-origin-isolated page to run on. it took a while to get right. the script comments explain all of it — read them before changing anything.

## layout

```
simittag/   spec, GF(256) RS codec, payload modes, detector
marker/     png + svg generators
rust/       simittag-core, simittag-cli (gates + bench), simittag-wasm
fixtures/   pinned reference outputs, the parity contract
```

every python module self-tests: `python -m simittag.codec` etc.

## license

tbd.
