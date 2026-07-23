Robustness and Failure-Mode Detail
==================================

This document holds the measurement detail behind the robustness claims in the README. Numbers come from the synthetic A4 rig defined in the README's Range section unless noted otherwise.

Decode Retry Ladder
===================
When a candidate fails to decode, the detector retries with four mechanisms. Each runs only after a failure, so healthy frames pay nothing, and every retry result still has to pass the sync, Reed-Solomon, CRC, and decode-verify gates.

1. Defocus deconvolution. At long range the limit is not finding the tag, since the outer ring is detected far past the decode floor. The limit is inter-symbol interference: defocus bleeds neighboring data cells into each other. The detector deconvolves the tag patch with a Wiener filter against an assumed Gaussian point spread and retries. Candidates up to 160 px are retried, with point-spread widths up to 2.4 px. At a defocus of sigma 2.0 on the A4 rig this raises the 90%-decode range of s16m from 2.6 m to 5.0 m and of sdata from 2.5 m to 3.9 m, with s256 improving from 6.0 m to 6.9 m.
2. Motion smear. Under motion blur the point spread is a line, not a Gaussian. When a failed candidate shows directional smear, measured by structure-tensor coherence, the detector deconvolves a line PSF along the estimated blur axis and retries. On a 180 px tag this roughly doubles the tolerated smear length: s16m decodes through 30 px of smear instead of 18, sdata through 24 instead of 12, and s256 through about 40 instead of 30.
3. Hard shadow edges. A shadow edge across the tag defeats the global black/white reference pair, which then misclassifies the shadowed half of the grid. The retry rethresholds every cell against an illumination plane fitted to the tag's own quiet rings. Measured on half-plane shadows at 0.4x, 0.3x, and 0.25x brightness: decodes went from 14, 9, and 11 of 20 to 20 of 20.
4. Occlusion. Handled by geometry rather than deconvolution; see the next section.

Occlusion
=========
When an occluder breaks the outer-ring contour, the intact bullseye is fitted as its own candidate and recovers the same projective geometry after scaling by its known radius. Small lone disks, down to a fitted radius of 4 px, are admitted into this fallback only, so normal frames pay nothing for it. Below about 55 px the bullseye ellipse is too small to carry the data grid, and occlusion tolerance ends.

Measured with a straight-edge occluder on the A4 rig at 15 degrees of tilt, standard degradation, 60 trials per cell. Decode counts at 5, 10, 15, 20, and 30% area occlusion:

| Tag size | Variant | 5% | 10% | 15% | 20% | 30% |
|---:|---|--:|--:|--:|--:|--:|
| 96 px | s4k | 59 | 59 | 58 | 57 | 57 |
| 96 px | s256 | 60 | 60 | 54 | 52 | 23 |
| 96 px | s64k | 58 | 59 | 58 | 56 | 38 |
| 96 px | s16m | 58 | 59 | 59 | 56 | 39 |
| 80 px | s4k | 39 | 33 | 35 | 27 | 26 |
| 80 px | s256 | 58 | 55 | 51 | 47 | 18 |
| 80 px | s64k | 46 | 37 | 44 | 31 | 24 |
| 80 px | s16m | 41 | 32 | 29 | 24 | 17 |
| 64 px | s4k | 22 | 15 | 9 | 8 | 8 |
| 64 px | s256 | 8 | 5 | 4 | 7 | 1 |
| 64 px | s64k | 20 | 15 | 6 | 3 | 1 |
| 64 px | s16m | 29 | 15 | 9 | 5 | 6 |

At 40% occlusion and above, no variant decodes. At 48 px, no variant decodes under any occlusion. On the same frames AprilTag and ArUco decode at most 2 of 60 at any occlusion of 5% or more, at every size. The sweep logged zero wrong IDs.

Pose Conventions and Accuracy
=============================
The camera frame has its origin at the camera center. The z-axis points out of the lens, x is to the right in the image, and y is down. This matches the ROS optical frame convention (REP-103), so the pose drops into a ROS pipeline without a frame conversion. The tag frame is centered on the tag. From the viewer's perspective, x is to the right, y is down, and z points into the tag surface.

An ellipse admits two pose interpretations. This is the circular counterpart of the planar pose ambiguity that square tags have. The detector evaluates both interpretations and picks the one confirmed by the decoded data grid. The two solutions converge as the tag becomes fronto-parallel, so the ambiguity is harmless exactly where it is hardest to distinguish.

Median pose accuracy on realistically degraded synthetic frames, variant s16m, tilts from 0 to 70 degrees: 0.01 to 0.03 degrees of tilt error, 0.07 degrees of full rotation error, 0.04% depth error, and about 0.6 px of center reprojection error.

Pose quality degrades before decoding does. Near the decode floor (tags 22 to 40 px across) the median tilt error grows to about 2 degrees, with a systematic underestimate of up to 3 degrees, because blur rounds the ellipse. The fitted tag scale also runs 1.3% large at 30 px and 0.5% large at 36 px, with no measurable bias at 44 px and above. If you decode at extreme range, trust the payload more than the tilt.

Lens Distortion
===============
The pose math assumes a pinhole camera. Under radial distortion an off-center circle does not project to an ellipse, and the pose becomes biased. The effect is worst with wide lenses and tags near the edge of the frame. Pass your distortion coefficients to correct for it:

```python
detect.detect(gray, DEFAULT, K=K, dist=(k1, k2, p1, p2, k3))
```

The frame is undistorted once, with cached maps. With a typical webcam lens and the tag near the frame edge, the uncorrected detector loses 20% of its decodes and misreads rotation by 9.5 degrees. The corrected detector matches the pinhole control.

Accept Gates
============
Two accept gates guard the decode search. A sync-ring correlation gate filters non-tag grids before Reed-Solomon runs. After any successful decode, the observed grid is correlated against the re-encoded decoded pattern, a matched filter of the image against what was decoded, and the result is rejected below 0.73.

The 0.73 gate was calibrated against clutter. In calibration, correct decodes scored at least 0.807. Across 600 procedurally generated ring-like clutter frames, CRC-valid wrong-decode candidates scored at most 0.673 and none passed the gate. This leaves a measured empty interval between false and correct candidates while preserving margin for degraded real tags. The margin is re-measured whenever codec behavior changes and currently stands at 0.6752.

Per-variant floors sit above the global gate where measurements demand it. The nibble variants s4k and s64k carry a decode-verify floor of 0.78. It was set from measured wrong-variant decode distributions: the worst same-grid wrong-variant survivor scored 0.759 across more than 43,000 trials, with a p99 of 0.664, so 0.78 rejects every observed event with margin. The floor was motivated by a single measured leak: one inverted s256 tag read as s64k at 40 px through the deconvolution retry, scoring 0.759, one event in about 38,000 trials before the fix and zero in about 102,000 after. That frame is pinned as a regression test. s256 carries a floor of 0.76, shipped together with disabling its ranked erasures, a pairing that gained 8.1% floor recall and cut wrong IDs by 19%.

Cross-Variant Rejection
=======================
The three 3x16 variants share one grid and one print geometry, so a wrong-variant decode samples exactly the same cells. Telling them apart rests on their synchronization patterns and codecs rather than on geometry. The sync patterns were chosen jointly for worst-case cross-correlation margin: the worst cross-correlation over all shifts and polarities between any pair is 6 of 16, against a gate that requires 12.

Cross-rejection was measured directly. Across roughly 280,000 trials of each same-grid variant's tags against the others' decoders, at sizes from 16 to 128 px, in both polarities, with degradations up to heavy defocus with strong noise, there were zero cross-decodes. The 600-frame clutter suite also remains at zero false positives with all five variants enabled.

Deploy at most one 3x16 variant per physical environment and configure the detector to match. Explicit multi-variant sets are supported for fleet migrations.
