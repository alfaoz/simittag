Changelog
=========

v0.2.0
======

Variants and naming
-------------------
- Two-layer variant names: canonical `sim<cells>c<payload bits>` names plus
  short aliases (sim48c8/s256, sim96c32/s16m, sim180c88/sdata). The T, M, and
  D names remain accepted as input aliases and are deprecated.
- New variant sim48c12/s4k: 3x16 grid, 12-bit ID (4,096 IDs), RS(8,4)
  correcting 2 errors or 3 erasures, 0.78 decode-verify floor.
- New variant sim48c16/s64k: 3x16 grid, 16-bit ID (65,536 IDs) on GF(16)
  nibbles, RS(8,5) correcting 1 error or 2 erasures.
- New default detection set: s4k + s16m + sdata. s256 and s64k decode when
  selected explicitly. Policy: deploy at most one 3x16 variant per physical
  environment and configure the detector to match.

Fixes
-----
- Headless `detect()` (Python and Rust) now sorts the two conic pose
  interpretations by the bullseye origin error. Off-axis tags previously took
  the wrong mirror in about 8 to 10 percent of poses; the mass1200 regression
  tilt p95 went from 19.2 to 0.19 degrees with decode stats unchanged.

Erasure and verify configuration
--------------------------------
- s256: ranked erasures disabled (CONF_ERASURE=0) and verify floor 0.76.
  Measured against the old config: floor recall +8.1%, wrong IDs -19%.
- s4k and s64k: ranked-erasure threshold raised to 0.40. The erasure audit
  showed erasures are load-bearing for the nibble variants under occlusion
  (+27 to +44 decodes of 600 at 30% occlusion, zero added wrong IDs).

Calibration
-----------
- Board descriptor v2 carries the tag variant; generated boards now default
  to s4k. `calibrate()` self-configures from the sheet's descriptor tag.
  v1 descriptors are byte-frozen as s256 and keep calibrating.

WebAssembly
-----------
- Rebuilt and measured: single-threaded 22.5 ms (previously quoted about
  35 ms), multi-threaded 19.1 ms on the default six-tag frame. The
  multi-threaded build loses to single-threaded on failure-heavy frames, so
  prefer single-threaded when variants beyond the default set are enabled.

Documentation
-------------
- s256 range figures corrected to reproducible numbers (the previously quoted
  20.6 px decode floor came from an unmerged intermediate build; the
  reproducible floor is about 22 px, 13.3 m on the A4 rig at 1920 px).
- README rewritten around the five-variant family; robustness and
  failure-mode detail moved to docs/robustness.md.

Compatibility
-------------
- All five tag formats are frozen. Printed tags of any variant, from any
  release, decode forever.
- T, M, and D variant names keep working as input aliases.
- Version 1 calibration board descriptors keep calibrating.
