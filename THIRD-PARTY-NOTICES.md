# Third-party notices

Simittag as a whole is licensed under the BSD 2-Clause License (see
[LICENSE](LICENSE)). The component below is derived from third-party code and
additionally carries its original license.

## OpenCV (Apache License 2.0)

`rust/simittag-core/src/fitellipse.rs` is a Rust port of the `fitEllipse`
(`fitEllipseNoDirect`) algorithm from OpenCV 4.13.0,
`modules/imgproc/src/shapedescr.cpp`.

Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
And other OpenCV contributors; see the OpenCV repository for the full list.

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use that file except in compliance with the License. You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

Modifications relative to the OpenCV original: translated from C++ to Rust;
the SVD least-squares solves use a Hestenes one-sided Jacobi routine instead
of OpenCV's; the rank-degenerate perturbation branch uses a fixed-seed jitter
instead of OpenCV's process-global RNG. The numerical behavior is pinned
against the OpenCV implementation by this repository's parity fixtures.
