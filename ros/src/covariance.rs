//! Empirical diagonal pose covariance.
//!
//! Fitted 2026-07-22 against the reference implementation's accuracy rig
//! (sim/exp_cov.py in the simittag dev tree): per-component robust error
//! std devs measured over a distance x tilt x degradation grid (variants T
//! and M, 640 px frames, ellipse diameters ~100-450 px, tilts 2-65 deg),
//! then fitted as power laws in the detected ellipse pixel diameter.
//! Cells with ellipse diameter > 450 px were excluded (frame-clipping
//! artifacts of the rig, not detector behavior).
//!
//! Published sigmas are 2x the fitted median, which covers the measured
//! envelope (worst cell / fit was x1.8 for depth, x2.6 for yaw, x2.2 for
//! out-of-plane). Lateral error measured as a near-constant fraction of the
//! tag diameter; its p90 is published directly. The out-of-plane multiplier
//! table is measured, interpolated in tilt: near fronto the rotation
//! direction is ill-conditioned and the error grows fast.
//!
//! COEFFICIENTS ARE FIT OUTPUTS. Regenerate with the harness, do not tune
//! by hand.

/// sigma = A * (px/100)^B, px clamped to the fitted range.
struct PowerLaw {
    a: f64,
    b: f64,
}

impl PowerLaw {
    fn eval(&self, px: f64) -> f64 {
        self.a * (px.clamp(30.0, 450.0) / 100.0).powf(self.b)
    }
}

/// depth: sigma_z / z (fit a=8.80e-4 b=-1.07, x2 safety)
const SIGMA_Z_FRAC: PowerLaw = PowerLaw { a: 1.76e-3, b: -1.07 };
/// in-plane rotation: radians, px-independent (fit 1.19e-3 rad, x2)
const SIGMA_YAW_RAD: f64 = 2.4e-3;
/// out-of-plane rotation at 45 deg tilt: radians (fit a=1.55e-3 b=-1.28, x2)
const SIGMA_OOP_RAD: PowerLaw = PowerLaw { a: 3.1e-3, b: -1.28 };
/// lateral: sigma_xy as a fraction of tag diameter (measured p90)
const SIGMA_XY_DIAM_FRAC: f64 = 5.0e-3;
/// measured out-of-plane multiplier vs detected tilt (deg), log-interpolated
const OOP_TILT_MULT: [(f64, f64); 5] =
    [(2.0, 82.4), (10.0, 4.19), (25.0, 2.0), (45.0, 1.0), (65.0, 0.69)];

fn oop_mult(tilt_deg: f64) -> f64 {
    let t = tilt_deg.clamp(OOP_TILT_MULT[0].0, OOP_TILT_MULT[4].0);
    for w in OOP_TILT_MULT.windows(2) {
        let ((t0, m0), (t1, m1)) = (w[0], w[1]);
        if t <= t1 {
            let a = (t - t0) / (t1 - t0);
            return (m0.ln() * (1.0 - a) + m1.ln() * a).exp();
        }
    }
    OOP_TILT_MULT[4].1
}

/// Row-major 6x6 for geometry_msgs/PoseWithCovariance:
/// (x, y, z, rot_x, rot_y, rot_z), diagonal only.
pub fn pose_covariance(
    px_diameter: f64,
    tilt_deg: f64,
    z_m: f64,
    tag_radius_m: f64,
    _fx: f64,
) -> [f64; 36] {
    let z = z_m.abs().max(1e-6);

    let s_z = SIGMA_Z_FRAC.eval(px_diameter) * z;
    let s_xy = SIGMA_XY_DIAM_FRAC * tag_radius_m * 2.0;
    let s_yaw = SIGMA_YAW_RAD;
    let s_oop = SIGMA_OOP_RAD.eval(px_diameter) * oop_mult(tilt_deg);

    let mut c = [0.0f64; 36];
    c[0] = s_xy * s_xy; // x
    c[7] = s_xy * s_xy; // y
    c[14] = s_z * s_z; // z
    // Out-of-plane rotation lives mostly on the optical x/y axes; in-plane
    // rotation (about the tag normal) mostly on z at moderate tilt. This
    // split is the usable approximation for a diagonal model.
    c[21] = s_oop * s_oop; // rot x
    c[28] = s_oop * s_oop; // rot y
    c[35] = s_yaw * s_yaw; // rot z
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_sane() {
        let c = pose_covariance(200.0, 45.0, 2.0, 0.05, 600.0);
        // depth sigma ~ 0.9e-3 * 2m * 2 -> ~3.4mm, variance ~1e-5
        assert!(c[14] > 1e-7 && c[14] < 1e-3);
        assert!(c[0] > 0.0 && c[35] > 0.0);
        // fronto must be much less certain out-of-plane than steep
        let fronto = pose_covariance(200.0, 2.0, 2.0, 0.05, 600.0);
        assert!(fronto[21] > 100.0 * c[21]);
        // covariance grows as the tag shrinks
        let far = pose_covariance(60.0, 45.0, 6.0, 0.05, 600.0);
        assert!(far[14] > c[14]);
    }
}
