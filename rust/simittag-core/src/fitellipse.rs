//! cv2.fitEllipse, ported line-for-line from OpenCV 4.13.0's
//! fitEllipseNoDirect (modules/imgproc/src/shapedescr.cpp) -- the Dr. Daniel
//! Weiss algorithm. This is parity risk #1: every tuned threshold in the
//! detector (roundness 0.03, the 0.18/0.25 dedup, sanity gates) was measured
//! against THIS fit, not a textbook Fitzgibbon fit. Faithful details:
//!
//! * points pass through f32 (Point2f), including the centroid accumulation;
//! * scale = 100/sum(|p-c|_1), rows built as [-px^2, -py^2, -px*py, px, py],
//!   b = 10000, solved by SVD least squares (Hestenes one-sided Jacobi here --
//!   same pseudo-inverse, well inside the ±0.1px candidate gate);
//! * 2x2 center solve, 3-var re-fit with b = 1;
//! * angle = -0.5*atan2(C, B-A); width/height from |A+B∓t|; swap-with-+90 only
//!   when width > height (else angle stays 0 -- true to the original);
//! * results cast to f32 exactly like the RotatedRect the Python side saw.
//!
//! The rank-degenerate perturbation branch (wd[0]*FLT_EPS > wd[4], random
//! point jitter) is ported with a fixed-seed jitter; it is unreachable for the
//! detector's use (round contour arcs are never rank-deficient), and parity
//! would be lost there anyway since OpenCV draws from its process-global RNG.

use crate::pose::EllipseGeom;

const FLT_EPSILON: f64 = f32::EPSILON as f64;

/// Least squares via one-sided Jacobi SVD. Returns (x, singular_desc).
/// Also used by the detector's quadratic peak fit (numpy.linalg.lstsq stand-in).
pub fn lstsq(a: &[f64], rows: usize, cols: usize, b: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // column-major working copy of A, plus V accumulator
    let mut u: Vec<f64> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| a[i * cols + j]))
        .collect();
    let mut v = vec![0f64; cols * cols];
    for j in 0..cols {
        v[j * cols + j] = 1.0;
    }
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..cols - 1 {
            for q in p + 1..cols {
                let (mut alpha, mut beta, mut gamma) = (0f64, 0f64, 0f64);
                for i in 0..rows {
                    let up = u[p * rows + i];
                    let uq = u[q * rows + i];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma.abs() <= 1e-300 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..rows {
                    let up = u[p * rows + i];
                    let uq = u[q * rows + i];
                    u[p * rows + i] = c * up - s * uq;
                    u[q * rows + i] = s * up + c * uq;
                }
                for i in 0..cols {
                    let vp = v[p * cols + i];
                    let vq = v[q * cols + i];
                    v[p * cols + i] = c * vp - s * vq;
                    v[q * cols + i] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut w: Vec<(f64, usize)> = (0..cols)
        .map(|j| {
            let n: f64 = (0..rows).map(|i| u[j * rows + i] * u[j * rows + i]).sum();
            (n.sqrt(), j)
        })
        .collect();
    w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let wmax = w[0].0;
    let tol = f64::EPSILON * wmax * rows.max(cols) as f64;
    let mut x = vec![0f64; cols];
    for &(wj, j) in &w {
        if wj <= tol {
            continue;
        }
        // coefficient = (u_j . b) / w_j, then x += coef * v_j
        let mut d = 0f64;
        for i in 0..rows {
            d += u[j * rows + i] * b[i];
        }
        d /= wj * wj; // u col is unnormalized (norm = w_j): divide twice
        for i in 0..cols {
            x[i] += d * v[j * cols + i];
        }
    }
    (x, w.iter().map(|&(wj, _)| wj).collect())
}

/// Port of fitEllipseNoDirect for integer contour pixels (CV_32S path).
pub fn fit_ellipse(pts: &[(i32, i32)]) -> EllipseGeom {
    let ptsf: Vec<(f32, f32)> = pts.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    fit_ellipse_pts(&ptsf)
}

/// fitEllipseNoDirect on float32 points (CV_32F path -- what _refine_ellipse
/// feeds it). Both paths meet at Point2f, exactly like OpenCV.
pub fn fit_ellipse_pts(ptsf: &[(f32, f32)]) -> EllipseGeom {
    let n = ptsf.len();
    assert!(n >= 5, "need at least 5 points");
    let mut cx = 0f32;
    let mut cy = 0f32;
    for &(x, y) in ptsf {
        cx += x;
        cy += y;
    }
    cx /= n as f32;
    cy /= n as f32;

    let mut s = 0f64;
    for &(x, y) in ptsf {
        s += ((x - cx) as f64).abs() + ((y - cy) as f64).abs();
    }
    let scale = 100.0 / if s > FLT_EPSILON { s } else { FLT_EPSILON };

    let build5 = |pf: &[(f32, f32)], a: &mut Vec<f64>, b: &mut Vec<f64>| {
        a.clear();
        b.clear();
        for &(x, y) in pf {
            let px = (x - cx) as f64 * scale;
            let py = (y - cy) as f64 * scale;
            b.push(10000.0);
            a.extend_from_slice(&[-px * px, -py * py, -px * py, px, py]);
        }
    };
    let mut a5 = Vec::new();
    let mut b5 = Vec::new();
    build5(&ptsf, &mut a5, &mut b5);
    let (mut gfp, w) = lstsq(&a5, n, 5, &b5);
    if w[0] * FLT_EPSILON > w[4] {
        // degenerate rank: OpenCV jitters the points from its global RNG. Use a
        // fixed-seed xorshift instead; unreachable for real ring contours.
        let eps = (s / (n as f64 * 2.0) * 1e-3) as f32;
        let mut st = 0x2545f491u64;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f32 / (1u64 << 53) as f32 * 2.0 * eps - eps
        };
        let jit: Vec<(f32, f32)> = ptsf.iter().map(|&(x, y)| (x + rnd(), y + rnd())).collect();
        build5(&jit, &mut a5, &mut b5);
        let r = lstsq(&a5, n, 5, &b5);
        gfp = r.0;
    }

    // 2x2 center solve
    let a2 = [2.0 * gfp[0], gfp[2], gfp[2], 2.0 * gfp[1]];
    let b2 = [gfp[3], gfp[4]];
    let (rp01, _) = lstsq(&a2, 2, 2, &b2);
    let (rp0, rp1) = (rp01[0], rp01[1]);

    // 3-var re-fit with the center fixed
    let mut a3 = Vec::with_capacity(n * 3);
    let mut b3 = Vec::with_capacity(n);
    for &(x, y) in ptsf {
        let px = (x - cx) as f64 * scale;
        let py = (y - cy) as f64 * scale;
        b3.push(1.0);
        a3.extend_from_slice(&[
            (px - rp0) * (px - rp0),
            (py - rp1) * (py - rp1),
            (px - rp0) * (py - rp1),
        ]);
    }
    let (g3, _) = lstsq(&a3, n, 3, &b3);

    let min_eps = 1e-8;
    let rp4 = -0.5 * g3[2].atan2(g3[1] - g3[0]);
    let t = if g3[2].abs() > min_eps {
        g3[2] / (-2.0 * rp4).sin()
    } else {
        g3[1] - g3[0]
    };
    let mut rp2 = (g3[0] + g3[1] - t).abs();
    if rp2 > min_eps {
        rp2 = (2.0 / rp2).sqrt();
    }
    let mut rp3 = (g3[0] + g3[1] + t).abs();
    if rp3 > min_eps {
        rp3 = (2.0 / rp3).sqrt();
    }

    let bcx = (rp0 / scale) as f32 + cx;
    let bcy = (rp1 / scale) as f32 + cy;
    let mut width = (rp2 * 2.0 / scale) as f32;
    let mut height = (rp3 * 2.0 / scale) as f32;
    let mut angle = 0f32;
    if width > height {
        std::mem::swap(&mut width, &mut height);
        angle = (90.0 + rp4 * 180.0 / std::f64::consts::PI) as f32;
    }
    if angle < -180.0 {
        angle += 360.0;
    }
    if angle > 360.0 {
        angle -= 360.0;
    }
    EllipseGeom {
        cx: bcx as f64,
        cy: bcy as f64,
        major: width as f64,  // cv2 returns (width, height); pose uses them as-is
        minor: height as f64,
        angle_deg: angle as f64,
    }
}
