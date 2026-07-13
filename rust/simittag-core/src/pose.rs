//! Conic -> perspective transform. Port of simittag/pose.py (itself a port of
//! Cantag's TransformEllipseFull). Gated at 1e-9 against fixtures/geometry.json.
//!
//! The eigenvector sign canonicalization matches pose.py exactly (col 2: normal
//! toward camera; col 1: largest-|component| positive; col 0: det=+1) -- that
//! rule exists precisely so this port and LAPACK land on the SAME homography.

use crate::mat::{self, M3, V3};

#[derive(Debug, Clone, Copy)]
pub struct EllipseGeom {
    pub cx: f64,
    pub cy: f64,
    pub major: f64, // full axis, cv2.fitEllipse convention (either may be larger)
    pub minor: f64,
    pub angle_deg: f64,
}

/// cv2.fitEllipse geometric form -> 3x3 conic C with [x y 1] C [x y 1]^T = 0.
pub fn ellipse_to_conic(g: &EllipseGeom) -> M3 {
    let a = g.major / 2.0;
    let b = g.minor / 2.0;
    let th = g.angle_deg.to_radians();
    let (s, c) = th.sin_cos();
    // M = R diag(1/a^2, 1/b^2) R^T
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let m00 = c * c * ia + s * s * ib;
    let m01 = c * s * ia - s * c * ib;
    let m11 = s * s * ia + c * c * ib;
    let mc = [
        m00 * g.cx + m01 * g.cy,
        m01 * g.cx + m11 * g.cy,
    ];
    [
        [m00, m01, -mc[0]],
        [m01, m11, -mc[1]],
        [
            -mc[0],
            -mc[1],
            g.cx * mc[0] + g.cy * mc[1] - 1.0,
        ],
    ]
}

/// Conic (normalized camera coords) -> two 3x4-equivalent transforms, returned
/// as the two homography column triples (cols 0,1,3 of the 4x4), matching
/// Python's H_norm = T[:3][:, [0,1,3]].
fn transforms_from_conic(c: &M3) -> [M3; 2] {
    let (mut w, mut v) = mat::eigh3(c);
    if w.iter().filter(|&&x| x < 0.0).count() > 1 {
        for x in &mut w {
            *x = -*x;
        }
    }
    // sort descending: l1 >= l2 >= l3
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| w[j].partial_cmp(&w[i]).unwrap());
    let ws = [w[idx[0]], w[idx[1]], w[idx[2]]];
    let mut vs = [[0.0f64; 3]; 3];
    for (col, &i) in idx.iter().enumerate() {
        for row in 0..3 {
            vs[row][col] = v[row][i];
        }
    }
    v = vs;
    if v[2][2] < 0.0 {
        for row in &mut v {
            row[2] = -row[2];
        }
    }
    let mut j = 0;
    for r in 1..3 {
        if v[r][1].abs() > v[j][1].abs() {
            j = r;
        }
    }
    if v[j][1] < 0.0 {
        for row in &mut v {
            row[1] = -row[1];
        }
    }
    if mat::det3(&v) < 0.0 {
        for row in &mut v {
            row[0] = -row[0];
        }
    }
    let (l1, l2, l3) = (ws[0], ws[1], ws[2]);

    let denom = l3 - l1;
    let pmcos = ((l3 - l2) / denom).max(0.0).sqrt();
    let pmsin = ((l2 - l1) / denom).max(0.0).sqrt();
    let tx = ((l2 - l1) * (l3 - l2)).max(0.0).sqrt() / l2;
    let scale = (-l1 * l3 / (l2 * l2)).max(0.0).sqrt(); // bullseye_size = 1

    let mut out = [[[0.0f64; 3]; 3]; 2];
    for (oi, sgn) in [(0usize, 1.0f64), (1, -1.0)] {
        // R1 @ r2 @ trans, keeping only columns 0, 1, 3 of the 4x4 product.
        // r2 rotates about y: [[pmcos,0,-sgn*pmsin],[0,1,0],[sgn*pmsin,0,pmcos]]
        // trans: x += sgn*tx/scale, z += 1/scale.
        let r2 = [
            [pmcos, 0.0, -sgn * pmsin],
            [0.0, 1.0, 0.0],
            [sgn * pmsin, 0.0, pmcos],
        ];
        let r12 = mat::matmul(&v, &r2);
        let tcol = [sgn * tx / scale, 0.0, 1.0 / scale];
        let tcam = mat::matvec(&r12, &tcol);
        out[oi] = [
            [r12[0][0], r12[0][1], tcam[0]],
            [r12[1][0], r12[1][1], tcam[1]],
            [r12[2][0], r12[2][1], tcam[2]],
        ];
    }
    out
}

/// Ellipse (pixel coords) + K -> up to 2 homographies mapping marker-plane
/// (X, Y, 1) -> image pixels, unit circle = the fitted ellipse.
pub fn pose_homographies(geom: &EllipseGeom, k: &M3) -> Vec<M3> {
    let c_pix = ellipse_to_conic(geom);
    let c_norm = mat::matmul(&mat::matmul(&mat::transpose(k), &c_pix), k);
    let mut out = Vec::with_capacity(2);
    for h_norm in transforms_from_conic(&c_norm) {
        let mut h_pix = mat::matmul(k, &h_norm);
        if h_pix[2][2].abs() > 1e-12 {
            let s = 1.0 / h_pix[2][2];
            for row in &mut h_pix {
                for e in row {
                    *e *= s;
                }
            }
        }
        out.push(h_pix);
    }
    out
}

pub fn apply_h(h: &M3, x: f64, y: f64) -> (f64, f64) {
    let p = mat::matvec(h, &[x, y, 1.0]);
    (p[0] / p[2], p[1] / p[2])
}

/// Homography -> (R, t) in camera coordinates; scale in unit-circle radii.
pub fn decompose_h(h: &M3, k: &M3) -> (M3, V3) {
    let kinv = mat::inv3(k);
    let mut l = mat::matmul(&kinv, h);
    let c0 = [l[0][0], l[1][0], l[2][0]];
    let c1 = [l[0][1], l[1][1], l[2][1]];
    let lam = 2.0 / (mat::norm(&c0) + mat::norm(&c1));
    for row in &mut l {
        for e in row {
            *e *= lam;
        }
    }
    let mut r1 = [l[0][0], l[1][0], l[2][0]];
    let mut r2 = [l[0][1], l[1][1], l[2][1]];
    let mut t = [l[0][2], l[1][2], l[2][2]];
    let n1 = mat::norm(&r1);
    let n1 = if n1 != 0.0 { n1 } else { 1.0 }; // Python: `or 1.0`
    for e in &mut r1 {
        *e /= n1;
    }
    let d = mat::dot(&r1, &r2);
    for i in 0..3 {
        r2[i] -= r1[i] * d;
    }
    let n2 = mat::norm(&r2);
    let n2 = if n2 != 0.0 { n2 } else { 1.0 };
    for e in &mut r2 {
        *e /= n2;
    }
    let mut r3 = mat::cross(&r1, &r2);
    if t[2] < 0.0 {
        for i in 0..3 {
            r1[i] = -r1[i];
            r2[i] = -r2[i];
            t[i] = -t[i];
        }
        r3 = mat::cross(&r1, &r2);
    }
    let r = [
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]],
    ];
    (r, t)
}

/// Approximate tilt (deg) of the marker plane normal vs the camera axis.
pub fn tilt_from_h(h: &M3, k: &M3) -> f64 {
    let kinv = mat::inv3(k);
    let h1 = mat::matvec(&kinv, &[h[0][0], h[1][0], h[2][0]]);
    let h2 = mat::matvec(&kinv, &[h[0][1], h[1][1], h[2][1]]);
    let n1n = mat::norm(&h1);
    let n2n = mat::norm(&h2);
    let n1 = [h1[0] / n1n, h1[1] / n1n, h1[2] / n1n];
    let n2 = [h2[0] / n2n, h2[1] / n2n, h2[2] / n2n];
    let mut nrm = mat::cross(&n1, &n2);
    let nn = mat::norm(&nrm);
    for e in &mut nrm {
        *e /= nn;
    }
    nrm[2].abs().min(1.0).acos().to_degrees()
}
