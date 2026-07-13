//! Minimal 3x3 / vector linear algebra -- just what pose.py's numpy calls need.
//!
//! eigh3 is a cyclic Jacobi eigensolver for symmetric 3x3 matrices, returning
//! eigenvalues ASCENDING with matching eigenvector columns (numpy.linalg.eigh
//! convention). Jacobi converges to ~1e-14 relative here, comfortably inside
//! the 1e-9 fixture gate; eigenvector SIGNS are pinned by the caller
//! (transforms_from_conic), not here, mirroring the Python canonicalization.

pub type M3 = [[f64; 3]; 3];
pub type V3 = [f64; 3];

pub fn matmul(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for (k, bk) in b.iter().enumerate() {
                c[i][j] += a[i][k] * bk[j];
            }
        }
    }
    c
}

pub fn matvec(a: &M3, v: &V3) -> V3 {
    let mut o = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i] += a[i][j] * v[j];
        }
    }
    o
}

pub fn transpose(a: &M3) -> M3 {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

pub fn det3(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Inverse by adjugate; fine for the well-conditioned K / H matrices used here.
pub fn inv3(a: &M3) -> M3 {
    let d = det3(a);
    let id = 1.0 / d;
    [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id,
        ],
    ]
}

pub fn cross(a: &V3, b: &V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(v: &V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Symmetric 3x3 eigendecomposition, eigenvalues ascending (numpy eigh order).
pub fn eigh3(a: &M3) -> (V3, M3) {
    let mut m = *a;
    let mut v: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        // largest off-diagonal element
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        if off < 1e-300 {
            break;
        }
        let scale = m[0][0].abs().max(m[1][1].abs()).max(m[2][2].abs()).max(1e-300);
        if off / scale < 1e-15 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if m[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // rotate m
            let mpp = m[p][p] - t * m[p][q];
            let mqq = m[q][q] + t * m[p][q];
            let r = 3 - p - q; // the untouched index
            let mrp = c * m[r][p] - s * m[r][q];
            let mrq = s * m[r][p] + c * m[r][q];
            m[p][p] = mpp;
            m[q][q] = mqq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            m[r][p] = mrp;
            m[p][r] = mrp;
            m[r][q] = mrq;
            m[q][r] = mrq;
            // accumulate eigenvectors (columns)
            for row in &mut v {
                let vp = c * row[p] - s * row[q];
                let vq = s * row[p] + c * row[q];
                row[p] = vp;
                row[q] = vq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| m[i][i].partial_cmp(&m[j][j]).unwrap());
    let w = [m[idx[0]][idx[0]], m[idx[1]][idx[1]], m[idx[2]][idx[2]]];
    let mut vs = [[0.0; 3]; 3];
    for (col, &i) in idx.iter().enumerate() {
        for row in 0..3 {
            vs[row][col] = v[row][i];
        }
    }
    (w, vs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_reconstructs() {
        let a: M3 = [[4.0, 1.0, -2.0], [1.0, 3.0, 0.5], [-2.0, 0.5, 1.0]];
        let (w, v) = eigh3(&a);
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        // A v_i = w_i v_i
        for i in 0..3 {
            let vi = [v[0][i], v[1][i], v[2][i]];
            let av = matvec(&a, &vi);
            for r in 0..3 {
                assert!((av[r] - w[i] * vi[r]).abs() < 1e-12, "{} {}", i, r);
            }
        }
    }

    #[test]
    fn inv_roundtrip() {
        let a: M3 = [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]];
        let ai = inv3(&a);
        let id = matmul(&a, &ai);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id[i][j] - want).abs() < 1e-12);
            }
        }
    }
}
