//! Minimal power-of-two FFT for the Wiener-deconvolution retry.
//!
//! Matches np.fft conventions: forward transform unnormalized, inverse
//! divides by N. Only what the deconvolution needs: a 2-D real-input
//! round trip through a real (Gaussian) frequency-domain filter.

/// In-place iterative radix-2 Cooley-Tukey. `re`/`im` length must be a
/// power of two. `inv` = inverse transform WITHOUT the 1/N scale (the
/// caller applies it once for the 2-D case).
fn fft1d(re: &mut [f64], im: &mut [f64], inv: bool) {
    let n = re.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n - 1 {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }
    // butterflies
    let sign = if inv { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ar, ai) = (re[i + k], im[i + k]);
                let (br, bi) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (tr, ti) = (br * cr - bi * ci, br * ci + bi * cr);
                re[i + k] = ar + tr;
                im[i + k] = ai + ti;
                re[i + k + len / 2] = ar - tr;
                im[i + k + len / 2] = ai - ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Apply a REAL frequency-domain filter to a real image of pow2
/// dimensions (h, w): out = irfft2( rfft2(img) * filt ), with `filt` a
/// full h*w array of real filter values on the np.fft frequency grid.
/// Output is the filtered real image (imaginary residue discarded).
pub fn filter2d_real(img: &[f64], h: usize, w: usize, filt: &[f64]) -> Vec<f64> {
    debug_assert!(h.is_power_of_two() && w.is_power_of_two());
    let mut re: Vec<f64> = img.to_vec();
    let mut im = vec![0f64; h * w];
    // rows forward
    let mut row_r = vec![0f64; w];
    let mut row_i = vec![0f64; w];
    for y in 0..h {
        row_r.copy_from_slice(&re[y * w..(y + 1) * w]);
        row_i.copy_from_slice(&im[y * w..(y + 1) * w]);
        fft1d(&mut row_r, &mut row_i, false);
        re[y * w..(y + 1) * w].copy_from_slice(&row_r);
        im[y * w..(y + 1) * w].copy_from_slice(&row_i);
    }
    // cols forward, filter, cols inverse
    let mut col_r = vec![0f64; h];
    let mut col_i = vec![0f64; h];
    for x in 0..w {
        for y in 0..h {
            col_r[y] = re[y * w + x];
            col_i[y] = im[y * w + x];
        }
        fft1d(&mut col_r, &mut col_i, false);
        for y in 0..h {
            let f = filt[y * w + x];
            col_r[y] *= f;
            col_i[y] *= f;
        }
        fft1d(&mut col_r, &mut col_i, true);
        for y in 0..h {
            re[y * w + x] = col_r[y];
            im[y * w + x] = col_i[y];
        }
    }
    // rows inverse + 1/(h*w)
    let scale = 1.0 / (h * w) as f64;
    for y in 0..h {
        row_r.copy_from_slice(&re[y * w..(y + 1) * w]);
        row_i.copy_from_slice(&im[y * w..(y + 1) * w]);
        fft1d(&mut row_r, &mut row_i, true);
        for x in 0..w {
            re[y * w + x] = row_r[x] * scale;
        }
    }
    re
}

/// np.fft.fftfreq(n): [0, 1, ..., n/2-1, -n/2, ..., -1] / n
pub fn fftfreq(n: usize) -> Vec<f64> {
    let mut out = vec![0f64; n];
    let half = (n - 1) / 2 + 1;
    for (i, o) in out.iter_mut().enumerate().take(half) {
        *o = i as f64 / n as f64;
    }
    for i in half..n {
        out[i] = (i as f64 - n as f64) / n as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_filter_roundtrips() {
        let (h, w) = (8, 16);
        let img: Vec<f64> = (0..h * w).map(|i| (i % 251) as f64).collect();
        let filt = vec![1.0; h * w];
        let out = filter2d_real(&img, h, w, &filt);
        for (a, b) in img.iter().zip(&out) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn matches_direct_dft_on_small_case() {
        // non-trivial real filter, compared against a naive O(n^2) DFT
        let (h, w) = (4, 8);
        let img: Vec<f64> = (0..h * w).map(|i| ((i * 7 + 3) % 17) as f64).collect();
        let fy: Vec<f64> = fftfreq(h).iter().map(|f| (-3.0 * f * f).exp()).collect();
        let fx: Vec<f64> = fftfreq(w).iter().map(|f| (-5.0 * f * f).exp()).collect();
        let filt: Vec<f64> = (0..h * w).map(|i| fy[i / w] * fx[i % w]).collect();
        let fast = filter2d_real(&img, h, w, &filt);
        // naive: F(u,v) -> filter -> inverse
        let pi2 = 2.0 * std::f64::consts::PI;
        let mut slow = vec![0f64; h * w];
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = 0f64;
                for u in 0..h {
                    for v in 0..w {
                        // frequency-domain coefficient F[u,v]
                        let (mut fr, mut fi) = (0f64, 0f64);
                        for y in 0..h {
                            for x in 0..w {
                                let ph = -pi2
                                    * (u as f64 * y as f64 / h as f64
                                        + v as f64 * x as f64 / w as f64);
                                fr += img[y * w + x] * ph.cos();
                                fi += img[y * w + x] * ph.sin();
                            }
                        }
                        let g = fy[u] * fx[v];
                        let ph = pi2
                            * (u as f64 * oy as f64 / h as f64
                                + v as f64 * ox as f64 / w as f64);
                        acc += g * (fr * ph.cos() - fi * ph.sin());
                    }
                }
                slow[oy * w + ox] = acc / (h * w) as f64;
            }
        }
        for (a, b) in fast.iter().zip(&slow) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}
