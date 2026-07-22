//! Grayscale image + the cv2 filter primitives the detector uses, each ported
//! with EMPIRICALLY VERIFIED semantics (probed against OpenCV 4.13 on random
//! images before porting -- see the phase-3 notes):
//!
//! * GaussianBlur(u8, sigma=1.0, auto ksize 7): OpenCV routes 8U through its
//!   bit-exact FIXED-POINT path -- kernel quantized to 8 fractional bits
//!   (round(k*256), sums to 256), full-precision integer accumulation through
//!   both separable passes, one final round: (acc + 2^15) >> 16. Verified
//!   0-diff; a float64 port differs on ~6% of pixels.
//! * addWeighted / adaptiveThreshold mean (ksize > 7): plain float64 with
//!   round-half-even matches bit-exact.
//! * adaptiveThreshold GAUSSIAN_C BINARY_INV: dst = (src <= mean_u8 - C) * 255
//!   with the mean rounded to u8 first. Verified 0-diff.
//! * medianBlur 3x3: replicate border. Verified 0-diff.

#[derive(Clone)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

/// Run `body(y, row)` over row-chunks of `buf`, in parallel when the
/// `parallel` feature is on. Rows are independent in every filter here, so
/// splitting at row boundaries cannot change any per-row arithmetic order --
/// which is what keeps the bitwise parity gates green under parallelism.
#[inline]
pub(crate) fn for_rows<F: Fn(usize, &mut [T]) + Sync, T: Send>(
    buf: &mut [T],
    w: usize,
    body: F,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| body(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    buf.chunks_mut(w).enumerate().for_each(|(y, row)| body(y, row));
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Gray { w, h, px: vec![0; w * h] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }
}

/// cv2.getGaussianKernel for sigma > 0 (double precision, normalized).
/// ksize <= 7 with sigma <= 0 would hit OpenCV's hardcoded small tables; the
/// detector never does that (sharpen passes sigma=1.0; adaptive uses ksize 11+).
pub fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f64> {
    let sigma = if sigma > 0.0 {
        sigma
    } else {
        0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8
    };
    let scale2x = -0.5 / (sigma * sigma);
    let c = (ksize as f64 - 1.0) * 0.5;
    let mut k: Vec<f64> = (0..ksize)
        .map(|i| ((i as f64 - c) * (i as f64 - c) * scale2x).exp())
        .collect();
    let s: f64 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

#[inline]
fn reflect101(i: i64, n: i64) -> i64 {
    // BORDER_REFLECT_101: ...cba|abc|cba... (no edge duplicate)
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i;
        }
    }
}

#[inline]
fn replicate(i: i64, n: i64) -> i64 {
    i.clamp(0, n - 1)
}

/// GaussianBlur sigma=1.0 on u8, OpenCV's bit-exact fixed-point path.
/// Strip-processed separable passes: the row pass pads each row once, the
/// column pass accumulates whole rows (t-outer), so the inner loops are
/// sequential, branch-free, and autovectorize. i32 is ample: 8.16 fixed peaks
/// at 255*65536 < 2^24.
pub fn gaussian_blur_u8_fixed(src: &Gray, sigma: f64) -> Gray {
    let ksize = (((sigma * 3.0 * 2.0 + 1.0).round() as i64) | 1) as usize; // 8U: sigma*3
    let kf = gaussian_kernel(ksize, sigma);
    let k: Vec<i32> = kf.iter().map(|v| (v * 256.0).round() as i32).collect();
    let r = ksize / 2;
    let (w, h) = (src.w, src.h);
    // row pass: 8.8 fixed
    let mut tmp = vec![0i32; w * h];
    for_rows(&mut tmp, w, |y, orow| {
        let row = &src.px[y * w..(y + 1) * w];
        let mut padded = vec![0i32; w + 2 * r];
        for i in 0..r {
            padded[i] = row[reflect101(i as i64 - r as i64, w as i64) as usize] as i32;
            padded[w + r + i] = row[reflect101((w + i) as i64, w as i64) as usize] as i32;
        }
        for x in 0..w {
            padded[r + x] = row[x] as i32;
        }
        for (t, &kv) in k.iter().enumerate() {
            let win = &padded[t..t + w];
            for (o, &p) in orow.iter_mut().zip(win) {
                *o += kv * p;
            }
        }
    });
    // column pass, t-outer over source rows, one final rounding (8.16 fixed)
    let mut out = Gray::new(w, h);
    let tmp_ref = &tmp;
    for_rows(&mut out.px, w, |y, orow| {
        let mut acc = vec![0i32; w];
        for (t, &kv) in k.iter().enumerate() {
            let yy = reflect101(y as i64 + t as i64 - r as i64, h as i64) as usize;
            let srow = &tmp_ref[yy * w..(yy + 1) * w];
            for (a, &s) in acc.iter_mut().zip(srow) {
                *a += kv * s;
            }
        }
        for x in 0..w {
            orow[x] = ((acc[x] + (1 << 15)) >> 16).clamp(0, 255) as u8;
        }
    });
    out
}

/// cv2.addWeighted(a, alpha, b, beta, 0) on u8: f64, round-half-even, saturate.
/// Row-parallel; each pixel is independent, so this is bit-exact.
pub fn add_weighted(a: &Gray, alpha: f64, b: &Gray, beta: f64) -> Gray {
    let mut out = Gray::new(a.w, a.h);
    let (apx, bpx, w) = (&a.px, &b.px, a.w);
    for_rows(&mut out.px, w, |y, orow| {
        let ar = &apx[y * w..y * w + w];
        let br = &bpx[y * w..y * w + w];
        for x in 0..w {
            let v = ar[x] as f64 * alpha + br[x] as f64 * beta;
            orow[x] = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// detect._sharpen: unsharp mask, amount 0.6, sigma 1.0.
pub fn sharpen(gray: &Gray, amount: f64, sigma: f64) -> Gray {
    let blur = gaussian_blur_u8_fixed(gray, sigma);
    add_weighted(gray, 1.0 + amount, &blur, -amount)
}

/// cv2.adaptiveThreshold(GAUSSIAN_C, BINARY_INV, blk, C): float64 separable
/// gaussian mean (sigma from the ksize formula, BORDER_REPLICATE), rounded to
/// u8 half-even, then dst = 255 where src <= mean - C.
/// f32 throughout: probed bitwise-identical to cv2 (OpenCV's own path is f32),
/// and f32 gives 4-lane simd128 in wasm where f64 only gets 2.
pub fn adaptive_threshold_inv(src: &Gray, blk: usize, c_delta: i32) -> Gray {
    let k: Vec<f32> = gaussian_kernel(blk, -1.0).iter().map(|&v| v as f32).collect();
    let r = blk / 2;
    let (w, h) = (src.w, src.h);
    let mut tmp = vec![0f32; w * h];
    for_rows(&mut tmp, w, |y, orow| {
        let row = &src.px[y * w..(y + 1) * w];
        let mut padded = vec![0f32; w + 2 * r];
        for i in 0..r {
            padded[i] = row[0] as f32; // BORDER_REPLICATE
            padded[w + r + i] = row[w - 1] as f32;
        }
        for x in 0..w {
            padded[r + x] = row[x] as f32;
        }
        for (t, &kv) in k.iter().enumerate() {
            // window view: bounds checks gone, x loop vectorizes; each output
            // element still accumulates in the same t order -> bit-exact
            let win = &padded[t..t + w];
            for (o, &p) in orow.iter_mut().zip(win) {
                *o += kv * p;
            }
        }
    });
    let mut out = Gray::new(w, h);
    let tmp_ref = &tmp;
    for_rows(&mut out.px, w, |y, orow| {
        let mut acc = vec![0f32; w];
        for (t, &kv) in k.iter().enumerate() {
            let yy = replicate(y as i64 + t as i64 - r as i64, h as i64) as usize;
            let srow = &tmp_ref[yy * w..(yy + 1) * w];
            for (a, &s) in acc.iter_mut().zip(srow) {
                *a += kv * s;
            }
        }
        let irow = &src.px[y * w..(y + 1) * w];
        for x in 0..w {
            let mean = acc[x].round_ties_even().clamp(0.0, 255.0) as i32;
            orow[x] = if (irow[x] as i32) <= mean - c_delta { 255 } else { 0 };
        }
    });
    out
}

/// cv2.medianBlur ksize 3 (replicate border), branchless median-of-9 network.
pub fn median3(src: &Gray) -> Gray {
    #[inline(always)]
    fn med9(mut v: [u8; 9]) -> u8 {
        #[inline(always)]
        fn s(v: &mut [u8; 9], a: usize, b: usize) {
            let (x, y) = (v[a], v[b]);
            v[a] = x.min(y);
            v[b] = x.max(y);
        }
        // Paeth's 19-comparator median-of-9
        s(&mut v, 1, 2); s(&mut v, 4, 5); s(&mut v, 7, 8);
        s(&mut v, 0, 1); s(&mut v, 3, 4); s(&mut v, 6, 7);
        s(&mut v, 1, 2); s(&mut v, 4, 5); s(&mut v, 7, 8);
        s(&mut v, 0, 3); s(&mut v, 5, 8); s(&mut v, 4, 7);
        s(&mut v, 3, 6); s(&mut v, 1, 4); s(&mut v, 2, 5);
        s(&mut v, 4, 7); s(&mut v, 4, 2); s(&mut v, 6, 4);
        s(&mut v, 4, 2);
        v[4]
    }
    let (w, h) = (src.w, src.h);
    let mut out = Gray::new(w, h);
    for_rows(&mut out.px, w, |y, orow| {
        let ym = if y > 0 { y - 1 } else { 0 };
        let yp = if y + 1 < h { y + 1 } else { h - 1 };
        let (r0, r1, r2) = (
            &src.px[ym * w..ym * w + w],
            &src.px[y * w..y * w + w],
            &src.px[yp * w..yp * w + w],
        );
        // interior via shifted window slices: no bounds checks, and the
        // branchless min/max network vectorizes on u8 lanes (bit-exact --
        // med9 is elementwise)
        if w >= 2 {
            orow[0] = med9([
                r0[0], r0[0], r0[1],
                r1[0], r1[0], r1[1],
                r2[0], r2[0], r2[1],
            ]);
            orow[w - 1] = med9([
                r0[w - 2], r0[w - 1], r0[w - 1],
                r1[w - 2], r1[w - 1], r1[w - 1],
                r2[w - 2], r2[w - 1], r2[w - 1],
            ]);
            for (x, o) in orow[1..w - 1].iter_mut().enumerate() {
                *o = med9([
                    r0[x], r0[x + 1], r0[x + 2],
                    r1[x], r1[x + 1], r1[x + 2],
                    r2[x], r2[x + 1], r2[x + 2],
                ]);
            }
        } else {
            orow[0] = med9([
                r0[0], r0[0], r0[0],
                r1[0], r1[0], r1[0],
                r2[0], r2[0], r2[0],
            ]);
        }
    });
    out
}

/// The detector's adaptive block size: max(11, min(51, (min(w,h)//8)|1) | 1).
pub fn adaptive_block(w: usize, h: usize) -> usize {
    11usize.max(51usize.min((w.min(h) / 8) | 1) | 1)
}

/// detect._undistort: cv2.initUndistortRectifyMap(K, dist, None, K, CV_16SC2) +
/// remap(INTER_LINEAR). Emulates OpenCV's fixed-point path bit-exact (probed
/// 0-diff): source coords quantized to 1/32 px via floor(s*32 + 0.5), bilinear
/// weights as round(w*32768) with the residual added to the largest weight,
/// final (acc + 2^14) >> 15, out-of-image taps = 0 (BORDER_CONSTANT).
pub fn undistort(src: &Gray, k: &crate::mat::M3, dist: &[f64]) -> Gray {
    if dist.iter().all(|&d| d == 0.0) {
        return src.clone();
    }
    let (k1, k2, p1, p2, k3) = (
        dist[0],
        *dist.get(1).unwrap_or(&0.0),
        *dist.get(2).unwrap_or(&0.0),
        *dist.get(3).unwrap_or(&0.0),
        *dist.get(4).unwrap_or(&0.0),
    );
    let (fx, fy, cx, cy) = (k[0][0], k[1][1], k[0][2], k[1][2]);
    let (w, h) = (src.w as i64, src.h as i64);
    let mut out = Gray::new(src.w, src.h);
    for_rows(&mut out.px, src.w, |v, orow| {
        let v = v as i64;
        let y = (v as f64 - cy) / fy;
        for u in 0..w {
            let x = (u as f64 - cx) / fx;
            let r2 = x * x + y * y;
            let rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            let xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            let sx = fx * xd + cx;
            let sy = fy * yd + cy;
            let ix = (sx * 32.0 + 0.5).floor() as i64;
            let iy = (sy * 32.0 + 0.5).floor() as i64;
            let (x0, y0) = (ix >> 5, iy >> 5);
            let (fxq, fyq) = ((ix & 31) as f64 / 32.0, (iy & 31) as f64 / 32.0);
            let wf = [
                (1.0 - fxq) * (1.0 - fyq),
                fxq * (1.0 - fyq),
                (1.0 - fxq) * fyq,
                fxq * fyq,
            ];
            let mut iw: [i64; 4] = [0; 4];
            let mut imax = 0usize;
            for t in 0..4 {
                iw[t] = (wf[t] * 32768.0).round() as i64;
                if iw[t] > iw[imax] {
                    imax = t;
                }
            }
            iw[imax] += 32768 - iw.iter().sum::<i64>();
            let mut acc = 0i64;
            for (t, &(dy, dx)) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                let (yy, xx) = (y0 + dy, x0 + dx);
                let pv = if yy >= 0 && yy < h && xx >= 0 && xx < w {
                    src.px[(yy * w + xx) as usize] as i64
                } else {
                    0
                };
                acc += iw[t] * pv;
            }
            orow[u as usize] = ((acc + (1 << 14)) >> 15).clamp(0, 255) as u8;
        }
    });
    out
}
