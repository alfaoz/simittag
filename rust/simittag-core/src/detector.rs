//! The full detector: detect.py's sampling, decode search, sub-pixel edge
//! refinement, full-grid phase refinement, and detect_markers assembly.
//! Gate: identical decode decisions on the fixture frames + pose within
//! ±0.05 deg tilt / ±0.1% depth of the Python reference.

use crate::codec;
use crate::fitellipse::fit_ellipse_pts;
use crate::frontend::{find_marker_ellipses, Candidate};
use crate::image::{sharpen, Gray};
use crate::mat::{self, M3, V3};
use crate::payload::{self, Value};
use crate::pose::{self, EllipseGeom};
use crate::spec::MarkerSpec;

pub struct Detection {
    pub center: (f64, f64),
    pub axes: (f64, f64),
    pub angle: f64,
    pub r: M3,
    pub t: V3,
    pub h: M3, // chosen homography (marker plane -> pixels), what pose derives from
    pub tilt_deg: f64,
    pub inverted: bool,
    pub decoded: Option<(&'static str, &'static str, Value)>, // (variant, mode, value)
    pub info: Option<DecodeInfo>, // decode diagnostics, present iff decoded
}

/// Diagnostics for a successful decode (verbose HUD/readout). Report-only:
/// detection behavior never branches on these.
#[derive(Clone, Copy)]
pub struct DecodeInfo {
    pub rs_erasures: usize,
    pub rs_corrected: usize,
    pub verify_corr: f64,
    pub sync_score: f64,    // normalized -1..1; <0 when the spec has no sync ring
    pub path: &'static str, // direct | sticker | bullseye | deconv
}

// ---------------------------------------------------------------------------
// sampling through a homography (detect._project + _sample_many semantics)
// ---------------------------------------------------------------------------

#[inline]
fn project(h: &M3, x: f64, y: f64) -> (f64, f64) {
    let px = h[0][0] * x + h[0][1] * y + h[0][2];
    let py = h[1][0] * x + h[1][1] * y + h[1][2];
    let pw = h[2][0] * x + h[2][1] * y + h[2][2];
    if pw.abs() < 1e-12 {
        (f64::NAN, f64::NAN)
    } else {
        (px / pw, py / pw)
    }
}

/// Vectorized-bilinear equivalent: returns (value, valid). Invalid samples get
/// the same clamped-extrapolated junk value Python computes (it is never used
/// behind a valid mask, but keeping it identical keeps every branch identical).
#[inline]
fn sample(img: &Gray, x: f64, y: f64) -> (f64, bool) {
    let finite = x.is_finite() && y.is_finite();
    let xf = if finite { x } else { -1.0 };
    let yf = if finite { y } else { -1.0 };
    let x0 = xf.floor() as i64;
    let y0 = yf.floor() as i64;
    let w = img.w as i64;
    let h = img.h as i64;
    let valid = finite && x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h;
    let x0c = x0.clamp(0, w - 2);
    let y0c = y0.clamp(0, h - 2);
    let fx = xf - x0c as f64;
    let fy = yf - y0c as f64;
    let i = (y0c * w + x0c) as usize;
    let v = img.px[i] as f64 * (1.0 - fx) * (1.0 - fy)
        + img.px[i + 1] as f64 * fx * (1.0 - fy)
        + img.px[i + img.w] as f64 * (1.0 - fx) * fy
        + img.px[i + img.w + 1] as f64 * fx * fy;
    (v, valid)
}

fn rotate_h(h: &M3, dphi: f64) -> M3 {
    let (s, c) = dphi.sin_cos();
    let rz: M3 = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    mat::matmul(h, &rz)
}

fn scale_h(h: &M3, s: f64) -> M3 {
    let d: M3 = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]];
    mat::matmul(h, &d)
}

/// Project H's unit circle and fit the corresponding image ellipse. Used only
/// after the bullseye fallback decodes, to report the recovered outer geometry.
fn ellipse_from_h(h: &M3) -> EllipseGeom {
    let pts: Vec<(f32, f32)> = (0..128)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / 128.0;
            let (x, y) = project(h, a.cos(), a.sin());
            (x as f32, y as f32)
        })
        .collect();
    fit_ellipse_pts(&pts)
}

/// Cheap radial photometric gate for an expanded bullseye hypothesis. A real
/// marker has a dark outer annulus beside a white quiet ring; requiring that
/// contrast on only a majority of angles keeps partial occlusion admissible.
fn has_outer_ring_contrast(gray: &Gray, h: &M3, spec: &MarkerSpec) -> bool {
    let n = 48usize;
    let black_r = 0.5 * (spec.r_ring_in + 1.0);
    let quiet_r = 0.5 * (spec.r_data_out + spec.r_ring_in);
    let mut valid = 0usize;
    let mut dark = 0usize;
    for i in 0..n {
        let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let (co, si) = (a.cos(), a.sin());
        let (bx, by) = project(h, black_r * co, black_r * si);
        let (qx, qy) = project(h, quiet_r * co, quiet_r * si);
        let (bv, bok) = sample(gray, bx, by);
        let (qv, qok) = sample(gray, qx, qy);
        if bok && qok {
            valid += 1;
            dark += (qv - bv > 20.0) as usize;
        }
    }
    valid >= n / 2 && dark as f64 >= 0.55 * n as f64
}

fn ring_median(gray: &Gray, h: &M3, radius: f64) -> Option<f64> {
    let mut values = Vec::with_capacity(12);
    for i in 0..12 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 12.0;
        let (x, y) = project(h, radius * angle.cos(), radius * angle.sin());
        let (value, valid) = sample(gray, x, y);
        if valid {
            values.push(value);
        }
    }
    if values.len() < 6 {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

/// Decide whether a second, inverted frontend pass is warranted. The fitted
/// candidate's center is the bullseye; a center substantially brighter than
/// either the canonical quiet ring or the contour exterior is white-on-black
/// evidence. Sorting by the detected inner center rejects the wrong conic pose.
fn needs_inverted_view(gray: &Gray, candidates: &[Candidate], k: &M3) -> bool {
    for cand in candidates {
        let mut hs = pose::pose_homographies(&cand.outer, k);
        if hs.is_empty() {
            continue;
        }
        if let Some(inner) = &cand.inner {
            let origin_err = |h: &M3| -> f64 {
                let hinv = mat::inv3(h);
                let p = mat::matvec(&hinv, &[inner.cx, inner.cy, 1.0]);
                ((p[0] / p[2]).powi(2) + (p[1] / p[2]).powi(2)).sqrt()
            };
            hs.sort_by(|a, b| origin_err(a).partial_cmp(&origin_err(b)).unwrap());
        }
        let h = &hs[0];
        let (cx, cy) = project(h, 0.0, 0.0);
        let (center, valid) = sample(gray, cx, cy);
        if !valid {
            continue;
        }
        let mut reference = f64::INFINITY;
        // All v1 variants share r_bullseye=.22 and r_data_in=.30.
        if let Some(value) = ring_median(gray, h, 0.26) {
            reference = reference.min(value);
        }
        // Also catches a surviving white bullseye after outer-ring occlusion.
        if let Some(value) = ring_median(gray, h, 1.08) {
            reference = reference.min(value);
        }
        if reference.is_finite() && center - reference > 30.0 {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// cell sample patterns (decode 3x3; denser-angular for the phase refine)
// ---------------------------------------------------------------------------

pub struct SamplePattern {
    xy: Vec<(f64, f64)>, // (rings * sectors * sub), cell-major
    sub: usize,
    rho_q: f64,
}

// Patterns are pure functions of the (static) specs; building them per
// candidate was measurable in profiles. OnceLock pairs = (decode, refine).
static PATTERNS: std::sync::OnceLock<Vec<(&'static str, SamplePattern, SamplePattern)>> =
    std::sync::OnceLock::new();

fn patterns_for(spec: &MarkerSpec) -> (&'static SamplePattern, &'static SamplePattern) {
    let all = PATTERNS.get_or_init(|| {
        crate::spec::variants()
            .iter()
            .map(|sp| (sp.name, cell_sample_points(sp), refine_sample_points(sp)))
            .collect()
    });
    let e = all.iter().find(|(n, _, _)| *n == spec.name).unwrap();
    (&e.1, &e.2)
}

fn cell_sample_points(spec: &MarkerSpec) -> SamplePattern {
    let (_, ring_c, _) = spec.ring_radii();
    let step = 2.0 * std::f64::consts::PI / spec.sector_count as f64;
    let dr_step = ring_c[1] - ring_c[0];
    let drs = [-0.25, 0.0, 0.25];
    let dps = [-0.3 * step, 0.0, 0.3 * step];
    let mut xy = Vec::with_capacity(spec.ring_count * spec.sector_count * 9);
    for ring in 0..spec.ring_count {
        for s in 0..spec.sector_count {
            let phi0 = (s as f64 + 0.5) * step;
            // meshgrid indexing="ij": dr-major
            for &dr in &drs {
                for &dp in &dps {
                    let rho = ring_c[ring] + dr * dr_step;
                    let phi = phi0 + dp;
                    xy.push((rho * phi.cos(), rho * phi.sin()));
                }
            }
        }
    }
    SamplePattern {
        xy,
        sub: 9,
        rho_q: (spec.r_bullseye + spec.r_data_in) / 2.0,
    }
}

fn refine_sample_points(spec: &MarkerSpec) -> SamplePattern {
    let (_, ring_c, _) = spec.ring_radii();
    let step = 2.0 * std::f64::consts::PI / spec.sector_count as f64;
    let dr_step = ring_c[1] - ring_c[0];
    let drs = [-0.25, 0.0, 0.25];
    let n_ang = ((step.to_degrees() / 2.0).clamp(7.0, 13.0) as usize) | 1;
    let dps: Vec<f64> = (0..n_ang)
        .map(|i| (-0.38 + 0.76 * i as f64 / (n_ang - 1) as f64) * step)
        .collect();
    let mut xy = Vec::with_capacity(spec.ring_count * spec.sector_count * 3 * n_ang);
    for ring in 0..spec.ring_count {
        for s in 0..spec.sector_count {
            for &dr in &drs {
                for &dp in &dps {
                    let rho = ring_c[ring] + dr * dr_step;
                    let phi = (s as f64 + 0.5) * step + dp;
                    xy.push((rho * phi.cos(), rho * phi.sin()));
                }
            }
        }
    }
    SamplePattern {
        xy,
        sub: 3 * n_ang,
        rho_q: 0.0,
    }
}

// ---------------------------------------------------------------------------
// _build_grid
// ---------------------------------------------------------------------------

/// Black/white reference levels for a grid hypothesis: bullseye center vs the
/// white quiet ring. Returns (mid, span), or None on missing contrast.
fn grid_refs(gray: &Gray, h: &M3, pat: &SamplePattern) -> Option<(f64, f64)> {
    let (bx, by) = project(h, 0.0, 0.0);
    let (qx, qy) = project(h, pat.rho_q, 0.0);
    let (bv, bok) = sample(gray, bx, by);
    let (qv, qok) = sample(gray, qx, qy);
    if !bok || !qok || (qv - bv).abs() < 20.0 {
        return None;
    }
    Some((0.5 * (bv + qv), (qv - bv).abs() / 2.0))
}

/// Sample the cells in `range` into grid/conf. False = some cell had no valid
/// subsample (the whole-grid build fails in that case, exactly as before).
fn sample_cells(
    gray: &Gray,
    h: &M3,
    pat: &SamplePattern,
    mid: f64,
    span: f64,
    range: std::ops::Range<usize>,
    grid: &mut [u8],
    conf: &mut [f32],
) -> bool {
    for cell in range {
        let mut sum = 0f64;
        let mut cnt = 0usize;
        for k in 0..pat.sub {
            let (x, y) = pat.xy[cell * pat.sub + k];
            let (px, py) = project(h, x, y);
            let (v, ok) = sample(gray, px, py);
            if ok {
                sum += v;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return false;
        }
        let m = sum / cnt as f64;
        grid[cell] = (m < mid) as u8;
        conf[cell] = (((m - mid).abs() / span).min(1.0)) as f32;
    }
    true
}

// ---------------------------------------------------------------------------
// _refine_phase
// ---------------------------------------------------------------------------

/// Returns (refined theta, verify corr): the peak full-grid correlation is
/// also the decode-verify score (see VERIFY_MIN in the Python reference).
fn refine_phase(
    gray: &Gray,
    hbase: &M3,
    spec: &MarkerSpec,
    theta0: f64,
    ref_grid: &[u8],
    pat: &SamplePattern,
) -> (f64, f64) {
    let step = 2.0 * std::f64::consts::PI / spec.sector_count as f64;
    let ncells = spec.ring_count * spec.sector_count;
    let refv: Vec<f64> = ref_grid.iter().map(|&b| if b > 0 { -1.0 } else { 1.0 }).collect();
    let ref_norm: f64 = (ncells as f64).sqrt(); // all entries are +-1

    let corr = |ths: &[f64]| -> Option<Vec<f64>> {
        let mut out = Vec::with_capacity(ths.len());
        for &th in ths {
            let (s, c) = th.sin_cos();
            let mut m = vec![0f64; ncells];
            for cell in 0..ncells {
                let mut sum = 0f64;
                let mut cnt = 0usize;
                for k in 0..pat.sub {
                    let (x, y) = pat.xy[cell * pat.sub + k];
                    let (xr, yr) = (x * c - y * s, x * s + y * c);
                    let (px, py) = project(hbase, xr, yr);
                    let (v, ok) = sample(gray, px, py);
                    if ok {
                        sum += v;
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    return None;
                }
                m[cell] = sum / cnt as f64;
            }
            let mean: f64 = m.iter().sum::<f64>() / ncells as f64;
            let mut dot = 0f64;
            let mut nrm = 0f64;
            for i in 0..ncells {
                let d = m[i] - mean;
                dot += d * refv[i];
                nrm += d * d;
            }
            let nm = nrm.sqrt() * ref_norm;
            out.push(if nm > 1e-6 { dot / nm.max(1e-9) } else { -2.0 });
        }
        Some(out)
    };

    let lin = |a: f64, b: f64, n: usize| -> Vec<f64> {
        (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
    };
    let ths1: Vec<f64> = lin(theta0 - 0.5 * step, theta0 + 0.5 * step, 13);
    let cs1 = match corr(&ths1) {
        Some(v) => v,
        None => return (theta0, -1.0),
    };
    let mut pk = 0usize;
    for (i, &v) in cs1.iter().enumerate() {
        if v > cs1[pk] {
            pk = i;
        }
    }
    let t_pk = ths1[pk];
    let ths2: Vec<f64> = lin(t_pk - step / 10.0, t_pk + step / 10.0, 13);
    let cs2 = match corr(&ths2) {
        Some(v) => v,
        None => return (t_pk, cs1.iter().cloned().fold(f64::MIN, f64::max)),
    };
    let vc = cs2.iter().cloned().fold(f64::MIN, f64::max);
    // least-squares quadratic over the whole fine window
    let mut a = Vec::with_capacity(13 * 3);
    for &th in &ths2 {
        let x = th - t_pk;
        a.extend_from_slice(&[x * x, x, 1.0]);
    }
    let (coef, _) = crate::fitellipse::lstsq(&a, 13, 3, &cs2);
    let (a2, a1) = (coef[0], coef[1]);
    if a2 < -1e-9 {
        let v = t_pk - a1 / (2.0 * a2);
        if ths2[0] <= v && v <= ths2[12] {
            return (v, vc);
        }
    }
    let mut pk2 = 0usize;
    for (i, &v) in cs2.iter().enumerate() {
        if v > cs2[pk2] {
            pk2 = i;
        }
    }
    (ths2[pk2], vc)
}

// ---------------------------------------------------------------------------
// _refine_ellipse
// ---------------------------------------------------------------------------

fn refine_ellipse(gray: &Gray, geom: &EllipseGeom) -> Option<EllipseGeom> {
    const N_RAYS: usize = 128;
    const NS: usize = 13;
    let a = geom.major / 2.0;
    let b = geom.minor / 2.0;
    let th = geom.angle_deg.to_radians();
    let (s, c) = th.sin_cos();
    let w = (0.25 * a.min(b)).clamp(1.5, 3.0);
    let step = 2.0 * w / (NS - 1) as f64;

    let mut pxs: Vec<f64> = Vec::with_capacity(N_RAYS);
    let mut pys: Vec<f64> = Vec::with_capacity(N_RAYS);
    let mut n_ok = 0usize;
    for ray in 0..N_RAYS {
        let t = ray as f64 / N_RAYS as f64 * 2.0 * std::f64::consts::PI;
        let (st, ct) = t.sin_cos();
        let ex = geom.cx + a * ct * c - b * st * s;
        let ey = geom.cy + a * ct * s + b * st * c;
        let mut nx = ct / a * c - st / b * s;
        let mut ny = ct / a * s + st / b * c;
        let nn = (nx * nx + ny * ny).sqrt();
        nx /= nn;
        ny /= nn;
        let mut vals = [0f64; NS];
        let mut all_valid = true;
        for k in 0..NS {
            let off = -w + step * k as f64;
            let (v, ok) = sample(gray, ex + off * nx, ey + off * ny);
            vals[k] = v;
            all_valid &= ok;
        }
        // derivative at midpoints; first max (np.argmax)
        let mut best = 0usize;
        let mut dmax = vals[1] - vals[0];
        for k in 1..NS - 1 {
            let d = vals[k + 1] - vals[k];
            if d > dmax {
                dmax = d;
                best = k;
            }
        }
        let ok = all_valid && best > 0 && best < NS - 2 && dmax > 4.0;
        if !ok {
            continue;
        }
        n_ok += 1;
        let y0 = vals[best] - vals[best - 1];
        let y2 = vals[best + 2] - vals[best + 1];
        let denom = y0 - 2.0 * dmax + y2;
        let mut delta = if denom.abs() > 1e-9 {
            0.5 * (y0 - y2) / denom
        } else {
            0.0
        };
        if !delta.is_finite() {
            delta = 0.0;
        }
        delta = delta.clamp(-1.0, 1.0);
        let offs_i = -w + step * best as f64;
        let pos = offs_i + 0.5 * step + delta * step;
        pxs.push(ex + pos * nx);
        pys.push(ey + pos * ny);
    }
    if n_ok < 32 {
        // max(12, 128 // 4)
        return None;
    }
    let pts: Vec<(f32, f32)> = pxs
        .iter()
        .zip(&pys)
        .map(|(&x, &y)| (x as f32, y as f32))
        .collect();
    let r = fit_ellipse_f32(&pts);
    // sanity: big jump or axis-ratio blowup means it latched onto clutter
    let jump = ((r.cx - geom.cx).powi(2) + (r.cy - geom.cy).powi(2)).sqrt();
    if jump > 0.05 * a.max(b) + 1.0
        || !(0.8..1.25).contains(&(r.major / geom.major))
        || !(0.8..1.25).contains(&(r.minor / geom.minor))
    {
        return None;
    }
    Some(r)
}

fn fit_ellipse_f32(pts: &[(f32, f32)]) -> EllipseGeom {
    // cv2.fitEllipse on a float32 array goes down the same Point2f path
    fit_ellipse_pts(pts)
}

// ---------------------------------------------------------------------------
// _try_decode_spec
// ---------------------------------------------------------------------------

pub struct DecodeHit {
    pub variant: &'static str,
    pub mode: &'static str,
    pub value: Value,
    pub chosen_h: M3,
    pub rs: codec::RsStats,
    pub verify_corr: f64, // decode-verify matched-filter score (gate: VERIFY_MIN)
    pub sync_score: f64,  // normalized sync-ring correlation; <0 = spec has no sync
}

// Decode-verify gate + deconvolution retry: constants mirror the Python
// reference (detect.py), where their calibration data lives.
const VERIFY_MIN: f64 = 0.73;
const DECONV_MAX_PX: f64 = 80.0;
const DECONV_SIGMAS: [f64; 2] = [1.0, 1.6];
const DECONV_LAMBDA: f64 = 0.01;

/// Crop the candidate (1.5x its ellipse), pad edge-replicate to pow2, and
/// Wiener-deconvolve a Gaussian PSF. Returns (patch, T) with T the
/// image->patch translation homography, or None when the crop degenerates.
fn deconv_patch(gray: &Gray, geom: &EllipseGeom, sigma: f64) -> Option<(Gray, M3)> {
    let r = 0.75 * geom.major.max(geom.minor);
    let x0 = (geom.cx - r).max(0.0) as i64;
    let y0 = (geom.cy - r).max(0.0) as i64;
    let x1 = (geom.cx + r).min(gray.w as f64) as i64;
    let y1 = (geom.cy + r).min(gray.h as f64) as i64;
    if x1 - x0 < 12 || y1 - y0 < 12 {
        return None;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let (pw, ph) = (x1 as usize - x0, y1 as usize - y0);
    let fw = pw.next_power_of_two();
    let fh = ph.next_power_of_two();
    // edge-replicate pad (np.pad mode="edge", bottom/right only)
    let mut padded = vec![0f64; fh * fw];
    for y in 0..fh {
        let sy = y.min(ph - 1);
        for x in 0..fw {
            let sx = x.min(pw - 1);
            padded[y * fw + x] = gray.px[(y0 + sy) * gray.w + (x0 + sx)] as f64;
        }
    }
    let fys = crate::fft::fftfreq(fh);
    let fxs = crate::fft::fftfreq(fw);
    let two_pi2_s2 = 2.0 * std::f64::consts::PI.powi(2) * sigma * sigma;
    let filt: Vec<f64> = (0..fh * fw)
        .map(|i| {
            let g = (-two_pi2_s2 * (fys[i / fw].powi(2) + fxs[i % fw].powi(2))).exp();
            g / (g * g + DECONV_LAMBDA)
        })
        .collect();
    let out = crate::fft::filter2d_real(&padded, fh, fw, &filt);
    let mut px = vec![0u8; ph * pw];
    for y in 0..ph {
        for x in 0..pw {
            // np clip + astype(uint8): clamp then truncate toward zero
            px[y * pw + x] = out[y * fw + x].clamp(0.0, 255.0) as u8;
        }
    }
    let t = [[1.0, 0.0, -(x0 as f64)], [0.0, 1.0, -(y0 as f64)], [0.0, 0.0, 1.0]];
    Some((Gray { w: pw, h: ph, px }, t))
}

/// The ISI retry: deconvolve a small failed candidate and rerun the decode
/// search on the cleaned patch. `hs` are image-frame homographies; the hit's
/// chosen_h is mapped back to image frame.
fn deconv_retry(
    gray: &Gray,
    geom: &EllipseGeom,
    hs: &[M3],
    specs: &[&'static MarkerSpec],
    conf_erasure: f32,
) -> Option<DecodeHit> {
    if geom.major.max(geom.minor) >= DECONV_MAX_PX {
        return None;
    }
    for &sg in &DECONV_SIGMAS {
        let (patch, t) = deconv_patch(gray, geom, sg)?;
        let hp: Vec<M3> = hs.iter().map(|h| mat::matmul(&t, h)).collect();
        let tinv = mat::inv3(&t);
        for sp in specs {
            if let Some(mut hit) = try_decode_spec(&patch, &hp, sp, conf_erasure) {
                hit.chosen_h = mat::matmul(&tinv, &hit.chosen_h);
                return Some(hit);
            }
        }
    }
    None
}

/// If an occluder breaks the outer ring, contour detection can still return the
/// intact bullseye. Since it is a concentric circle on the same plane, scaling
/// its homography by the known bullseye radius recovers the tag geometry.
fn bullseye_retry(
    gray: &Gray,
    hs: &[M3],
    specs: &[&'static MarkerSpec],
    conf_erasure: f32,
) -> Option<DecodeHit> {
    for sp in specs {
        let expanded: Vec<M3> = hs
            .iter()
            .map(|h| scale_h(h, 1.0 / sp.r_bullseye))
            .filter(|h| has_outer_ring_contrast(gray, h, sp))
            .collect();
        if expanded.is_empty() {
            continue;
        }
        if let Some(hit) = try_decode_spec(gray, &expanded, sp, conf_erasure) {
            return Some(hit);
        }
    }
    None
}

fn try_decode_spec(
    gray: &Gray,
    hs: &[M3],
    spec: &'static MarkerSpec,
    conf_erasure: f32,
) -> Option<DecodeHit> {
    let step = 2.0 * std::f64::consts::PI / spec.sector_count as f64;
    let (pat, rpat) = patterns_for(spec);
    let sync_min = 0.70 * spec.sector_count as f64;
    for h in hs {
        for &scale in &[1.0, 1.06, 1.12, 0.94] {
            let hs_ = scale_h(h, scale);
            for k in 0..6 {
                let phi0 = step * k as f64 / 6.0;
                let hphi = rotate_h(&hs_, phi0);
                let (mid, span) = match grid_refs(gray, &hphi, pat) {
                    Some(v) => v,
                    None => continue,
                };
                let ncells = spec.ring_count * spec.sector_count;
                let mut grid = vec![0u8; ncells];
                let mut conf = vec![0f32; ncells];
                // Sync-first: sample only the sync ring, gate on it, and build
                // the rest of the grid only for survivors. Every attempt that
                // fails here also failed (as a continue) in the whole-grid
                // order, so decode decisions are unchanged; the bulk of
                // wrong-variant / clutter attempts now stop after 1/3-1/5 of
                // the sampling work.
                let sync_score = if spec.has_sync {
                    if !sample_cells(gray, &hphi, pat, mid, span,
                                     0..spec.sector_count, &mut grid, &mut conf) {
                        continue;
                    }
                    let (_, scores) = codec::find_rotation(&grid[..spec.sector_count], spec);
                    let best = *scores.iter().max().unwrap() as f64;
                    if best < sync_min {
                        continue;
                    }
                    if !sample_cells(gray, &hphi, pat, mid, span,
                                     spec.sector_count..ncells, &mut grid, &mut conf) {
                        continue;
                    }
                    best / spec.sector_count as f64
                } else {
                    if !sample_cells(gray, &hphi, pat, mid, span,
                                     0..ncells, &mut grid, &mut conf) {
                        continue;
                    }
                    -1.0
                };
                let (res, sh) = codec::decode_conf(&grid, spec, &conf, conf_erasure);
                if let Some((pb, rs)) = res {
                    let theta0 = phi0 + sh as f64 * step;
                    let ref_grid = codec::encode(&pb, spec).unwrap();
                    let (refined, vcorr) =
                        refine_phase(gray, &hs_, spec, theta0, &ref_grid, rpat);
                    // Decode-verify gate: matched filter of the image against
                    // the decoded codeword's full grid; rejects wrong-value
                    // RS+CRC collisions (see the Python reference for the
                    // measured calibration behind VERIFY_MIN).
                    if vcorr < VERIFY_MIN {
                        continue;
                    }
                    let d = (refined - theta0 + std::f64::consts::PI)
                        .rem_euclid(2.0 * std::f64::consts::PI)
                        - std::f64::consts::PI;
                    let theta = theta0 + d;
                    let chosen_h = rotate_h(&hs_, theta);
                    let (mode, value) = match payload::decode(&pb, spec) {
                        Ok((m, v)) => (m, v),
                        Err(_) => ("UNKNOWN", Value::Bytes(pb)),
                    };
                    return Some(DecodeHit {
                        variant: spec.name,
                        mode,
                        value,
                        chosen_h,
                        rs,
                        verify_corr: vcorr,
                        sync_score,
                    });
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// detect_markers
// ---------------------------------------------------------------------------

pub fn detect_markers(
    gray_raw: &Gray,
    k: &M3,
    specs: &[&'static MarkerSpec],
    conf_erasure: f32,
    pose_only: bool,
    dist: Option<&[f64]>,
) -> Vec<Detection> {
    let undistorted;
    let gray_in = match dist {
        Some(d) => {
            undistorted = crate::image::undistort(gray_raw, k, d);
            &undistorted
        }
        None => gray_raw,
    };
    let gray = sharpen(gray_in, 0.6, 1.0);
    let normal_cands = find_marker_ellipses(&gray);
    let inverse = if needs_inverted_view(&gray, &normal_cands, k) {
        let raw = Gray {
            w: gray_in.w,
            h: gray_in.h,
            px: gray_in.px.iter().map(|&v| 255 - v).collect(),
        };
        Some(sharpen(&raw, 0.6, 1.0))
    } else {
        None
    };
    let mut cands: Vec<(Candidate, bool)> = normal_cands
        .into_iter()
        .map(|candidate| (candidate, false))
        .collect();
    if let Some(inverse_view) = &inverse {
        cands.extend(
            find_marker_ellipses(inverse_view)
            .into_iter()
            .map(|candidate| (candidate, true)),
        );
    }
    // per-candidate work is independent; map preserves candidate order, so the
    // result is identical to the sequential loop (parity gates re-run green)
    let process = |entry: &(Candidate, bool)| -> Option<Detection> {
        let (cand, inverted) = entry;
        let work_gray = if *inverted {
            inverse.as_ref().unwrap()
        } else {
            &gray
        };
        let geom0 = cand.outer;
        let refined = refine_ellipse(work_gray, &geom0);
        let geom1 = refined.unwrap_or(geom0);
        let mut hs = pose::pose_homographies(&geom1, k);
        let mut hs_coarse = if refined.is_some() && geom1.major.max(geom1.minor) < 100.0 {
            pose::pose_homographies(&geom0, k)
        } else {
            Vec::new()
        };
        if hs.is_empty() {
            return None;
        }
        if let Some(inner) = &cand.inner {
            let origin_err = |h: &M3| -> f64 {
                let hinv = mat::inv3(h);
                let p = mat::matvec(&hinv, &[inner.cx, inner.cy, 1.0]);
                ((p[0] / p[2]).powi(2) + (p[1] / p[2]).powi(2)).sqrt()
            };
            hs.sort_by(|a, b| origin_err(a).partial_cmp(&origin_err(b)).unwrap());
            hs_coarse.sort_by(|a, b| origin_err(a).partial_cmp(&origin_err(b)).unwrap());
        }
        hs.extend(hs_coarse);
        let mut path = "direct";
        let mut hit = None;
        for sp in specs {
            hit = try_decode_spec(work_gray, &hs, sp, conf_erasure);
            if hit.is_some() {
                break;
            }
        }
        let mut geom_rep = geom1;
        // circular-sticker fallback: outer contour failed to decode; retry at
        // the largest suppressed round child (the tag ring inside a label)
        if hit.is_none() {
            if let Some(alt) = &cand.alt {
                let alt1 = refine_ellipse(work_gray, alt).unwrap_or(*alt);
                let mut hs_alt = pose::pose_homographies(&alt1, k);
                if let Some(inner) = &cand.inner {
                    let origin_err = |h: &M3| -> f64 {
                        let hinv = mat::inv3(h);
                        let p = mat::matvec(&hinv, &[inner.cx, inner.cy, 1.0]);
                        ((p[0] / p[2]).powi(2) + (p[1] / p[2]).powi(2)).sqrt()
                    };
                    hs_alt.sort_by(|a, b| origin_err(a).partial_cmp(&origin_err(b)).unwrap());
                }
                for sp in specs {
                    hit = try_decode_spec(work_gray, &hs_alt, sp, conf_erasure);
                    if hit.is_some() {
                        geom_rep = alt1; // report the TAG's geometry
                        path = "sticker";
                        break;
                    }
                }
            }
        }
        // Occlusion fallback: the candidate can be the surviving bullseye when
        // the marker's outer ring is no longer a closed contour.
        if hit.is_none() {
            hit = bullseye_retry(work_gray, &hs, specs, conf_erasure);
            if let Some(decoded) = &hit {
                geom_rep = ellipse_from_h(&decoded.chosen_h);
                path = "bullseye";
            }
        }
        // ISI retry: small candidate, all attempts failed -> deconvolve the
        // patch and search again (see deconv_retry above).
        if hit.is_none() {
            hit = deconv_retry(work_gray, &geom1, &hs, specs, conf_erasure);
            if hit.is_some() {
                path = "deconv";
            }
        }
        if hit.is_none() && (cand.inner.is_none() || !pose_only) {
            return None;
        }
        let chosen_h = hit.as_ref().map(|h| h.chosen_h).unwrap_or(hs[0]);
        let (r, t) = pose::decompose_h(&chosen_h, k);
        let info = hit.as_ref().map(|h| DecodeInfo {
            rs_erasures: h.rs.erasures,
            rs_corrected: h.rs.corrected,
            verify_corr: h.verify_corr,
            sync_score: h.sync_score,
            path,
        });
        Some(Detection {
            center: (geom_rep.cx, geom_rep.cy),
            axes: (geom_rep.major, geom_rep.minor),
            angle: geom_rep.angle_deg,
            r,
            t,
            h: chosen_h,
            tilt_deg: pose::tilt_from_h(&chosen_h, k),
            inverted: *inverted,
            decoded: hit.map(|h| (h.variant, h.mode, h.value)),
            info,
        })
    };
    #[cfg(feature = "parallel")]
    let mut out: Vec<Detection> = {
        use rayon::prelude::*;
        cands.par_iter().filter_map(process).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let mut out: Vec<Detection> = cands.iter().filter_map(process).collect();
    // de-dup: keep the larger-radius detection within a center-proximity cluster
    out.sort_by(|a, b| {
        b.decoded
            .is_some()
            .cmp(&a.decoded.is_some())
            .then_with(|| {
                b.axes
                    .0
                    .max(b.axes.1)
                    .partial_cmp(&a.axes.0.max(a.axes.1))
                    .unwrap()
            })
    });
    let mut kept: Vec<Detection> = Vec::new();
    for d in out {
        let r0 = d.axes.0.max(d.axes.1) / 2.0;
        let ok = kept.iter().all(|kd| {
            ((d.center.0 - kd.center.0).powi(2) + (d.center.1 - kd.center.1).powi(2)).sqrt()
                > 0.4 * r0
        });
        if ok {
            kept.push(d);
        }
    }
    kept
}

/// detect.py's headless `detect()`: decoded markers ONLY, no dedup, no pose-only
/// boxes -- what the validation harnesses (sweep/pose/range) consume.
pub fn detect(
    gray_raw: &Gray,
    k: &M3,
    specs: &[&'static MarkerSpec],
    conf_erasure: f32,
    dist: Option<&[f64]>,
) -> Vec<Detection> {
    let undistorted;
    let gray_in = match dist {
        Some(d) => {
            undistorted = crate::image::undistort(gray_raw, k, d);
            &undistorted
        }
        None => gray_raw,
    };
    let gray = sharpen(gray_in, 0.6, 1.0);
    let normal_cands = find_marker_ellipses(&gray);
    let inverse = if needs_inverted_view(&gray, &normal_cands, k) {
        let raw = Gray {
            w: gray_in.w,
            h: gray_in.h,
            px: gray_in.px.iter().map(|&v| 255 - v).collect(),
        };
        Some(sharpen(&raw, 0.6, 1.0))
    } else {
        None
    };
    let mut cands: Vec<(Candidate, bool)> = normal_cands
        .into_iter()
        .map(|candidate| (candidate, false))
        .collect();
    if let Some(inverse_view) = &inverse {
        cands.extend(
            find_marker_ellipses(inverse_view)
            .into_iter()
            .map(|candidate| (candidate, true)),
        );
    }
    let process = |entry: &(Candidate, bool)| -> Option<Detection> {
        let (cand, inverted) = entry;
        let work_gray = if *inverted {
            inverse.as_ref().unwrap()
        } else {
            &gray
        };
        let geom0 = cand.outer;
        let refined = refine_ellipse(work_gray, &geom0);
        let geom1 = refined.unwrap_or(geom0);
        let mut hs = pose::pose_homographies(&geom1, k);
        if refined.is_some() && geom1.major.max(geom1.minor) < 100.0 {
            hs.extend(pose::pose_homographies(&geom0, k));
        }
        if hs.is_empty() {
            return None;
        }
        let mut path = "direct";
        let mut found: Option<(DecodeHit, EllipseGeom)> = None;
        for sp in specs {
            if let Some(hit) = try_decode_spec(work_gray, &hs, sp, conf_erasure) {
                found = Some((hit, geom1));
                break;
            }
        }
        if found.is_none() {
            if let Some(alt) = &cand.alt {
                // circular-sticker fallback (see detect_markers)
                let alt1 = refine_ellipse(work_gray, alt).unwrap_or(*alt);
                let hs_alt = pose::pose_homographies(&alt1, k);
                for sp in specs {
                    if let Some(hit) = try_decode_spec(work_gray, &hs_alt, sp, conf_erasure) {
                        found = Some((hit, alt1));
                        path = "sticker";
                        break;
                    }
                }
            }
        }
        // Occlusion fallback: retry a surviving bullseye at outer-ring scale.
        if found.is_none() {
            found = bullseye_retry(work_gray, &hs, specs, conf_erasure).map(|hit| {
                let g = ellipse_from_h(&hit.chosen_h);
                path = "bullseye";
                (hit, g)
            });
        }
        // ISI retry: deconvolve small failed candidates (see detect_markers)
        if found.is_none() {
            found = deconv_retry(work_gray, &geom1, &hs, specs, conf_erasure)
                .map(|hit| {
                    path = "deconv";
                    (hit, geom1)
                });
        }
        found.map(|(hit, g)| {
            let (r, t) = pose::decompose_h(&hit.chosen_h, k);
            Detection {
                center: (g.cx, g.cy),
                axes: (g.major, g.minor),
                angle: g.angle_deg,
                r,
                t,
                h: hit.chosen_h,
                tilt_deg: pose::tilt_from_h(&hit.chosen_h, k),
                inverted: *inverted,
                info: Some(DecodeInfo {
                    rs_erasures: hit.rs.erasures,
                    rs_corrected: hit.rs.corrected,
                    verify_corr: hit.verify_corr,
                    sync_score: hit.sync_score,
                    path,
                }),
                decoded: Some((hit.variant, hit.mode, hit.value)),
            }
        })
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        cands.par_iter().filter_map(process).collect()
    }
    #[cfg(not(feature = "parallel"))]
    cands.iter().filter_map(process).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec;

    fn render_tag(sp: &MarkerSpec, value: u128, inverted: bool) -> Gray {
        let size = 360usize;
        let grid = codec::encode(&payload::encode_id(value, sp).unwrap(), sp).unwrap();
        let center = (size as f64 - 1.0) / 2.0;
        let radius_px = size as f64 / 2.0 * 0.88;
        let ring_width = (sp.r_data_out - sp.r_data_in) / sp.ring_count as f64;
        let sector_step = 2.0 * std::f64::consts::PI / sp.sector_count as f64;
        let mut px = vec![255u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f64 - center) / radius_px;
                let dy = (y as f64 - center) / radius_px;
                let radius = (dx * dx + dy * dy).sqrt();
                let mut value_px = 255u8;
                if (sp.r_ring_in..=1.0).contains(&radius) || radius <= sp.r_bullseye {
                    value_px = 0;
                } else if (sp.r_data_in..sp.r_data_out).contains(&radius) {
                    let ring = (((radius - sp.r_data_in) / ring_width) as usize)
                        .min(sp.ring_count - 1);
                    let angle = dy.atan2(dx).rem_euclid(2.0 * std::f64::consts::PI);
                    let sector = ((angle / sector_step) as usize).min(sp.sector_count - 1);
                    if grid[ring * sp.sector_count + sector] == 1 {
                        value_px = 0;
                    }
                }
                px[y * size + x] = if inverted { 255 - value_px } else { value_px };
            }
        }
        Gray { w: size, h: size, px }
    }

    #[test]
    fn detects_both_polarities_for_every_variant() {
        let size = 360.0;
        let f = (size / 2.0) / (std::f64::consts::PI / 6.0).tan();
        let k = [[f, 0.0, 179.5], [0.0, f, 179.5], [0.0, 0.0, 1.0]];
        for &(sp, value) in &[(&spec::T, 42), (&spec::M, 0xabcdef), (&spec::D, 123456789)] {
            for inverted in [false, true] {
                let image = render_tag(sp, value, inverted);
                let hits = detect(&image, &k, &[sp], 0.25, None);
                assert_eq!(hits.len(), 1, "{} inverted={}", sp.name, inverted);
                assert_eq!(hits[0].inverted, inverted);
                assert!(matches!(
                    hits[0].decoded,
                    Some((variant, "ID", Value::Int(got)))
                        if variant == sp.name && got == value
                ));
            }
        }
    }
}
