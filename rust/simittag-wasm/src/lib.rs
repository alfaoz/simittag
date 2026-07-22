//! WASM bindings: a flat-buffer API so a frame crosses the JS boundary as one
//! Uint8Array and results come back as one JSON string (parsed once per frame;
//! the payload is a handful of detections, so JSON beats a hand-rolled binary
//! layout on simplicity with no measurable cost at 30 fps).

use simittag_core::{detector, image::Gray, payload::Value, spec};
use std::sync::atomic::{AtomicU64, Ordering};
use wasm_bindgen::prelude::*;

// threaded build: JS awaits initThreadPool(n) once before the first detect
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

static NOISE_SEED: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

/// Honesty degradation, applied BEFORE detection. Order matches a real camera:
/// motion smear (shutter open while the scene moves) -> defocus blur -> sensor
/// noise. motion_px = horizontal smear length in pixels (conveyor page derives
/// it as speed * shutter * f / Z -- the actual physics); implemented as a
/// horizontal box blur, the exact PSF of linear motion at constant velocity.
/// The noise RNG re-seeds per call so it shimmers per frame like np.random.
#[wasm_bindgen]
pub fn degrade(
    gray: &[u8],
    w: usize,
    h: usize,
    blur_sigma: f64,
    noise_std: f64,
    motion_px: f64,
) -> Vec<u8> {
    let mut img = Gray {
        w,
        h,
        px: gray.to_vec(),
    };
    if motion_px >= 2.0 {
        let l = (motion_px.round() as usize).min(w / 2).max(2);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            let row = &img.px[y * w..(y + 1) * w];
            // prefix sums -> O(1) box mean per pixel, replicate border
            let mut pre = vec![0u32; w + 1];
            for x in 0..w {
                pre[x + 1] = pre[x] + row[x] as u32;
            }
            let half = l / 2;
            let orow = &mut out[y * w..(y + 1) * w];
            for x in 0..w {
                let a = x.saturating_sub(half);
                let b = (x + l - half).min(w);
                let sum = pre[b] - pre[a]
                    + (half.saturating_sub(x)) as u32 * row[0] as u32
                    + (x + l - half).saturating_sub(w) as u32 * row[w - 1] as u32;
                orow[x] = ((sum as f64 / l as f64).round() as i64).clamp(0, 255) as u8;
            }
        }
        img.px = out;
    }
    if blur_sigma > 0.0 {
        img = simittag_core::image::gaussian_blur_u8_fixed(&img, blur_sigma);
    }
    if noise_std > 0.0 {
        let mut s = NOISE_SEED.fetch_add(0x2545_f491_4f6c_dd1d, Ordering::Relaxed);
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut i = 0;
        while i < img.px.len() {
            // Box-Muller: two N(0,1) draws per transform
            let (u1, u2) = (next().max(1e-12), next());
            let r = (-2.0 * u1.ln()).sqrt();
            let (s2, c2) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            for &g in &[r * c2, r * s2] {
                if i < img.px.len() {
                    let v = img.px[i] as f64 + g * noise_std;
                    img.px[i] = v.clamp(0.0, 255.0) as u8;
                    i += 1;
                }
            }
        }
    }
    img.px
}

/// gray: w*h luma bytes. fx/fy/cx/cy: pinhole intrinsics for THIS resolution.
/// versions: "" = auto (T/M/D), else e.g. "M". pose_only mirrors the Python
/// flag (true = also return undecoded nested-ring candidates as pose-only).
#[wasm_bindgen]
pub fn detect(
    gray: &[u8],
    w: usize,
    h: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    versions: &str,
    pose_only: bool,
) -> String {
    let img = Gray {
        w,
        h,
        px: gray.to_vec(),
    };
    let k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]];
    let specs: Vec<&'static spec::MarkerSpec> = if versions.is_empty() || versions == "auto" {
        spec::default_variants().to_vec()
    } else {
        versions
            .split(',')
            .filter_map(spec::by_name)
            .collect()
    };
    let specs = if specs.is_empty() {
        spec::default_variants().to_vec()
    } else {
        specs
    };
    let dets = detector::detect_markers(&img, &k, &specs, 0.25, pose_only, None);

    let mut out = String::from("[");
    for (i, d) in dets.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let rflat: Vec<String> = d
            .r
            .iter()
            .flat_map(|row| row.iter().map(|v| format!("{:.9}", v)))
            .collect();
        let hflat: Vec<String> = d
            .h
            .iter()
            .flat_map(|row| row.iter().map(|v| format!("{:.9}", v)))
            .collect();
        out.push_str(&format!(
            "{{\"center\":[{:.3},{:.3}],\"axes\":[{:.3},{:.3}],\"angle\":{:.3},\
             \"R\":[{}],\"t\":[{:.6},{:.6},{:.6}],\"h\":[{}],\"tilt_deg\":{:.2},\"decoded\":{},\"inverted\":{}",
            d.center.0,
            d.center.1,
            d.axes.0,
            d.axes.1,
            d.angle,
            rflat.join(","),
            d.t[0],
            d.t[1],
            d.t[2],
            hflat.join(","),
            d.tilt_deg,
            d.decoded.is_some(),
            d.inverted
        ));
        if let Some(info) = &d.info {
            out.push_str(&format!(
                ",\"rs_corrected\":{},\"rs_erasures\":{},\"verify\":{:.3},\"path\":\"{}\"",
                info.rs_corrected, info.rs_erasures, info.verify_corr, info.path
            ));
            if info.sync_score >= 0.0 {
                out.push_str(&format!(",\"sync\":{:.3}", info.sync_score));
            }
        }
        if let Some((variant, mode, value)) = &d.decoded {
            let alias = spec::by_name(variant).map(|s| s.alias).unwrap_or("");
            out.push_str(&format!(
                ",\"variant\":\"{}\",\"alias\":\"{}\",\"mode\":\"{}\"",
                variant, alias, mode
            ));
            match value {
                Value::Int(v) => out.push_str(&format!(",\"value\":{}", v)),
                Value::Bytes(b) => {
                    let hex: String = b.iter().map(|v| format!("{:02x}", v)).collect();
                    out.push_str(&format!(",\"value_hex\":\"{}\"", hex));
                }
                Value::Geo { lat, lon, alt_m } => out.push_str(&format!(
                    ",\"geo\":{{\"lat\":{:.7},\"lon\":{:.7},\"alt_m\":{}}}",
                    lat, lon, alt_m
                )),
                Value::Tagged { namespace, id } => out.push_str(&format!(
                    ",\"tagged\":{{\"ns\":{},\"id\":{}}}",
                    namespace, id
                )),
            }
        }
        out.push('}');
    }
    out.push(']');
    out
}
