//! Parity/bench CLI. Phase gates run here:
//!
//!   simittag parity-spec  fixtures/spec.json     constants vs Python's tables
//!   simittag parity-codec fixtures/codec.json    bit-exact codec/payload vectors
//!   simittag cross-gen N                         emit N randomized decode cases
//!                                                (JSON lines) for Python to replay
//!
//! Every comparison is exact (bytes, ints, decisions); floats are compared by
//! reparsing the fixture literal to f64 and requiring bit equality, since the
//! Rust port performs the identical IEEE-754 operations as the Python reference.

use serde_json::Value as J;
use simittag_core::{codec, gf16, gf256, payload, spec};
use std::process::exit;

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn bytes_to_hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{:02x}", v)).collect()
}

struct Gate {
    name: &'static str,
    pass: usize,
    fail: usize,
}

impl Gate {
    fn new(name: &'static str) -> Self {
        Gate { name, pass: 0, fail: 0 }
    }
    fn check(&mut self, ok: bool, detail: impl Fn() -> String) {
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
            if self.fail <= 10 {
                eprintln!("  FAIL [{}] {}", self.name, detail());
            }
        }
    }
    fn report(&self) -> bool {
        println!(
            "  {:<14} {:>6} pass  {:>4} fail",
            self.name, self.pass, self.fail
        );
        self.fail == 0
    }
}

fn load(path: &str) -> J {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    serde_json::from_str(&data).unwrap()
}

// ---------------------------------------------------------------------------
// parity-spec
// ---------------------------------------------------------------------------

fn parity_spec(path: &str) -> bool {
    let fx = load(path);
    let mut g = Gate::new("spec");
    for sp in spec::variants() {
        let f = &fx[sp.name];
        let eq_f = |a: f64, b: &J| (a - b.as_f64().unwrap()).abs() < 1e-12;
        let eq_i = |a: usize, b: &J| a as i64 == b.as_i64().unwrap();
        g.check(eq_f(sp.r_bullseye, &f["R_BULLSEYE"]), || format!("{} R_BULLSEYE", sp.name));
        g.check(eq_f(sp.r_data_in, &f["R_DATA_IN"]), || format!("{} R_DATA_IN", sp.name));
        g.check(eq_f(sp.r_data_out, &f["R_DATA_OUT"]), || format!("{} R_DATA_OUT", sp.name));
        g.check(eq_f(sp.r_ring_in, &f["R_RING_IN"]), || format!("{} R_RING_IN", sp.name));
        g.check(eq_i(sp.ring_count, &f["RING_COUNT"]), || format!("{} RING_COUNT", sp.name));
        g.check(eq_i(sp.sector_count, &f["SECTOR_COUNT"]), || format!("{} SECTOR_COUNT", sp.name));
        g.check(eq_i(sp.rs_k, &f["RS_K"]), || format!("{} RS_K", sp.name));
        g.check(eq_i(sp.rs_nsym, &f["RS_NSYM"]), || format!("{} RS_NSYM", sp.name));
        g.check(
            sp.has_sync == f["HAS_SYNC"].as_bool().unwrap()
                && sp.use_header == f["USE_HEADER"].as_bool().unwrap(),
            || format!("{} flags", sp.name),
        );
        g.check(sp.alias == f["ALIAS"].as_str().unwrap(), || format!("{} ALIAS", sp.name));
        g.check(eq_i(sp.symbol_bits, &f["SYMBOL_BITS"]), || format!("{} SYMBOL_BITS", sp.name));
        g.check(eq_i(sp.payload_bits(), &f["payload_bits"]), || format!("{} payload_bits", sp.name));
        let max_err_ok = match (&sp.max_errors, &f["MAX_ERRORS"]) {
            (None, J::Null) => true,
            (Some(v), w) => Some(*v as i64) == w.as_i64(),
            _ => false,
        };
        g.check(max_err_ok, || format!("{} MAX_ERRORS", sp.name));
        let vmin_ok = match (&sp.verify_min, &f["VERIFY_MIN"]) {
            (None, J::Null) => true,
            (Some(v), w) => w.as_f64().map(|x| (x - v).abs() < 1e-12).unwrap_or(false),
            _ => false,
        };
        g.check(vmin_ok, || format!("{} VERIFY_MIN", sp.name));
        // exact at the field's working precision: the detector consumes an
        // f32, so the fixture's f64 literal must round to EXACTLY this f32
        // (no f32 lies between 0.4f64 and 0.4f32, so the erasure decision
        // itself is bit-identical across the representations)
        let ce_ok = match (&sp.conf_erasure, &f["CONF_ERASURE"]) {
            (None, J::Null) => true,
            (Some(v), w) => w.as_f64().map(|x| x as f32 == *v).unwrap_or(false),
            _ => false,
        };
        g.check(ce_ok, || format!("{} CONF_ERASURE", sp.name));
        let sync: Vec<u8> = f["SYNC"].as_array().unwrap().iter()
            .map(|v| v.as_i64().unwrap() as u8).collect();
        g.check(sync == sp.sync, || format!("{} SYNC", sp.name));
        g.check(eq_i(sp.data_ring_count(), &f["data_ring_count"]), || format!("{} drc", sp.name));
        g.check(eq_i(sp.first_data_ring(), &f["first_data_ring"]), || format!("{} fdr", sp.name));
        g.check(eq_i(sp.payload_bytes(), &f["payload_bytes"]), || format!("{} pb", sp.name));
        let (_, center, _) = sp.ring_radii();
        let fc: Vec<f64> = f["ring_center_radii"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect();
        g.check(
            center.len() == fc.len()
                && center.iter().zip(&fc).all(|(a, b)| (a - b).abs() < 1e-12),
            || format!("{} ring_center_radii", sp.name),
        );
        let ang = sp.sector_center_angles();
        let fa: Vec<f64> = f["sector_center_angles"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect();
        g.check(
            ang.iter().zip(&fa).all(|(a, b)| (a - b).abs() < 1e-12),
            || format!("{} sector angles", sp.name),
        );
    }
    g.report()
}

// ---------------------------------------------------------------------------
// parity-codec
// ---------------------------------------------------------------------------

fn grid_to_rows(grid: &[u8], sectors: usize) -> Vec<String> {
    grid.chunks(sectors)
        .map(|r| r.iter().map(|&b| char::from(b'0' + b)).collect())
        .collect()
}

fn num_str(v: &J) -> String {
    // arbitrary_precision keeps the exact literal; fall back to Display
    v.to_string()
}

fn parity_codec(path: &str) -> bool {
    let fx = load(path);
    let mut ok = true;

    let mut g = Gate::new("crc8");
    for c in fx["crc8"].as_array().unwrap() {
        let got = gf256::crc8(&hex_to_bytes(c["in"].as_str().unwrap()));
        g.check(got as i64 == c["out"].as_i64().unwrap(), || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("rs_encode");
    for c in fx["rs_encode"].as_array().unwrap() {
        let got = gf256::rs_encode(
            &hex_to_bytes(c["data"].as_str().unwrap()),
            c["nsym"].as_u64().unwrap() as usize,
        );
        g.check(bytes_to_hex(&got) == c["code"].as_str().unwrap(), || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("crc4");
    for c in fx["crc4"].as_array().unwrap() {
        let nibbles: Vec<u8> = c["in"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as u8).collect();
        g.check(gf16::crc4(&nibbles) as i64 == c["out"].as_i64().unwrap(), || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("rs16_encode");
    for c in fx["rs16_encode"].as_array().unwrap() {
        let data: Vec<u8> = c["data"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as u8).collect();
        let got = gf16::rs_encode(&data, c["nsym"].as_u64().unwrap() as usize);
        let want: Vec<u8> = c["code"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as u8).collect();
        g.check(got == want, || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("rs16_decode");
    for c in fx["rs16_decode"].as_array().unwrap() {
        let recv: Vec<u8> = c["recv"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as u8).collect();
        let era: Vec<usize> = c["erase"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as usize).collect();
        let max_err = c["max_errors"].as_u64().map(|v| v as usize);
        let got = gf16::rs_decode(&recv, c["nsym"].as_u64().unwrap() as usize,
                                  &era, max_err);
        let want = &c["out"];
        let matches = match (&got, want) {
            (Err(_), J::Null) => true,
            (Ok((d, _)), J::Array(a)) => {
                d.len() == a.len()
                    && d.iter().zip(a).all(|(x, w)| *x as i64 == w.as_i64().unwrap())
            }
            _ => false,
        };
        g.check(matches, || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("rs_decode");
    for c in fx["rs_decode"].as_array().unwrap() {
        let recv = hex_to_bytes(c["recv"].as_str().unwrap());
        let era: Vec<usize> = c["erase"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as usize).collect();
        let got = gf256::rs_decode(&recv, c["nsym"].as_u64().unwrap() as usize, &era, None);
        let want = &c["out"];
        let matches = match (&got, want) {
            (Err(_), J::Null) => true,
            (Ok((d, _)), J::String(s)) => bytes_to_hex(d) == *s,
            _ => false,
        };
        g.check(matches, || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("grid");
    for sp in spec::variants() {
        for case in fx["grids"][sp.name].as_array().unwrap() {
            let pb = hex_to_bytes(case["payload"].as_str().unwrap());
            let grid = codec::encode(&pb, sp).unwrap();
            g.check(
                grid_to_rows(&grid, sp.sector_count)
                    == case["grid"].as_array().unwrap().iter()
                        .map(|r| r.as_str().unwrap().to_string()).collect::<Vec<_>>(),
                || format!("{} encode {}", sp.name, case["payload"]),
            );
            for sub in case["cases"].as_array().unwrap() {
                let shift = sub["shift"].as_u64().unwrap() as usize;
                // np.roll(grid, +shift, axis=1)
                let n = sp.sector_count;
                let mut gmut = vec![0u8; grid.len()];
                for r in 0..sp.ring_count {
                    for i in 0..n {
                        gmut[r * n + (i + shift) % n] = grid[r * n + i];
                    }
                }
                for f in sub["flips"].as_array().unwrap() {
                    let (r, s) = (f[0].as_u64().unwrap() as usize, f[1].as_u64().unwrap() as usize);
                    gmut[r * n + s] ^= 1;
                }
                let eras = sub["erasures"].as_array().unwrap();
                let eg: Option<Vec<bool>> = if eras.is_empty() {
                    None
                } else {
                    let mut e = vec![false; grid.len()];
                    for p in eras {
                        e[p[0].as_u64().unwrap() as usize * n
                            + p[1].as_u64().unwrap() as usize] = true;
                    }
                    Some(e)
                };
                let (got, gsh) = codec::decode(&gmut, sp, eg.as_deref());
                let want = &sub["out"];
                let matches = match (&got, want) {
                    (None, J::Null) => true,
                    (Some(d), J::String(s)) => bytes_to_hex(d) == *s,
                    _ => false,
                } && gsh as i64 == sub["out_shift"].as_i64().unwrap();
                g.check(matches, || format!("{} case {}", sp.name, sub));
            }
        }
    }
    ok &= g.report();

    // ranked-confidence decode path (codec.decode conf_grid= in Python)
    if !fx["grids_conf"].is_null() {
        let mut g = Gate::new("grid_conf");
        for sp in spec::variants() {
            for case in fx["grids_conf"][sp.name].as_array().unwrap() {
                let pb = hex_to_bytes(case["payload"].as_str().unwrap());
                let grid = codec::encode(&pb, sp).unwrap();
                for sub in case["cases"].as_array().unwrap() {
                    let shift = sub["shift"].as_u64().unwrap() as usize;
                    let n = sp.sector_count;
                    let mut gmut = vec![0u8; grid.len()];
                    for r in 0..sp.ring_count {
                        for i in 0..n {
                            gmut[r * n + (i + shift) % n] = grid[r * n + i];
                        }
                    }
                    for f in sub["flips"].as_array().unwrap() {
                        let (r, s) =
                            (f[0].as_u64().unwrap() as usize, f[1].as_u64().unwrap() as usize);
                        gmut[r * n + s] ^= 1;
                    }
                    let conf: Vec<f32> = sub["conf"].as_array().unwrap().iter()
                        .flat_map(|row| row.as_array().unwrap().iter())
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    let (got, gsh) = codec::decode_conf(&gmut, sp, &conf, 0.25);
                    let got = got.map(|(pb, _)| pb);
                    let want = &sub["out"];
                    let matches = match (&got, want) {
                        (None, J::Null) => true,
                        (Some(d), J::String(s)) => bytes_to_hex(d) == *s,
                        _ => false,
                    } && gsh as i64 == sub["out_shift"].as_i64().unwrap();
                    g.check(matches, || format!("{} conf case {}", sp.name, sub));
                }
            }
        }
        ok &= g.report();
    }

    let mut g = Gate::new("mode_encode");
    for c in fx["mode_encode"].as_array().unwrap() {
        let sp = spec::by_name(c["variant"].as_str().unwrap()).unwrap();
        let args: Vec<String> = c["args"].as_array().unwrap().iter()
            .map(|a| a.as_str().unwrap().to_string()).collect();
        let got = call_encode(c["fn"].as_str().unwrap(), &args, sp);
        g.check(
            got.as_deref().map(bytes_to_hex).ok() == Some(c["out"].as_str().unwrap().into()),
            || c.to_string(),
        );
    }
    ok &= g.report();

    let mut g = Gate::new("mode_decode");
    for c in fx["mode_decode"].as_array().unwrap() {
        let sp = spec::by_name(c["variant"].as_str().unwrap()).unwrap();
        let pb = hex_to_bytes(c["payload"].as_str().unwrap());
        let got = payload::decode(&pb, sp);
        let want_mode = c["mode"].as_str().unwrap();
        let matches = match got {
            Ok((mode, val)) => mode == want_mode && value_matches(&val, &c["value"]),
            Err(_) => false,
        };
        g.check(matches, || c.to_string());
    }
    ok &= g.report();

    let mut g = Gate::new("mode_guards");
    for c in fx["mode_guards"].as_array().unwrap() {
        let sp = spec::by_name(c["variant"].as_str().unwrap()).unwrap();
        let args: Vec<String> = c["args"].as_array().unwrap().iter()
            .map(|a| a.as_str().unwrap().to_string()).collect();
        g.check(call_encode(c["fn"].as_str().unwrap(), &args, sp).is_err(), || c.to_string());
    }
    ok &= g.report();

    ok
}

fn call_encode(fnname: &str, args: &[String], sp: &spec::MarkerSpec) -> Result<Vec<u8>, String> {
    match fnname {
        "id" => payload::encode_id(args[0].parse().map_err(|_| "parse")?, sp),
        "raw" => payload::encode_raw(&hex_to_bytes(&args[0]), sp),
        "tagged" => payload::encode_tagged(
            args[0].parse().map_err(|_| "parse")?,
            args[1].parse().map_err(|_| "parse")?,
            sp,
        ),
        "geo" => payload::encode_geo(
            args[0].parse().map_err(|_| "parse")?,
            args[1].parse().map_err(|_| "parse")?,
            args[2].parse::<f64>().map_err(|_| "parse")? as i64,
            sp,
        ),
        other => panic!("unknown fn {}", other),
    }
}

fn value_matches(val: &payload::Value, want: &J) -> bool {
    use payload::Value::*;
    if let Some(n) = want.get("int") {
        return matches!(val, Int(v) if v.to_string() == num_str(n));
    }
    if let Some(h) = want.get("hex") {
        return matches!(val, Bytes(b) if bytes_to_hex(b) == h.as_str().unwrap());
    }
    if let Some(l) = want.get("list") {
        let l = l.as_array().unwrap();
        return match val {
            Geo { lat, lon, alt_m } => {
                l.len() == 3
                    && lat.to_bits() == num_str(&l[0]).parse::<f64>().unwrap().to_bits()
                    && lon.to_bits() == num_str(&l[1]).parse::<f64>().unwrap().to_bits()
                    && *alt_m as i64 == num_str(&l[2]).parse::<f64>().unwrap() as i64
            }
            Tagged { namespace, id } => {
                l.len() == 2
                    && *namespace as i64 == l[0].as_i64().unwrap()
                    && id.to_string() == num_str(&l[1])
            }
            _ => false,
        };
    }
    false
}

// ---------------------------------------------------------------------------
// parity-geometry: conic -> H -> (R, t) at 1e-9
// ---------------------------------------------------------------------------

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn parity_geometry(path: &str) -> bool {
    use simittag_core::mat::M3;
    use simittag_core::pose::{self, EllipseGeom};
    let fx = load(path);
    let tol = fx["tolerance"].as_f64().unwrap();
    let mut g = Gate::new("geometry");
    for case in fx["cases"].as_array().unwrap() {
        let e: Vec<f64> = case["ellipse"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect();
        let kf: Vec<f64> = case["K"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect();
        let k: M3 = [
            [kf[0], kf[1], kf[2]],
            [kf[3], kf[4], kf[5]],
            [kf[6], kf[7], kf[8]],
        ];
        let geom = EllipseGeom {
            cx: e[0], cy: e[1], major: e[2], minor: e[3], angle_deg: e[4],
        };
        let hs = pose::pose_homographies(&geom, &k);
        let sols = case["solutions"].as_array().unwrap();
        let mut all_ok = hs.len() == sols.len();
        for (h, sol) in hs.iter().zip(sols) {
            let hw: Vec<f64> = sol["H"].as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap()).collect();
            for r in 0..3 {
                for c in 0..3 {
                    all_ok &= close(h[r][c], hw[r * 3 + c], tol);
                }
            }
            let (rm, tv) = pose::decompose_h(h, &k);
            let rw: Vec<f64> = sol["R"].as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap()).collect();
            let tw: Vec<f64> = sol["t"].as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap()).collect();
            for r in 0..3 {
                for c in 0..3 {
                    all_ok &= close(rm[r][c], rw[r * 3 + c], tol);
                }
                all_ok &= close(tv[r], tw[r], tol);
            }
            all_ok &= close(pose::tilt_from_h(h, &k), sol["tilt"].as_f64().unwrap(), tol);
        }
        g.check(all_ok, || format!("ellipse {:?}", e));
    }
    g.report()
}

// ---------------------------------------------------------------------------
// parity-stages / parity-candidates: the imaging front-end gates
// ---------------------------------------------------------------------------

fn read_png_gray(path: &str) -> simittag_core::image::Gray {
    let dec = png::Decoder::new(std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {}", path, e)));
    let mut reader = dec.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert!(info.color_type == png::ColorType::Grayscale && info.bit_depth == png::BitDepth::Eight,
            "{}: expected 8-bit grayscale", path);
    simittag_core::image::Gray {
        w: info.width as usize,
        h: info.height as usize,
        px: buf[..(info.width * info.height) as usize].to_vec(),
    }
}

fn img_diff(a: &simittag_core::image::Gray, b: &simittag_core::image::Gray) -> usize {
    a.px.iter().zip(&b.px).filter(|(x, y)| x != y).count()
}

/// candidate comparison at the phase-3 gate tolerance
fn geom_close(a: &simittag_core::pose::EllipseGeom, g: &[f64], tol_px: f64) -> bool {
    let pos = (a.cx - g[0]).abs() <= tol_px
        && (a.cy - g[1]).abs() <= tol_px
        && (a.major - g[2]).abs() <= 2.0 * tol_px
        && (a.minor - g[3]).abs() <= 2.0 * tol_px;
    // angle via sin/cos of 2theta (axis flip = 180-period; near-circular angles drift)
    let ecc = 1.0 - (a.major.min(a.minor) / a.major.max(a.minor));
    let t1 = 2.0 * a.angle_deg.to_radians();
    let t2 = 2.0 * g[4].to_radians();
    let ang = if ecc < 0.02 {
        true // near-circular: angle is meaningless and solver-noise dominated
    } else {
        ((t1.sin() - t2.sin()).powi(2) + (t1.cos() - t2.cos()).powi(2)).sqrt() < 0.05
    };
    pos && ang
}

fn candidates_match(
    got: &[simittag_core::frontend::Candidate],
    want: &J,
    tol_px: f64,
) -> Result<(), String> {
    let want = want.as_array().unwrap();
    if got.len() != want.len() {
        return Err(format!("count {} != {}", got.len(), want.len()));
    }
    // order-independent matching (prune order can legally differ on r-ties)
    let mut used = vec![false; want.len()];
    for g in got {
        let mut hit = false;
        for (wi, w) in want.iter().enumerate() {
            if used[wi] {
                continue;
            }
            let o: Vec<f64> = w["outer"].as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap()).collect();
            if !geom_close(&g.outer, &o, tol_px) {
                continue;
            }
            let sub_ok = |got: &Option<simittag_core::pose::EllipseGeom>, want: &J| -> bool {
                match (got, want) {
                    (None, J::Null) => true,
                    (Some(gi), wv) if !wv.is_null() => {
                        let iv: Vec<f64> = wv.as_array().unwrap().iter()
                            .map(|v| v.as_f64().unwrap()).collect();
                        geom_close(gi, &iv, tol_px)
                    }
                    _ => false,
                }
            };
            // alt absent from older fixture files -> treat missing key as null
            let alt_want = w.get("alt").cloned().unwrap_or(J::Null);
            let small_want = w.get("bullseye_only")
                .and_then(|v| v.as_bool()).unwrap_or(false);
            if g.bullseye_only != small_want {
                continue;
            }
            if sub_ok(&g.inner, &w["inner"]) && sub_ok(&g.alt, &alt_want) {
                used[wi] = true;
                hit = true;
                break;
            }
        }
        if !hit {
            return Err(format!("unmatched candidate at ({:.1},{:.1}) r={:.1}",
                               g.outer.cx, g.outer.cy, g.outer_r));
        }
    }
    Ok(())
}

fn parity_stages(fixtures_dir: &str) -> bool {
    use simittag_core::{contours, fitellipse, frontend, image};
    let stages_dir = format!("{}/stages", fixtures_dir);
    let mut all_ok = true;
    let mut names: Vec<String> = std::fs::read_dir(&stages_dir).unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in &names {
        let sd = format!("{}/{}", stages_dir, name);
        let src = if std::path::Path::new(&format!("{}/undistorted.png", sd)).exists() {
            read_png_gray(&format!("{}/undistorted.png", sd))
        } else {
            read_png_gray(&format!("{}/frames/{}.png", fixtures_dir, name))
        };
        let mut g = Gate::new("stage");
        // 1. sharpen (bitwise)
        let sharp = image::sharpen(&src, 0.6, 1.0);
        let sharp_ref = read_png_gray(&format!("{}/sharpened.png", sd));
        g.check(img_diff(&sharp, &sharp_ref) == 0,
                || format!("{} sharpened: {} px differ", name, img_diff(&sharp, &sharp_ref)));
        // 2. adaptive threshold (bitwise), from OUR sharpened output
        let js: J = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/contours.json", sd)).unwrap()).unwrap();
        let blk = js["block"].as_u64().unwrap() as usize;
        assert_eq!(blk, image::adaptive_block(src.w, src.h), "{} blk", name);
        // Threshold/despeckle allow a FEW stray pixels: cv2's separable float
        // mean sums in a different order, so a value sitting exactly on a
        // rounding tie can flip one pixel (measured: 1 px in 400k). The strict
        // gates are the contour multiset and the candidate set below.
        let thr = image::adaptive_threshold_inv(&sharp, blk, 7);
        let thr_ref = read_png_gray(&format!("{}/threshold.png", sd));
        g.check(img_diff(&thr, &thr_ref) <= 8,
                || format!("{} threshold: {} px differ", name, img_diff(&thr, &thr_ref)));
        // 3. despeckle (near-bitwise, same rationale)
        let med = image::median3(&thr);
        let med_ref = read_png_gray(&format!("{}/despeckled.png", sd));
        g.check(img_diff(&med, &med_ref) <= 8,
                || format!("{} despeckled: {} px differ", name, img_diff(&med, &med_ref)));
        // 4. contours: point multisets + parent topology + area + fit + gates
        let cnts = contours::find_contours(&med);
        let want = js["contours"].as_array().unwrap();
        g.check(cnts.len() == want.len(),
                || format!("{} contour count {} != {}", name, cnts.len(), want.len()));
        if cnts.len() == want.len() {
            // match by sorted point multiset
            let key = |pts: &[(i32, i32)]| {
                let mut v = pts.to_vec();
                v.sort_unstable();
                v
            };
            use std::collections::HashMap;
            let mut by_key: HashMap<Vec<(i32, i32)>, Vec<usize>> = HashMap::new();
            for (wi, wc) in want.iter().enumerate() {
                let flat: Vec<i64> = wc["pts"].as_array().unwrap().iter()
                    .map(|v| v.as_i64().unwrap()).collect();
                let pts: Vec<(i32, i32)> = flat.chunks(2)
                    .map(|c| (c[0] as i32, c[1] as i32)).collect();
                by_key.entry(key(&pts)).or_default().push(wi);
            }
            let mut mapping: Vec<i64> = vec![-1; cnts.len()]; // rust idx -> py idx
            let mut matched = 0usize;
            for (ci, c) in cnts.iter().enumerate() {
                if let Some(list) = by_key.get_mut(&key(&c.pts)) {
                    if let Some(wi) = list.pop() {
                        mapping[ci] = wi as i64;
                        matched += 1;
                    }
                }
            }
            g.check(matched == cnts.len(),
                    || format!("{} matched {}/{} contours by point multiset",
                               name, matched, cnts.len()));
            if matched == cnts.len() {
                let mut topo_ok = true;
                let mut fit_ok = true;
                for (ci, c) in cnts.iter().enumerate() {
                    let wc = &want[mapping[ci] as usize];
                    // parent topology: parent's python index must equal python's parent
                    let wparent = wc["hier"][3].as_i64().unwrap();
                    let gparent = if c.parent >= 0 { mapping[c.parent as usize] } else { -1 };
                    if gparent != wparent {
                        topo_ok = false;
                    }
                    let area = contours::contour_area(&c.pts);
                    if (area - wc["area"].as_f64().unwrap()).abs() > 1e-6 {
                        fit_ok = false;
                    }
                    if let Some(fit) = wc.get("fit").filter(|f| !f.is_null()) {
                        let fv: Vec<f64> = fit.as_array().unwrap().iter()
                            .map(|v| v.as_f64().unwrap()).collect();
                        let gfit = fitellipse::fit_ellipse(&c.pts);
                        if !geom_close(&gfit, &fv, 0.1) {
                            fit_ok = false;
                            if g.fail < 3 {
                                eprintln!("  fit diff {}: got ({:.3},{:.3},{:.3},{:.3},{:.2}) want {:?}",
                                          name, gfit.cx, gfit.cy, gfit.major, gfit.minor,
                                          gfit.angle_deg, fv);
                            }
                        }
                    }
                }
                g.check(topo_ok, || format!("{} hierarchy topology", name));
                g.check(fit_ok, || format!("{} area/fitEllipse", name));
            }
        }
        // 5. full front-end candidates
        let cands = frontend::find_marker_ellipses(&sharp);
        match candidates_match(&cands, &js["candidates"], 0.1) {
            Ok(()) => g.check(true, || String::new()),
            Err(e) => g.check(false, || format!("{} candidates: {}", name, e)),
        }
        print!("{:<22}", name);
        all_ok &= g.report();
    }
    all_ok
}

fn parity_candidates(fixtures_dir: &str) -> bool {
    use simittag_core::{frontend, image};
    let fx = load(&format!("{}/frames.json", fixtures_dir));
    let mut g = Gate::new("candidates");
    let mut skipped = 0;
    for e in fx["entries"].as_array().unwrap() {
        if !e["dist"].is_null() {
            skipped += 1; // undistort lands in phase 4
            continue;
        }
        let file = e["file"].as_str().unwrap();
        let src = read_png_gray(&format!("{}/{}", fixtures_dir, file));
        let sharp = image::sharpen(&src, 0.6, 1.0);
        let cands = frontend::find_marker_ellipses(&sharp);
        match candidates_match(&cands, &e["candidates"], 0.1) {
            Ok(()) => g.check(true, || String::new()),
            Err(err) => g.check(false, || format!("{}: {}", file, err)),
        }
    }
    println!("  ({} dist frames deferred to phase 4)", skipped);
    g.report()
}

// ---------------------------------------------------------------------------
// parity-detect: the phase-4 gate. Identical decode decisions on every frame,
// pose within ±0.05 deg tilt / ±0.1% depth of the Python reference.
// ---------------------------------------------------------------------------

fn value_matches_det(val: &simittag_core::payload::Value, want: &J) -> bool {
    use simittag_core::payload::Value::*;
    if let Some(n) = want.get("int") {
        return matches!(val, Int(v) if v.to_string() == num_str(n));
    }
    if let Some(hx) = want.get("hex") {
        return matches!(val, Bytes(b) if bytes_to_hex(b) == hx.as_str().unwrap());
    }
    if let Some(l) = want.get("list") {
        let l = l.as_array().unwrap();
        return match val {
            Geo { lat, lon, alt_m } => {
                l.len() == 3
                    && (lat - num_str(&l[0]).parse::<f64>().unwrap()).abs() < 1e-12
                    && (lon - num_str(&l[1]).parse::<f64>().unwrap()).abs() < 1e-12
                    && *alt_m as i64 == num_str(&l[2]).parse::<f64>().unwrap() as i64
            }
            Tagged { namespace, id } => {
                l.len() == 2
                    && *namespace as i64 == l[0].as_i64().unwrap()
                    && id.to_string() == num_str(&l[1])
            }
            _ => false,
        };
    }
    false
}

fn parity_detect(fixtures_dir: &str) -> bool {
    use simittag_core::{detector, mat::M3, spec};
    let fx = load(&format!("{}/frames.json", fixtures_dir));
    let mut g = Gate::new("detect");
    let mut worst_tilt = 0f64;
    let mut worst_depth = 0f64;
    for e in fx["entries"].as_array().unwrap() {
        let file = e["file"].as_str().unwrap();
        let src = read_png_gray(&format!("{}/{}", fixtures_dir, file));
        let kf: Vec<f64> = e["K"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect();
        let k: M3 = [
            [kf[0], kf[1], kf[2]],
            [kf[3], kf[4], kf[5]],
            [kf[6], kf[7], kf[8]],
        ];
        let dist: Option<Vec<f64>> = e["dist"].as_array().map(|a| {
            a.iter().map(|v| v.as_f64().unwrap()).collect()
        });
        let specs: Vec<&'static spec::MarkerSpec> = match &e["versions"] {
            J::String(s) => vec![spec::by_name(s).unwrap()],
            J::Array(a) => a.iter()
                .map(|v| spec::by_name(v.as_str().unwrap()).unwrap())
                .collect(),
            _ => spec::default_variants().to_vec(),
        };
        let dets = detector::detect_markers(&src, &k, &specs, 0.25, true, dist.as_deref());
        let want = e["detections"].as_array().unwrap();
        if dets.len() != want.len() {
            g.check(false, || format!("{}: {} dets != {}", file, dets.len(), want.len()));
            continue;
        }
        // order-independent center matching
        let mut used = vec![false; want.len()];
        let mut frame_ok = true;
        let mut why = String::new();
        for d in &dets {
            let mut hit = false;
            for (wi, w) in want.iter().enumerate() {
                if used[wi] {
                    continue;
                }
                let wc: Vec<f64> = w["center"].as_array().unwrap().iter()
                    .map(|v| v.as_f64().unwrap()).collect();
                if ((d.center.0 - wc[0]).powi(2) + (d.center.1 - wc[1]).powi(2)).sqrt() > 2.0 {
                    continue;
                }
                // decode decision must be IDENTICAL
                let wdec = w["decoded"].as_bool().unwrap();
                let dec_ok = match (&d.decoded, wdec) {
                    (None, false) => true,
                    (Some((var, mode, val)), true) => {
                        *var == w["variant"].as_str().unwrap()
                            && *mode == w["mode"].as_str().unwrap()
                            && value_matches_det(val, &w["value"])
                    }
                    _ => false,
                };
                if !dec_ok {
                    why = format!("{}: decode mismatch at ({:.0},{:.0})", file, wc[0], wc[1]);
                    continue;
                }
                // pose gate
                let wtilt = w["tilt_deg"].as_f64().unwrap();
                let wt: Vec<f64> = w["t"].as_array().unwrap().iter()
                    .map(|v| v.as_f64().unwrap()).collect();
                let dt_tilt = (d.tilt_deg - wtilt).abs();
                let wz = wt[2];
                let dz = (d.t[2] - wz).abs() / wz.abs().max(1e-9);
                worst_tilt = worst_tilt.max(dt_tilt);
                worst_depth = worst_depth.max(dz);
                if dt_tilt > 0.05 || dz > 0.001 {
                    why = format!("{}: pose off (dtilt {:.4} deg, ddepth {:.4}%)",
                                  file, dt_tilt, dz * 100.0);
                    continue;
                }
                used[wi] = true;
                hit = true;
                break;
            }
            if !hit {
                frame_ok = false;
                if why.is_empty() {
                    why = format!("{}: unmatched detection at ({:.0},{:.0})",
                                  file, d.center.0, d.center.1);
                }
                break;
            }
        }
        g.check(frame_ok, || why.clone());

        // The same fixture, luminance-inverted, must produce identical decode
        // and pose decisions through the white-on-black frontend.
        let inverted_src = simittag_core::image::Gray {
            w: src.w,
            h: src.h,
            px: src.px.iter().map(|&v| 255 - v).collect(),
        };
        let inverted_dets = detector::detect_markers(
            &inverted_src, &k, &specs, 0.25, true, dist.as_deref());
        let normal_decoded: Vec<_> = dets.iter().filter(|d| d.decoded.is_some()).collect();
        let inverted_decoded: Vec<_> = inverted_dets
            .iter()
            .filter(|d| d.decoded.is_some())
            .collect();
        let mut inverted_ok = inverted_decoded.len() == normal_decoded.len();
        let mut inverted_why = if inverted_ok {
            String::new()
        } else {
            format!(
                "{} inverted: {} decoded != {}",
                file,
                inverted_decoded.len(),
                normal_decoded.len()
            )
        };
        let mut used_normal = vec![false; normal_decoded.len()];
        for d in inverted_decoded {
            let matched = normal_decoded.iter().enumerate().position(|(i, normal)| {
                !used_normal[i]
                    && ((d.center.0 - normal.center.0).powi(2)
                        + (d.center.1 - normal.center.1).powi(2))
                    .sqrt() < 0.1
                    && d.decoded == normal.decoded
                    && (d.tilt_deg - normal.tilt_deg).abs() < 0.05
                    && (d.t[2] - normal.t[2]).abs() / normal.t[2].abs().max(1e-9) < 0.001
                    && d.inverted
            });
            if let Some(i) = matched {
                used_normal[i] = true;
            } else {
                inverted_ok = false;
                inverted_why = format!(
                    "{} inverted: unmatched detection at ({:.0},{:.0})",
                    file, d.center.0, d.center.1);
                break;
            }
        }
        g.check(inverted_ok, || inverted_why.clone());

        // Headless detect() expectations (the frames.json "headless" section):
        // decode identity + the same pose gates for detector::detect(), which
        // previously had no parity coverage at all -- the gap that let the
        // R3.13 pose-mirror bug ship in the headless path only.
        if let Some(heads) = e["headless"].as_array() {
            let hdets = detector::detect(&src, &k, &specs, 0.25, dist.as_deref());
            if hdets.len() != heads.len() {
                g.check(false, || format!("{} headless: {} dets != {}",
                                          file, hdets.len(), heads.len()));
                continue;
            }
            let mut used = vec![false; heads.len()];
            let mut ok_all = true;
            let mut why = String::new();
            for d in &hdets {
                let mut hit = false;
                for (wi, w) in heads.iter().enumerate() {
                    if used[wi] {
                        continue;
                    }
                    let wc: Vec<f64> = w["center"].as_array().unwrap().iter()
                        .map(|v| v.as_f64().unwrap()).collect();
                    if ((d.center.0 - wc[0]).powi(2) + (d.center.1 - wc[1]).powi(2))
                        .sqrt() > 2.0 {
                        continue;
                    }
                    let dec_ok = match &d.decoded {
                        Some((var, mode, val)) => {
                            *var == w["variant"].as_str().unwrap()
                                && *mode == w["mode"].as_str().unwrap()
                                && value_matches_det(val, &w["value"])
                                && d.inverted == w["inverted"].as_bool().unwrap()
                        }
                        None => false,
                    };
                    if !dec_ok {
                        why = format!("{} headless: decode mismatch at ({:.0},{:.0})",
                                      file, wc[0], wc[1]);
                        continue;
                    }
                    let wtilt = w["tilt_deg"].as_f64().unwrap();
                    let wt: Vec<f64> = w["t"].as_array().unwrap().iter()
                        .map(|v| v.as_f64().unwrap()).collect();
                    let dt_tilt = (d.tilt_deg - wtilt).abs();
                    let dz = (d.t[2] - wt[2]).abs() / wt[2].abs().max(1e-9);
                    worst_tilt = worst_tilt.max(dt_tilt);
                    worst_depth = worst_depth.max(dz);
                    if dt_tilt > 0.05 || dz > 0.001 {
                        why = format!("{} headless: pose off (dtilt {:.4} deg, ddepth {:.4}%)",
                                      file, dt_tilt, dz * 100.0);
                        continue;
                    }
                    used[wi] = true;
                    hit = true;
                    break;
                }
                if !hit {
                    ok_all = false;
                    if why.is_empty() {
                        why = format!("{} headless: unmatched detection at ({:.0},{:.0})",
                                      file, d.center.0, d.center.1);
                    }
                    break;
                }
            }
            g.check(ok_all, || why.clone());
        }
    }
    println!("  worst tilt diff {:.4} deg, worst depth diff {:.4}%",
             worst_tilt, worst_depth * 100.0);
    g.report()
}

// ---------------------------------------------------------------------------
// cross-gen: randomized cases for Python to replay (10k gate)
// ---------------------------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn cross_gen(n: usize) {
    let mut rng = XorShift(0x5eed_cafe_f00d_1234);
    let vars = spec::variants();
    for i in 0..n {
        let sp = vars[i % 3];
        let pb: Vec<u8> = (0..sp.payload_bytes()).map(|_| rng.below(256) as u8).collect();
        let grid = codec::encode(&pb, sp).unwrap();
        let shift = rng.below(sp.sector_count);
        let nsec = sp.sector_count;
        let mut gmut = vec![0u8; grid.len()];
        for r in 0..sp.ring_count {
            for s in 0..nsec {
                gmut[r * nsec + (s + shift) % nsec] = grid[r * nsec + s];
            }
        }
        let nflip = rng.below(sp.rs_nsym + 2);
        let mut flips = Vec::new();
        for _ in 0..nflip {
            let r = sp.first_data_ring() + rng.below(sp.ring_count - sp.first_data_ring());
            let s = rng.below(nsec);
            gmut[r * nsec + s] ^= 1;
            flips.push((r, s));
        }
        let nera = rng.below(4);
        let mut eras: Vec<(usize, usize)> = Vec::new();
        for _ in 0..nera {
            let e = (rng.below(sp.ring_count), rng.below(nsec));
            if !eras.contains(&e) {
                eras.push(e);
            }
        }
        eras.sort_unstable();
        let eg: Option<Vec<bool>> = if eras.is_empty() {
            None
        } else {
            let mut e = vec![false; grid.len()];
            for &(r, s) in &eras {
                e[r * nsec + s] = true;
            }
            Some(e)
        };
        let (out, out_shift) = codec::decode(&gmut, sp, eg.as_deref());
        println!(
            "{{\"v\":\"{}\",\"payload\":\"{}\",\"shift\":{},\"flips\":{:?},\"erasures\":{:?},\"out\":{},\"out_shift\":{}}}",
            sp.name,
            bytes_to_hex(&pb),
            shift,
            flips.iter().map(|&(r, s)| [r, s]).collect::<Vec<_>>(),
            eras.iter().map(|&(r, s)| [r, s]).collect::<Vec<_>>(),
            out.map(|d| format!("\"{}\"", bytes_to_hex(&d))).unwrap_or("null".into()),
            out_shift
        );
    }
}

// ---------------------------------------------------------------------------
// serve: persistent frame-decode loop for the Python harness bridge.
// Protocol per frame: one JSON header line on stdin
//   {"w":..,"h":..,"fx":..,"fy":..,"cx":..,"cy":..,"versions":"M"|""|null,
//    "mode":"detect"|"markers","pose_only":bool,"dist":[..]|null}
// followed by w*h raw grayscale bytes; one JSON line out per frame.
// ---------------------------------------------------------------------------

fn det_to_json(d: &simittag_core::detector::Detection) -> String {
    use simittag_core::payload::Value;
    let flat = |m: &simittag_core::mat::M3| -> String {
        m.iter()
            .flat_map(|r| r.iter().map(|v| format!("{:.17e}", v)))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut s = format!(
        "{{\"center\":[{:.10},{:.10}],\"axes\":[{:.10},{:.10}],\"angle\":{:.10},\
         \"R\":[{}],\"t\":[{:.17e},{:.17e},{:.17e}],\"H\":[{}],\"tilt_deg\":{:.10},\"decoded\":{},\"inverted\":{}",
        d.center.0, d.center.1, d.axes.0, d.axes.1, d.angle,
        flat(&d.r), d.t[0], d.t[1], d.t[2], flat(&d.h), d.tilt_deg,
        d.decoded.is_some(), d.inverted
    );
    if let Some((variant, mode, value)) = &d.decoded {
        let alias = simittag_core::spec::by_name(variant).map(|s| s.alias).unwrap_or("");
        s.push_str(&format!(
            ",\"variant\":\"{}\",\"alias\":\"{}\",\"mode\":\"{}\"",
            variant, alias, mode
        ));
        match value {
            Value::Int(v) => s.push_str(&format!(",\"value\":{}", v)),
            Value::Bytes(b) => s.push_str(&format!(",\"value_hex\":\"{}\"", bytes_to_hex(b))),
            Value::Geo { lat, lon, alt_m } => s.push_str(&format!(
                ",\"geo\":[{:.17e},{:.17e},{}]", lat, lon, alt_m)),
            Value::Tagged { namespace, id } => s.push_str(&format!(
                ",\"tagged\":[{},{}]", namespace, id)),
        }
    }
    s.push('}');
    s
}

fn serve() {
    use simittag_core::{detector, image::Gray, spec};
    use std::io::{BufRead, Read, Write};
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return; // EOF
        }
        if line.trim().is_empty() {
            continue;
        }
        let hdr: J = serde_json::from_str(&line).unwrap();
        let (w, h) = (hdr["w"].as_u64().unwrap() as usize, hdr["h"].as_u64().unwrap() as usize);
        let mut px = vec![0u8; w * h];
        reader.read_exact(&mut px).unwrap();
        let img = Gray { w, h, px };
        let k = [
            [hdr["fx"].as_f64().unwrap(), 0.0, hdr["cx"].as_f64().unwrap()],
            [0.0, hdr["fy"].as_f64().unwrap(), hdr["cy"].as_f64().unwrap()],
            [0.0, 0.0, 1.0],
        ];
        let specs: Vec<&'static spec::MarkerSpec> = match &hdr["versions"] {
            J::String(s) if !s.is_empty() && s != "auto" => {
                s.split(',').filter_map(spec::by_name).collect()
            }
            J::Array(a) => a.iter()
                .filter_map(|v| v.as_str().and_then(spec::by_name)).collect(),
            _ => spec::default_variants().to_vec(),
        };
        let dist: Option<Vec<f64>> = hdr["dist"].as_array()
            .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect());
        let dets = if hdr["mode"].as_str() == Some("markers") {
            detector::detect_markers(&img, &k, &specs, 0.25,
                                     hdr["pose_only"].as_bool().unwrap_or(true),
                                     dist.as_deref())
        } else {
            detector::detect(&img, &k, &specs, 0.25, dist.as_deref())
        };
        let body: Vec<String> = dets.iter().map(det_to_json).collect();
        writeln!(out, "[{}]", body.join(",")).unwrap();
        out.flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: simittag <parity-spec|parity-codec> <fixture.json> | cross-gen <n>";
    match args.get(1).map(|s| s.as_str()) {
        Some("parity-spec") => {
            if !parity_spec(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("parity-codec") => {
            if !parity_codec(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("parity-geometry") => {
            if !parity_geometry(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("parity-stages") => {
            if !parity_stages(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("parity-candidates") => {
            if !parity_candidates(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("parity-detect") => {
            if !parity_detect(args.get(2).expect(usage)) {
                exit(1);
            }
        }
        Some("bench") => {
            use simittag_core::{detector, spec};
            let path = args.get(2).expect(usage);
            let src = read_png_gray(path);
            let f = (src.w as f64 / 2.0) / (30f64.to_radians()).tan();
            let k = [
                [f, 0.0, (src.w as f64 - 1.0) / 2.0],
                [0.0, f, (src.h as f64 - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ];
            let pinned: Option<&str> = args.get(3).map(|s| s.as_str());
            let specs: Vec<&'static spec::MarkerSpec> = match pinned {
                Some(names) => names.split(',').map(|n| spec::by_name(n).unwrap()).collect(),
                None => spec::default_variants().to_vec(),
            };
            let n = 20;
            let t0 = std::time::Instant::now();
            let mut ndet = 0;
            for _ in 0..n {
                ndet = detector::detect_markers(&src, &k, &specs, 0.25, true, None).len();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            println!("{}x{} {} dets  {:.2} ms/frame  ({:.0} Hz)",
                     src.w, src.h, ndet, ms, 1000.0 / ms);
        }
        Some("cross-gen") => {
            cross_gen(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000))
        }
        // detect <image.png> [variant] [fov_deg] -- decode a frame, one JSON
        // line per decoded tag. Without calibration the pose is approximate
        // (default K assumes the given horizontal FOV, 60 deg like the Python
        // reference's default_K); decoding itself is K-robust.
        Some("detect") => {
            use simittag_core::{detector, spec};
            let path = args.get(2).expect("usage: simittag detect <image.png> [variant] [fov_deg]");
            let src = read_png_gray(path);
            let fov: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(60.0);
            let f = (src.w as f64 / 2.0) / (fov.to_radians() / 2.0).tan();
            let k = [
                [f, 0.0, (src.w as f64 - 1.0) / 2.0],
                [0.0, f, (src.h as f64 - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ];
            let specs: Vec<&'static spec::MarkerSpec> = match args.get(3).map(|s| s.as_str()) {
                Some(name) if name != "auto" => {
                    vec![spec::by_name(name).unwrap_or_else(|| {
                        eprintln!("unknown variant {name} (canonical name, alias, comma-list, deprecated T/M/D letter, or auto; experimental s64k/s4k are explicit-only)");
                        exit(2);
                    })]
                }
                _ => spec::default_variants().to_vec(),
            };
            let dets = detector::detect(&src, &k, &specs, 0.25, None);
            for d in &dets {
                println!("{}", det_to_json(d));
            }
            if dets.is_empty() {
                eprintln!("no simittag decoded");
                exit(1);
            }
        }
        Some("serve") => serve(),
        _ => {
            eprintln!("{}", usage);
            exit(2);
        }
    }
}
