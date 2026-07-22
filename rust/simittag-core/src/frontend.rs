//! detect._find_marker_ellipses port: threshold -> despeckle -> contour tree ->
//! fitEllipse -> roundness gate -> ancestor suppression -> dedup/cap.
//! Gate: identical candidate sets (±0.1px) on every fixture frame.

use crate::contours::{contour_area, find_contours};
use crate::fitellipse::fit_ellipse;
use crate::image::{adaptive_block, adaptive_threshold_inv, median3, Gray};
use crate::pose::EllipseGeom;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub outer: EllipseGeom,
    pub outer_r: f64, // max(MA, ma) / 2
    pub inner: Option<EllipseGeom>,
    /// Largest suppressed round child clearly smaller than the outer contour
    /// (<= 0.9 r): the decode fallback for a tag inside a circular sticker,
    /// where the sticker edge -- not the tag ring -- is the outermost contour.
    pub alt: Option<EllipseGeom>,
}

struct Fitted {
    geom: EllipseGeom,
    r: f64,
}

pub fn find_marker_ellipses(sharp: &Gray) -> Vec<Candidate> {
    let blk = adaptive_block(sharp.w, sharp.h);
    let thr = adaptive_threshold_inv(sharp, blk, 7);
    let med = median3(&thr);
    let cnts = find_contours(&med);

    // Fit + roundness-gate each contour independently. The per-contour
    // arithmetic is untouched and the indexed collect preserves order, so the
    // parallel map is bit-identical to the sequential loop (same parity story
    // as the detector's candidate map).
    let fit_one = |c: &crate::contours::Contour| -> Option<Fitted> {
        if c.pts.len() < 6 || contour_area(&c.pts) < 25.0 {
            return None;
        }
        let g = fit_ellipse(&c.pts);
        let a = g.major / 2.0;
        let b = g.minor / 2.0;
        if a < 1.0 || b < 1.0 {
            return None;
        }
        // tilt-invariant roundness: mean |r_norm - 1| in the ellipse's frame
        let th = g.angle_deg.to_radians();
        let (s, co) = th.sin_cos();
        let mut acc = 0f64;
        for &(px, py) in &c.pts {
            let dx = px as f64 - g.cx;
            let dy = py as f64 - g.cy;
            let u = (dx * co + dy * s) / a;
            let v = (-dx * s + dy * co) / b;
            acc += ((u * u + v * v).sqrt() - 1.0).abs();
        }
        if acc / c.pts.len() as f64 > 0.03 {
            return None;
        }
        let r = g.major.max(g.minor) / 2.0;
        Some(Fitted { geom: g, r })
    };
    #[cfg(feature = "parallel")]
    let ell: Vec<Option<Fitted>> = {
        use rayon::prelude::*;
        cnts.par_iter().map(fit_one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let ell: Vec<Option<Fitted>> = cnts.iter().map(fit_one).collect();

    // suppress candidates nested under a larger round ancestor (inner ring of an
    // already-kept tag); topological, perspective-invariant
    let has_round_ancestor = |i: usize| -> bool {
        let mut p = cnts[i].parent;
        while p >= 0 {
            if let Some(f) = &ell[p as usize] {
                if f.r >= 8.0 {
                    return true;
                }
            }
            p = cnts[p as usize].parent;
        }
        false
    };
    // all descendants of i in the contour tree
    fn descend(cnts: &[crate::contours::Contour], i: usize, out: &mut Vec<usize>) {
        for &ch in &cnts[i].children {
            out.push(ch);
            descend(cnts, ch, out);
        }
    }

    let mut cands: Vec<Candidate> = Vec::new();
    for i in 0..cnts.len() {
        let f = match &ell[i] {
            Some(f) if f.r >= 8.0 => f,
            _ => continue,
        };
        if has_round_ancestor(i) {
            continue;
        }
        let mut kids = Vec::new();
        descend(&cnts, i, &mut kids);
        // first-minimum / first-maximum on ties, like Python's min()/max()
        let mut inner: Option<&Fitted> = None;
        let mut alt: Option<&Fitted> = None;
        for &j in &kids {
            if let Some(f2) = &ell[j] {
                if inner.map_or(true, |cur| f2.r < cur.r) {
                    inner = Some(f2);
                }
                if f2.r <= 0.9 * f.r && alt.map_or(true, |cur| f2.r > cur.r) {
                    alt = Some(f2);
                }
            }
        }
        cands.push(Candidate {
            outer: f.geom,
            outer_r: f.r,
            inner: inner.map(|f2| f2.geom),
            alt: alt.map(|f2| f2.geom),
        });
    }
    // largest outer edge first (stable, like Python's list.sort)
    cands.sort_by(|a, b| b.outer_r.partial_cmp(&a.outer_r).unwrap());
    let mut pruned: Vec<Candidate> = Vec::new();
    for e in cands {
        let dup = pruned.iter().any(|p| {
            (e.outer_r - p.outer_r).abs() / p.outer_r < 0.18
                && ((e.outer.cx - p.outer.cx).powi(2) + (e.outer.cy - p.outer.cy).powi(2))
                    .sqrt()
                    < 0.25 * p.outer_r
        });
        if !dup {
            pruned.push(e);
        }
    }
    // max-simultaneous-tags cap, mirrored from detect.py (512 covers dense
    // calibration boards; an 18x13 grid is 234 tags)
    pruned.truncate(512);
    pruned
}
