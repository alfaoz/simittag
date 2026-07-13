//! Suzuki-Abe border following (cv2.findContours, RETR_TREE + CHAIN_APPROX_NONE,
//! 8-connectivity), with the full hierarchy -- the nesting tree is load-bearing
//! for the detector (ring-stack suppression, bullseye disambiguation, multitag).
//!
//! Semantics: pixels outside the image are background (we run on a zero-padded
//! copy). Every border pixel is emitted once per traversal visit, so spur pixels
//! can appear twice -- same as OpenCV's CHAIN_APPROX_NONE. Parity is gated on
//! the fixture stage dumps: identical point multisets, counts, areas, and parent
//! topology on real threshold outputs.

use crate::image::Gray;

pub struct Contour {
    pub pts: Vec<(i32, i32)>, // (x, y)
    pub is_hole: bool,
    pub parent: i32, // index into the contour list, -1 = top level
    pub children: Vec<usize>,
}

/// 8-neighborhood, counterclockwise, starting East (x right, y down).
const NBR: [(i64, i64); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn dir_of(from: (i64, i64), to: (i64, i64)) -> usize {
    let d = (to.0 - from.0, to.1 - from.1);
    NBR.iter().position(|&n| n == d).unwrap()
}

pub fn find_contours(binary: &Gray) -> Vec<Contour> {
    let w = binary.w as i64 + 2;
    let h = binary.h as i64 + 2;
    // label grid, zero-padded; 1 = unvisited foreground
    let mut f = vec![0i32; (w * h) as usize];
    for y in 0..binary.h {
        for x in 0..binary.w {
            if binary.px[y * binary.w + x] != 0 {
                f[(y as i64 + 1) as usize * w as usize + (x as i64 + 1) as usize] = 1;
            }
        }
    }
    let idx = |x: i64, y: i64| (y * w + x) as usize;

    let mut contours: Vec<Contour> = Vec::new();
    // border bookkeeping: for each NBD (2..), its contour index and hole flag.
    // NBD 1 is the virtual frame border (outer, no contour).
    let mut nbd_info: Vec<(i32, bool)> = vec![(-1, false), (-1, false)]; // [0], [1]
    let mut nbd: i32 = 1;

    for y in 1..h - 1 {
        let mut lnbd: i32 = 1;
        for x in 1..w - 1 {
            let fxy = f[idx(x, y)];
            if fxy == 0 {
                continue;
            }
            let outer = fxy == 1 && f[idx(x - 1, y)] == 0;
            let hole = fxy >= 1 && f[idx(x + 1, y)] == 0;
            if outer || hole {
                nbd += 1;
                let is_hole = !outer; // outer test takes precedence (Suzuki step 1)
                let start_nbr = if is_hole { (x + 1, y) } else { (x - 1, y) };
                if is_hole && fxy > 1 {
                    lnbd = fxy;
                }
                // parent from Suzuki's table via LNBD
                let (lidx, lhole) = nbd_info[lnbd.unsigned_abs() as usize];
                let parent = if is_hole != lhole {
                    lidx
                } else if lidx >= 0 {
                    contours[lidx as usize].parent
                } else {
                    -1
                };
                let ci = contours.len();
                let mut pts: Vec<(i32, i32)> = Vec::new();

                // --- border following (Suzuki appendix, 8-connectivity) ---
                // 3.1: from start_nbr, search CLOCKWISE around (x,y)
                let d0 = dir_of((x, y), (start_nbr.0 - x + x, start_nbr.1)); // dir to start_nbr
                let mut found = None;
                for k in 0..8 {
                    let d = (d0 + 8 - k) % 8; // clockwise
                    let (dx, dy) = NBR[d];
                    if f[idx(x + dx, y + dy)] != 0 {
                        found = Some((x + dx, y + dy));
                        break;
                    }
                }
                match found {
                    None => {
                        // isolated pixel
                        f[idx(x, y)] = -nbd;
                        pts.push(((x - 1) as i32, (y - 1) as i32));
                    }
                    Some(p1) => {
                        let mut p2 = p1; // (i2,j2)
                        let mut p3 = (x, y); // (i3,j3)
                        loop {
                            // 3.3: search COUNTERclockwise around p3, starting from
                            // the element after p2 (counterclockwise), tracking
                            // whether (x+1, y) of p3 was examined and was 0.
                            let dstart = (dir_of(p3, p2) + 1) % 8;
                            let mut p4 = None;
                            let mut right_zero = false;
                            for k in 0..8 {
                                let d = (dstart + k) % 8;
                                let (dx, dy) = NBR[d];
                                let q = (p3.0 + dx, p3.1 + dy);
                                if f[idx(q.0, q.1)] != 0 {
                                    p4 = Some(q);
                                    break;
                                }
                                if d == 0 {
                                    right_zero = true; // examined East and it was 0
                                }
                            }
                            let p4 = p4.unwrap(); // p2 is nonzero, loop always finds one
                            pts.push(((p3.0 - 1) as i32, (p3.1 - 1) as i32));
                            // 3.4 marking
                            let fi = idx(p3.0, p3.1);
                            if right_zero {
                                f[fi] = -nbd;
                            } else if f[fi] == 1 {
                                f[fi] = nbd;
                            }
                            // 3.5 termination: back at start heading to first pixel
                            if p4 == (x, y) && p3 == p1 {
                                break;
                            }
                            p2 = p3;
                            p3 = p4;
                        }
                    }
                }
                nbd_info.push((ci as i32, is_hole));
                contours.push(Contour {
                    pts,
                    is_hole,
                    parent,
                    children: Vec::new(),
                });
            }
            // step 4
            let fxy = f[idx(x, y)];
            if fxy != 1 {
                lnbd = fxy.abs();
            }
        }
    }
    for i in 0..contours.len() {
        let p = contours[i].parent;
        if p >= 0 {
            contours[p as usize].children.push(i);
        }
    }
    contours
}

/// cv2.contourArea: Green's formula, |sum|/2, f64 over the point sequence.
pub fn contour_area(pts: &[(i32, i32)]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut a = 0f64;
    let mut prev = pts[pts.len() - 1];
    for &p in pts {
        a += prev.0 as f64 * p.1 as f64 - p.0 as f64 * prev.1 as f64;
        prev = p;
    }
    (a * 0.5).abs()
}
