//! Reed-Solomon over GF(256) with erasure support.
//!
//! Direct port of simittag/gf256.py (field 0x11D, generator alpha=2, fcr=0;
//! Berlekamp-Massey + Chien + Forney extended for erasures). Polynomial
//! convention matches Python: index 0 = highest-degree coefficient. Every code
//! path is gated bit-exact against fixtures/codec.json, INCLUDING the decode
//! failures -- a decoder that fails on different inputs than the reference is
//! a different detector.

pub const PRIM: u16 = 0x11D;
pub const GEN: u8 = 2;

const fn build_tables() -> ([u8; 512], [u8; 256]) {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u16 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= PRIM;
        }
        i += 1;
    }
    let mut j = 255;
    while j < 512 {
        exp[j] = exp[j - 255];
        j += 1;
    }
    (exp, log)
}

static TABLES: ([u8; 512], [u8; 256]) = build_tables();

#[inline]
pub fn gmul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        TABLES.0[TABLES.1[a as usize] as usize + TABLES.1[b as usize] as usize]
    }
}

#[inline]
pub fn gdiv(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0);
    if a == 0 {
        0
    } else {
        let d = TABLES.1[a as usize] as i32 - TABLES.1[b as usize] as i32;
        TABLES.0[d.rem_euclid(255) as usize]
    }
}

#[inline]
pub fn gpow(a: u8, n: i64) -> u8 {
    // Python: _EXP[(_LOG[a] * n) % 255] with Python's non-negative modulo
    let e = (TABLES.1[a as usize] as i64 * n).rem_euclid(255);
    TABLES.0[e as usize]
}

#[inline]
pub fn ginv(a: u8) -> u8 {
    TABLES.0[(255 - TABLES.1[a as usize] as usize) % 255]
}

fn poly_scale(p: &[u8], x: u8) -> Vec<u8> {
    p.iter().map(|&c| gmul(c, x)).collect()
}

fn poly_add(p: &[u8], q: &[u8]) -> Vec<u8> {
    let n = p.len().max(q.len());
    let mut r = vec![0u8; n];
    for (i, &v) in p.iter().enumerate() {
        r[i + n - p.len()] = v;
    }
    for (i, &v) in q.iter().enumerate() {
        r[i + n - q.len()] ^= v;
    }
    r
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; p.len() + q.len() - 1];
    for (i, &pi) in p.iter().enumerate() {
        if pi != 0 {
            for (j, &qj) in q.iter().enumerate() {
                r[i + j] ^= gmul(pi, qj);
            }
        }
    }
    r
}

fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut y = 0u8;
    for &c in p {
        y = gmul(y, x) ^ c;
    }
    y
}

fn poly_div(dividend: &[u8], divisor: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut out = dividend.to_vec();
    let sep = divisor.len() - 1;
    for i in 0..dividend.len() - sep {
        let coef = out[i];
        if coef != 0 {
            for j in 1..divisor.len() {
                if divisor[j] != 0 {
                    out[i + j] ^= gmul(divisor[j], coef);
                }
            }
        }
    }
    let rem = out.split_off(out.len() - sep);
    (out, rem)
}

fn generator_poly(nsym: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..nsym {
        g = poly_mul(&g, &[1, gpow(GEN, i as i64)]);
    }
    g
}

/// Systematic RS: data + nsym parity bytes.
pub fn rs_encode(data: &[u8], nsym: usize) -> Vec<u8> {
    let gen = generator_poly(nsym);
    let mut padded = data.to_vec();
    padded.extend(std::iter::repeat(0u8).take(nsym));
    let (_, remainder) = poly_div(&padded, &gen);
    let mut out = data.to_vec();
    out.extend(remainder);
    out
}

fn calc_syndromes(msg: &[u8], nsym: usize) -> Vec<u8> {
    let mut s = vec![0u8];
    for i in 0..nsym {
        s.push(poly_eval(msg, gpow(GEN, i as i64)));
    }
    s
}

fn errata_locator(e_pos: &[usize]) -> Vec<u8> {
    let mut loc = vec![1u8];
    for &i in e_pos {
        loc = poly_mul(&loc, &poly_add(&[1], &[gpow(GEN, i as i64), 0]));
    }
    loc
}

fn error_evaluator(synd: &[u8], err_loc: &[u8], nsym: usize) -> Vec<u8> {
    let mut divisor = vec![0u8; nsym + 2];
    divisor[0] = 1;
    let (_, rem) = poly_div(&poly_mul(synd, err_loc), &divisor);
    rem
}

fn find_error_locator(synd: &[u8], nsym: usize, erase_count: usize) -> Result<Vec<u8>, ()> {
    let mut err_loc = vec![1u8];
    let mut old_loc = vec![1u8];
    let synd_shift = if synd.len() > nsym { synd.len() - nsym } else { 0 };
    for i in 0..nsym.saturating_sub(erase_count) {
        let k = i + synd_shift;
        let mut delta = synd[k];
        for j in 1..err_loc.len() {
            delta ^= gmul(err_loc[err_loc.len() - 1 - j], synd[k - j]);
        }
        old_loc.push(0);
        if delta != 0 {
            if old_loc.len() > err_loc.len() {
                let new_loc = poly_scale(&old_loc, delta);
                old_loc = poly_scale(&err_loc, ginv(delta));
                err_loc = new_loc;
            }
            err_loc = poly_add(&err_loc, &poly_scale(&old_loc, delta));
        }
    }
    while !err_loc.is_empty() && err_loc[0] == 0 {
        err_loc.remove(0);
    }
    let errs = err_loc.len() as i64 - 1;
    // Python computes this SIGNED: (errs - erase_count)*2 + erase_count > nsym
    if (errs - erase_count as i64) * 2 + erase_count as i64 > nsym as i64 {
        return Err(());
    }
    Ok(err_loc)
}

fn find_errors(err_loc: &[u8], nmess: usize) -> Result<Vec<usize>, ()> {
    let errs = err_loc.len() as i64 - 1; // may be -1 if the locator collapsed
    let mut pos = Vec::new();
    for i in 0..nmess {
        if poly_eval(err_loc, gpow(GEN, i as i64)) == 0 {
            pos.push(nmess - 1 - i);
        }
    }
    if pos.len() as i64 != errs {
        return Err(());
    }
    Ok(pos)
}

fn forney_syndromes(synd: &[u8], pos: &[usize], nmess: usize) -> Vec<u8> {
    let mut fsynd = synd[1..].to_vec();
    for &p in pos {
        let x = gpow(GEN, (nmess - 1 - p) as i64);
        for j in 0..fsynd.len() - 1 {
            fsynd[j] = gmul(fsynd[j], x) ^ fsynd[j + 1];
        }
    }
    fsynd
}

fn correct_errata(msg: &[u8], synd: &[u8], err_pos: &[usize]) -> Vec<u8> {
    let coef_pos: Vec<usize> = err_pos.iter().map(|&p| msg.len() - 1 - p).collect();
    let err_loc = errata_locator(&coef_pos);
    let synd_rev: Vec<u8> = synd.iter().rev().cloned().collect();
    let mut err_eval = error_evaluator(&synd_rev, &err_loc, err_loc.len() - 1);
    err_eval.reverse();
    let xs: Vec<u8> = coef_pos
        .iter()
        .map(|&p| gpow(GEN, -(255i64 - p as i64)))
        .collect();
    let mut e = vec![0u8; msg.len()];
    for (i, &xi) in xs.iter().enumerate() {
        let xi_inv = ginv(xi);
        let mut prime = 1u8;
        for (j, &xj) in xs.iter().enumerate() {
            if j != i {
                prime = gmul(prime, 1 ^ gmul(xi_inv, xj));
            }
        }
        let eval_rev: Vec<u8> = err_eval.iter().rev().cloned().collect();
        let mut y = poly_eval(&eval_rev, xi_inv);
        y = gmul(xi, y);
        e[err_pos[i]] = gdiv(y, prime);
    }
    poly_add(msg, &e)
}

/// Correct a received codeword; erase_pos = byte indices known-bad.
/// Returns (data, full_codeword) or Err on uncorrectable input.
pub fn rs_decode(
    msg_in: &[u8],
    nsym: usize,
    erase_pos: &[usize],
) -> Result<(Vec<u8>, Vec<u8>), ()> {
    let mut msg = msg_in.to_vec();
    for &e in erase_pos {
        msg[e] = 0;
    }
    if erase_pos.len() > nsym {
        return Err(());
    }
    let synd = calc_syndromes(&msg, nsym);
    if synd.iter().max().copied().unwrap_or(0) == 0 {
        let data = msg[..msg.len() - nsym].to_vec();
        return Ok((data, msg));
    }
    let fsynd = forney_syndromes(&synd, erase_pos, msg.len());
    let err_loc = find_error_locator(&fsynd, nsym, erase_pos.len())?;
    let err_loc_rev: Vec<u8> = err_loc.iter().rev().cloned().collect();
    let err_pos = find_errors(&err_loc_rev, msg.len())?;
    let mut all_pos = erase_pos.to_vec();
    all_pos.extend(&err_pos);
    let msg = correct_errata(&msg, &synd, &all_pos);
    if calc_syndromes(&msg, nsym).iter().max().copied().unwrap_or(0) != 0 {
        return Err(());
    }
    let data = msg[..msg.len() - nsym].to_vec();
    Ok((data, msg))
}

pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_errors_and_erasures() {
        // mirrors the Python __main__ self-test structure (deterministic cases)
        let data = b"\x12\x34\x56\x78\x9a";
        let code = rs_encode(data, 4);
        assert_eq!(&code[..5], data);
        // 2 blind errors (t = nsym/2)
        let mut c = code.clone();
        c[1] ^= 0x5c;
        c[7] ^= 0x11;
        let (d, _) = rs_decode(&c, 4, &[]).unwrap();
        assert_eq!(d, data);
        // 3 erasures
        let mut c = code.clone();
        c[0] ^= 0xff;
        c[4] ^= 0x33;
        c[8] ^= 0x77;
        let (d, _) = rs_decode(&c, 4, &[0, 4, 8]).unwrap();
        assert_eq!(d, data);
        // 3 blind errors must FAIL (or mis-locate -> Err)
        let mut c = code.clone();
        c[0] ^= 1;
        c[3] ^= 2;
        c[6] ^= 3;
        assert!(rs_decode(&c, 4, &[]).is_err() || rs_decode(&c, 4, &[]).unwrap().0 != data);
    }
}
