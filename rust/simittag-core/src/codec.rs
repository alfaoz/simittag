//! Payload <-> cell grid. Port of simittag/codec.py.
//!
//! Grid = flat `Vec<u8>` (0/1), row-major: `grid[ring * sector_count + sector]`.
//! Data-cell linear order is sector-major: k = sector*data_rings + (ring - first),
//! bits packed MSB-first -- exactly the Python ordering, or nothing decodes.

use crate::gf16;
use crate::gf256;
use crate::spec::MarkerSpec;

/// Pack bits MSB-first into symbols of sb bits (sb=8 -> byte values).
fn bits_to_syms(bits: &[u8], sb: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() / sb);
    for chunk in bits.chunks(sb) {
        let mut b = 0u8;
        for &bit in chunk {
            b = (b << 1) | (bit & 1);
        }
        out.push(b);
    }
    out
}

fn syms_to_bits(syms: &[u8], sb: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(syms.len() * sb);
    for &v in syms {
        for j in (0..sb).rev() {
            bits.push((v >> j) & 1);
        }
    }
    bits
}

/// Canonical payload bytes -> data symbols (big-endian; high pad bits must
/// be zero when payload_bits is not a byte multiple). Mirrors codec.py.
fn payload_to_syms(payload: &[u8], spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    if payload.len() != spec.payload_bytes() {
        return Err(format!(
            "payload must be {} bytes, got {}",
            spec.payload_bytes(),
            payload.len()
        ));
    }
    let mut value = 0u128;
    for &b in payload {
        value = (value << 8) | b as u128;
    }
    if spec.payload_bits() < 128 && value >> spec.payload_bits() != 0 {
        return Err(format!("payload exceeds {} bits", spec.payload_bits()));
    }
    let sb = spec.symbol_bits;
    let nsyms = spec.payload_bits() / sb;
    Ok((0..nsyms)
        .map(|i| ((value >> (sb * (nsyms - 1 - i))) as u8) & ((1u16 << sb) - 1) as u8)
        .collect())
}

/// Data symbols -> canonical payload bytes (big-endian, zero pad bits).
fn syms_to_payload(syms: &[u8], spec: &MarkerSpec) -> Vec<u8> {
    let sb = spec.symbol_bits;
    let mut value = 0u128;
    for &v in syms {
        value = (value << sb) | v as u128;
    }
    let n = spec.payload_bytes();
    (0..n).rev().map(|i| (value >> (8 * i)) as u8).collect()
}

fn crc_of(data_syms: &[u8], spec: &MarkerSpec) -> u8 {
    if spec.symbol_bits == 4 {
        gf16::crc4(data_syms)
    } else {
        gf256::crc8(data_syms)
    }
}

fn rs_encode_spec(data: &[u8], spec: &MarkerSpec) -> Vec<u8> {
    if spec.symbol_bits == 4 {
        gf16::rs_encode(data, spec.rs_nsym)
    } else {
        gf256::rs_encode(data, spec.rs_nsym)
    }
}

fn rs_decode_spec(
    code: &[u8],
    spec: &MarkerSpec,
    erase_pos: &[usize],
) -> Result<(Vec<u8>, Vec<u8>), ()> {
    if spec.symbol_bits == 4 {
        gf16::rs_decode(code, spec.rs_nsym, erase_pos, spec.max_errors)
    } else {
        gf256::rs_decode(code, spec.rs_nsym, erase_pos, spec.max_errors)
    }
}

/// (k, ring, sector) for the data cells in codeword-bit order.
fn cell_order(spec: &MarkerSpec) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    let dr = spec.data_ring_count();
    let r0 = spec.first_data_ring();
    (0..spec.sector_count)
        .flat_map(move |s| (0..dr).map(move |rd| (s * dr + rd, r0 + rd, s)))
}

/// payload bytes -> +CRC -> RS encode -> bits -> grid (sync ring set).
pub fn encode(payload: &[u8], spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    let syms = payload_to_syms(payload, spec)?;
    let mut data = syms;
    data.push(crc_of(&data, spec));
    let code = rs_encode_spec(&data, spec);
    let bits = syms_to_bits(&code, spec.symbol_bits);

    let mut grid = vec![0u8; spec.ring_count * spec.sector_count];
    if spec.has_sync {
        grid[..spec.sector_count].copy_from_slice(spec.sync);
    }
    for (k, ring, sector) in cell_order(spec) {
        grid[ring * spec.sector_count + sector] = bits[k];
    }
    Ok(grid)
}

/// Best sector rotation aligning the observed sync ring to spec.sync via circular
/// cross-correlation. Returns (shift, scores); np.argmax semantics = FIRST max.
pub fn find_rotation(ring0: &[u8], spec: &MarkerSpec) -> (usize, Vec<i32>) {
    let n = spec.sector_count;
    let obs: Vec<i32> = ring0.iter().map(|&b| if b > 0 { 1 } else { -1 }).collect();
    let refv: Vec<i32> = spec.sync.iter().map(|&b| if b > 0 { 1 } else { -1 }).collect();
    let mut scores = Vec::with_capacity(n);
    for s in 0..n {
        let mut acc = 0i32;
        for i in 0..n {
            acc += refv[i] * obs[(i + s) % n]; // np.roll(obs, -s)[i]
        }
        scores.push(acc);
    }
    let mut best = 0usize;
    for (i, &v) in scores.iter().enumerate() {
        if v > scores[best] {
            best = i;
        }
    }
    (best, scores)
}

fn decode_aligned(
    grid: &[u8],
    spec: &MarkerSpec,
    erasure: Option<&[bool]>,
) -> Option<Vec<u8>> {
    let sb = spec.symbol_bits;
    let nbits = spec.data_ring_count() * spec.sector_count;
    let mut bits = vec![0u8; nbits];
    let mut unreliable = vec![false; nbits];
    for (k, ring, sector) in cell_order(spec) {
        bits[k] = grid[ring * spec.sector_count + sector];
        if let Some(eg) = erasure {
            unreliable[k] = eg[ring * spec.sector_count + sector];
        }
    }
    let code = bits_to_syms(&bits, sb);
    let mut erase_pos: Vec<usize> = Vec::new();
    if erasure.is_some() {
        for k in 0..nbits {
            if unreliable[k] && !erase_pos.contains(&(k / sb)) {
                erase_pos.push(k / sb);
            }
        }
        erase_pos.sort_unstable();
        // Cap at NSYM-1, never NSYM: with NSYM erasures RS always "succeeds"
        // (fills to the all-zeros codeword, CRC8(0)=0 -> phantom ID 0). One
        // syndrome of margin forces the surviving cells to be consistent.
        let cap = spec.rs_nsym.saturating_sub(1);
        erase_pos.truncate(cap);
    }
    let (data, _) = rs_decode_spec(&code, spec, &erase_pos).ok()?;
    let nds = spec.rs_k - spec.crc_bytes;
    let (payload_syms, crc) = (&data[..nds], data[nds]);
    if crc_of(payload_syms, spec) != crc {
        return None;
    }
    Some(syms_to_payload(payload_syms, spec))
}

/// Reed-Solomon work done for a successful decode. Report-only diagnostics:
/// nothing downstream branches on these.
#[derive(Clone, Copy, Default)]
pub struct RsStats {
    pub erasures: usize,  // codeword bytes erased up front (weak-confidence cells)
    pub corrected: usize, // codeword bytes whose value RS actually changed
}

/// Per-codeword-symbol reliability = weakest cell confidence in the symbol.
fn sym_reliability(conf: &[f32], spec: &MarkerSpec) -> Vec<f64> {
    let sb = spec.symbol_bits;
    let nbits = spec.data_ring_count() * spec.sector_count;
    let mut rel = vec![f64::INFINITY; nbits / sb];
    for (k, ring, sector) in cell_order(spec) {
        let c = conf[ring * spec.sector_count + sector] as f64;
        if c < rel[k / sb] {
            rel[k / sb] = c;
        }
    }
    rel
}

/// Decode a rotation-aligned grid with confidence-RANKED erasures: the
/// NSYM-1 cap keeps the WEAKEST under-threshold bytes, not the
/// lowest-indexed ones. One RS attempt, like the boolean path.
fn decode_aligned_ranked(
    grid: &[u8],
    spec: &MarkerSpec,
    conf: &[f32],
    conf_erasure: f32,
) -> Option<(Vec<u8>, RsStats)> {
    let nbits = spec.data_ring_count() * spec.sector_count;
    let mut bits = vec![0u8; nbits];
    for (k, ring, sector) in cell_order(spec) {
        bits[k] = grid[ring * spec.sector_count + sector];
    }
    let code = bits_to_syms(&bits, spec.symbol_bits);

    let rel = sym_reliability(conf, spec);
    let mut order: Vec<usize> = (0..rel.len()).collect();
    order.sort_by(|&a, &b| rel[a].partial_cmp(&rel[b]).unwrap()); // stable
    let cap = spec.rs_nsym.saturating_sub(1);
    let mut erase_pos: Vec<usize> = order
        .iter()
        .take(cap)
        .copied()
        .filter(|&i| rel[i] < conf_erasure as f64)
        .collect();
    erase_pos.sort_unstable();

    let (data, msg) = rs_decode_spec(&code, spec, &erase_pos).ok()?;
    let nds = spec.rs_k - spec.crc_bytes;
    let (payload_syms, crc) = (&data[..nds], data[nds]);
    if crc_of(payload_syms, spec) != crc {
        return None;
    }
    // corrected counts symbols whose value changed vs the codeword as
    // sampled, so an erased symbol that was right does not inflate the number
    let corrected = code.iter().zip(msg.iter()).filter(|(a, b)| a != b).count();
    Some((
        syms_to_payload(payload_syms, spec),
        RsStats {
            erasures: erase_pos.len(),
            corrected,
        },
    ))
}

fn roll_conf(conf: &[f32], spec: &MarkerSpec, shift: usize) -> Vec<f32> {
    let n = spec.sector_count;
    let mut out = vec![0f32; conf.len()];
    for r in 0..spec.ring_count {
        for i in 0..n {
            out[r * n + i] = conf[r * n + (i + shift) % n];
        }
    }
    out
}

/// Ranked-erasure variant of `decode` (mirrors Python's conf_grid path).
/// The success value carries the RS diagnostics alongside the payload.
pub fn decode_conf(
    grid: &[u8],
    spec: &MarkerSpec,
    conf: &[f32],
    conf_erasure: f32,
) -> (Option<(Vec<u8>, RsStats)>, usize) {
    let shifts: Vec<usize> = if spec.has_sync {
        let (shift, _) = find_rotation(&grid[..spec.sector_count], spec);
        vec![shift]
    } else {
        (0..spec.sector_count).collect()
    };
    for &shift in &shifts {
        let aligned = roll_grid(grid, spec, shift);
        let conf_s = roll_conf(conf, spec, shift);
        if let Some(hit) = decode_aligned_ranked(&aligned, spec, &conf_s, conf_erasure) {
            return (Some(hit), shift);
        }
    }
    (None, if spec.has_sync { shifts[0] } else { 0 })
}

fn roll_grid(grid: &[u8], spec: &MarkerSpec, shift: usize) -> Vec<u8> {
    // np.roll(grid, -shift, axis=1): out[r][i] = grid[r][(i + shift) % n]
    let n = spec.sector_count;
    let mut out = vec![0u8; grid.len()];
    for r in 0..spec.ring_count {
        for i in 0..n {
            out[r * n + i] = grid[r * n + (i + shift) % n];
        }
    }
    out
}

fn roll_erasure(eg: &[bool], spec: &MarkerSpec, shift: usize) -> Vec<bool> {
    let n = spec.sector_count;
    let mut out = vec![false; eg.len()];
    for r in 0..spec.ring_count {
        for i in 0..n {
            out[r * n + i] = eg[r * n + (i + shift) % n];
        }
    }
    out
}

/// Decode a cell grid -> (payload or None, rotation applied). Mirrors Python
/// including the no-sync brute-force branch (unused by current variants but
/// part of the pinned behavior).
pub fn decode(
    grid: &[u8],
    spec: &MarkerSpec,
    erasure: Option<&[bool]>,
) -> (Option<Vec<u8>>, usize) {
    let shifts: Vec<usize> = if spec.has_sync {
        let (shift, _) = find_rotation(&grid[..spec.sector_count], spec);
        vec![shift]
    } else {
        (0..spec.sector_count).collect()
    };
    for &shift in &shifts {
        let aligned = roll_grid(grid, spec, shift);
        let eg_s = erasure.map(|eg| roll_erasure(eg, spec, shift));
        if let Some(pb) = decode_aligned(&aligned, spec, eg_s.as_deref()) {
            return (Some(pb), shift);
        }
    }
    (None, if spec.has_sync { shifts[0] } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec;

    #[test]
    fn roundtrip_all_variants_with_rotation() {
        for sp in spec::variants() {
            let payload: Vec<u8> = (0..sp.payload_bytes()).map(|i| (i * 37 + 5) as u8).collect();
            let grid = encode(&payload, sp).unwrap();
            let (got, sh) = decode(&grid, sp, None);
            assert_eq!(got.as_deref(), Some(&payload[..]), "{} clean", sp.name);
            assert_eq!(sh, 0);
            // rotate by 5 sectors (np.roll +5 = our roll_grid with shift n-5)
            let rot = roll_grid(&grid, sp, sp.sector_count - 5);
            let (got, sh) = decode(&rot, sp, None);
            assert_eq!(got.as_deref(), Some(&payload[..]), "{} rotated", sp.name);
            assert_eq!(sh, sp.sector_count - (sp.sector_count - 5));
        }
    }
}
