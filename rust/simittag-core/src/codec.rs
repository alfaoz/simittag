//! Payload <-> cell grid. Port of simittag/codec.py.
//!
//! Grid = flat `Vec<u8>` (0/1), row-major: `grid[ring * sector_count + sector]`.
//! Data-cell linear order is sector-major: k = sector*data_rings + (ring - first),
//! bits packed MSB-first -- exactly the Python ordering, or nothing decodes.

use crate::gf256;
use crate::spec::MarkerSpec;

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks(8) {
        let mut b = 0u8;
        for &bit in chunk {
            b = (b << 1) | (bit & 1);
        }
        out.push(b);
    }
    out
}

fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &b in data {
        for j in (0..8).rev() {
            bits.push((b >> j) & 1);
        }
    }
    bits
}

/// (k, ring, sector) for the data cells in codeword-bit order.
fn cell_order(spec: &MarkerSpec) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    let dr = spec.data_ring_count();
    let r0 = spec.first_data_ring();
    (0..spec.sector_count)
        .flat_map(move |s| (0..dr).map(move |rd| (s * dr + rd, r0 + rd, s)))
}

/// payload bytes -> +CRC8 -> RS encode -> bits -> grid (sync ring set).
pub fn encode(payload: &[u8], spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    if payload.len() != spec.payload_bytes() {
        return Err(format!(
            "payload must be {} bytes, got {}",
            spec.payload_bytes(),
            payload.len()
        ));
    }
    let mut data = payload.to_vec();
    data.push(gf256::crc8(payload));
    let code = gf256::rs_encode(&data, spec.rs_nsym);
    let bits = bytes_to_bits(&code);

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
    let nbits = spec.data_ring_count() * spec.sector_count;
    let mut bits = vec![0u8; nbits];
    let mut unreliable = vec![false; nbits];
    for (k, ring, sector) in cell_order(spec) {
        bits[k] = grid[ring * spec.sector_count + sector];
        if let Some(eg) = erasure {
            unreliable[k] = eg[ring * spec.sector_count + sector];
        }
    }
    let code = bits_to_bytes(&bits);
    let mut erase_pos: Vec<usize> = Vec::new();
    if erasure.is_some() {
        for k in 0..nbits {
            if unreliable[k] && !erase_pos.contains(&(k / 8)) {
                erase_pos.push(k / 8);
            }
        }
        erase_pos.sort_unstable();
        // Cap at NSYM-1, never NSYM: with NSYM erasures RS always "succeeds"
        // (fills to the all-zeros codeword, CRC8(0)=0 -> phantom ID 0). One
        // syndrome of margin forces the surviving cells to be consistent.
        let cap = spec.rs_nsym.saturating_sub(1);
        erase_pos.truncate(cap);
    }
    let (data, _) = gf256::rs_decode(&code, spec.rs_nsym, &erase_pos).ok()?;
    let (payload, crc) = (&data[..spec.payload_bytes()], data[spec.payload_bytes()]);
    if gf256::crc8(payload) != crc {
        return None;
    }
    Some(payload.to_vec())
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
