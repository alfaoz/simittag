//! Simittag marker format -- Rust mirror of simittag/spec.py.
//!
//! Constants are cross-checked at parity time against fixtures/spec.json so a
//! transcription typo in a SYNC pattern or radius fails loudly, not as a
//! mysterious decode-rate regression three phases later.

#[derive(Debug)]
pub struct MarkerSpec {
    /// canonical technical name: sim<total cells incl. sync>c<payload bits>
    pub name: &'static str,
    /// human alias (s256/s16m/sdata); accepted as input, reported in output
    pub alias: &'static str,
    // geometry (normalized radii, outer edge = 1.0)
    pub r_bullseye: f64,
    pub r_data_in: f64,
    pub r_data_out: f64,
    pub r_ring_in: f64,
    // data grid
    pub ring_count: usize,
    pub sector_count: usize,
    pub has_sync: bool,
    pub use_header: bool,
    // codec
    /// bits per RS symbol: 8 = bytes over GF(256) + CRC8 (v1 variants),
    /// 4 = nibbles over GF(16) + CRC4 (small-grid v2 variants)
    pub symbol_bits: usize,
    pub rs_k: usize,
    pub rs_nsym: usize,
    pub crc_bytes: usize,
    /// cap on BLIND RS error corrections (None = full floor(NSYM/2));
    /// erasure corrections are never capped by this
    pub max_errors: Option<usize>,
    /// per-variant decode-verify floor (None = the detector's global 0.73
    /// gate); same-grid variants carry a higher floor -- see the Python
    /// reference (spec.py / NOTES R3.4) for the measured calibration
    pub verify_min: Option<f64>,
    /// per-variant ranked-erasure confidence threshold (None = the caller's
    /// conf_erasure; 0.0 disables ranked erasures). For RS(4,2) one erasure
    /// forfeits the whole blind budget -- see spec.py for the measurement.
    pub conf_erasure: Option<f32>,
    pub sync: &'static [u8],
}

impl MarkerSpec {
    pub fn data_ring_count(&self) -> usize {
        if self.has_sync {
            self.ring_count - 1
        } else {
            self.ring_count
        }
    }

    pub fn first_data_ring(&self) -> usize {
        if self.has_sync {
            1
        } else {
            0
        }
    }

    pub fn payload_bits(&self) -> usize {
        (self.rs_k - self.crc_bytes) * self.symbol_bits
    }

    /// Bytes in the canonical payload representation (big-endian; high pad
    /// bits zero when payload_bits is not a byte multiple).
    pub fn payload_bytes(&self) -> usize {
        (self.payload_bits() + 7) / 8
    }

    /// (inner, center, outer) normalized radius of each data ring.
    pub fn ring_radii(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let w = (self.r_data_out - self.r_data_in) / self.ring_count as f64;
        let inner: Vec<f64> = (0..self.ring_count)
            .map(|i| self.r_data_in + w * i as f64)
            .collect();
        let outer: Vec<f64> = inner.iter().map(|r| r + w).collect();
        let center: Vec<f64> = inner.iter().map(|r| r + w / 2.0).collect();
        (inner, center, outer)
    }

    /// Angle (rad) of each sector center, sector 0 centered on +x, CCW.
    pub fn sector_center_angles(&self) -> Vec<f64> {
        let step = 2.0 * std::f64::consts::PI / self.sector_count as f64;
        (0..self.sector_count).map(|s| (s as f64 + 0.5) * step).collect()
    }
}

pub static SIM48C8: MarkerSpec = MarkerSpec {
    symbol_bits: 8,
    max_errors: None,
    verify_min: Some(0.76),
    conf_erasure: Some(0.0),
    name: "sim48c8",
    alias: "s256",
    r_bullseye: 0.22,
    r_data_in: 0.30,
    r_data_out: 0.78,
    r_ring_in: 0.86,
    ring_count: 3,
    sector_count: 16,
    has_sync: true,
    use_header: false,
    rs_k: 2,
    rs_nsym: 2,
    crc_bytes: 1,
    sync: &[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
};

pub static SIM96C32: MarkerSpec = MarkerSpec {
    symbol_bits: 8,
    max_errors: None,
    verify_min: None,
    conf_erasure: None,
    name: "sim96c32",
    alias: "s16m",
    r_bullseye: 0.22,
    r_data_in: 0.30,
    r_data_out: 0.78,
    r_ring_in: 0.86,
    ring_count: 4,
    sector_count: 24,
    has_sync: true,
    use_header: true,
    rs_k: 5,
    rs_nsym: 4,
    crc_bytes: 1,
    sync: &[
        1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
    ],
};

pub static SIM180C88: MarkerSpec = MarkerSpec {
    symbol_bits: 8,
    max_errors: None,
    verify_min: None,
    conf_erasure: None,
    name: "sim180c88",
    alias: "sdata",
    r_bullseye: 0.22,
    r_data_in: 0.30,
    r_data_out: 0.78,
    r_ring_in: 0.86,
    ring_count: 5,
    sector_count: 36,
    has_sync: true,
    use_header: true,
    rs_k: 12,
    rs_nsym: 6,
    crc_bytes: 1,
    sync: &[
        1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 1, 0,
    ],
};

/// sim48c16 / s64k -- EXPERIMENTAL: sim48c8's 3x16 grid carrying 8 GF(16)
/// nibbles (4 ID + CRC4 + RS(8,5) parity; 1 err / 2 ranked erasures) for a
/// 65,536-ID tracking tag at near-s256 range. Same-grid disambiguation vs
/// sim48c8 rests on the sync patterns (chosen jointly for cross-correlation
/// margin: worst |cross| 6 vs the sync gate's 12) + codec + verify gate.
pub static SIM48C16: MarkerSpec = MarkerSpec {
    symbol_bits: 4,
    max_errors: None,
    verify_min: Some(0.78),
    conf_erasure: Some(0.40),
    name: "sim48c16",
    alias: "s64k",
    r_bullseye: 0.22,
    r_data_in: 0.30,
    r_data_out: 0.78,
    r_ring_in: 0.86,
    ring_count: 3,
    sector_count: 16,
    has_sync: true,
    use_header: false,
    rs_k: 5,
    rs_nsym: 3,
    crc_bytes: 1,
    sync: &[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
};

/// sim48c12 / s4k -- EXPERIMENTAL 12-bit-ID tracking tag at s256-class
/// range: 3x16 grid, 3 ID nibbles + CRC4 + RS(8,4) parity (2 errors / 3
/// ranked erasures). The heavier code measured CLEANER than RS(8,5) on
/// wrong-values (see the Python reference / NOTES R3.6). Same raised
/// verify floor and jointly-chosen sync as sim48c16.
pub static SIM48C12: MarkerSpec = MarkerSpec {
    symbol_bits: 4,
    max_errors: None,
    verify_min: Some(0.78),
    conf_erasure: Some(0.40),
    name: "sim48c12",
    alias: "s4k",
    r_bullseye: 0.22,
    r_data_in: 0.30,
    r_data_out: 0.78,
    r_ring_in: 0.86,
    ring_count: 3,
    sector_count: 16,
    has_sync: true,
    use_header: false,
    rs_k: 4,
    rs_nsym: 4,
    crc_bytes: 1,
    sync: &[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
};

/// Every registered variant, auto-detect order (matches Python's VARIANTS
/// dict order): v1 variants first, experimental v2 variants appended. This
/// is the REGISTRY (parity gates, sample patterns) — the default detection
/// set is default_variants().
pub fn variants() -> [&'static MarkerSpec; 5] {
    [&SIM48C8, &SIM96C32, &SIM180C88, &SIM48C16, &SIM48C12]
}

/// The DEFAULT auto-detect set: ONE 3x16 variant + sim96c32 + sim180c88.
/// The 3x16 slot belongs to sim48c12/s4k (run 3); sim48c8/s256 and
/// sim48c16/s64k are selected explicitly (printed s256 fleets, migrations).
/// Mirrors spec.DEFAULT_VERSIONS — see the Python reference for the
/// measured tradeoffs behind the swap.
pub fn default_variants() -> [&'static MarkerSpec; 3] {
    [&SIM48C12, &SIM96C32, &SIM180C88]
}

/// Resolve any accepted spelling: canonical name, human alias, or the
/// deprecated pre-0.2 letters T/M/D (input-only; never emitted).
pub fn by_name(name: &str) -> Option<&'static MarkerSpec> {
    match name {
        "sim48c8" | "s256" | "T" => Some(&SIM48C8),
        "sim96c32" | "s16m" | "M" => Some(&SIM96C32),
        "sim180c88" | "sdata" | "D" => Some(&SIM180C88),
        "sim48c16" | "s64k" => Some(&SIM48C16),
        "sim48c12" | "s4k" => Some(&SIM48C12),
        _ => None,
    }
}
