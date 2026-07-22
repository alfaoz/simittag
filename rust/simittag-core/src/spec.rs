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
    pub rs_k: usize,
    pub rs_nsym: usize,
    pub crc_bytes: usize,
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

    pub fn payload_bytes(&self) -> usize {
        self.rs_k - self.crc_bytes
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

/// Auto-detect order (matches Python's VARIANTS dict order).
pub fn variants() -> [&'static MarkerSpec; 3] {
    [&SIM48C8, &SIM96C32, &SIM180C88]
}

/// Resolve any accepted spelling: canonical name, human alias, or the
/// deprecated pre-0.2 letters T/M/D (input-only; never emitted).
pub fn by_name(name: &str) -> Option<&'static MarkerSpec> {
    match name {
        "sim48c8" | "s256" | "T" => Some(&SIM48C8),
        "sim96c32" | "s16m" | "M" => Some(&SIM96C32),
        "sim180c88" | "sdata" | "D" => Some(&SIM180C88),
        _ => None,
    }
}
