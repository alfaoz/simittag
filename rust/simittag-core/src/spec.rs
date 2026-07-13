//! Simittag marker format -- Rust mirror of simittag/spec.py.
//!
//! Constants are cross-checked at parity time against fixtures/spec.json so a
//! transcription typo in a SYNC pattern or radius fails loudly, not as a
//! mysterious decode-rate regression three phases later.

#[derive(Debug)]
pub struct MarkerSpec {
    pub name: &'static str,
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

pub static T: MarkerSpec = MarkerSpec {
    name: "T",
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

pub static M: MarkerSpec = MarkerSpec {
    name: "M",
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

pub static D: MarkerSpec = MarkerSpec {
    name: "D",
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

/// T, M, D -- the auto-detect order (matches Python's VARIANTS dict order).
pub fn variants() -> [&'static MarkerSpec; 3] {
    [&T, &M, &D]
}

pub fn by_name(name: &str) -> Option<&'static MarkerSpec> {
    match name {
        "T" => Some(&T),
        "M" => Some(&M),
        "D" => Some(&D),
        _ => None,
    }
}
