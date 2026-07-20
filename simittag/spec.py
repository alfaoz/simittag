"""
Simittag marker format — single source of truth.

A Simittag marker is concentric, read from the center out:

    r in [0.00, R_BULLSEYE]          solid black disk  (bullseye: locks center + scale)
    r in [R_BULLSEYE, R_DATA_IN]     white quiet ring
    r in [R_DATA_IN, R_DATA_OUT]     DATA: RING_COUNT rings x SECTOR_COUNT sectors
    r in [R_DATA_OUT, R_RING_IN]     white quiet ring
    r in [R_RING_IN, 1.0]            solid black outer ring (primary detection contour)

Radii are normalized so the outer edge of the outer ring = 1.0.

Detection contours (concentric ellipses under perspective):
  - outer ring outer edge  (r=1.0)        <- primary, biggest, most edge points
  - outer ring inner edge  (r=R_RING_IN)
  - bullseye edge          (r=R_BULLSEYE)
Three concentric circles overdetermine the supporting plane -> we can resolve the
conic pose 2-fold ambiguity geometrically, not only via the decode.

Data cell (ring, sector): black = 1, white = 0.
  ring 0  (innermost data ring) = SYNC pattern (known sequence) -> rotation by
          circular cross-correlation, one-shot, no brute force over the ECC.
  rings 1..RING_COUNT-1          = payload + Reed-Solomon parity.

This file is imported by the generator, the detector, and the simulators so all
three agree on geometry, cell ordering, and ECC parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class MarkerSpec:
    # --- identity ---
    NAME: str = "M"              # variant label (T/M/D); used by auto-detect

    # --- geometry (normalized radii, outer edge = 1.0) ---
    R_BULLSEYE: float = 0.22
    R_DATA_IN: float = 0.30
    R_DATA_OUT: float = 0.78
    R_RING_IN: float = 0.86

    # --- data grid ---
    RING_COUNT: int = 4          # total data rings (incl. the sync ring if HAS_SYNC)
    SECTOR_COUNT: int = 24       # angular cells per ring

    # --- rotation lock ---
    # HAS_SYNC=True: ring 0 carries a known sync pattern, rotation found in one shot
    #   by circular cross-correlation (M, D).
    # HAS_SYNC=False: no sync ring -- every ring is data, rotation found by brute
    #   force over the SECTOR_COUNT shifts with RS+CRC arbitrating (T: tiny grid, so
    #   the brute force is cheap, and the saved ring buys range/data).
    HAS_SYNC: bool = True

    # --- codec ---
    # Reed-Solomon over GF(256). The DATA rings carry the codeword.
    # M: rings 1..3 * 24 sectors = 72 cells = 9 bytes = RS_K data + RS_NSYM parity.
    RS_K: int = 5                # data bytes (incl. CRC byte)  -> 4 payload + 1 CRC8
    RS_NSYM: int = 4             # parity bytes -> corrects 2 errors OR 3 erasures
                                 # (erasures capped at NSYM-1 for a detection margin;
                                 # see codec.decode)
    CRC_BYTES: int = 1           # of the RS_K data bytes, this many are CRC8

    # payload-mode header: M/D spend 1 byte on a (version<<4)|mode header so one
    # marker can be ID/GEO/TEXT/RAW. T is a pure tracking tag (ID only) with just 2
    # payload bytes, so it skips the header and uses all bytes as a raw ID
    # (65 536 IDs instead of 256). See payload.py.
    USE_HEADER: bool = True

    # sync sequence (length must == SECTOR_COUNT when HAS_SYNC). Chosen for low
    # circular-autocorrelation sidelobes (validated in spec self-test). 0/1 per sector.
    SYNC: tuple = (1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
                   1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0)

    # which data ring carries the sync pattern (0 = innermost, the v1 layout).
    # Cell arc length grows with ring radius, so an outermost sync ring keeps
    # the gate readable deeper into the range floor (v2 experiment).
    SYNC_RING: int = 0

    def __post_init__(self):
        assert self.R_BULLSEYE < self.R_DATA_IN < self.R_DATA_OUT < self.R_RING_IN < 1.0
        assert 0 <= self.SYNC_RING < self.RING_COUNT
        if self.HAS_SYNC:
            assert len(self.SYNC) == self.SECTOR_COUNT, "SYNC length must equal SECTOR_COUNT"
        total_data_cells = self.data_ring_count * self.SECTOR_COUNT
        assert total_data_cells == 8 * (self.RS_K + self.RS_NSYM), (
            f"{total_data_cells} data cells must equal 8*(RS_K+RS_NSYM)="
            f"{8*(self.RS_K + self.RS_NSYM)}"
        )

    # ---- derived geometry helpers ----
    @property
    def data_ring_count(self) -> int:
        return self.RING_COUNT - 1 if self.HAS_SYNC else self.RING_COUNT

    @property
    def first_data_ring(self) -> int:
        return 1 if self.HAS_SYNC else 0  # ring index where the codeword starts

    @property
    def data_rings(self) -> tuple:
        """Ring indices carrying the codeword, in radial order (skips SYNC_RING)."""
        if not self.HAS_SYNC:
            return tuple(range(self.RING_COUNT))
        return tuple(r for r in range(self.RING_COUNT) if r != self.SYNC_RING)

    def ring_radii(self):
        """Inner/center/outer normalized radius of each data ring (ring 0..RING_COUNT-1)."""
        w = (self.R_DATA_OUT - self.R_DATA_IN) / self.RING_COUNT
        inner = self.R_DATA_IN + w * np.arange(self.RING_COUNT)
        outer = inner + w
        center = inner + w / 2
        return inner, center, outer

    def sector_center_angles(self):
        """Angle (rad) of each sector center, sector 0 centered on +x, increasing CCW."""
        return (np.arange(self.SECTOR_COUNT) + 0.5) * (2 * np.pi / self.SECTOR_COUNT)

    @property
    def payload_bytes(self) -> int:
        return self.RS_K - self.CRC_BYTES

    @property
    def payload_bits(self) -> int:
        return self.payload_bytes * 8


# ---------------------------------------------------------------------------
# Variant family T / M / D
#
# All three share the SAME radial layout (bullseye, quiet rings, data band, outer
# ring) and therefore the SAME detection + conic pose. They differ ONLY in the data
# grid: ring count, sector count, sync, and ECC budget. Fewer/larger cells (T) stay
# readable smaller and farther -> tracking; more cells (D) carry more bytes -> data.
#
# AUTO-DETECT: every variant carries its own sync ring with a distinct, low-sidelobe
# pattern. The detector tries each candidate variant's grid; a wrong variant samples
# the tag at the wrong ring/sector count, so its sync ring fails to correlate and the
# grid is rejected before RS/CRC. The correct variant's sync locks, CRC confirms. This
# sync-gating is what makes auto-detect robust -- not the sector counts (16/24/36 are
# kept distinct but the disambiguation does NOT rely on angular-frequency probing,
# which proved unreliable under perspective).
# ---------------------------------------------------------------------------

# T -- tracking: 3x16 with sync ring, 32 data cells = 4 bytes. Fewest rings + biggest
#   cells -> longest range / smallest print. Headerless 1-byte ID = 256 IDs (ample for
#   tracking a handful of physical tags, like AprilTag families).
#   RS(4,2): 1 payload + 1 CRC8 + 2 parity -> corrects 1 byte error OR 1 erasure.
#   (Earlier sync-LESS T was dropped: brute-forcing every rotation with only CRC8 is
#   inherently false-accept-prone -- erasure-fill makes RS always "succeed", leaving
#   1/256 CRC as the sole gate over hundreds of tries. A sync ring locks rotation in
#   one shot AND lets the detector reject wrong-variant grids by sync correlation,
#   which is what makes T/M/D auto-detect robust.)
T_SPEC = MarkerSpec(
    NAME="T", RING_COUNT=3, SECTOR_COUNT=16, HAS_SYNC=True,
    RS_K=2, RS_NSYM=2, USE_HEADER=False,
    SYNC=(1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1),
)

# M -- balanced (default): 4x24, sync ring, 72 cells = 9 bytes.
#   RS(9,5): 4 payload + 1 CRC8 + 4 parity -> corrects 2 byte errors OR 3 erasures.
M_SPEC = MarkerSpec(NAME="M")  # the canonical defaults above

# D -- data: 5x36, sync ring, 144 cells = 18 bytes.
#   RS(18,12): 11 payload + 1 CRC8 + 6 parity -> corrects 3 byte errors OR 5 erasures.
D_SPEC = MarkerSpec(
    NAME="D", RING_COUNT=5, SECTOR_COUNT=36, HAS_SYNC=True,
    RS_K=12, RS_NSYM=6,
    SYNC=(1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
          1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0),
)

VARIANTS = {s.NAME: s for s in (T_SPEC, M_SPEC, D_SPEC)}

DEFAULT = M_SPEC


def resolve_specs(versions=None):
    """
    versions=None      -> all variants (auto-detect among T/M/D).
    versions="M" or ["M","D"] -> only those (faster, no ambiguity).
    Returns a list of MarkerSpec in T,M,D order.
    """
    if versions is None:
        names = list(VARIANTS)
    elif isinstance(versions, str):
        names = [versions]
    else:
        names = list(versions)
    return [VARIANTS[n] for n in VARIANTS if n in names] or [DEFAULT]


def autocorr_sidelobe(seq) -> int:
    """Max absolute circular autocorrelation at nonzero lag, mapping 0/1 -> -1/+1."""
    s = np.where(np.asarray(seq) > 0, 1, -1)
    n = len(s)
    return max(abs(int(np.dot(s, np.roll(s, k)))) for k in range(1, n))


if __name__ == "__main__":
    print("Simittag variant family:")
    print(f"  {'var':>3} {'grid':>6} {'sync':>5} {'cells':>5} {'RS':>7} "
          f"{'payload':>7} {'IDs':>12}  corrects")
    for name, sp in VARIANTS.items():
        cells = sp.data_ring_count * sp.SECTOR_COUNT
        id_bytes = sp.payload_bytes - (1 if sp.USE_HEADER else 0)  # ID-mode body
        ids = 1 << (8 * id_bytes)
        sl = f"sl{autocorr_sidelobe(sp.SYNC)}" if sp.HAS_SYNC else "none"
        print(f"  {name:>3} {f'{sp.RING_COUNT}x{sp.SECTOR_COUNT}':>6} {sl:>5} "
              f"{cells:>5} {f'RS({sp.RS_K+sp.RS_NSYM},{sp.RS_K})':>7} "
              f"{f'{sp.payload_bytes}B':>7} {ids:>12,}  "
              f"{sp.RS_NSYM//2} err / {sp.RS_NSYM-1} eras")
    print("\n  every variant has a sync ring -> auto-detect by sync-gated decode")
