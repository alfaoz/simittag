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
    # Canonical technical name: sim<total cells incl. sync ring>c<payload bits>.
    # Used by auto-detect and reported as `variant` in every detection.
    NAME: str = "sim96c32"
    # Human alias (s256 / s16m / sdata ...), reported as `alias` alongside the
    # canonical name and accepted anywhere a variant is selected.
    ALIAS: str = "s16m"

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
    #   by circular cross-correlation (all current variants).
    # HAS_SYNC=False: no sync ring -- every ring is data, rotation found by brute
    #   force over the SECTOR_COUNT shifts with RS+CRC arbitrating (unused since the
    #   sync-less tracking design was dropped; kept as pinned behavior).
    HAS_SYNC: bool = True

    # --- codec ---
    # Reed-Solomon over GF(2^SYMBOL_BITS). The DATA rings carry the codeword.
    # SYMBOL_BITS=8: bytes over GF(256) with CRC8 (the v1 variants).
    # SYMBOL_BITS=4: nibbles over GF(16) with CRC4 (the small-grid v2 variants,
    #   where a byte symbol would span a quarter of the grid).
    # sim96c32: rings 1..3 * 24 sectors = 72 cells = 9 bytes = RS_K + RS_NSYM.
    SYMBOL_BITS: int = 8
    RS_K: int = 5                # data symbols (incl. CRC symbol) -> 4 payload + CRC8
    RS_NSYM: int = 4             # parity symbols -> corrects 2 errors OR 3 erasures
                                 # (erasures capped at NSYM-1 for a detection margin;
                                 # see codec.decode)
    CRC_BYTES: int = 1           # of the RS_K data symbols, this many are CRC
                                 # (one CRC8 byte or one CRC4 nibble)
    # Cap on BLIND RS error corrections (None = the code's full floor(NSYM/2)).
    # Erasure corrections are never capped by this. Small codes can trade a
    # sliver of recall for a lower wrong-value rate here; see codec.decode.
    MAX_ERRORS: object = None
    # Per-variant ranked-erasure confidence threshold (None = the detector's
    # conf_erasure parameter, default 0.25; 0.0 disables ranked erasures).
    # For RS(4,2) one erasure consumes the ENTIRE blind-correction budget
    # ((NSYM-erasures)//2 = 0), so erasing the weakest byte FORFEITS fixing
    # any other byte — measured on paired frames: disabling erasures for
    # sim48c8 gains +9% floor-band recall and never hurts occlusion (NOTES
    # R3.9). Unique to NSYM=2; the other variants keep ranked erasures.
    CONF_ERASURE: object = None
    # Per-variant decode-verify floor (None = the detector's global gate,
    # 0.73). Same-grid variants need a higher floor: the global gate was
    # calibrated against CLUTTER, but a wrong-variant read of a REAL same-grid
    # tag through a misregistered deconvolved view can self-consistently
    # decode and land inside the clutter-calibrated margin (measured worst
    # 0.759 for inverted s256 -> s64k over ~43k stress trials; wrong-variant
    # survivor p99 = 0.664). See detect.VERIFY_MIN and NOTES R3.4.
    VERIFY_MIN: object = None

    # payload-mode header: s16m/sdata spend 1 byte on a (version<<4)|mode header
    # so one marker can be ID/GEO/RAW/TAGGED. s256 is a pure tracking tag (ID
    # only) with one payload byte, so it skips the header and uses that byte as
    # a raw ID (256 IDs). See payload.py.
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
        assert self.SYMBOL_BITS in (4, 8)
        total_data_cells = self.data_ring_count * self.SECTOR_COUNT
        assert total_data_cells == self.SYMBOL_BITS * (self.RS_K + self.RS_NSYM), (
            f"{total_data_cells} data cells must equal SYMBOL_BITS*(RS_K+RS_NSYM)="
            f"{self.SYMBOL_BITS*(self.RS_K + self.RS_NSYM)}"
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
    def payload_bits(self) -> int:
        return (self.RS_K - self.CRC_BYTES) * self.SYMBOL_BITS

    @property
    def payload_bytes(self) -> int:
        """Bytes in the canonical payload representation. Payloads are always
        handled as big-endian byte strings; when payload_bits is not a byte
        multiple (nibble variants), the high pad bits of the first byte are
        zero (enforced by codec.encode)."""
        return (self.payload_bits + 7) // 8


# ---------------------------------------------------------------------------
# Variant family sim48c8 (s256) / sim96c32 (s16m) / sim180c88 (sdata)
#
# All three share the SAME radial layout (bullseye, quiet rings, data band, outer
# ring) and therefore the SAME detection + conic pose. They differ ONLY in the data
# grid: ring count, sector count, sync, and ECC budget. Fewer/larger cells (s256)
# stay readable smaller and farther -> tracking; more cells (sdata) carry more
# bytes -> data.
#
# Naming: canonical = sim<total cells incl. sync ring>c<payload bits>; alias =
# a short human name keyed to the ID space (s256 = 256 IDs, s16m = 16.7M IDs,
# sdata = data payloads). The pre-0.2 letters T/M/D map to s256/s16m/sdata and
# remain accepted as DEPRECATED input everywhere a variant is selected; they
# no longer appear in any output. Printed tags carry sync patterns, not names,
# so the rename does not affect any physical tag.
#
# AUTO-DETECT: every variant carries its own sync ring with a distinct, low-sidelobe
# pattern. The detector tries each candidate variant's grid; a wrong variant samples
# the tag at the wrong ring/sector count, so its sync ring fails to correlate and the
# grid is rejected before RS/CRC. The correct variant's sync locks, CRC confirms. This
# sync-gating is what makes auto-detect robust -- not the sector counts (16/24/36 are
# kept distinct but the disambiguation does NOT rely on angular-frequency probing,
# which proved unreliable under perspective).
# ---------------------------------------------------------------------------

# sim48c8 / s256 -- tracking: 3x16 with sync ring, 32 data cells = 4 bytes. Fewest
#   rings + biggest cells -> longest range / smallest print. Headerless 1-byte ID =
#   256 IDs (ample for tracking a handful of physical tags, like AprilTag families).
#   RS(4,2): 1 payload + 1 CRC8 + 2 parity -> corrects 1 byte error OR 1 erasure.
#   (Earlier sync-LESS design was dropped: brute-forcing every rotation with only
#   CRC8 is inherently false-accept-prone -- erasure-fill makes RS always "succeed",
#   leaving 1/256 CRC as the sole gate over hundreds of tries. A sync ring locks
#   rotation in one shot AND lets the detector reject wrong-variant grids by sync
#   correlation, which is what makes the family auto-detect robust.)
#   Integrity/recall config (run 3, measured on paired frames — NOTES R3.9):
#   CONF_ERASURE=0.0 (see field comment) + VERIFY_MIN=0.76. Together they
#   BEAT the previous config on both axes: +8% decode recall in the floor
#   band and -19% wrong-ID accepts, with healthy-size recall and occlusion
#   tolerance unchanged. Wrong IDs remain possible below the reliable floor
#   (~0.4% of near-floor trials at std degradation, all at <=20px); tracking
#   users who need zero measured wrongs should prefer sim48c12/s4k.
T_SPEC = MarkerSpec(
    NAME="sim48c8", ALIAS="s256", RING_COUNT=3, SECTOR_COUNT=16, HAS_SYNC=True,
    RS_K=2, RS_NSYM=2, USE_HEADER=False,
    SYNC=(1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1),
    CONF_ERASURE=0.0, VERIFY_MIN=0.76,
)

# sim96c32 / s16m -- balanced (default): 4x24, sync ring, 72 cells = 9 bytes.
#   RS(9,5): 4 payload + 1 CRC8 + 4 parity -> corrects 2 byte errors OR 3 erasures.
M_SPEC = MarkerSpec()  # the canonical defaults above

# sim180c88 / sdata -- data: 5x36, sync ring, 144 cells = 18 bytes.
#   RS(18,12): 11 payload + 1 CRC8 + 6 parity -> corrects 3 byte errors OR 5 erasures.
D_SPEC = MarkerSpec(
    NAME="sim180c88", ALIAS="sdata", RING_COUNT=5, SECTOR_COUNT=36, HAS_SYNC=True,
    RS_K=12, RS_NSYM=6,
    SYNC=(1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
          1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0),
)

# sim48c16 / s64k -- EXPERIMENTAL tracking tag with a 16-bit ID space: the
#   same 3x16 grid and radial layout as sim48c8 (so near-s256 range), but the
#   32 data cells carry 8 GF(16) NIBBLES instead of 4 bytes: 4 ID nibbles +
#   CRC4 + RS(8,5) parity (corrects 1 symbol error or 2 ranked erasures).
#   65,536 IDs at tracking-tag range -- 256x s256's ID space for ~1px of
#   decode floor (lab-measured px90 23.5 vs s256's 22.1 at std tilt 15).
#   Nibble symbols matter at this grid size: one GF(256) byte would span a
#   quarter of the grid, so any localized smear would burn multiple symbols.
#   Sharing the 3x16 grid means auto-detect between s256/s64k rests ONLY on
#   the sync patterns + codec + verify gate; SYNC below was chosen jointly
#   with the family for cross-correlation margin (worst |cross| 6 vs the
#   sync gate's 12, all shifts and polarities) and autocorr sidelobe 4.
S64K_SPEC = MarkerSpec(
    NAME="sim48c16", ALIAS="s64k", RING_COUNT=3, SECTOR_COUNT=16, HAS_SYNC=True,
    SYMBOL_BITS=4, RS_K=5, RS_NSYM=3, USE_HEADER=False,
    SYNC=(0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0),
    VERIFY_MIN=0.78,
)

# sim48c12 / s4k -- EXPERIMENTAL 12-bit-ID tracking tag at s256-class range:
#   the 3x16 grid carrying 8 GF(16) nibbles as 3 ID nibbles + CRC4 + RS(8,4)
#   parity (corrects 2 symbol errors or 3 ranked erasures). 4,096 IDs with a
#   HEAVIER code than either sibling: measured px90 22.9 vs s256's 21.9 at
#   std tilt 15 (deep lab run, ~6k trials), zero wrong-value decodes where
#   s256 logged 11 -- the extra parity nibble outweighs the stronger blind
#   correction for false accepts (NOTES R3.6). Sync chosen jointly with the
#   family for cross-correlation margin; carries the same raised verify
#   floor as sim48c16.
S4K_SPEC = MarkerSpec(
    NAME="sim48c12", ALIAS="s4k", RING_COUNT=3, SECTOR_COUNT=16, HAS_SYNC=True,
    SYMBOL_BITS=4, RS_K=4, RS_NSYM=4, USE_HEADER=False,
    SYNC=(1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0),
    VERIFY_MIN=0.78,
)

# Deprecated pre-0.2 letters, still accepted as input (never emitted).
LEGACY_NAMES = {"T": "sim48c8", "M": "sim96c32", "D": "sim180c88"}


def normalize_variant(name: str) -> str:
    """Canonical spec key for any accepted spelling: the canonical name, the
    human alias, or a deprecated T/M/D letter. Raises KeyError on unknown."""
    if isinstance(name, str):
        if dict.__contains__(VARIANTS, name):
            return name
        if name in ALIASES:
            return ALIASES[name]
        if name in LEGACY_NAMES:
            return LEGACY_NAMES[name]
    accepted = [n for sp in VARIANTS.values() for n in (sp.NAME, sp.ALIAS)]
    raise KeyError(f"unknown variant {name!r}; accepted: {', '.join(accepted)} "
                   f"(and deprecated {'/'.join(LEGACY_NAMES)})")


class _VariantMap(dict):
    """Canonical-name -> MarkerSpec. Lookup also accepts aliases and the
    deprecated T/M/D letters so pre-rename callers keep working."""
    def __missing__(self, key):
        return dict.__getitem__(self, normalize_variant(key))

    def __contains__(self, key):
        try:
            normalize_variant(key)
            return True
        except KeyError:
            return False


# Auto-detect order: v1 variants first (a candidate that decodes as one of
# them never pays for the specs after it), experimental v2 variants appended.
VARIANTS = _VariantMap({s.NAME: s
                        for s in (T_SPEC, M_SPEC, D_SPEC, S64K_SPEC, S4K_SPEC)})

# The DEFAULT auto-detect set carries ONE 3x16 variant plus s16m and sdata.
# As of run 3 that slot belongs to sim48c12/s4k — the new default tracking
# tag (16x s256's ID space, the family's strongest code, zero measured wrong
# IDs) at a measured cost of ~0.6-0.7m range at 0-15 deg tilt and more under
# heavy blur (NOTES R3.10). sim48c8/s256 and sim48c16/s64k remain fully
# supported by EXPLICIT selection — pinned alone or in any explicit set;
# existing printed s256 fleets select it (or calibrate via BOARD_VERSIONS,
# which pins the printed-board reality independently of this default).
# Intended usage: at most one 3x16 variant per environment; explicit
# multi-3x16 sets remain supported (fleet migrations) and measured safe
# (zero cross-decodes in ~280k adversarial trials, NOTES R3.4/R3.8).
DEFAULT_VERSIONS = ("sim48c12", "sim96c32", "sim180c88")
ALIASES = {s.ALIAS: s.NAME for s in VARIANTS.values()}
ALIAS_OF = {s.NAME: s.ALIAS for s in VARIANTS.values()}

DEFAULT = M_SPEC


def resolve_specs(versions=None):
    """
    versions=None -> the DEFAULT auto-detect set (v1 trio; experimental
    variants are explicit-only, see DEFAULT_VERSIONS).
    versions="s16m" or ["sim96c32","sdata"] -> only those (faster, no
    ambiguity). Canonical names, aliases, and the deprecated T/M/D letters are
    all accepted. Returns a list of MarkerSpec in auto-detect order.
    """
    if versions is None:
        names = list(DEFAULT_VERSIONS)
    elif isinstance(versions, str):
        names = [versions]
    else:
        names = list(versions)
    normalized = set()
    for n in names:
        try:
            normalized.add(normalize_variant(n))
        except KeyError:
            continue  # unknown names fall through to the DEFAULT fallback
    return [dict.__getitem__(VARIANTS, n) for n in VARIANTS
            if n in normalized] or [DEFAULT]


def autocorr_sidelobe(seq) -> int:
    """Max absolute circular autocorrelation at nonzero lag, mapping 0/1 -> -1/+1."""
    s = np.where(np.asarray(seq) > 0, 1, -1)
    n = len(s)
    return max(abs(int(np.dot(s, np.roll(s, k)))) for k in range(1, n))


if __name__ == "__main__":
    print("Simittag variant family:")
    print(f"  {'variant':>10} {'alias':>6} {'grid':>6} {'sync':>5} {'cells':>5} "
          f"{'RS':>9} {'payload':>7} {'IDs':>12}  corrects")
    for name, sp in VARIANTS.items():
        cells = sp.data_ring_count * sp.SECTOR_COUNT
        id_bits = sp.payload_bits - (8 if sp.USE_HEADER else 0)  # ID-mode body
        ids = 1 << id_bits
        sl = f"sl{autocorr_sidelobe(sp.SYNC)}" if sp.HAS_SYNC else "none"
        print(f"  {name:>10} {sp.ALIAS:>6} {f'{sp.RING_COUNT}x{sp.SECTOR_COUNT}':>6} "
              f"{sl:>5} {cells:>5} {f'RS({sp.RS_K+sp.RS_NSYM},{sp.RS_K})':>9} "
              f"{f'{sp.payload_bits}b':>7} {ids:>12,}  "
              f"{sp.RS_NSYM//2} err / {sp.RS_NSYM-1} eras")
    # sanity: every accepted spelling resolves, deprecated letters map
    for legacy, canon in LEGACY_NAMES.items():
        assert VARIANTS[legacy].NAME == canon
        assert normalize_variant(VARIANTS[canon].ALIAS) == canon
    assert resolve_specs("s256")[0] is T_SPEC
    assert resolve_specs(["M", "sdata"]) == [M_SPEC, D_SPEC]
    # every variant shares the v1 radial layout: detection, conic pose, the
    # inverted-view probe radii (0.26 / 1.08), and the bullseye fallback all
    # assume it (detect._needs_inverted_view hardcodes those radii)
    for sp in VARIANTS.values():
        assert (sp.R_BULLSEYE, sp.R_DATA_IN, sp.R_DATA_OUT, sp.R_RING_IN) \
            == (0.22, 0.30, 0.78, 0.86), sp.NAME
    # same-grid variants (3x16) are disambiguated by sync alone: pin the
    # cross-correlation margin so a future sync choice can't silently erode it
    same_grid = [sp for sp in VARIANTS.values()
                 if (sp.RING_COUNT, sp.SECTOR_COUNT) == (3, 16)]
    for i, a in enumerate(same_grid):
        for b in same_grid[i + 1:]:
            sa = np.where(np.asarray(a.SYNC) > 0, 1, -1)
            sb = np.where(np.asarray(b.SYNC) > 0, 1, -1)
            worst = max(abs(int(np.dot(sa, np.roll(sb, k))))
                        for k in range(len(sb)))
            assert worst <= 6, (a.NAME, b.NAME, worst)  # gate needs corr >= 12
    print("\n  every variant has a sync ring -> auto-detect by sync-gated decode")
    print("  aliases + deprecated T/M/D letters accepted as input everywhere")
    print("  same-grid sync cross-correlation margin pinned (worst |corr| <= 6)")
