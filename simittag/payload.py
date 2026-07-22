"""
Payload modes (semantic-density layer) sitting on top of the RS+CRC codec.

The codec carries `spec.payload_bytes` of opaque user bytes. This module structures
those bytes as: [1 header byte][body], header = (version<<4)|mode.

v1 modes:
  ID     (0): body = big-endian unsigned int, fills the whole body.
  GEO    (1): sdata-only. body = u32 (lat+90)*1e7, u32 (lon+180)*1e7, i16 altitude in
              METERS -- exactly 10 bytes. ~1 cm angular resolution; 1 m altitude
              resolution covers +-32 km (GPS accuracy is coarser anyway). A GEO tag
              + the free 6-DoF pose = the camera knows its own absolute position
              from one glance at one tag.
  RAW    (3): body = [len:1][bytes], remaining body zero-padded.
  TAGGED (4): body = [namespace:1][big-endian id] -- deployments don't collide
              (s16m: 256 namespaces x 65 536 ids; sdata: 256 x 2^72).

Mode 2 (was TEXT) is PERMANENTLY reserved-unused: a packed-charset text mode was
designed and dropped -- RAW covers short text, and the density win didn't justify a
second text representation. URL was rejected outright: a URL does not fit in 10
bytes, and hardcoding a shortener domain into a marker spec is how specs die. URL
use cases are TAGGED + an application-side resolver.

s256 (USE_HEADER=False) is headerless: the whole payload IS a raw ID, no modes, ever.
"""
from __future__ import annotations
from .spec import MarkerSpec, DEFAULT

VERSION = 0
MODE_ID = 0
MODE_GEO = 1
MODE_RAW = 3
MODE_TAGGED = 4
# mode 2 permanently reserved (dropped TEXT); never reuse.

_NAMES = {MODE_ID: "ID", MODE_GEO: "GEO", MODE_RAW: "RAW", MODE_TAGGED: "TAGGED"}


def _header(mode): return bytes([((VERSION & 0xF) << 4) | (mode & 0xF)])


def _body_bytes(spec):
    return spec.payload_bytes - (1 if spec.USE_HEADER else 0)


def encode_id(value: int, spec: MarkerSpec = DEFAULT) -> bytes:
    # USE_HEADER=False (s256/s64k): pure tracking tag, the whole payload IS the
    # raw ID. The ID space is payload_bits, which for nibble variants can be
    # narrower than the byte representation (sim48c12: 12 bits in 2 bytes).
    body = _body_bytes(spec)
    id_bits = spec.payload_bits - (8 if spec.USE_HEADER else 0)
    if value < 0 or value >= (1 << id_bits):
        mx = (1 << id_bits) - 1
        raise ValueError(
            f"ID too big for variant {spec.NAME}: max 0x{mx:x} ({mx}). "
            f"Use sim96c32/s16m (16.7M IDs) or sim180c88/sdata for larger values.")
    head = _header(MODE_ID) if spec.USE_HEADER else b""
    return head + value.to_bytes(body, "big")


def encode_geo(lat: float, lon: float, alt_m: int = 0,
               spec: MarkerSpec = DEFAULT) -> bytes:
    if not spec.USE_HEADER:
        raise ValueError(f"variant {spec.NAME} is ID-only (no payload-mode header)")
    if _body_bytes(spec) < 10:
        raise ValueError(f"GEO needs 10 body bytes; variant {spec.NAME} has "
                         f"{_body_bytes(spec)}. Use sim180c88 (sdata).")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("lat out of range [-90, 90]")
    if lon == 180.0:                      # one canonical antimeridian
        lon = -180.0
    if not (-180.0 <= lon < 180.0):
        raise ValueError("lon out of range [-180, 180)")
    alt_m = int(round(alt_m))
    if not (-32768 <= alt_m <= 32767):
        raise ValueError("altitude out of range [-32768, 32767] m")
    ulat = int(round((lat + 90.0) * 1e7))
    ulon = int(round((lon + 180.0) * 1e7))
    out = (_header(MODE_GEO) + ulat.to_bytes(4, "big") + ulon.to_bytes(4, "big")
           + alt_m.to_bytes(2, "big", signed=True))
    return out.ljust(spec.payload_bytes, b"\x00")


def encode_raw(data: bytes, spec: MarkerSpec = DEFAULT) -> bytes:
    if not spec.USE_HEADER:
        raise ValueError(f"variant {spec.NAME} is ID-only (no payload-mode header)")
    body = _body_bytes(spec)
    if len(data) > body - 1:
        raise ValueError(f"RAW too long: {len(data)} > {body-1} bytes for this size")
    out = _header(MODE_RAW) + bytes([len(data)]) + bytes(data)
    return out.ljust(spec.payload_bytes, b"\x00")


def encode_tagged(namespace: int, value: int, spec: MarkerSpec = DEFAULT) -> bytes:
    if not spec.USE_HEADER:
        raise ValueError(f"variant {spec.NAME} is ID-only (no payload-mode header)")
    if not (0 <= namespace <= 255):
        raise ValueError("namespace out of range [0, 255]")
    idb = _body_bytes(spec) - 1
    if value < 0 or value >= (1 << (8 * idb)):
        mx = (1 << (8 * idb)) - 1
        raise ValueError(f"TAGGED id too big for variant {spec.NAME}: max 0x{mx:x}")
    return _header(MODE_TAGGED) + bytes([namespace]) + value.to_bytes(idb, "big")


def decode(payload: bytes, spec: MarkerSpec = DEFAULT):
    """Return (mode_name, value). Raises on unknown/reserved mode.
    GEO value = (lat, lon, alt_m); TAGGED value = (namespace, id)."""
    if len(payload) != spec.payload_bytes:
        raise ValueError("wrong payload length")
    if not spec.USE_HEADER:                 # s256: headerless raw ID
        return "ID", int.from_bytes(payload, "big")
    h = payload[0]
    version, mode = h >> 4, h & 0xF
    body = payload[1:]
    if version != VERSION:
        raise ValueError(f"payload version {version} is not supported (current {VERSION})")
    if mode == MODE_ID:
        return "ID", int.from_bytes(body, "big")
    if mode == MODE_GEO:
        if len(body) < 10:
            raise ValueError(f"GEO body is truncated: {len(body)} < 10 bytes")
        lat = int.from_bytes(body[0:4], "big") / 1e7 - 90.0
        lon = int.from_bytes(body[4:8], "big") / 1e7 - 180.0
        alt = int.from_bytes(body[8:10], "big", signed=True)
        return "GEO", (lat, lon, alt)
    if mode == MODE_RAW:
        if not body:
            raise ValueError("RAW body has no length byte")
        n = body[0]
        if n > len(body) - 1:
            raise ValueError(f"RAW length {n} exceeds body capacity {len(body)-1}")
        return "RAW", bytes(body[1:1 + n])
    if mode == MODE_TAGGED:
        if len(body) < 2:
            raise ValueError("TAGGED body needs a namespace and ID")
        return "TAGGED", (body[0], int.from_bytes(body[1:], "big"))
    raise ValueError(f"mode {mode} not implemented in v{VERSION}")


if __name__ == "__main__":
    from .spec import VARIANTS
    sp = DEFAULT
    p = encode_id(0xABCDEF, sp)
    print("ID    ", p.hex(), "->", decode(p, sp))
    p = encode_raw(b"hi", sp)
    print("RAW   ", p.hex(), "->", decode(p, sp))
    p = encode_tagged(12, 0x1F4, sp)
    print("TAGGED", p.hex(), "->", decode(p, sp))
    D = VARIANTS["sim180c88"]
    cases = [(52.520008, 13.404954, 34), (-33.86882, 151.20930, 58),
             (89.9999999, -179.9999999, -430), (-90.0, -180.0, 32767), (0.0, 0.0, 0)]
    for lat, lon, alt in cases:
        p = encode_geo(lat, lon, alt, D)
        m, (la, lo, al) = decode(p, D)
        assert m == "GEO" and abs(la - lat) < 5.1e-8 and abs(lo - lon) < 5.1e-8 \
            and al == alt, (lat, lon, alt, la, lo, al)
        print(f"GEO   {p.hex()} -> {la:.7f},{lo:.7f} alt {al}m  (err "
              f"{abs(la-lat)*111e6:.0f}mm,{abs(lo-lon)*111e6:.0f}mm)")
    # guards
    for fn, args, spv in [(encode_geo, (0, 0, 0), sp),            # M too small
                          (encode_geo, (91, 0, 0), D),            # lat range
                          (encode_tagged, (256, 0), D),           # ns range
                          (encode_tagged, (0, 1 << 16), sp)]:     # M id range
        try:
            fn(*args, spv); raise AssertionError("guard failed")
        except ValueError:
            pass
    p = encode_geo(48.858370, 2.294481, 330, D)   # decodes on a real grid too
    from . import codec
    back, _ = codec.decode(codec.encode(p, D), D)
    assert back == p
    print("guards OK, codec round-trip OK")
