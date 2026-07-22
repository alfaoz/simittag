//! Payload modes (v1) on top of the RS+CRC codec. Port of simittag/payload.py.
//!
//! Header byte = (version<<4)|mode. Modes: ID(0), GEO(1), RAW(3), TAGGED(4);
//! mode 2 permanently reserved (dropped TEXT). s256 (use_header=false) is a
//! headerless raw ID. GEO float math uses the identical IEEE-754 double ops as
//! Python, so decoded lat/lon compare EXACTLY against the fixtures.

use crate::spec::MarkerSpec;

pub const VERSION: u8 = 0;
pub const MODE_ID: u8 = 0;
pub const MODE_GEO: u8 = 1;
pub const MODE_RAW: u8 = 3;
pub const MODE_TAGGED: u8 = 4;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(u128),
    Bytes(Vec<u8>),
    Geo { lat: f64, lon: f64, alt_m: i32 },
    Tagged { namespace: u8, id: u128 },
}

fn header(mode: u8) -> u8 {
    ((VERSION & 0xF) << 4) | (mode & 0xF)
}

fn body_bytes(spec: &MarkerSpec) -> usize {
    spec.payload_bytes() - if spec.use_header { 1 } else { 0 }
}

fn int_to_be(value: u128, n: usize) -> Vec<u8> {
    (0..n).rev().map(|i| (value >> (8 * i)) as u8).collect()
}

fn be_to_int(b: &[u8]) -> u128 {
    b.iter().fold(0u128, |acc, &v| (acc << 8) | v as u128)
}

pub fn encode_id(value: u128, spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    let body = body_bytes(spec);
    if body < 16 && value >> (8 * body) != 0 {
        return Err(format!("ID too big for variant {}", spec.name));
    }
    let mut out = Vec::new();
    if spec.use_header {
        out.push(header(MODE_ID));
    }
    out.extend(int_to_be(value, body));
    Ok(out)
}

pub fn encode_geo(lat: f64, lon: f64, alt_m: i64, spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    if !spec.use_header {
        return Err(format!("variant {} is ID-only", spec.name));
    }
    if body_bytes(spec) < 10 {
        return Err(format!("GEO needs 10 body bytes; use sim180c88, not {}", spec.name));
    }
    if !(-90.0..=90.0).contains(&lat) {
        return Err("lat out of range [-90, 90]".into());
    }
    let lon = if lon == 180.0 { -180.0 } else { lon }; // canonical antimeridian
    if !(-180.0..180.0).contains(&lon) {
        return Err("lon out of range [-180, 180)".into());
    }
    if !(-32768..=32767).contains(&alt_m) {
        return Err("altitude out of range [-32768, 32767] m".into());
    }
    let ulat = ((lat + 90.0) * 1e7).round() as u32;
    let ulon = ((lon + 180.0) * 1e7).round() as u32;
    let mut out = vec![header(MODE_GEO)];
    out.extend(ulat.to_be_bytes());
    out.extend(ulon.to_be_bytes());
    out.extend((alt_m as i16).to_be_bytes());
    out.resize(spec.payload_bytes(), 0);
    Ok(out)
}

pub fn encode_raw(data: &[u8], spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    if !spec.use_header {
        return Err(format!("variant {} is ID-only", spec.name));
    }
    let body = body_bytes(spec);
    if data.len() > body - 1 {
        return Err(format!("RAW too long: {} > {} bytes", data.len(), body - 1));
    }
    let mut out = vec![header(MODE_RAW), data.len() as u8];
    out.extend(data);
    out.resize(spec.payload_bytes(), 0);
    Ok(out)
}

pub fn encode_tagged(namespace: u16, value: u128, spec: &MarkerSpec) -> Result<Vec<u8>, String> {
    if !spec.use_header {
        return Err(format!("variant {} is ID-only", spec.name));
    }
    if namespace > 255 {
        return Err("namespace out of range [0, 255]".into());
    }
    let idb = body_bytes(spec) - 1;
    if idb < 16 && value >> (8 * idb) != 0 {
        return Err(format!("TAGGED id too big for variant {}", spec.name));
    }
    let mut out = vec![header(MODE_TAGGED), namespace as u8];
    out.extend(int_to_be(value, idb));
    Ok(out)
}

/// (mode_name, value); Err on unknown/reserved mode or wrong length.
pub fn decode(payload: &[u8], spec: &MarkerSpec) -> Result<(&'static str, Value), String> {
    if payload.len() != spec.payload_bytes() {
        return Err("wrong payload length".into());
    }
    if !spec.use_header {
        return Ok(("ID", Value::Int(be_to_int(payload))));
    }
    let version = payload[0] >> 4;
    let mode = payload[0] & 0xF;
    let body = &payload[1..];
    if version != VERSION {
        return Err(format!(
            "payload version {} is not supported (current {})",
            version, VERSION
        ));
    }
    match mode {
        MODE_ID => Ok(("ID", Value::Int(be_to_int(body)))),
        MODE_GEO => {
            if body.len() < 10 {
                return Err(format!("GEO body is truncated: {} < 10 bytes", body.len()));
            }
            let ulat = u32::from_be_bytes(body[0..4].try_into().unwrap());
            let ulon = u32::from_be_bytes(body[4..8].try_into().unwrap());
            let alt = i16::from_be_bytes(body[8..10].try_into().unwrap());
            Ok((
                "GEO",
                Value::Geo {
                    lat: ulat as f64 / 1e7 - 90.0,
                    lon: ulon as f64 / 1e7 - 180.0,
                    alt_m: alt as i32,
                },
            ))
        }
        MODE_RAW => {
            let Some((&length, data)) = body.split_first() else {
                return Err("RAW body has no length byte".into());
            };
            let n = length as usize;
            if n > data.len() {
                return Err(format!("RAW length {} exceeds body capacity {}", n, data.len()));
            }
            Ok(("RAW", Value::Bytes(data[..n].to_vec())))
        }
        MODE_TAGGED => {
            if body.len() < 2 {
                return Err("TAGGED body needs a namespace and ID".into());
            }
            Ok((
                "TAGGED",
                Value::Tagged {
                    namespace: body[0],
                    id: be_to_int(&body[1..]),
                },
            ))
        }
        m => Err(format!("mode {} not implemented in v{}", m, VERSION)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec;

    #[test]
    fn geo_roundtrip_exact() {
        let p = encode_geo(48.858370, 2.294481, 330, &spec::SIM180C88).unwrap();
        match decode(&p, &spec::SIM180C88).unwrap() {
            ("GEO", Value::Geo { lat, lon, alt_m }) => {
                assert!((lat - 48.858370).abs() < 5.1e-8);
                assert!((lon - 2.294481).abs() < 5.1e-8);
                assert_eq!(alt_m, 330);
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn guards() {
        assert!(encode_geo(0.0, 0.0, 0, &spec::SIM96C32).is_err()); // sim96c32 too small
        assert!(encode_geo(91.0, 0.0, 0, &spec::SIM180C88).is_err());
        assert!(encode_tagged(256, 0, &spec::SIM180C88).is_err());
        assert!(encode_tagged(0, 1 << 16, &spec::SIM96C32).is_err());
        assert!(encode_raw(&[0], &spec::SIM48C8).is_err()); // sim48c8 headerless
        assert!(decode(&[0x01, 0, 0, 0], &spec::SIM96C32).is_err()); // GEO needs sim180c88
        assert!(decode(&[0x13, 0, 0, 0], &spec::SIM96C32).is_err()); // future version
        assert!(decode(&[0x03, 3, 0, 0], &spec::SIM96C32).is_err()); // RAW length > capacity
    }
}
