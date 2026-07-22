//! simittag_node: ROS 2 wrapper around simittag-core.
//!
//! Subscribes an image stream plus CameraInfo, publishes verified tag
//! detections as vision_msgs/Detection3DArray and (optionally) TF frames.
//!
//! Frame conventions: the input image header frame is assumed to be a
//! REP-103 camera optical frame (x right, y down, z forward). The published
//! tag frame follows the apriltag_ros convention: x right, y up, z out of
//! the tag surface toward the viewer. The simittag marker frame is x
//! print-right, y print-down, z into the surface, so the fixed conversion
//! is diag(1,-1,-1).

mod covariance;

use futures::executor::LocalPool;
use futures::stream::StreamExt;
use futures::task::LocalSpawnExt;
use r2r::QosProfile;
use simittag_core::detector::{self, Detection};
use simittag_core::image::Gray;
use simittag_core::mat::M3;
use simittag_core::payload::Value;
use simittag_core::spec::{self, MarkerSpec};
use std::cell::RefCell;
use std::rc::Rc;

const CONF_ERASURE: f32 = 0.25; // reference-implementation default
const FOV_GUESS_DEG: f64 = 60.0; // matches the Python reference default_K
const CAMERA_INFO_WARN_FRAMES: u64 = 30;

struct Calib {
    k: M3,
    dist: Option<Vec<f64>>,
}

struct Config {
    tag_radius_m: f64,
    specs: Vec<&'static MarkerSpec>,
    pose_only: bool,
    publish_tf: bool,
    frame_id_override: Option<String>,
    detect_width: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "simittag", "")?;

    // -- parameters --------------------------------------------------------
    let image_topic = p_str(&node, "image_topic", "/camera/image_raw");
    let camera_info_topic = p_str(&node, "camera_info_topic", "/camera/camera_info");
    let detections_topic = p_str(&node, "detections_topic", "/simittag/detections");
    let tag_diameter_m = p_f64(&node, "tag_diameter_m", 0.0);
    let variant = p_str(&node, "variant", "auto");
    let publish_tf = p_bool(&node, "publish_tf", true);
    let pose_only = p_bool(&node, "pose_only", false);
    let debug_image = p_bool(&node, "debug_image", false);
    let detect_width = p_f64(&node, "detect_width", 0.0) as usize;
    let frame_id_override = {
        let s = p_str(&node, "frame_id", "");
        if s.is_empty() { None } else { Some(s) }
    };

    let logger = node.logger().to_string();
    if tag_diameter_m <= 0.0 {
        r2r::log_error!(
            &logger,
            "tag_diameter_m must be set to the printed outer-ring outer diameter \
             in meters; poses will be published in tag-radius units until it is."
        );
    }
    let specs: Vec<&'static MarkerSpec> = match variant.as_str() {
        "auto" | "" => spec::default_variants().to_vec(),
        name => match spec::by_name(name) {
            Some(s) => vec![s],
            None => {
                r2r::log_error!(&logger, "unknown variant '{}', using auto", name);
                spec::default_variants().to_vec()
            }
        },
    };
    let cfg = Rc::new(Config {
        tag_radius_m: if tag_diameter_m > 0.0 { tag_diameter_m / 2.0 } else { 2.0 },
        specs,
        pose_only,
        publish_tf,
        frame_id_override,
        detect_width,
    });

    // -- pubs / subs -------------------------------------------------------
    let sensor_qos = QosProfile::sensor_data();
    let image_sub =
        node.subscribe::<r2r::sensor_msgs::msg::Image>(&image_topic, sensor_qos.clone())?;
    let info_sub =
        node.subscribe::<r2r::sensor_msgs::msg::CameraInfo>(&camera_info_topic, sensor_qos)?;
    let det_pub = node.create_publisher::<r2r::vision_msgs::msg::Detection3DArray>(
        &detections_topic,
        QosProfile::default(),
    )?;
    let tf_pub =
        node.create_publisher::<r2r::tf2_msgs::msg::TFMessage>("/tf", QosProfile::default())?;
    let dbg_pub = if debug_image {
        Some(node.create_publisher::<r2r::sensor_msgs::msg::Image>(
            "/simittag/debug_image",
            QosProfile::default(),
        )?)
    } else {
        None
    };

    let calib: Rc<RefCell<Option<Calib>>> = Rc::new(RefCell::new(None));

    let mut pool = LocalPool::new();
    let spawner = pool.spawner();

    // CameraInfo intake: keep the latest K and distortion.
    {
        let calib = calib.clone();
        let mut info_sub = info_sub;
        spawner.spawn_local(async move {
            while let Some(msg) = info_sub.next().await {
                let k = [
                    [msg.k[0], msg.k[1], msg.k[2]],
                    [msg.k[3], msg.k[4], msg.k[5]],
                    [msg.k[6], msg.k[7], msg.k[8]],
                ];
                let mut d: Vec<f64> = msg.d.clone();
                d.resize(5, 0.0);
                let dist = if d.iter().all(|v| v.abs() < 1e-12) {
                    None
                } else {
                    Some(d)
                };
                *calib.borrow_mut() = Some(Calib { k, dist });
            }
        })?;
    }

    // Image intake: detect and publish.
    {
        let calib = calib.clone();
        let cfg = cfg.clone();
        let logger = logger.clone();
        let mut image_sub = image_sub;
        spawner.spawn_local(async move {
            let mut frames: u64 = 0;
            let mut warned_no_info = false;
            let mut warned_encoding = false;
            while let Some(msg) = image_sub.next().await {
                frames += 1;
                let Some(gray) = to_gray(&msg) else {
                    if !warned_encoding {
                        r2r::log_warn!(
                            &logger,
                            "unsupported image encoding '{}' (need mono8/rgb8/bgr8); \
                             skipping frames with this encoding",
                            msg.encoding
                        );
                        warned_encoding = true;
                    }
                    continue;
                };
                let (gray, scale_back) = maybe_downscale(gray, cfg.detect_width);
                let (k, dist) = match calib.borrow().as_ref() {
                    Some(c) => (scale_k(&c.k, scale_back), c.dist.clone()),
                    None => {
                        if frames == CAMERA_INFO_WARN_FRAMES && !warned_no_info {
                            r2r::log_warn!(
                                &logger,
                                "no CameraInfo after {} frames; using a {}-degree-FOV \
                                 guess (decode is unaffected, pose is approximate)",
                                frames,
                                FOV_GUESS_DEG
                            );
                            warned_no_info = true;
                        }
                        (guess_k(gray.w, gray.h), None)
                    }
                };
                let dets = detector::detect_markers(
                    &gray,
                    &k,
                    &cfg.specs,
                    CONF_ERASURE,
                    cfg.pose_only,
                    dist.as_deref(),
                );
                let frame_id = cfg
                    .frame_id_override
                    .clone()
                    .unwrap_or_else(|| msg.header.frame_id.clone());
                let header = r2r::std_msgs::msg::Header {
                    stamp: msg.header.stamp.clone(),
                    frame_id: frame_id.clone(),
                };
                publish_detections(&det_pub, &tf_pub, &header, &dets, &cfg, &k);
                if let Some(ref dbg) = dbg_pub {
                    let img = draw_debug(&gray, &dets);
                    let mut out = r2r::sensor_msgs::msg::Image::default();
                    out.header = header;
                    out.height = gray.h as u32;
                    out.width = gray.w as u32;
                    out.encoding = "bgr8".to_string();
                    out.step = (gray.w * 3) as u32;
                    out.data = img;
                    let _ = dbg.publish(&out);
                }
            }
        })?;
    }

    loop {
        node.spin_once(std::time::Duration::from_millis(10));
        pool.run_until_stalled();
    }
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

fn p_get(node: &r2r::Node, name: &str) -> Option<r2r::ParameterValue> {
    node.params
        .lock()
        .unwrap()
        .get(name)
        .map(|p| p.value.clone())
}

fn p_str(node: &r2r::Node, name: &str, default: &str) -> String {
    match p_get(node, name) {
        Some(r2r::ParameterValue::String(s)) => s,
        _ => default.to_string(),
    }
}

fn p_f64(node: &r2r::Node, name: &str, default: f64) -> f64 {
    match p_get(node, name) {
        Some(r2r::ParameterValue::Double(v)) => v,
        Some(r2r::ParameterValue::Integer(v)) => v as f64,
        _ => default,
    }
}

fn p_bool(node: &r2r::Node, name: &str, default: bool) -> bool {
    match p_get(node, name) {
        Some(r2r::ParameterValue::Bool(v)) => v,
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// image handling
// ---------------------------------------------------------------------------

fn to_gray(msg: &r2r::sensor_msgs::msg::Image) -> Option<Gray> {
    let w = msg.width as usize;
    let h = msg.height as usize;
    let step = msg.step as usize;
    let mut px = vec![0u8; w * h];
    match msg.encoding.as_str() {
        "mono8" | "8UC1" => {
            for y in 0..h {
                let row = &msg.data[y * step..y * step + w];
                px[y * w..(y + 1) * w].copy_from_slice(row);
            }
        }
        "rgb8" | "bgr8" => {
            // integer BT.601, same as the wasm path
            let (ri, gi, bi) = if msg.encoding == "rgb8" { (0, 1, 2) } else { (2, 1, 0) };
            for y in 0..h {
                let row = &msg.data[y * step..y * step + w * 3];
                for x in 0..w {
                    let p = &row[x * 3..x * 3 + 3];
                    let (r, g, b) = (p[ri] as u32, p[gi] as u32, p[bi] as u32);
                    px[y * w + x] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
                }
            }
        }
        _ => return None,
    }
    Some(Gray { w, h, px })
}

/// Optional integer-factor box downscale for weak boards. Returns the image
/// and the factor needed to scale detected pixel coordinates back up.
fn maybe_downscale(gray: Gray, target_w: usize) -> (Gray, f64) {
    if target_w == 0 || gray.w <= target_w {
        return (gray, 1.0);
    }
    let f = gray.w.div_ceil(target_w); // integer factor >= 2
    let (w, h) = (gray.w / f, gray.h / f);
    let mut px = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc: u32 = 0;
            for dy in 0..f {
                for dx in 0..f {
                    acc += gray.px[(y * f + dy) * gray.w + x * f + dx] as u32;
                }
            }
            px[y * w + x] = (acc / (f * f) as u32) as u8;
        }
    }
    (Gray { w, h, px }, f as f64)
}

/// When detecting on a downscaled image, K must shrink with it. `scale_back`
/// is the original/detect size ratio (>= 1).
fn scale_k(k: &M3, scale_back: f64) -> M3 {
    if scale_back == 1.0 {
        return *k;
    }
    let s = 1.0 / scale_back;
    [
        [k[0][0] * s, k[0][1] * s, (k[0][2] + 0.5) * s - 0.5],
        [k[1][0] * s, k[1][1] * s, (k[1][2] + 0.5) * s - 0.5],
        [0.0, 0.0, 1.0],
    ]
}

fn guess_k(w: usize, h: usize) -> M3 {
    let f = (w as f64 / 2.0) / (FOV_GUESS_DEG.to_radians() / 2.0).tan();
    [
        [f, 0.0, (w as f64 - 1.0) / 2.0],
        [0.0, f, (h as f64 - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// publishing
// ---------------------------------------------------------------------------

fn publish_detections(
    det_pub: &r2r::Publisher<r2r::vision_msgs::msg::Detection3DArray>,
    tf_pub: &r2r::Publisher<r2r::tf2_msgs::msg::TFMessage>,
    header: &r2r::std_msgs::msg::Header,
    dets: &[Detection],
    cfg: &Config,
    k: &M3,
) {
    use r2r::vision_msgs::msg::*;
    let mut arr = Detection3DArray::default();
    arr.header = header.clone();
    let mut tfs: Vec<r2r::geometry_msgs::msg::TransformStamped> = Vec::new();

    for d in dets {
        let label = d.decoded.as_ref().map(|(v, m, val)| format_value(v, m, val));
        let (pos, quat) = metric_pose(d, cfg.tag_radius_m);
        let px_diam = d.axes.0.max(d.axes.1);
        let cov = covariance::pose_covariance(
            px_diam,
            d.tilt_deg,
            pos[2],
            cfg.tag_radius_m,
            k[0][0],
        );

        let mut det = Detection3D::default();
        det.header = header.clone();
        if let Some(ref label) = label {
            det.id = label.clone();
            let mut hyp = ObjectHypothesisWithPose::default();
            hyp.hypothesis.class_id = label.clone();
            hyp.hypothesis.score = d.info.as_ref().map(|i| i.verify_corr).unwrap_or(0.0);
            hyp.pose.pose = pose_msg(&pos, &quat);
            hyp.pose.covariance = cov.to_vec();
            det.results.push(hyp);
        } else {
            // pose-only candidate (only present when the pose_only param is on):
            // no payload, no TF, pose still carried in the bbox center.
        }
        det.bbox.center = pose_msg(&pos, &quat);
        det.bbox.size.x = cfg.tag_radius_m * 2.0;
        det.bbox.size.y = cfg.tag_radius_m * 2.0;
        det.bbox.size.z = 1e-3;
        arr.detections.push(det);

        if cfg.publish_tf {
            if let Some(ref label) = label {
                let mut tf = r2r::geometry_msgs::msg::TransformStamped::default();
                tf.header = header.clone();
                tf.child_frame_id = format!("simittag/{}", sanitize_frame(label));
                tf.transform.translation.x = pos[0];
                tf.transform.translation.y = pos[1];
                tf.transform.translation.z = pos[2];
                tf.transform.rotation = quat_msg(&quat);
                tfs.push(tf);
            }
        }
    }

    let _ = det_pub.publish(&arr);
    if cfg.publish_tf && !tfs.is_empty() {
        let tfm = r2r::tf2_msgs::msg::TFMessage { transforms: tfs };
        let _ = tf_pub.publish(&tfm);
    }
}

/// Marker-frame pose -> metric ROS pose in the optical frame, tag frame in
/// the apriltag_ros convention (x right, y up, z toward the viewer).
fn metric_pose(d: &Detection, tag_radius_m: f64) -> ([f64; 3], [f64; 4]) {
    let pos = [
        d.t[0] * tag_radius_m,
        d.t[1] * tag_radius_m,
        d.t[2] * tag_radius_m,
    ];
    // R_pub = R_marker * diag(1,-1,-1)
    let r = &d.r;
    let m = [
        [r[0][0], -r[0][1], -r[0][2]],
        [r[1][0], -r[1][1], -r[1][2]],
        [r[2][0], -r[2][1], -r[2][2]],
    ];
    (pos, quat_from_mat(&m))
}

fn quat_from_mat(m: &M3) -> [f64; 4] {
    // returns (x, y, z, w), Shepperd's method
    let tr = m[0][0] + m[1][1] + m[2][2];
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    }
}

fn pose_msg(pos: &[f64; 3], quat: &[f64; 4]) -> r2r::geometry_msgs::msg::Pose {
    let mut p = r2r::geometry_msgs::msg::Pose::default();
    p.position.x = pos[0];
    p.position.y = pos[1];
    p.position.z = pos[2];
    p.orientation = quat_msg(quat);
    p
}

fn quat_msg(q: &[f64; 4]) -> r2r::geometry_msgs::msg::Quaternion {
    let mut m = r2r::geometry_msgs::msg::Quaternion::default();
    m.x = q[0];
    m.y = q[1];
    m.z = q[2];
    m.w = q[3];
    m
}

/// "sim48c8:id:0x2a", "sim180c88:geo:48.858370,2.294481,+330",
/// "sim96c32:tag:7:0x2a", "sim180c88:raw:c0ffee"
fn format_value(variant: &str, mode: &str, value: &Value) -> String {
    let body = match value {
        Value::Int(v) => format!("id:0x{:x}", v),
        Value::Geo { lat, lon, alt_m } => {
            format!("geo:{:.6},{:.6},{:+}", lat, lon, alt_m)
        }
        Value::Tagged { namespace, id } => format!("tag:{}:0x{:x}", namespace, id),
        Value::Bytes(b) => {
            let hex: String = b.iter().map(|v| format!("{:02x}", v)).collect();
            format!("raw:{}", hex)
        }
    };
    let _ = mode; // mode is implied by the body prefix
    format!("{}:{}", variant, body)
}

fn sanitize_frame(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// debug image
// ---------------------------------------------------------------------------

fn draw_debug(gray: &Gray, dets: &[Detection]) -> Vec<u8> {
    let (w, h) = (gray.w, gray.h);
    let mut img = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let v = gray.px[i];
        img[i * 3] = v;
        img[i * 3 + 1] = v;
        img[i * 3 + 2] = v;
    }
    let mut set = |x: f64, y: f64, c: (u8, u8, u8)| {
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (px, py) = (xi + dx, yi + dy);
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                    let i = (py as usize * w + px as usize) * 3;
                    img[i] = c.0;
                    img[i + 1] = c.1;
                    img[i + 2] = c.2;
                }
            }
        }
    };
    for d in dets {
        let color = if d.decoded.is_some() { (0, 200, 0) } else { (0, 160, 255) }; // bgr
        for s in 0..180 {
            let a = s as f64 * std::f64::consts::TAU / 180.0;
            let (x, y) = simittag_core::pose::apply_h(&d.h, a.cos(), a.sin());
            set(x, y, color);
        }
        // in-plane axes through the homography: x red, y blue
        for s in 0..40 {
            let t = s as f64 / 40.0;
            let (x, y) = simittag_core::pose::apply_h(&d.h, t, 0.0);
            set(x, y, (0, 0, 255));
            let (x, y) = simittag_core::pose::apply_h(&d.h, 0.0, t);
            set(x, y, (255, 0, 0));
        }
    }
    img
}
