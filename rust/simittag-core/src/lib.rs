//! simittag-core: dependency-free port of the Simittag pipeline, built to run
//! identically on native and wasm32-unknown-unknown. Ported module-by-module
//! from the Python reference, each gated against fixtures/ (see fixtures/README
//! and simittag-cli's `parity-*` subcommands) before the next layer goes on.

pub mod codec;
pub mod contours;
pub mod detector;
pub mod fft;
pub mod fitellipse;
pub mod frontend;
pub mod gf16;
pub mod gf256;
pub mod image;
pub mod mat;
pub mod payload;
pub mod pose;
pub mod spec;
