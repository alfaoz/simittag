#!/bin/sh
# Full verification: Rust build + unit tests + all six parity gates + Python
# reference self-tests. CI runs exactly this; run it locally before pushing.
set -e
cd "$(dirname "$0")"
PATH="$HOME/.cargo/bin:$PATH"

echo "== rust: build + unit tests =="
cargo build --release -p simittag-cli --manifest-path rust/Cargo.toml
cargo test --release -p simittag-core --manifest-path rust/Cargo.toml

echo "== rust: parity gates (fixtures are the contract) =="
B=rust/target/release/simittag
"$B" parity-spec fixtures/spec.json
"$B" parity-codec fixtures/codec.json
"$B" parity-geometry fixtures/geometry.json
"$B" parity-stages fixtures
"$B" parity-candidates fixtures
"$B" parity-detect fixtures

echo "== python reference self-tests =="
python3 -m simittag.gf256
python3 -m simittag.spec
python3 -m simittag.codec
python3 -m simittag.payload
python3 -m unittest discover -s tests -v

echo "ALL CHECKS PASSED"
