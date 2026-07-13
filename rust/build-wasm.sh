#!/bin/zsh
# Build both WASM variants into rust/dist/.
#
#   ./build-wasm.sh          # single-thread (dist/wasm/) + threaded (dist/wasm-mt/)
#
# THREADED BUILD NOTES (hard-won; do not simplify):
#  * needs rustup NIGHTLY with rust-src, and ~/.cargo/bin FIRST in PATH --
#    Homebrew's stable rustc shadows the shims otherwise and -Z build-std dies
#    (or worse, silently links a non-atomics std).
#  * -C target-feature=+atomics alone is NOT enough on current nightlies: rustc
#    does not auto-add the wasm-ld shared-memory args, so the module gets
#    NON-shared memory and wasm-bindgen-rayon fails at runtime with
#    "Memory could not be cloned". The link-args below force it.
#  * the TLS/heap --export args are required by wasm-bindgen's thread
#    transform ("failed to find __heap_base" without them).
#  * wasm-bindgen CLI is run directly (not wasm-pack) for the mt build so the
#    link args survive; wasm-opt is skipped for mt (fine, SIMD does the work).
#  * simittag-wasm's `parallel` feature enables wasm-bindgen-rayon with
#    no-bundler (plain ES-module serving; bundler default hangs initThreadPool).
#  * pages serving the mt build MUST be cross-origin isolated (COOP: same-origin
#    + COEP: credentialless on every response).
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

echo "=== single-thread build (dist/wasm) ==="
( cd simittag-wasm && RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --release --target web --out-dir ../dist/wasm )

echo "=== threaded build (dist/wasm-mt) ==="
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--import-memory \
-C link-arg=--max-memory=1073741824 \
-C link-arg=--export=__heap_base -C link-arg=--export=__data_end \
-C link-arg=--export=__tls_base -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__wasm_init_tls" \
rustup run nightly cargo build --release --target wasm32-unknown-unknown \
  -p simittag-wasm --features parallel -Z build-std=panic_abort,std

WB=$(ls ~/Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-*/wasm-bindgen | tail -1)
"$WB" target/wasm32-unknown-unknown/release/simittag_wasm.wasm \
  --out-dir dist/wasm-mt --target web

# verify the threaded module really has shared memory (the silent failure mode)
python3 - <<'EOF'
d = open('dist/wasm-mt/simittag_wasm_bg.wasm', 'rb').read()
i = 8
def leb(d, i):
    r = s = 0
    while True:
        b = d[i]; i += 1; r |= (b & 0x7f) << s; s += 7
        if not b & 0x80: return r, i
ok = False
while i < len(d):
    sec, i0 = leb(d, i); ln, i0 = leb(d, i0)
    if sec == 2:
        n, j = leb(d, i0)
        for _ in range(n):
            ml, j = leb(d, j); j += ml
            nl, j = leb(d, j); j += nl
            k = d[j]; j += 1
            if k == 2:
                ok = d[j] & 2 != 0
                j += 1; _, j = leb(d, j)
                if ok: break
            elif k == 0: _, j = leb(d, j)
            elif k == 1: j += 1; _, j = leb(d, j)
            elif k == 3: j += 2
        break
    i = i0 + ln
assert ok, "wasm-mt memory is NOT shared -- threaded build broken"
print("wasm-mt: shared memory OK")
EOF
echo "=== done: rust/dist/{wasm,wasm-mt} ==="
