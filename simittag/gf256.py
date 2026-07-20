"""
Reed-Solomon over GF(256) with erasure support.

Field: primitive polynomial 0x11D (x^8+x^4+x^3+x^2+1), generator alpha = 2,
first consecutive root exponent fcr = 0. Decoder is the standard
Berlekamp-Massey + Chien + Forney pipeline extended for erasures (the well-known
"Reed-Solomon for coders" formulation). The __main__ self-test injects errors AND
erasures and confirms recovery, so we don't fool ourselves about the ECC.

Polynomial convention: lists with index 0 = highest-degree coefficient.
"""
from __future__ import annotations

PRIM = 0x11D
GEN = 2

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIM
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def gmul(a, b):
    return 0 if (a == 0 or b == 0) else _EXP[_LOG[a] + _LOG[b]]


def gdiv(a, b):
    if b == 0:
        raise ZeroDivisionError
    return 0 if a == 0 else _EXP[(_LOG[a] - _LOG[b]) % 255]


def gpow(a, n):
    return _EXP[(_LOG[a] * n) % 255]


def ginv(a):
    return _EXP[(255 - _LOG[a]) % 255]


def poly_scale(p, x):
    return [gmul(c, x) for c in p]


def poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi:
            for j, qj in enumerate(q):
                r[i + j] ^= gmul(pi, qj)
    return r


def poly_eval(p, x):
    y = 0
    for c in p:
        y = gmul(y, x) ^ c
    return y


def poly_div(dividend, divisor):
    out = list(dividend)
    for i in range(len(dividend) - (len(divisor) - 1)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    out[i + j] ^= gmul(divisor[j], coef)
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]


def _generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [1, gpow(GEN, i)])
    return g


def rs_encode(data, nsym):
    """Systematic RS: data + nsym parity bytes."""
    gen = _generator_poly(nsym)
    _, remainder = poly_div(list(data) + [0] * nsym, gen)
    return bytes(data) + bytes(remainder)


def _calc_syndromes(msg, nsym):
    return [0] + [poly_eval(msg, gpow(GEN, i)) for i in range(nsym)]


def _errata_locator(e_pos):
    loc = [1]
    for i in e_pos:
        loc = poly_mul(loc, poly_add([1], [gpow(GEN, i), 0]))
    return loc


def _error_evaluator(synd, err_loc, nsym):
    _, rem = poly_div(poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return rem


def _find_error_locator(synd, nsym, erase_count=0):
    err_loc = [1]
    old_loc = [1]
    synd_shift = len(synd) - nsym if len(synd) > nsym else 0
    for i in range(nsym - erase_count):
        K = i + synd_shift
        delta = synd[K]
        for j in range(1, len(err_loc)):
            delta ^= gmul(err_loc[-(j + 1)], synd[K - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = poly_scale(old_loc, delta)
                old_loc = poly_scale(err_loc, ginv(delta))
                err_loc = new_loc
            err_loc = poly_add(err_loc, poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    errs = len(err_loc) - 1
    if (errs - erase_count) * 2 + erase_count > nsym:
        raise ValueError("too many errors to correct")
    return err_loc


def _find_errors(err_loc, nmess):
    errs = len(err_loc) - 1
    pos = [nmess - 1 - i for i in range(nmess)
           if poly_eval(err_loc, gpow(GEN, i)) == 0]
    if len(pos) != errs:
        raise ValueError("could not locate errors")
    return pos


def _forney_syndromes(synd, pos, nmess):
    fsynd = list(synd[1:])
    for p in pos:
        x = gpow(GEN, nmess - 1 - p)
        for j in range(len(fsynd) - 1):
            fsynd[j] = gmul(fsynd[j], x) ^ fsynd[j + 1]
    return fsynd


def _correct_errata(msg, synd, err_pos):
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    err_loc = _errata_locator(coef_pos)
    err_eval = _error_evaluator(synd[::-1], err_loc, len(err_loc) - 1)[::-1]
    X = [gpow(GEN, -(255 - p)) for p in coef_pos]
    E = [0] * len(msg)
    for i, Xi in enumerate(X):
        Xi_inv = ginv(Xi)
        prime = 1
        for j in range(len(X)):
            if j != i:
                prime = gmul(prime, 1 ^ gmul(Xi_inv, X[j]))
        y = poly_eval(err_eval[::-1], Xi_inv)
        y = gmul(Xi, y)
        E[err_pos[i]] = gdiv(y, prime)
    return poly_add(msg, E)


def rs_decode(msg, nsym, erase_pos=None, max_errors=None):
    """
    Correct a received codeword. erase_pos = list of byte indices known-bad.
    max_errors: if set, raise when more than this many *blind* (non-erasure) symbol
    errors had to be corrected. Sync-less variants pass max_errors=0 so a garbage grid
    can't be "corrected" into a valid-looking codeword (false-accept guard); erasure
    correction is still allowed.
    Returns (data_bytes, full_codeword). Raises on uncorrectable input.
    """
    msg = list(msg)
    erase_pos = list(erase_pos or [])
    for e in erase_pos:
        msg[e] = 0
    if len(erase_pos) > nsym:
        raise ValueError("too many erasures")
    synd = _calc_syndromes(msg, nsym)
    if max(synd) == 0:
        return bytes(msg[:-nsym]), bytes(msg)
    fsynd = _forney_syndromes(synd, erase_pos, len(msg))
    err_loc = _find_error_locator(fsynd, nsym, erase_count=len(erase_pos))
    err_pos = _find_errors(err_loc[::-1], len(msg))
    if max_errors is not None and len(err_pos) > max_errors:
        raise ValueError(f"{len(err_pos)} blind errors > max_errors={max_errors}")
    msg = _correct_errata(msg, synd, erase_pos + err_pos)
    if max(_calc_syndromes(msg, nsym)) != 0:
        raise ValueError("decode failed (residual syndrome)")
    return bytes(msg[:-nsym]), bytes(msg)


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


if __name__ == "__main__":
    import random
    random.seed(0)
    nsym = 4
    ok_err = ok_era = 0
    trials = 3000
    for _ in range(trials):
        data = bytes(random.randint(0, 255) for _ in range(5))
        code = rs_encode(data, nsym)
        # case A: up to t=nsym/2 random errors (positions unknown)
        c = bytearray(code)
        for p in random.sample(range(len(c)), random.randint(0, nsym // 2)):
            c[p] ^= random.randint(1, 255)
        try:
            ok_err += (rs_decode(bytes(c), nsym)[0] == data)
        except Exception:
            pass
        # case B: up to nsym erasures (positions known)
        c = bytearray(code)
        epos = random.sample(range(len(c)), random.randint(0, nsym))
        for p in epos:
            c[p] ^= random.randint(1, 255)
        try:
            ok_era += (rs_decode(bytes(c), nsym, erase_pos=epos)[0] == data)
        except Exception:
            pass
    print(f"RS(9,5) self-test over {trials} trials:")
    print(f"  errors  (<= {nsym // 2}): {ok_err}/{trials} recovered")
    print(f"  erasures(<= {nsym}): {ok_era}/{trials} recovered")
    print(f"  crc8('simit') = {crc8(b'simit'):#04x}")
    assert ok_err == trials and ok_era == trials, "gf256 self-test FAILED"
