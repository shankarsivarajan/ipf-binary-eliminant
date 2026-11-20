"""Exact corner-entry eliminant E_{11}(x) of Theorem 4.7 over Q via CRT and rational reconstruction."""
from fractions import Fraction as F
from math import lcm
import flint
from .atkinson import build_exact, delta
from .corner_pencil import monic_mod_p
from .mle_regular_part import ratrec


def eliminant_T(T, n, maxp=600):
    """Compute the monic eliminant E_{11}(x) in Q[x] for seed table T and counts n."""
    J, K = T.shape[1], T.shape[2]
    eqs = build_exact(T, n)
    Ds = [delta(eqs, q) for q in range(J + K)]
    s11 = F(int(T[0, 0, 0]), int(T[1, 0, 0]))
    m11 = int(n[0, 0, 0] + n[1, 0, 0])

    p = 1 << 62
    crt = None
    mod = 1
    prev = None
    rec = None
    stable = 0
    npr = 0
    N = None

    while npr < maxp:
        p -= 1
        while not flint.fmpz(p).is_prime():
            p -= 1
        try:
            cs = flint.nmod(s11.numerator, p) / flint.nmod(s11.denominator, p)
            Nn, co = monic_mod_p(Ds, J, flint.nmod(m11, p), cs, p)
        except Exception:
            continue
        if N is None:
            N = Nn
        if Nn != N:
            continue
        npr += 1
        if crt is None:
            crt, mod = co, p
        else:
            inv = pow(mod, -1, p)
            crt = [a + mod * (((b - a) * inv) % p) for a, b in zip(crt, co)]
            mod *= p
        rec = [ratrec(c, mod) for c in crt]
        if None not in rec and rec == prev:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        prev = rec

    if rec is None or None in rec:
        raise RuntimeError(f"Rational reconstruction did not stabilize within {maxp} primes")
    return flint.fmpq_poly([flint.fmpq(c.numerator, c.denominator) for c in rec]), npr


def roots_sorted(P):
    """Return complex root approximations of an fmpq_poly sorted by (real, imag)."""
    L = 1
    for c in P.coeffs():
        L = lcm(L, int(c.q))
    Z = flint.fmpz_poly([int(c * L) for c in P.coeffs()])
    raw = Z.complex_roots()
    roots = [
        complex(r.real.mid(), r.imag.mid()) if hasattr(r, "real") else complex(r[0].real.mid(), r[0].imag.mid())
        for r in raw
    ]
    return sorted(roots, key=lambda z: (z.real, z.imag))
