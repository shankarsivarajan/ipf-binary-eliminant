"""Reproduce every numerical claim of Appendix A of the paper from the data printed there.

Each check recomputes the quantity independently and compares it with the value stated in the paper.
Real-root counts use certified root isolation (Arb balls in python-flint); polynomials over Q are exact.

Usage:
    python reproduce.py          # full verification (Examples A.1-A.7)
    python reproduce.py --fast   # skip Example A.5 (2x3x4)
"""
import argparse
import itertools
import sys
import time
from fractions import Fraction as F
from math import lcm

import flint
import numpy as np

from ipf_binary_eliminant import (
    R_mod_p,
    build_exact,
    corner_eliminant,
    delta,
    eliminant_22k as eliminant_22K,
    ipf_ntwi,
    orbit_basis,
    pi_matrix,
    ratrec,
)

RESULTS = []
T0 = time.time()


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ""), flush=True)


def close(a, b, digits=10):
    return abs(a - b) <= 0.6 * 10 ** (-digits) * max(1.0, abs(b))


def to_fmpz(P):
    """Convert an fmpq_poly (or coefficient list) to a primitive integer fmpz_poly."""
    cs = P.coeffs() if hasattr(P, "coeffs") else P
    L = 1
    for c in cs:
        q = int(c.q) if hasattr(c, "q") else 1
        L = lcm(L, q)
    return flint.fmpz_poly([int(c * L) if hasattr(c, "q") else int(c) for c in cs])


def real_roots_certified(Z):
    """Certified real roots of a squarefree fmpz_poly: a root is real iff its isolating ball meets R."""
    roots = Z.complex_roots()
    real = sorted(float(z.real.mid()) for z, _ in roots if z.imag.contains(0))
    nonreal = sum(1 for z, _ in roots if not z.imag.contains(0))
    assert len(real) + nonreal == Z.degree()
    return real


def irreducible(Z):
    _, fac = Z.factor()
    return len(fac) == 1 and fac[0][1] == 1


def table22(t_slices, n_slices):
    """Convert K slices (x_{ij}) to shape-(2, 2, K) float arrays."""
    return (
        np.stack([np.array(s, dtype=float) for s in t_slices], axis=2),
        np.stack([np.array(s, dtype=float) for s in n_slices], axis=2),
    )


def common_odds_ratio(P, T):
    or_ratios = P[0, 0] * P[1, 1] / (P[0, 1] * P[1, 0]) / (T[0, 0] * T[1, 1] / (T[0, 1] * T[1, 0]))
    assert np.allclose(or_ratios, or_ratios[0])
    return float(or_ratios[0])


def main():
    parser = argparse.ArgumentParser(description="Verify Examples A.1-A.7 from the paper.")
    parser.add_argument("--fast", action="store_true", help="Skip Example A.5 (2x3x4).")
    args = parser.parse_args()

    # ------------------------------------------------------------ Example A.1
    T, n = table22(
        [[[2, 1], [1, 5]], [[1, 4], [3, 1]], [[3, 1], [2, 2]]],
        [[[12, 5], [8, 10]], [[7, 14], [11, 6]], [[9, 6], [4, 13]]],
    )
    E = eliminant_22K(T.astype(int), n.astype(int))
    Z = to_fmpz([E[i] for i in range(E.degree() + 1)])
    check("A.1: E has degree 12", Z.degree() == 12)
    check("A.1: E irreducible over Q", irreducible(Z))
    rr = real_roots_certified(Z)
    stated = [0.1734984020, 0.1779707973, 0.9717681410, 1.2197185992, 1.9477252408, 2.8146073184]
    check("A.1: real roots as stated", len(rr) == 6 and all(close(a, b) for a, b in zip(rr, stated)), f"{rr}")
    psi_hat = common_odds_ratio(ipf_ntwi(T, n), T)
    check("A.1: psi-hat = 1.2197185992", close(psi_hat, 1.2197185992), f"{psi_hat:.10f}")

    # ------------------------------------------------------------ Example A.2
    T1 = np.ones_like(T)
    E = eliminant_22K(T1.astype(int), n.astype(int))
    Z = to_fmpz([E[i] for i in range(E.degree() + 1)])
    x1 = flint.fmpz_poly([-1, 1])
    e = 0
    while Z.degree() > 0 and Z % x1 == 0:
        Z = Z // x1
        e += 1
    c = Z.content()
    Fp = Z / c if c != 0 else Z
    if Fp.coeffs()[-1] < 0:
        Fp = -Fp
    stated_F = [-16963914240, -54554310492, -75784685929, -45268267044, 5810133454, 27735604391, 17421462240, 4236232000]
    check("A.2: E = const * (psi-1)^5 * F", e == 5 and Fp.degree() == 7, f"power {e}, deg F {Fp.degree()}")
    check("A.2: F coefficients as stated", [int(v) for v in Fp.coeffs()] == stated_F)
    check("A.2: F irreducible over Q", irreducible(Fp))
    rr = real_roots_certified(Fp)
    psi_hat = common_odds_ratio(ipf_ntwi(T1, n), T1)
    check(
        "A.2: single real root = MLE 1.4210239993",
        len(rr) == 1 and close(rr[0], 1.4210239993) and close(psi_hat, 1.4210239993),
        f"{rr}, IPF {psi_hat:.10f}",
    )

    # ------------------------------------------------------------ Example A.3
    T3 = np.array([[[3, 3, 4], [5, 1, 1], [5, 5, 2]], [[2, 5, 3], [2, 5, 2], [3, 4, 3]]])
    n3 = np.array([[[1, 1, 8], [7, 8, 5], [8, 3, 5]], [[8, 2, 3], [2, 5, 9], [2, 4, 4]]])
    P11, _ = corner_eliminant(T3, n3)
    Z = to_fmpz(P11)
    check("A.3: E_11 has degree 81", Z.degree() == 81)
    check("A.3: E_11 irreducible over Q", irreducible(Z))
    rr = real_roots_certified(Z)
    m11 = int(n3[0, 0, 0] + n3[1, 0, 0])
    inside = [r for r in rr if 0 < r < m11]
    stated = [2.2887616252, 2.7354972215, 3.2838372433, 4.2591282778, 4.5510375640]
    check("A.3: 33 real roots", len(rr) == 33, f"{len(rr)}")
    check(
        "A.3: m11 = 9 and five roots in (0, 9) as stated",
        m11 == 9 and len(inside) == 5 and all(close(a, b) for a, b in zip(inside, stated)),
        f"{inside}",
    )
    x_hat = float(ipf_ntwi(T3.astype(float), n3.astype(float))[0, 0, 0])
    check("A.3: fitted P_111 = 3.2838372433", close(x_hat, 3.2838372433), f"{x_hat:.10f}")

    # ---------------------------------------------------- Examples A.4 & A.5
    def mle_case(n_arr, label, stated_R, stated_deg, stated_root):
        J, K = n_arr.shape[1], n_arr.shape[2]
        T_ones = np.ones((2, J, K), dtype=int)
        eqs = build_exact(T_ones, n_arr)
        Ds = [delta(eqs, q) for q in range(J + K)]
        m11_val = int(n_arr[0, 0, 0] + n_arr[1, 0, 0])
        p = 1 << 62
        crt = None
        mod = 1
        prev = None
        rec = None
        stable = 0
        deg = None
        npr = 0
        Rfirst = None

        def simple_part(co, mod_p):
            P = flint.nmod_poly(co, mod_p)
            out = flint.nmod_poly([1], mod_p)
            for f, mu in P.factor_squarefree()[1]:
                if mu == 1:
                    out *= f
            return [int(v) for v in out.coeffs()]

        while npr < 400:
            p -= 1
            while not flint.fmpz(p).is_prime():
                p -= 1
            try:
                _, _, co, _ = R_mod_p(Ds, J, F(1), m11_val, p)
            except ZeroDivisionError:
                continue
            if Rfirst is None:
                Rfirst = (p, co)
            sp = simple_part(co, p)
            if deg is None:
                deg = len(sp) - 1
            if len(sp) - 1 != deg:
                continue
            npr += 1
            if crt is None:
                crt, mod = sp, p
            else:
                inv = pow(mod, -1, p)
                crt = [a + mod * (((b - a) * inv) % p) for a, b in zip(crt, sp)]
                mod *= p
            rec = [ratrec(cc, mod) for cc in crt]
            if None not in rec and rec == prev:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = rec

        S = flint.fmpq_poly([flint.fmpq(c.numerator, c.denominator) for c in rec])
        x_mle = float(ipf_ntwi(T_ones.astype(float), n_arr.astype(float))[0, 0, 0])
        best = None
        for f, _ in S.factor()[1]:
            Zf = to_fmpz(flint.fmpq_poly(f))
            val = abs(sum(float(cf) * x_mle ** i for i, cf in enumerate(Zf.coeffs()))) / sum(
                abs(float(cf)) * abs(x_mle) ** i for i, cf in enumerate(Zf.coeffs())
            )
            if best is None or val < best[0]:
                best = (val, Zf)
        Eml = best[1]

        p0, co0 = Rfirst
        Rp = flint.nmod_poly(co0, p0)
        Ep = flint.nmod_poly([int(c) % p0 for c in Eml.coeffs()], p0)
        mult = 0
        Q = Rp
        while Q.degree() >= Ep.degree():
            qq, rr_ = divmod(Q, Ep)
            if rr_ != 0:
                break
            Q = qq
            mult += 1

        check(f"{label}: R has degree {stated_R}", Rp.degree() == stated_R, f"{Rp.degree()}")
        check(f"{label}: E_ML degree {stated_deg}, irreducible", Eml.degree() == stated_deg and irreducible(Eml), f"{Eml.degree()}")
        check(f"{label}: E_ML divides R with multiplicity one (mod p)", mult == 1, f"{mult}")
        rr_ml = real_roots_certified(Eml)
        check(
            f"{label}: single real root = fitted P_111 = {stated_root}",
            len(rr_ml) == 1 and close(rr_ml[0], stated_root) and close(x_mle, stated_root),
            f"{rr_ml}, IPF {x_mle:.10f}",
        )

    mle_case(n3, "A.4", 89, 31, 4.1615312803)
    if args.fast:
        print("[SKIP] A.5 (--fast mode)", flush=True)
    else:
        n5 = np.array([[[8, 3, 5, 8], [2, 3, 2, 5], [9, 2, 4, 4]], [[9, 2, 5, 3], [1, 7, 1, 3], [5, 5, 2, 9]]])
        mle_case(n5, "A.5", 477, 115, 9.9706390967)

    # ------------------------------------------------------------ Example A.6
    T6, n6 = table22(
        [[[197, 53], [98, 102]], [[23, 5], [23, 57]], [[142, 79], [85, 48]]],
        [[[132, 430], [663, 34]], [[77, 439], [472, 34]], [[107, 518], [785, 3493]]],
    )
    E = eliminant_22K(T6.astype(int), n6.astype(int))
    Z = to_fmpz([E[i] for i in range(E.degree() + 1)])
    check("A.6: E has degree 12", Z.degree() == 12)
    roots = [complex(z.real.mid(), z.imag.mid()) for z, _ in Z.complex_roots()]
    r_ = n6[0].sum(0)
    rp = n6[1].sum(0)
    c_ = n6[:, 0, :].sum(0)
    N_tot = n6[0, 0, :].sum()
    rho = T6[0, 0] * T6[1, 1] / (T6[0, 1] * T6[1, 0])

    def solution(psi):
        xs = [
            np.roots([1 - psi * rho[k], rp[k] - c_[k] + psi * rho[k] * (r_[k] + c_[k]), -psi * rho[k] * r_[k] * c_[k]])
            for k in range(3)
        ]
        best = min((abs(sum(xs[k][ch[k]] for k in range(3)) - N_tot), ch) for ch in itertools.product(range(2), repeat=3))
        x = np.array([xs[k][best[1][k]] for k in range(3)])
        S = np.zeros((2, 2, 3), complex)
        S[0, 0] = x
        S[0, 1] = r_ - x
        S[1, 0] = c_ - x
        S[1, 1] = rp - c_ + x
        return S

    def radius(S):
        I_ = np.eye(S.size)
        B = orbit_basis(S.shape)
        D = (I_ - pi_matrix(S, 0)) @ (I_ - pi_matrix(S, 1)) @ (I_ - pi_matrix(S, 2))
        return float(np.max(np.abs(np.linalg.eigvals(np.linalg.pinv(B) @ D @ B))))

    psi_hat = common_odds_ratio(ipf_ntwi(T6, n6), T6)
    check("A.6: psi-hat = 0.0103229485", close(psi_hat, 0.0103229485), f"{psi_hat:.10f}")
    pair = [
        z for z in roots
        if abs(z - complex(-0.0429312847, 0.0067384684)) < 1e-8
        or abs(z - complex(-0.0429312847, -0.0067384684)) < 1e-8
    ]
    check("A.6: conjugate pair psi = -0.0429312847 +- 0.0067384684i is a root", len(pair) == 2)
    rads = [radius(solution(z)) for z in pair]
    check("A.6: spectral radius 0.5347 at the pair", all(abs(r - 0.5347) < 5e-5 for r in rads), f"{rads}")
    check(
        "A.6: spectral radius 0.7310 at the fitted table",
        abs(radius(solution(psi_hat)) - 0.7310) < 5e-5,
        f"{radius(solution(psi_hat)):.6f}",
    )

    # ------------------------------------------------------------ Example A.7
    T7, n7 = table22([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[10, 1], [3, 1]], [[1, 1], [50, 10]]])
    E = eliminant_22K(T7.astype(int), n7.astype(int))
    Z = to_fmpz([E[i] for i in range(E.degree() + 1)])
    x1 = flint.fmpz_poly([-1, 1])
    e = 0
    while Z.degree() > 0 and Z % x1 == 0:
        Z = Z // x1
        e += 1
    c = Z.content()
    Fp = Z / c if c != 0 else Z
    if Fp.coeffs()[-1] < 0:
        Fp = -Fp
    r7 = n7[0].sum(0)
    c7 = n7[:, 0, :].sum(0)
    N7 = n7[0, 0, :].sum()
    Nk7 = n7.sum((0, 1))
    check("A.7: N = 11 differs from sum r_k c_k / N_k", N7 == 11 and abs(N7 - (r7 * c7 / Nk7).sum()) > 1e-6)
    check("A.7: E = const * (psi-1) * F, F as stated", e == 1 and [int(v) for v in Fp.coeffs()] == [-2420, -5687, 3686, 10812])
    check("A.7: F irreducible over Q", irreducible(Fp))
    rr = real_roots_certified(Fp)
    check(
        "A.7: three real roots as stated",
        len(rr) == 3 and all(close(a, b) for a, b in zip(rr, [-0.5882641171, -0.5054368472, 0.7527834652])),
        f"{rr}",
    )
    rp7 = n7[1].sum(0)
    allreal = True
    for psi in rr:
        xs = [np.roots([1 - psi, rp7[k] - c7[k] + psi * (r7[k] + c7[k]), -psi * r7[k] * c7[k]]) for k in range(2)]
        ok = any(
            abs(xs[0][a] + xs[1][b] - N7) < 1e-8 * N7
            and abs(np.imag(xs[0][a])) < 1e-12
            and abs(np.imag(xs[1][b])) < 1e-12
            for a in range(2)
            for b in range(2)
        )
        allreal &= ok
    check("A.7: each real root carries a solution with real entries", allreal)
    check("A.7: psi-hat = 0.7527834652", close(common_odds_ratio(ipf_ntwi(T7, n7), T7), 0.7527834652))

    nfail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nfail}/{len(RESULTS)} checks passed in {(time.time() - T0) / 60:.1f} min")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
