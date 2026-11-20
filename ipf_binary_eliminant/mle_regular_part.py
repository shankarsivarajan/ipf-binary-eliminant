"""Regular part R(x) of the singular corner pencil at T = 1 (Section 4, maximum likelihood case).

For L(x) = x (Delta_{a'_1} + s_{11} Delta_{b_1}) - m_{11} s_{11} Delta_{b_1}, R(x) mod p
is computed as the gcd of det(L(x) + rank-completing perturbation) over two random
perturbations of rank kappa = dim(H) - nrank(L).
"""
from fractions import Fraction as F
import numpy as np
import flint


def nm(A, p):
    """Convert an integer array-like matrix A to an nmod_mat modulo p."""
    return flint.nmod_mat([[int(x) % p for x in row] for row in A], p)


def R_mod_p(Ds, J, s11, m11, p, seed=3):
    """Return (dim H, normal rank, monic R(x) coeffs mod p, infinite-root multiplicity)."""
    rng = np.random.default_rng(seed)
    cs = flint.nmod(s11.numerator, p) / flint.nmod(s11.denominator, p)
    Da, Db = nm(Ds[0], p), nm(Ds[J], p)
    B = Da + Db * cs
    A = Db * (cs * flint.nmod(m11, p))  # L(x) = x B - A
    n = A.nrows()
    x0 = flint.nmod(int(rng.integers(1, p)), p)
    x1 = flint.nmod(7919, p)
    r = (B * x0 - A).rank()
    k = n - r

    polys = []
    for _ in range(2):
        U = nm(rng.integers(0, p, size=(n, k)), p)
        V = nm(rng.integers(0, p, size=(k, n)), p)
        DA = flint.nmod_mat(k, k, p)
        DB = flint.nmod_mat(k, k, p)
        for i in range(k):
            DA[i, i] = int(rng.integers(1, p))
            DB[i, i] = int(rng.integers(1, p))
        At = A + U * DA * V
        Bt = B + U * DB * V
        # Substituting y = 1/(x - x1) maps det(x Bt - At) up to scale to det(y I - C^{-1} Bt), C = At - x1 Bt.
        C = At - Bt * x1
        polys.append((C.inv() * Bt).charpoly())

    Ry = polys[0].gcd(polys[1])
    c = [int(v) for v in Ry.coeffs()]
    m0 = next(i for i, v in enumerate(c) if v)  # y^m0 corresponds to roots at x = infinity
    gz = flint.nmod_poly(c[m0:][::-1], p)       # polynomial in z = x - x1
    gx = gz(flint.nmod_poly([-int(x1), 1], p))  # polynomial in x
    gx = gx * (flint.nmod(1, p) / gx.coeffs()[-1])
    return n, r, [int(v) for v in gx.coeffs()], m0


def ratrec(a, m):
    """Rational reconstruction of a mod m via the extended Euclidean algorithm."""
    a %= m
    r0, r1, s0, s1 = m, a, 0, 1
    bound = int(flint.fmpz(m // 2).isqrt())
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return F(r1, s1) if s1 > 0 else F(-r1, -s1)
