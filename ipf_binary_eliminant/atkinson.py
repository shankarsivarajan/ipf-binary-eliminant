"""Atkinson operator determinants Delta_p for the 2 x J x K multiparameter eigenvalue problem."""
import itertools
import math
from fractions import Fraction
from functools import reduce
import numpy as np
import flint


def perm_sign(perm):
    """Return the parity (+1 or -1) of a permutation of 0..n-1."""
    p = list(perm)
    sgn = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sgn = -sgn
    return sgn


def build_exact(T, n):
    """Build integer-scaled pencil coefficient matrices A_{i,q} for W_1..W_K, V_1..V_{J-1}."""
    J, K = T.shape[1], T.shape[2]
    F = Fraction
    s = [[F(int(T[0, j, k]), int(T[1, j, k])) for k in range(K)] for j in range(J)]
    m = [[int(n[0, j, k] + n[1, j, k]) for k in range(K)] for j in range(J)]
    R = [int(n[0, j, :].sum()) for j in range(J)]
    C = [int(n[0, :, k].sum()) for k in range(K)]
    npar = J + K
    eqs = []

    for k in range(K):
        A = [[[F(0)] * J for _ in range(J)] for _ in range(npar)]
        for j in range(J):
            A[j][j][j] = -1 / s[j][k]
        for r in range(J):
            for c in range(J):
                A[J + k][r][c] = (-1 if r == c else 0) + F(m[r][k], C[k])
        eqs.append(A)

    for j in range(J - 1):
        A = [[[F(0)] * K for _ in range(K)] for _ in range(npar)]
        for r in range(K):
            A[j][r][r] = F(-1)
        for k in range(K):
            for c in range(K):
                A[J + k][k][c] = s[j][k] * ((-1 if k == c else 0) + F(m[j][k], R[j]))
        eqs.append(A)

    out = []
    for A in eqs:
        lcm_den = 1
        for M in A:
            for row in M:
                for val in row:
                    lcm_den = lcm_den * val.denominator // math.gcd(lcm_den, val.denominator)
        out.append([np.array([[int(val * lcm_den) for val in row] for row in M], dtype=object) for M in A])
    return out


def delta(eqs, p):
    """Compute the Atkinson operator determinant Delta_p with 0-based parameter index p."""
    npar = len(eqs[0])
    cols = [q for q in range(npar) if q != p]
    D = None
    for sigma in itertools.permutations(range(len(cols))):
        mats = [eqs[i][cols[sigma[i]]] for i in range(len(eqs))]
        if any(not M.any() for M in mats):
            continue
        term = perm_sign(sigma) * reduce(np.kron, mats)
        D = term if D is None else D + term
    return (-1) ** p * D


def det_pencil(A, B, log=print):
    """Integer polynomial det(B - lam A) via CRT over nmod_mat characteristic polynomials."""
    n = A.shape[0]
    Al = [[int(x) for x in row] for row in A]
    Bl = [[int(x) for x in row] for row in B]
    count = 0
    mod = 1
    prev = None
    prev_sym = None
    p = 1 << 62
    stable = 0
    sgn = 1 if n % 2 == 0 else -1

    while True:
        p -= 1
        while not flint.fmpz(p).is_prime():
            p -= 1
        Am = flint.nmod_mat(Al, p)
        Bm = flint.nmod_mat(Bl, p)
        dA = Am.det()
        if int(dA) == 0:
            continue
        cp = (Am.inv() * Bm).charpoly()  # det(lam I - A^{-1} B)
        # det(B - lam A) = det(A) * (-1)^n * charpoly(lam)
        coeffs = [(int(dA) * sgn * int(c)) % p for c in cp.coeffs()]
        count += 1
        if prev is None:
            cur = coeffs
            mod = p
        else:
            inv = pow(mod, -1, p)
            cur = [a + mod * (((b - a) * inv) % p) for a, b in zip(prev, coeffs)]
            mod *= p
        sym = [c - mod if c > mod // 2 else c for c in cur]
        if prev_sym is not None and sym == prev_sym:
            stable += 1
            if stable >= 2:
                return sym
        else:
            stable = 0
        prev = cur
        prev_sym = sym
        if count % 50 == 0:
            log(f"  {count} primes, {mod.bit_length()} bits")
