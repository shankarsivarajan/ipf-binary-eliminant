"""Exact eliminant E(psi) for 2 x 2 x K tables via evaluation/interpolation."""
import flint


def kron(matrices):
    """Return the Kronecker product of a nonempty sequence of fmpq_mat matrices."""
    out = matrices[0]
    for M in matrices[1:]:
        r1, c1 = out.nrows(), out.ncols()
        r2, c2 = M.nrows(), M.ncols()
        Z = flint.fmpq_mat(r1 * r2, c1 * c2)
        for i in range(r1):
            for j in range(c1):
                if out[i, j] != 0:
                    for p in range(r2):
                        for q in range(c2):
                            Z[i * r2 + p, j * c2 + q] = out[i, j] * M[p, q]
        out = Z
    return out


def data(T, n):
    """Extract per-stratum margins (r_k, r'_k, c_k, rho_k) and total N = n_{11+}."""
    K = T.shape[2]
    strata = []
    for k in range(K):
        r1 = int(n[0, :, k].sum())
        r2 = int(n[1, :, k].sum())
        c1 = int(n[:, 0, k].sum())
        rho = flint.fmpq(int(T[0, 0, k] * T[1, 1, k]), int(T[0, 1, k] * T[1, 0, k]))
        strata.append((r1, r2, c1, rho))
    return strata, int(n[0, 0, :].sum())


def E_at(strata, n11, psi):
    """Evaluate the Kronecker-sum determinant E(psi) at a rational value psi."""
    As, Bs = [], []
    for r1, r2, c1, rho in strata:
        u = 1 - psi * rho
        v = r2 - c1 + psi * rho * (r1 + c1)
        w = -psi * rho * r1 * c1
        As.append(flint.fmpq_mat(2, 2, [0, -w, 1, -v]))
        Bs.append(flint.fmpq_mat(2, 2, [1, 0, 0, u]))
    K = len(strata)
    M = kron(Bs) * n11
    for k in range(K):
        M -= kron([As[j] if j == k else Bs[j] for j in range(K)])
    return M.det()


def eliminant(T, n):
    """Compute the integer polynomial E(psi) (up to clearing denominators) for a 2 x 2 x K table."""
    strata, n11 = data(T, n)
    K = len(strata)
    deg = K * 2 ** (K - 1)  # Theorem 3.2(2): exact degree of E(psi)
    xs = list(range(deg + 1))
    ys = [E_at(strata, n11, flint.fmpq(x)) for x in xs]
    if hasattr(flint.fmpq_poly, "interpolate"):
        P = flint.fmpq_poly.interpolate(xs, ys)
    else:
        P = flint.fmpq_poly(0)
        for i, xi in enumerate(xs):
            L = flint.fmpq_poly([1])
            for j, xj in enumerate(xs):
                if j != i:
                    L *= flint.fmpq_poly([-xj, 1]) / (xi - xj)
            P += L * ys[i]
    return P.numer()
