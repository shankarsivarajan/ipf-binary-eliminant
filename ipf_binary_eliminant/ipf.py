"""Iterative proportional fitting for the no-three-way-interaction (NTWI) model."""
import numpy as np


def margins2(n):
    """Return the three two-way margins (n_{ij+}, n_{i+k}, n_{+jk}) of a 3-way array n."""
    return n.sum(axis=2), n.sum(axis=1), n.sum(axis=0)


def ipf_ntwi(T, n, iters=20000, tol=1e-15):
    """Fit the NTWI model with seed T to the two-way margins of n by cyclic rescaling."""
    m12, m13, m23 = margins2(n)
    P = np.array(T, dtype=float, copy=True)
    for _ in range(iters):
        P *= (m12 / P.sum(axis=2))[:, :, None]
        P *= (m13 / P.sum(axis=1))[:, None, :]
        P *= (m23 / P.sum(axis=0))[None, :, :]
        err = max(abs(P.sum(axis=2) - m12).max(), abs(P.sum(axis=1) - m13).max())
        if err < tol:
            break
    return P
