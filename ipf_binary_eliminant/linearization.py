"""Linearization D_S = (I - Pi_3)(I - Pi_2)(I - Pi_1) of one IPF cycle at a fixed point S
on the tangent space V = L_1 + L_2 + L_3 (Section 5).
"""
import itertools
import numpy as np


def pi_matrix(S, ax):
    """Matrix of the S-weighted averaging operator along axis `ax`."""
    sh = S.shape
    N = S.size
    sums = S.sum(axis=ax)
    M = np.zeros((N, N), complex)
    for c in itertools.product(*[range(d) for d in sh]):
        fiber_sum = sums[tuple(x for i, x in enumerate(c) if i != ax)]
        r_idx = np.ravel_multi_index(c, sh)
        for t in range(sh[ax]):
            c2 = list(c)
            c2[ax] = t
            c2 = tuple(c2)
            M[r_idx, np.ravel_multi_index(c2, sh)] = S[c2] / fiber_sum
    return M


def orbit_basis(sh):
    """Orthonormal basis matrix for the tangent space V = L_1 + L_2 + L_3."""
    cols = []
    for ax in range(3):
        for idx in itertools.product(*[range(d) for i, d in enumerate(sh) if i != ax]):
            v = np.zeros(sh)
            s = list(idx)
            s.insert(ax, slice(None))
            v[tuple(s)] = 1
            cols.append(v.ravel())
    B = np.array(cols).T
    U, sv, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, sv > 1e-9]


def sweep_radius(S):
    """Spectral radius of D_S restricted to V for the IPF cycle (ij+, i+k, +jk)."""
    I_ = np.eye(S.size)
    # Apply Phi_1 (sum over k=2) first, then Phi_2 (sum over j=1), then Phi_3 (sum over i=0).
    D = (I_ - pi_matrix(S, 0)) @ (I_ - pi_matrix(S, 1)) @ (I_ - pi_matrix(S, 2))
    B = orbit_basis(S.shape)
    Mr = np.linalg.pinv(B) @ D @ B
    return float(np.max(np.abs(np.linalg.eigvals(Mr))))
