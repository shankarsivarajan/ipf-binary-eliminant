"""Corner pencil of Theorem 4.7 for 2 x J x K tables, restricted to U = range(Gamma_*^{dim H}).

Here Gamma_q = Delta_{a'_J}^{-1} Delta_q commute, and Gamma_* = prod_{j<J} Gamma_{a'_j}
is nilpotent on boundary eigenspaces and invertible on U (the complex solutions).
For the corner entry x = P_{111} = m_{11} s_{11} b_1 / (a'_1 + s_{11} b_1),
E_{11}(x) is computed from det(x (Gamma_{a'_1} + s_{11} Gamma_{b_1})|_U - m_{11} s_{11} Gamma_{b_1}|_U)
modulo primes.
"""
import flint


def nm(A, p):
    """Convert an integer array-like matrix A to an nmod_mat modulo p."""
    return flint.nmod_mat([[int(x) % p for x in row] for row in A], p)


def _pivot_cols(W):
    """Return the pivot column indices of an nmod_mat."""
    R, rank = W.rref()
    piv = []
    r = 0
    for c in range(R.ncols()):
        if r < rank and int(R[r, c]) == 1 and all(int(R[i, c]) == 0 for i in range(R.nrows()) if i != r):
            piv.append(c)
            r += 1
    return piv


def colbasis(W):
    """Return a basis matrix for the column space of W (via pivot columns) and its rank."""
    piv = _pivot_cols(W)
    cols = [[int(W[i, c]) for c in piv] for i in range(W.nrows())]
    return flint.nmod_mat(cols, W.modulus()), len(piv)


def restrict(Aop, U, rows):
    """Express Aop|_U in the basis given by the columns of U, using pivot rows."""
    AU = Aop * U
    Ur = flint.nmod_mat([[int(U[i, j]) for j in range(U.ncols())] for i in rows], U.modulus())
    AUr = flint.nmod_mat([[int(AU[i, j]) for j in range(AU.ncols())] for i in rows], U.modulus())
    return Ur.inv() * AUr


def monic_mod_p(Ds, J, coef_m, coef_s, p):
    """Return (dim U, coeffs of monic E_{11}(x) mod p) from Atkinson matrices Ds."""
    D0 = nm(Ds[J - 1], p)
    D0i = D0.inv()
    G = [D0i * nm(D, p) for D in Ds]

    M = G[0]
    for j in range(1, J - 1):
        M = M * G[j]
    W = M
    rk = W.rank()
    while True:
        W2 = W * W
        r2 = W2.rank()
        if r2 == rk:
            break
        W, rk = W2, r2

    U, N = colbasis(W)
    rows = _pivot_cols(U.transpose())
    A = restrict(G[0] + G[J] * coef_s, U, rows)     # Gamma_{a'_1} + s_{11} Gamma_{b_1}
    B = restrict(G[J] * (coef_m * coef_s), U, rows)  # m_{11} s_{11} Gamma_{b_1}
    cp = (A.inv() * B).charpoly()                    # monic polynomial in x
    return N, [int(c) for c in cp.coeffs()]
