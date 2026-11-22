# IPF Binary Eliminant

Companion code for

> Shankar Sivarajan, *An explicit determinantal formula for limits of iterative proportional fitting with a binary variable*, [hexagon:2611.00705](https://hexagonmath.org/2611.00705) (2026).

## Overview

Iterative proportional fitting (IPF) adjusts a three-way table $T$ cyclically until its three two-way margins match those of an observed table $n$. Started from the uniform seed $T = \mathbf{1}$, the limit $\hat{P}$ is the maximum likelihood estimate under the no-three-way-interaction model; for a general positive seed $T$, it is the unique positive table in the scaled model with the target margins. In general $\hat{P}$ is an algebraic function of the margins of degree greater than one, with no closed form.

This package implements the explicit determinantal descriptions of these limits when one variable is binary ($2\times J\times K$ tables):

* **Common odds ratio ($2\times 2\times K$):** The common odds ratio $\hat{\psi}$ is a root of an explicit $2^K\times 2^K$ determinant $E(\psi)$, built as a Kronecker sum of one $2\times 2$ companion pencil per stratum. For $T = \mathbf{1}$ and margins with $N \neq \sum_k r_k c_k/N_k$, $E(\psi) = (\psi - 1)^{K2^{K-1} - 2^K + 1}F(\psi)$ with $\deg F = 2^K - 1$, and the roots of $F$ are the common odds ratios of all complex solutions.
* **General binary tables ($2\times J\times K$):** Writing the margin equations in inverse row parameters $\alpha'_j = 1/\alpha_j$ and column parameters $\beta_k$ turns the fit into a multiparameter eigenvalue problem in Atkinson's sense. Restricting the commuting Atkinson operators $\Gamma_q = \Delta_{\alpha'_J}^{-1}\Delta_q$ to an explicit invariant subspace $U$ of dimension $J^{K-1}K^{J-1}$ gives a matrix pencil whose eigenvalues are the corner entries $P_{111}$ at all complex solutions. For $T = \mathbf{1}$, the pencil is singular and its finite regular eigenvalues contain the maximum likelihood eliminant $E_{\mathrm{ML}}$.
* **Fixed points ($I\times J\times K$):** Every complex solution is a fixed point of the IPF sweep. The package computes the linearization $D_S = (I - \Pi_3)(I - \Pi_2)(I - \Pi_1)$ on the tangent space $\mathcal{V}$ and its spectral radius, which decides whether a solution is linearly attracting (Section 5: $\hat P$ always is; for $2\times 2\times K$ tables and generic margins, no other real solution is).

The determinants give a structural algebraic description of the limit and all complex solutions, not a faster numerical replacement for IPF.

## Quickstart

For $2\times 2\times K$ tables, `eliminant_22k` computes the exact integer polynomial $E(\psi)$ directly and `ipf_ntwi` computes the fitted table $\hat{P}$:

```python
import numpy as np
from ipf_binary_eliminant import eliminant_22k, ipf_ntwi

# Example A.8: 2x2x2 table with uniform seed T = 1
T = np.ones((2, 2, 2), dtype=int)
n = np.stack([[[10, 1], [3, 1]], [[1, 1], [50, 10]]], axis=2)

E = eliminant_22k(T, n)         # exact fmpz_poly in psi; divisible by (psi - 1)
P_hat = ipf_ntwi(T, n)          # fitted table via IPF
psi_hat = (P_hat[0, 0, 0] * P_hat[1, 1, 0]) / (P_hat[0, 1, 0] * P_hat[1, 0, 0])
```

For general $2\times J\times K$ tables with a generic seed $T$, `corner_eliminant(T, n)` computes the monic corner-entry eliminant $E_{11}(x) \in \mathbb{Q}[x]$ via multimodular reduction and rational reconstruction. (For $T = \mathbf{1}$ the pencil is singular, so `R_mod_p` computes the regular part $\mathcal{R}(x)$ modulo a prime instead; see `mle_case` in `reproduce.py`.)

## Paper-to-code guide

| Paper | Entry point / file | What it computes |
|---|---|---|
| Section 2 | `ipf_ntwi` (`ipf.py`) | IPF for the no-three-way-interaction model |
| Theorems 3.2 and 3.5 | `eliminant_22k` (`eliminant_22k.py`) | Exact $E(\psi) \in \mathbb{Z}[\psi]$ for $2\times 2\times K$ tables |
| Section 4 | `build_exact`, `delta` (`atkinson.py`) | Atkinson operator determinants $\Delta_p$ for $2\times J\times K$ tables |
| Theorem 4.7 | `monic_mod_p` (`corner_pencil.py`), `corner_eliminant` (`corner_eliminant.py`) | Restricted corner pencil and exact $E_{11}(x) \in \mathbb{Q}[x]$ |
| Theorem 4.8 | `build_exact`, `delta` (`atkinson.py`); Example A.4 in `reproduce.py` | Joint eigenvalues outside $U$ (boundary subproblems) and their multiplicities |
| Theorem 4.9, Conjecture 4.10 | `R_mod_p` (`mle_regular_part.py`) | Regular part $\mathcal{R}(x)$ of the singular $T = \mathbf{1}$ pencil mod $p$ |
| Section 5 | `pi_matrix`, `orbit_basis`, `sweep_radius` (`linearization.py`) | Linearization $D_S$ on $\mathcal{V}$ and its spectral radius |
| Appendix A | `reproduce.py` | Verifies every numerical claim in Examples A.1–A.8 |

## Reproducing the paper

Requirements: Python 3.10+, `numpy`, `python-flint`.

```bash
pip install -e .             # or: pip install numpy python-flint
python reproduce.py          # full verification (Examples A.1-A.8)
python reproduce.py --fast   # skip Example A.6 (2x3x4)
```

Each check prints `[PASS]` or `[FAIL]` with the recomputed value. Polynomials over $\mathbb{Q}$ are computed exactly, and real-root counts use certified root isolation (Arb balls in `python-flint`).

## Related

The case of one-dimensional margins (matrix and tensor scaling) is treated in the companion repository [`sinkhorn-eliminant`](https://github.com/shankarsivarajan/sinkhorn-eliminant) and the paper

> Shankar Sivarajan, *An explicit determinantal formula for Sinkhorn limits of matrices and tensors*, [hexagon:2611.00433](https://hexagonmath.org/2611.00433) (2026).
