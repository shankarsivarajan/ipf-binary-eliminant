"""Exact determinantal eliminants for limits of iterative proportional fitting
of 2 x J x K tables under the no-three-way-interaction model.
"""
from .ipf import ipf_ntwi
from .eliminant_22k import eliminant as eliminant_22k
from .corner_eliminant import eliminant_T as corner_eliminant
from .atkinson import build_exact, delta
from .mle_regular_part import R_mod_p, ratrec
from .linearization import pi_matrix, orbit_basis, sweep_radius

__all__ = [
    "ipf_ntwi",
    "eliminant_22k",
    "corner_eliminant",
    "build_exact",
    "delta",
    "R_mod_p",
    "ratrec",
    "pi_matrix",
    "orbit_basis",
    "sweep_radius",
]
