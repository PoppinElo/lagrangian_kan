"""Módulo para trabajar con Lagrangianos y ecuaciones de Euler-Lagrange."""

from .solver import EulerLagrangeSolver
from .derivatives import compute_lagrangian_derivatives
from .integrator import TrajectoryIntegrator

__all__ = [
    "EulerLagrangeSolver",
    "compute_lagrangian_derivatives",
    "TrajectoryIntegrator",
]
