"""Utilidades para visualización y métricas."""

from .metrics import (
    compute_energy,
    compute_conservation_error,
    compute_lagrangian_error,
    compute_acceleration_error,
)
from .visualization import plot_trajectory, plot_phase_space

__all__ = [
    "compute_energy",
    "compute_conservation_error",
    "compute_lagrangian_error",
    "compute_acceleration_error",
    "plot_trajectory",
    "plot_phase_space",
]
