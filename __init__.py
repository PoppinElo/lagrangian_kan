"""
lagrangian_kan: Framework para aprendizaje y descubrimiento de Lagrangianos.

Prioridades:
1. Modularidad
2. Legibilidad
3. Didáctica
4. Optimicidad
"""

__version__ = "0.1.0"

# Imports principales
from .systems.base import LagrangianSystem
from .systems.harmonic_oscillator import HarmonicOscillator
from .systems.simple_pendulum import SimplePendulum
from .lagrangian.solver import EulerLagrangeSolver
from .lagrangian.derivatives import compute_lagrangian_derivatives
from .lagrangian.integrator import TrajectoryIntegrator
from .losses.lagrangian_losses import (
    EulerLagrangeResidualLoss,
    AccelerationRegularizer,
    MassRegularizer,
    InteractionRegularizer,
)
from .training.trainer import LagrangianTrainer
from .data.generators import generate_trajectory_data
from .utils.metrics import (
    compute_lagrangian_error,
    compute_acceleration_error,
    compute_energy,
    compute_conservation_error,
)

__all__ = [
    "__version__",
    # Systems
    "LagrangianSystem",
    "HarmonicOscillator",
    "SimplePendulum",
    # Lagrangian
    "EulerLagrangeSolver",
    "compute_lagrangian_derivatives",
    "TrajectoryIntegrator",
    # Loss (solo el principal)
    "EulerLagrangeResidualLoss",
    # Regularizadores
    "AccelerationRegularizer",
    "MassRegularizer",
    "InteractionRegularizer",
    # Training
    "LagrangianTrainer",
    # Data
    "generate_trajectory_data",
    # Métricas
    "compute_lagrangian_error",
    "compute_acceleration_error",
    "compute_energy",
    "compute_conservation_error",
]
