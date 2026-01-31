"""
Visualizaciones para sistemas Lagrangianos.

Funciones para visualizar trayectorias, espacio de fase, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_trajectory(
    q_traj: torch.Tensor,
    qdot_traj: Optional[torch.Tensor] = None,
    t: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
    title: str = "Trayectoria"
):
    """
    Visualiza trayectorias del sistema.
    
    Args:
        q_traj: [n_points, n_dof] - posiciones
        qdot_traj: [n_points, n_dof] - velocidades (opcional)
        t: [n_points] - tiempos (opcional)
        labels: Lista de labels para cada dimensión
        title: Título del gráfico
    """
    q_traj_np = q_traj.detach().cpu().numpy()
    n_points, n_dof = q_traj_np.shape
    
    if t is None:
        t = np.arange(n_points)
    
    if labels is None:
        labels = [f"q_{i}" for i in range(n_dof)]
    
    fig, axes = plt.subplots(n_dof, 1, figsize=(10, 3*n_dof), sharex=True)
    if n_dof == 1:
        axes = [axes]
    
    for i in range(n_dof):
        axes[i].plot(t, q_traj_np[:, i], label=labels[i])
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel("Tiempo")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_phase_space(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_true: Optional[torch.Tensor] = None,
    qdot_true: Optional[torch.Tensor] = None,
    labels: Optional[list] = None,
    title: str = "Espacio de Fase"
):
    """
    Visualiza espacio de fase (q vs q̇).
    
    Args:
        q: [n_points, n_dof] - posiciones predichas
        qdot: [n_points, n_dof] - velocidades predichas
        q_true: [n_points, n_dof] - posiciones verdaderas (opcional)
        qdot_true: [n_points, n_dof] - velocidades verdaderas (opcional)
        labels: Lista de labels
        title: Título del gráfico
    """
    q_np = q.detach().cpu().numpy()
    qdot_np = qdot.detach().cpu().numpy()
    n_dof = q_np.shape[1]
    
    if labels is None:
        labels = [f"q_{i}" for i in range(n_dof)]
    
    # Para sistemas 1D, un solo plot
    if n_dof == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(q_np[:, 0], qdot_np[:, 0], 'b-', label='Predicho', alpha=0.7)
        if q_true is not None and qdot_true is not None:
            q_true_np = q_true.detach().cpu().numpy()
            qdot_true_np = qdot_true.detach().cpu().numpy()
            ax.plot(q_true_np[:, 0], qdot_true_np[:, 0], 'r--', label='Verdadero', alpha=0.7)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(f"q̇_{0}")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    # Para sistemas multidimensionales, subplots
    n_cols = min(2, n_dof)
    n_rows = (n_dof + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_dof == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n_dof):
        ax = axes[i]
        ax.plot(q_np[:, i], qdot_np[:, i], 'b-', label='Predicho', alpha=0.7)
        if q_true is not None and qdot_true is not None:
            q_true_np = q_true.detach().cpu().numpy()
            qdot_true_np = qdot_true.detach().cpu().numpy()
            ax.plot(q_true_np[:, i], qdot_true_np[:, i], 'r--', label='Verdadero', alpha=0.7)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(f"q̇_{i}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig
