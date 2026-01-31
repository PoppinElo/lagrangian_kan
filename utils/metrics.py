"""
Métricas físicas para evaluar modelos Lagrangianos.

Incluye métricas como conservación de energía, momento, etc.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def compute_energy(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    system: Optional[object] = None
) -> torch.Tensor:
    """
    Calcula la energía total del sistema.
    
    Para sistemas conservativos: E = T + V
    donde T es energía cinética y V es energía potencial.
    
    Si el Lagrangiano es L = T - V, entonces:
    E = T + V = (L + 2*V) o se puede calcular directamente.
    
    Args:
        model: Modelo que predice L(q, q̇)
        q: [batch, n_dof] - posiciones
        qdot: [batch, n_dof] - velocidades
        system: Sistema físico (opcional, para cálculo exacto)
        
    Returns:
        E: [batch, 1] - energía total
    """
    # Por ahora, retornamos el Lagrangiano como aproximación
    # (esto debería mejorarse con cálculo explícito de T y V)
    X = torch.cat([q, qdot], dim=1)
    L = model(X)
    
    # Nota: Para sistemas conservativos, E debería calcularse
    # explícitamente como T + V, no solo L
    # Esto es una aproximación simple
    return L


def compute_conservation_error(
    energies: torch.Tensor,
    relative: bool = True
) -> float:
    """
    Calcula el error en la conservación de energía.
    
    Para sistemas conservativos, la energía debería ser constante.
    Este error mide cuánto varía la energía a lo largo de la trayectoria.
    
    Args:
        energies: [n_points] - energía en cada punto
        relative: Si True, retorna error relativo
        
    Returns:
        error: Escalar - error de conservación
    """
    if relative:
        mean_energy = energies.mean()
        if abs(mean_energy) < 1e-10:
            return torch.std(energies).item()
        return (torch.std(energies) / abs(mean_energy)).item()
    else:
        return torch.std(energies).item()


def compute_lagrangian_error(
    model: nn.Module,
    X: torch.Tensor,
    L_true: torch.Tensor,
    criterion: Optional[nn.Module] = None
) -> float:
    """
    Métrica: Error entre L predicho y L verdadero.
    
    Esta es una métrica de monitoreo, NO un loss de entrenamiento.
    Útil cuando el Lagrangiano verdadero es conocido.
    
    Args:
        model: Modelo que predice L(q, q̇)
        X: [batch, 2*n_dof] - input concatenado [q, qdot]
        L_true: [batch, 1] - Lagrangiano verdadero
        criterion: Función de pérdida (default: MSELoss)
        
    Returns:
        error: Escalar - error en L
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    with torch.no_grad():
        L_pred = model(X)
        error = criterion(L_pred, L_true).item()
    
    return error


def compute_acceleration_error(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    qddot_true: torch.Tensor,
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Métrica: Error entre q̈ predicho y q̈ verdadero.
    
    Args:
        model: Modelo que predice L(q, q̇)
        q: [batch, n_dof] - posiciones
        qdot: [batch, n_dof] - velocidades
        qddot_true: [batch, n_dof] - aceleraciones verdaderas
        criterion: Función de pérdida (default: MSELoss)
        
    Returns:
        dict con 'MSE' y 'MAE'
    """
    from ..lagrangian.solver import EulerLagrangeSolver
    
    if criterion is None:
        criterion = nn.MSELoss()
    
    solver = EulerLagrangeSolver()
    
    # Guardar estado original del modelo
    was_training = model.training
    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    
    # Necesitamos habilitar gradientes para calcular derivadas del Lagrangiano
    # pero no queremos que fluyan hacia los parámetros del modelo
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Calcular aceleración (necesita gradientes en q y qdot)
    # compute_lagrangian_derivatives ya clona q y qdot y les pone requires_grad=True
    with torch.enable_grad():
        qddot_pred = solver.compute_acceleration(model, q, qdot)
    
    # Calcular métricas sin gradientes
    with torch.no_grad():
        mse = criterion(qddot_pred, qddot_true).item()
        mae = nn.L1Loss()(qddot_pred, qddot_true).item()
    
    # Restaurar estado original del modelo
    model.train(was_training)
    for name, param in model.named_parameters():
        param.requires_grad = original_requires_grad[name]
    
    return {'MSE': mse, 'MAE': mae}
