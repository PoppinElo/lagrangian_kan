"""
Cálculo de derivadas del Lagrangiano usando autograd.

Este módulo proporciona funciones para calcular todas las derivadas
necesarias para resolver las ecuaciones de Euler-Lagrange.
"""

import torch
import torch.nn as nn
from typing import Tuple


def compute_lagrangian_derivatives(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula todas las derivadas necesarias del Lagrangiano L(q, q̇).
    
    Calcula:
    - L: Lagrangiano
    - ∂L/∂q: Gradiente respecto a q
    - ∂L/∂q̇: Gradiente respecto a q̇
    - ∂²L/∂q̇²: Matriz Hessiana respecto a q̇ (para sistemas multidimensionales)
    - ∂²L/(∂q∂q̇): Matriz de derivadas mixtas
    
    Args:
        model: Modelo que predice L(q, q̇)
               Input: [batch, 2*n_dof] = [q, qdot] concatenados
               Output: [batch, 1] = L
        q: [batch, n_dof] - coordenadas generalizadas
        qdot: [batch, n_dof] - velocidades generalizadas
        
    Returns:
        tuple: (L, dL_dq, dL_dqdot, d2L_dqdot2, d2L_dqdqdot)
            - L: [batch, 1]
            - dL_dq: [batch, n_dof]
            - dL_dqdot: [batch, n_dof]
            - d2L_dqdot2: [batch, n_dof, n_dof] - matriz Hessiana
            - d2L_dqdqdot: [batch, n_dof, n_dof] - matriz de derivadas mixtas
    """
    batch_size, n_dof = q.shape
    
    # Asegurar que q y qdot requieren gradientes
    # Usar detach() para desconectar del grafo anterior, luego requires_grad_(True)
    q = q.clone().detach().requires_grad_(True)
    qdot = qdot.clone().detach().requires_grad_(True)
    
    # Crear input concatenando q y qdot
    x = torch.cat([q, qdot], dim=1)  # [batch, 2*n_dof]
    
    # Predecir L - asegurarse de que se ejecute en contexto con gradientes habilitados
    # Aunque los parámetros del modelo no requieren gradientes, el output debe tener grad_fn
    # porque los inputs (q, qdot) requieren gradientes
    # Usar torch.set_grad_enabled(True) para asegurar que los gradientes estén habilitados
    with torch.set_grad_enabled(True):
        L = model(x)  # [batch, 1]
    
    # Calcular ∂L/∂q
    dL_dq = torch.autograd.grad(
        outputs=L.sum(),
        inputs=q,
        create_graph=True,
        retain_graph=True
    )[0]  # [batch, n_dof]
    
    # Calcular ∂L/∂q̇
    dL_dqdot = torch.autograd.grad(
        outputs=L.sum(),
        inputs=qdot,
        create_graph=True,
        retain_graph=True
    )[0]  # [batch, n_dof]
    
    # Calcular ∂²L/∂q̇² (matriz Hessiana)
    # Para cada componente de qdot, calcular ∂(∂L/∂q̇)/∂qdot
    d2L_dqdot2 = torch.zeros(batch_size, n_dof, n_dof, device=q.device, dtype=q.dtype)
    for i in range(n_dof):
        grad_i = torch.autograd.grad(
            outputs=dL_dqdot[:, i].sum(),
            inputs=qdot,
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, n_dof]
        d2L_dqdot2[:, i, :] = grad_i
    
    # Calcular ∂²L/(∂q∂q̇) (derivadas mixtas)
    # Para cada componente de qdot, calcular ∂(∂L/∂q̇)/∂q
    d2L_dqdqdot = torch.zeros(batch_size, n_dof, n_dof, device=q.device, dtype=q.dtype)
    for i in range(n_dof):
        grad_i = torch.autograd.grad(
            outputs=dL_dqdot[:, i].sum(),
            inputs=q,
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, n_dof]
        d2L_dqdqdot[:, i, :] = grad_i
    
    return L, dL_dq, dL_dqdot, d2L_dqdot2, d2L_dqdqdot
