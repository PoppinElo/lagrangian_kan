"""
Resuelve las ecuaciones de Euler-Lagrange para obtener q̈ a partir de L(q, q̇).

Ecuación de Euler-Lagrange:
    d/dt(∂L/∂q̇) = ∂L/∂q

Expandiendo la derivada temporal:
    ∂²L/∂q̇² * q̈ + ∂²L/(∂q∂q̇) * q̇ = ∂L/∂q

Para sistemas multidimensionales, esto se resuelve como un sistema lineal:
    H * q̈ = b

donde Hᵢⱼ = ∂²L/∂q̇ᵢ∂q̇ⱼ (matriz Hessiana)
y bᵢ = ∂L/∂qᵢ - Σⱼ (∂²L/∂qᵢ∂q̇ⱼ) * q̇ⱼ
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
from .derivatives import compute_lagrangian_derivatives


class EulerLagrangeSolver:
    """
    Resuelve q̈ usando el Lagrangiano L(q, q̇).
    
    Soporta sistemas multidimensionales resolviendo el sistema lineal
    H * q̈ = b donde H es la matriz Hessiana de L respecto a q̇.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Valor pequeño para estabilidad numérica al invertir H
        """
        self.epsilon = epsilon
    
    def compute_acceleration(
        self,
        model: nn.Module,
        q: torch.Tensor,
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula q̈ usando un modelo que predice L(q, q̇).
        
        Args:
            model: Modelo que predice L(q, q̇)
                   Input: [batch, 2*n_dof]
                   Output: [batch, 1]
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            
        Returns:
            qddot: [batch, n_dof] - aceleraciones
        """
        # Guardar estado original del modelo
        was_training = model.training
        model.eval()
        
        # Necesitamos habilitar gradientes para calcular derivadas del Lagrangiano
        # No necesitamos deshabilitar requires_grad en los parámetros del modelo porque
        # torch.autograd.grad solo calculará gradientes respecto a los inputs (q, qdot)
        # especificados explícitamente, no respecto a los parámetros del modelo
        
        # Calcular todas las derivadas necesarias (necesita gradientes en q y qdot)
        # compute_lagrangian_derivatives ya clona q y qdot y les pone requires_grad=True
        # Usar torch.set_grad_enabled(True) para asegurar que los gradientes estén habilitados
        # incluso si se llama desde un contexto no_grad()
        with torch.set_grad_enabled(True):
            L, dL_dq, dL_dqdot, d2L_dqdot2, d2L_dqdqdot = compute_lagrangian_derivatives(
                model, q, qdot
            )
        
        # Resolver aceleración (sin gradientes para el resultado final)
        with torch.no_grad():
            qddot = self._solve_acceleration(
                dL_dq, dL_dqdot, d2L_dqdot2, d2L_dqdqdot, qdot
            )
        
        # Restaurar estado original del modelo
        model.train(was_training)
        
        return qddot
    
    def compute_acceleration_from_lagrangian(
        self,
        lagrangian_func: Callable,
        q: torch.Tensor,
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula q̈ usando una función Lagrangiano directamente.
        
        Útil cuando tienes el Lagrangiano analítico.
        
        Args:
            lagrangian_func: Función que calcula L(q, qdot) -> [batch, 1]
            q: [batch, n_dof]
            qdot: [batch, n_dof]
            
        Returns:
            qddot: [batch, n_dof]
        """
        # Crear un modelo wrapper temporal
        class LagrangianWrapper(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func
            
            def forward(self, x):
                # x: [batch, 2*n_dof] = [q, qdot] concatenados
                n_dof = x.shape[1] // 2
                q = x[:, :n_dof]
                qdot = x[:, n_dof:]
                return self.func(q, qdot)
        
        wrapper = LagrangianWrapper(lagrangian_func)
        return self.compute_acceleration(wrapper, q, qdot)
    
    def _solve_acceleration(
        self,
        dL_dq: torch.Tensor,
        dL_dqdot: torch.Tensor,
        d2L_dqdot2: torch.Tensor,
        d2L_dqdqdot: torch.Tensor,
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Resuelve el sistema lineal H * q̈ = b.
        
        Args:
            dL_dq: [batch, n_dof] - ∂L/∂q
            dL_dqdot: [batch, n_dof] - ∂L/∂q̇
            d2L_dqdot2: [batch, n_dof, n_dof] - ∂²L/∂q̇² (matriz Hessiana)
            d2L_dqdqdot: [batch, n_dof, n_dof] - ∂²L/(∂q∂q̇)
            qdot: [batch, n_dof] - velocidades
            
        Returns:
            qddot: [batch, n_dof] - aceleraciones
        """
        batch_size, n_dof = qdot.shape
        
        # Calcular vector b: bᵢ = ∂L/∂qᵢ - Σⱼ (∂²L/∂qᵢ∂q̇ⱼ) * q̇ⱼ
        # d2L_dqdqdot[:, i, j] = ∂²L/(∂qᵢ∂q̇ⱼ)
        # Necesitamos: Σⱼ (∂²L/∂qᵢ∂q̇ⱼ) * q̇ⱼ para cada i
        mixed_term = torch.einsum('bij,bj->bi', d2L_dqdqdot, qdot)  # [batch, n_dof]
        b = dL_dq - mixed_term  # [batch, n_dof]
        
        # Resolver H * q̈ = b para cada muestra en el batch
        qddot = torch.zeros_like(qdot)
        
        for i in range(batch_size):
            H_i = d2L_dqdot2[i]  # [n_dof, n_dof]
            b_i = b[i]  # [n_dof]
            
            # Agregar epsilon a la diagonal para estabilidad
            H_i_stable = H_i + self.epsilon * torch.eye(
                n_dof, device=H_i.device, dtype=H_i.dtype
            )
            
            # Resolver sistema lineal
            try:
                qddot_i = torch.linalg.solve(H_i_stable, b_i)  # [n_dof]
            except RuntimeError:
                # Si falla, usar pseudoinversa
                qddot_i = torch.linalg.pinv(H_i_stable) @ b_i
            
            qddot[i] = qddot_i
        
        return qddot
