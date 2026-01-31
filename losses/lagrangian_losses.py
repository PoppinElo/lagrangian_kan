"""
Funciones de pérdida y regularizadores para entrenar modelos que aprenden Lagrangianos.

Estructura:
- EulerLagrangeResidualLoss: Único loss principal de entrenamiento
- Regularizadores opcionales:
  - AccelerationRegularizer: Regulariza q̈ predicho vs q̈ verdadero
  - MassRegularizer: Regulariza la matriz de masa M = ∂²L/∂q̇²
  - InteractionRegularizer: Regulariza interacciones ∂²L/(∂q∂q̇)
"""

import torch
import torch.nn as nn
from typing import Optional
from ..lagrangian.derivatives import compute_lagrangian_derivatives
from ..lagrangian.solver import EulerLagrangeSolver


class EulerLagrangeResidualLoss(nn.Module):
    """
    Loss principal basado en el residuo de Euler-Lagrange.
    
    Ecuación de Euler-Lagrange: d/dt(∂L/∂q̇) = ∂L/∂q
    
    Expandiendo: ∂²L/∂q̇² * q̈ + ∂²L/(∂q∂q̇) * q̇ = ∂L/∂q
    
    Residuo: R = ∂²L/∂q̇² * q̈ + ∂²L/(∂q∂q̇) * q̇ - ∂L/∂q
    
    Loss: ||R||²
    
    Este es el único loss real de entrenamiento. No requiere conocer L verdadero,
    solo requiere tener trayectorias (q, q̇, q̈).
    """
    
    def __init__(self, criterion: Optional[nn.Module] = None):
        """
        Args:
            criterion: Función de pérdida (default: MSELoss)
        """
        super().__init__()
        self.criterion = criterion if criterion is not None else nn.MSELoss()
    
    def forward(
        self,
        model: nn.Module,
        q: torch.Tensor,
        qdot: torch.Tensor,
        qddot_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model: Modelo que predice L(q, q̇)
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            qddot_true: [batch, n_dof] - aceleraciones verdaderas
            
        Returns:
            loss: Escalar
        """
        # Calcular derivadas del Lagrangiano
        L, dL_dq, dL_dqdot, d2L_dqdot2, d2L_dqdqdot = compute_lagrangian_derivatives(
            model, q, qdot
        )
        
        # Calcular residuo de Euler-Lagrange
        # R = ∂²L/∂q̇² * q̈ + ∂²L/(∂q∂q̇) * q̇ - ∂L/∂q
        
        # Término 1: ∂²L/∂q̇² * q̈
        # d2L_dqdot2: [batch, n_dof, n_dof]
        # qddot_true: [batch, n_dof]
        term1 = torch.einsum('bij,bj->bi', d2L_dqdot2, qddot_true)  # [batch, n_dof]
        
        # Término 2: ∂²L/(∂q∂q̇) * q̇
        # d2L_dqdqdot: [batch, n_dof, n_dof]
        # qdot: [batch, n_dof]
        term2 = torch.einsum('bij,bj->bi', d2L_dqdqdot, qdot)  # [batch, n_dof]
        
        # Residuo
        residual = term1 + term2 - dL_dq  # [batch, n_dof]
        
        # Loss: ||R||²
        return self.criterion(residual, torch.zeros_like(residual))


class AccelerationRegularizer(nn.Module):
    """
    Regularizador opcional: ||q̈_pred - q̈_true||²
    
    Usa el Lagrangiano aprendido para predecir q̈ y compara con q̈ verdadero.
    Puede ayudar a estabilizar el entrenamiento.
    
    Args:
        weight: Peso del regularizador (default: 0.1)
    """
    
    def __init__(self, weight: float = 0.1, criterion: Optional[nn.Module] = None):
        """
        Args:
            weight: Peso del regularizador
            criterion: Función de pérdida (default: MSELoss)
        """
        super().__init__()
        self.weight = weight
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.solver = EulerLagrangeSolver()
    
    def forward(
        self,
        model: nn.Module,
        q: torch.Tensor,
        qdot: torch.Tensor,
        qddot_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model: Modelo que predice L(q, q̇)
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            qddot_true: [batch, n_dof] - aceleraciones verdaderas
            
        Returns:
            regularization_term: Escalar (ya multiplicado por weight)
        """
        qddot_pred = self.solver.compute_acceleration(model, q, qdot)
        loss = self.criterion(qddot_pred, qddot_true)
        return self.weight * loss


class MassRegularizer(nn.Module):
    """
    Regularizador de masa: E[ReLU(ε - tr(M))]
    
    Donde M(q, q̇) = ∂²L/∂q̇² es la matriz de masa (Hessiana respecto a q̇).
    
    Este regularizador asegura que la traza de la matriz de masa sea positiva
    y mayor que un umbral ε, lo cual es físicamente necesario para que el sistema
    tenga masa efectiva positiva.
    
    Fórmula: L_mass = E[ReLU(ε - tr(M))]
    
    Args:
        epsilon: Umbral mínimo para la traza de M (default: 0.01)
        weight: Peso del regularizador (default: 0.1)
    """
    
    def __init__(self, epsilon: float = 0.01, weight: float = 0.1):
        """
        Args:
            epsilon: Umbral mínimo para tr(M)
            weight: Peso del regularizador
        """
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
    
    def forward(
        self,
        model: nn.Module,
        q: torch.Tensor,
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model: Modelo que predice L(q, q̇)
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            
        Returns:
            regularization_term: Escalar (ya multiplicado por weight)
        """
        _, _, _, d2L_dqdot2, _ = compute_lagrangian_derivatives(model, q, qdot)
        
        # Calcular traza de M: tr(M) = Σᵢ Mᵢᵢ
        trace_M = torch.diagonal(d2L_dqdot2, dim1=-2, dim2=-1).sum(dim=-1)  # [batch]
        
        # ReLU(ε - tr(M)) penaliza cuando tr(M) < ε
        penalty = torch.relu(self.epsilon - trace_M).mean()
        
        return self.weight * penalty


class InteractionRegularizer(nn.Module):
    """
    Regularizador de interacciones: ||∂²L/(∂q∂q̇)||²
    
    Este regularizador penaliza las interacciones entre coordenadas y velocidades,
    promoviendo Lagrangianos más simples y mejor estructurados.
    
    Fórmula: L_cross = ||∂²L/(∂q∂q̇)||²
    
    Args:
        weight: Peso del regularizador (default: 0.01)
    """
    
    def __init__(self, weight: float = 0.01):
        """
        Args:
            weight: Peso del regularizador
        """
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        model: nn.Module,
        q: torch.Tensor,
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model: Modelo que predice L(q, q̇)
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            
        Returns:
            regularization_term: Escalar (ya multiplicado por weight)
        """
        _, _, _, _, d2L_dqdqdot = compute_lagrangian_derivatives(model, q, qdot)
        
        # Norma al cuadrado (Frobenius): ||A||²_F = Σᵢⱼ Aᵢⱼ²
        norm_squared = (d2L_dqdqdot ** 2).sum(dim=(-2, -1)).mean()
        
        return self.weight * norm_squared
