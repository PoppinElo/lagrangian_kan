"""
Clase base para sistemas físicos descritos por un Lagrangiano.

Todos los sistemas físicos deben heredar de esta clase e implementar
el método lagrangian() y generate_trajectories().
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class LagrangianSystem(ABC):
    """
    Sistema físico con Lagrangiano L(q, q̇).
    
    Soporta múltiples dimensiones:
    - q: [batch, n_dof] donde n_dof = número de grados de libertad
    - q̇: [batch, n_dof]
    - L: [batch, 1] (escalar)
    
    El Lagrangiano puede ser:
    1. Conocido analíticamente (para generar datos de entrenamiento)
    2. Desconocido (para descubrimiento desde trayectorias)
    """
    
    def __init__(self, n_dof: int = 1, name: str = "LagrangianSystem"):
        """
        Args:
            n_dof: Número de grados de libertad (dimensiones)
            name: Nombre del sistema físico
        """
        self.n_dof = n_dof
        self.name = name
    
    @abstractmethod
    def lagrangian(
        self, 
        q: torch.Tensor, 
        qdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula el Lagrangiano L(q, q̇).
        
        Para sistemas conservativos: L = T - V
        donde T es la energía cinética y V es la energía potencial.
        
        Args:
            q: [batch, n_dof] - coordenadas generalizadas
            qdot: [batch, n_dof] - velocidades generalizadas
            
        Returns:
            L: [batch, 1] - Lagrangiano (escalar)
        """
        pass
    
    @abstractmethod
    def generate_trajectories(
        self, 
        n_trajectories: int,
        n_points_per_traj: int = 50,
        t_max: float = 10.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Genera trayectorias del sistema para entrenamiento.
        
        Args:
            n_trajectories: Número de trayectorias diferentes
            n_points_per_traj: Puntos por trayectoria
            t_max: Tiempo máximo por trayectoria
            **kwargs: Parámetros específicos del sistema
            
        Returns:
            dict con las siguientes keys:
                - 'q': [n_points, n_dof] - posiciones
                - 'qdot': [n_points, n_dof] - velocidades
                - 'qddot': [n_points, n_dof] - aceleraciones
                - 'L': [n_points, 1] - Lagrangiano
                - 'trajectory_ids': [n_points] - ID de trayectoria
        """
        pass
    
    def compute_acceleration_from_lagrangian(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        qddot: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula q̈ usando las ecuaciones de Euler-Lagrange.
        
        Ecuación: d/dt(∂L/∂q̇) = ∂L/∂q
        
        Para sistemas multidimensionales, esto se resuelve como:
        H * q̈ = b
        
        donde Hᵢⱼ = ∂²L/∂q̇ᵢ∂q̇ⱼ (matriz Hessiana)
        y bᵢ = ∂L/∂qᵢ - Σⱼ (∂²L/∂qᵢ∂q̇ⱼ) * q̇ⱼ
        
        Args:
            q: [batch, n_dof] - posiciones
            qdot: [batch, n_dof] - velocidades
            qddot: [batch, n_dof] - aceleraciones (opcional, se calculan si None)
            
        Returns:
            qddot: [batch, n_dof] - aceleraciones calculadas
        """
        # Import local para evitar circular
        from ..lagrangian.solver import EulerLagrangeSolver
        
        solver = EulerLagrangeSolver()
        return solver.compute_acceleration_from_lagrangian(
            self.lagrangian, q, qdot
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_dof={self.n_dof}, name='{self.name}')"
