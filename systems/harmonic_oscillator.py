"""
Oscilador armónico: L = Σᵢ (mᵢ*q̇ᵢ²/2 - kᵢ*qᵢ²/2)

Soporta sistemas 1D, 2D, 3D, etc.
Cada dimensión puede tener su propia masa y constante del resorte.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple
from .base import LagrangianSystem


class HarmonicOscillator(LagrangianSystem):
    """
    Oscilador armónico multidimensional.
    
    Lagrangiano: L = Σᵢ (mᵢ*q̇ᵢ²/2 - kᵢ*qᵢ²/2)
    
    Ecuación de movimiento: q̈ᵢ = -(kᵢ/mᵢ) * qᵢ
    
    Solución analítica: qᵢ(t) = Aᵢ * cos(ωᵢ*t + φᵢ)
    donde ωᵢ = √(kᵢ/mᵢ)
    """
    
    def __init__(
        self, 
        n_dof: int = 1,
        m: Union[float, torch.Tensor] = 1.0,
        k: Union[float, torch.Tensor] = 1.0,
        name: str = "HarmonicOscillator"
    ):
        """
        Args:
            n_dof: Número de grados de libertad
            m: Masa (escalar o vector [n_dof])
            k: Constante del resorte (escalar o vector [n_dof])
            name: Nombre del sistema
        """
        super().__init__(n_dof=n_dof, name=name)
        
        # Convertir a tensores
        if isinstance(m, (int, float)):
            self.m = torch.tensor([m] * n_dof, dtype=torch.float32)
        else:
            self.m = torch.tensor(m, dtype=torch.float32)
            if len(self.m) != n_dof:
                raise ValueError(f"m debe tener longitud {n_dof}")
        
        if isinstance(k, (int, float)):
            self.k = torch.tensor([k] * n_dof, dtype=torch.float32)
        else:
            self.k = torch.tensor(k, dtype=torch.float32)
            if len(self.k) != n_dof:
                raise ValueError(f"k debe tener longitud {n_dof}")
        
        # Frecuencias angulares
        self.omega = torch.sqrt(self.k / self.m)  # [n_dof]
    
    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Calcula L = Σᵢ (mᵢ*q̇ᵢ²/2 - kᵢ*qᵢ²/2)
        
        Args:
            q: [batch, n_dof]
            qdot: [batch, n_dof]
            
        Returns:
            L: [batch, 1]
        """
        # Energía cinética: T = Σᵢ mᵢ*q̇ᵢ²/2
        T = 0.5 * (self.m * qdot ** 2).sum(dim=1, keepdim=True)
        
        # Energía potencial: V = Σᵢ kᵢ*qᵢ²/2
        V = 0.5 * (self.k * q ** 2).sum(dim=1, keepdim=True)
        
        # Lagrangiano: L = T - V
        return T - V
    
    def generate_trajectories(
        self,
        n_trajectories: int = 10,
        n_points_per_traj: int = 50,
        t_max: float = 10.0,
        A_range: tuple = (0.5, 1.5),
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Genera trayectorias del oscilador armónico.
        
        Solución analítica: qᵢ(t) = Aᵢ * cos(ωᵢ*t + φᵢ)
        q̇ᵢ(t) = -Aᵢ*ωᵢ * sin(ωᵢ*t + φᵢ)
        q̈ᵢ(t) = -Aᵢ*ωᵢ² * cos(ωᵢ*t + φᵢ)
        
        Args:
            n_trajectories: Número de trayectorias
            n_points_per_traj: Puntos por trayectoria
            t_max: Tiempo máximo
            A_range: Tupla (min, max) para amplitudes aleatorias
            seed: Semilla aleatoria
            
        Returns:
            dict con 'q', 'qdot', 'qddot', 'L', 'trajectory_ids'
        """
        if seed is not None:
            np.random.seed(seed)
        
        all_q = []
        all_qdot = []
        all_qddot = []
        all_L = []
        trajectory_ids = []
        
        for traj_id in range(n_trajectories):
            # Condiciones iniciales aleatorias para cada dimensión
            A = np.random.uniform(A_range[0], A_range[1], size=self.n_dof)
            phi = np.random.uniform(0, 2*np.pi, size=self.n_dof)
            
            # Tiempos uniformes
            t = np.linspace(0, t_max, n_points_per_traj)
            
            # Calcular trayectoria para cada dimensión
            q_traj = np.zeros((n_points_per_traj, self.n_dof))
            qdot_traj = np.zeros((n_points_per_traj, self.n_dof))
            qddot_traj = np.zeros((n_points_per_traj, self.n_dof))
            
            for i in range(self.n_dof):
                omega_i = self.omega[i].item()
                A_i = A[i]
                phi_i = phi[i]
                
                q_traj[:, i] = A_i * np.cos(omega_i * t + phi_i)
                qdot_traj[:, i] = -A_i * omega_i * np.sin(omega_i * t + phi_i)
                qddot_traj[:, i] = -A_i * omega_i**2 * np.cos(omega_i * t + phi_i)
            
            # Calcular Lagrangiano
            q_tensor = torch.tensor(q_traj, dtype=torch.float32)
            qdot_tensor = torch.tensor(qdot_traj, dtype=torch.float32)
            L_traj = self.lagrangian(q_tensor, qdot_tensor)
            
            # Guardar
            all_q.append(q_traj)
            all_qdot.append(qdot_traj)
            all_qddot.append(qddot_traj)
            all_L.append(L_traj.numpy())
            trajectory_ids.extend([traj_id] * n_points_per_traj)
        
        # Convertir a tensores
        return {
            'q': torch.tensor(np.vstack(all_q), dtype=torch.float32),
            'qdot': torch.tensor(np.vstack(all_qdot), dtype=torch.float32),
            'qddot': torch.tensor(np.vstack(all_qddot), dtype=torch.float32),
            'L': torch.tensor(np.vstack(all_L), dtype=torch.float32),
            'trajectory_ids': np.array(trajectory_ids)
        }
    
    def analytical_solution(
        self,
        t: np.ndarray,
        A: np.ndarray,
        phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solución analítica del oscilador armónico.
        
        Args:
            t: [n_points] - tiempos
            A: [n_dof] - amplitudes
            phi: [n_dof] - fases
            
        Returns:
            q, qdot, qddot: cada uno [n_points, n_dof]
        """
        q = np.zeros((len(t), self.n_dof))
        qdot = np.zeros((len(t), self.n_dof))
        qddot = np.zeros((len(t), self.n_dof))
        
        for i in range(self.n_dof):
            omega_i = self.omega[i].item()
            q[:, i] = A[i] * np.cos(omega_i * t + phi[i])
            qdot[:, i] = -A[i] * omega_i * np.sin(omega_i * t + phi[i])
            qddot[:, i] = -A[i] * omega_i**2 * np.cos(omega_i * t + phi[i])
        
        return q, qdot, qddot
