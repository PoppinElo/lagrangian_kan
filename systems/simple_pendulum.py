"""
Péndulo simple: L = (1/2) * m * l² * θ̇² - m * g * l * (1 - cos(θ))

Sistema no lineal con un grado de libertad.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple
from scipy.integrate import odeint
from .base import LagrangianSystem


class SimplePendulum(LagrangianSystem):
    """
    Péndulo simple.
    
    Lagrangiano: L = (1/2) * m * l² * θ̇² - m * g * l * (1 - cos(θ))
    
    Ecuación de movimiento: θ̈ = -(g/l) * sin(θ)
    
    Nota: No tiene solución analítica cerrada, se usa integración numérica.
    """
    
    def __init__(
        self, 
        m: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        name: str = "SimplePendulum"
    ):
        """
        Args:
            m: Masa del péndulo
            l: Longitud del péndulo
            g: Aceleración gravitacional
            name: Nombre del sistema
        """
        super().__init__(n_dof=1, name=name)
        
        self.m = torch.tensor(m, dtype=torch.float32)
        self.l = torch.tensor(l, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.float32)
        
        # Frecuencia natural para pequeñas oscilaciones: ω₀ = √(g/l)
        self.omega0 = torch.sqrt(self.g / self.l)
    
    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Calcula L = (1/2) * m * l² * θ̇² - m * g * l * (1 - cos(θ))
        
        Args:
            q: [batch, 1] - ángulo θ
            qdot: [batch, 1] - velocidad angular θ̇
            
        Returns:
            L: [batch, 1]
        """
        # Energía cinética: T = (1/2) * m * l² * θ̇²
        T = 0.5 * self.m * self.l ** 2 * qdot ** 2
        
        # Energía potencial: V = m * g * l * (1 - cos(θ))
        # Usamos (1 - cos(θ)) para que V = 0 en θ = 0
        V = self.m * self.g * self.l * (1 - torch.cos(q))
        
        # Lagrangiano: L = T - V
        return T - V
    
    def _pendulum_ode(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Ecuación diferencial del péndulo: d²θ/dt² = -(g/l) * sin(θ)
        
        Args:
            y: [θ, θ̇]
            t: tiempo
            
        Returns:
            [θ̇, θ̈]
        """
        theta, theta_dot = y
        omega0_sq = (self.g / self.l).item()
        theta_ddot = -omega0_sq * np.sin(theta)
        return [theta_dot, theta_ddot]
    
    
    def generate_trajectories(
        self,
        n_trajectories: int = 10,
        n_points_per_traj: int = 50,
        t_max: float = 10.0,
        theta_range: tuple = (-np.pi/2, np.pi/2),
        theta_dot_range: Optional[tuple] = None,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Genera trayectorias del péndulo simple usando integración numérica.
        
        Restricción: E_total = T + V ≤ m*g*l (no puede alcanzar el punto más alto)
        Esto limita el ángulo a [-π/2, π/2] y las velocidades angulares.
        
        Args:
            n_trajectories: Número de trayectorias
            n_points_per_traj: Puntos por trayectoria
            t_max: Tiempo máximo
            theta_range: Tupla (min, max) para ángulos iniciales (default: [-π/2, π/2])
            theta_dot_range: Tupla (min, max) para velocidades angulares iniciales.
                            Si None, se calcula automáticamente para respetar E ≤ m*g*l
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
        
        # Energía máxima permitida: E_max = m*g*l
        m_val = self.m.item()
        g_val = self.g.item()
        l_val = self.l.item()
        E_max = m_val * g_val * l_val
        
        # Tiempos uniformes
        t = np.linspace(0, t_max, n_points_per_traj)
        
        for traj_id in range(n_trajectories):
            # Generar condiciones iniciales que respeten E ≤ m*g*l
            max_attempts = 100
            theta0 = None
            theta_dot0 = None
            
            for attempt in range(max_attempts):
                # Generar ángulo inicial en el rango especificado
                theta0 = np.random.uniform(theta_range[0], theta_range[1])
                
                # Calcular velocidad angular máxima permitida para este ángulo
                # E = (1/2) * m * l² * θ̇² + m * g * l * (1 - cos(θ)) ≤ m * g * l
                # (1/2) * m * l² * θ̇² ≤ m * g * l * cos(θ)
                # θ̇² ≤ 2g/l * cos(θ)
                cos_theta = np.cos(theta0)
                if cos_theta <= 0:
                    continue  # Ángulo fuera de [-π/2, π/2]
                
                max_theta_dot_sq = 2 * g_val / l_val * cos_theta
                max_theta_dot = np.sqrt(max_theta_dot_sq)
                
                # Generar velocidad angular inicial
                if theta_dot_range is None:
                    # Usar rango automático basado en la restricción de energía
                    theta_dot0 = np.random.uniform(-max_theta_dot, max_theta_dot)
                else:
                    # Usar rango especificado, pero verificar que respete la restricción
                    theta_dot0 = np.random.uniform(theta_dot_range[0], theta_dot_range[1])
                    if abs(theta_dot0) > max_theta_dot:
                        continue  # No respeta la restricción, intentar de nuevo
                
                # Verificar que la energía total sea ≤ m*g*l
                T0 = 0.5 * m_val * l_val**2 * theta_dot0**2
                V0 = m_val * g_val * l_val * (1 - np.cos(theta0))
                E_total = T0 + V0
                
                if E_total <= E_max:
                    break  # Condiciones iniciales válidas
            else:
                # Si no se encontraron condiciones válidas después de max_attempts intentos,
                # usar condiciones conservadoras (poca energía)
                theta0 = np.random.uniform(theta_range[0], theta_range[1]) * 0.5
                theta_dot0 = 0.0
            
            # Integrar ecuación diferencial
            y0 = [theta0, theta_dot0]
            sol = odeint(self._pendulum_ode, y0, t)
            
            # Extraer trayectorias
            q_traj = sol[:, 0:1].copy()  # [n_points, 1]
            qdot_traj = sol[:, 1:2].copy()  # [n_points, 1]
            
            # Detectar si el ángulo cruza los límites [-π/2, π/2] y emitir warning
            import warnings
            for i in range(1, len(q_traj)):
                theta_prev = q_traj[i-1, 0]
                theta_curr = q_traj[i, 0]
                
                # Detectar cruce de π/2 (de abajo hacia arriba)
                if theta_prev <= np.pi/2 and theta_curr > np.pi/2:
                    warnings.warn(
                        f"Trayectoria {traj_id}: Ángulo cruzó el límite superior π/2 "
                        f"en t={t[i]:.3f} (θ={theta_curr:.3f} rad). "
                        f"La restricción de energía puede no estar siendo respetada.",
                        UserWarning
                    )
                # Detectar cruce de -π/2 (de arriba hacia abajo)
                elif theta_prev >= -np.pi/2 and theta_curr < -np.pi/2:
                    warnings.warn(
                        f"Trayectoria {traj_id}: Ángulo cruzó el límite inferior -π/2 "
                        f"en t={t[i]:.3f} (θ={theta_curr:.3f} rad). "
                        f"La restricción de energía puede no estar siendo respetada.",
                        UserWarning
                    )
            
            # Calcular aceleraciones usando la ecuación de movimiento
            omega0_sq = (self.g / self.l).item()
            qddot_traj = -omega0_sq * np.sin(q_traj)  # [n_points, 1]
            
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
    
    def compute_acceleration_from_lagrangian(self, q, qdot, qddot=None):
        """
        Calcula aceleración usando el Lagrangiano.
        
        Para el péndulo: θ̈ = -(g/l) * sin(θ)
        """
        from ..lagrangian.solver import EulerLagrangeSolver
        solver = EulerLagrangeSolver()
        return solver.compute_acceleration_from_lagrangian(self.lagrangian, q, qdot)
