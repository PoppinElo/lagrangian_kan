"""
Integrador numérico para trayectorias usando el Lagrangiano aprendido.

Soporta dos métodos:
1. Método ODE (Euler explícito): Resuelve q̈ y integra directamente
2. Método variacional puro: Discretiza el principio de acción estacionaria
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .solver import EulerLagrangeSolver


class TrajectoryIntegrator:
    """
    Integra trayectorias usando un modelo que predice L(q, q̇).
    
    Métodos disponibles:
    - 'euler': Método de Euler explícito basado en ODE (rápido, no conserva energía exactamente)
    - 'variational': Integrador variacional puro (más lento, preserva energía y estructura simpléctica)
    """
    
    def __init__(
        self, 
        solver: Optional[EulerLagrangeSolver] = None,
        variational_tol: float = 1e-8,
        variational_max_iter: int = 10
    ):
        """
        Args:
            solver: Solver de Euler-Lagrange (se crea uno nuevo si None)
            variational_tol: Tolerancia para el método variacional (default: 1e-8)
            variational_max_iter: Máximo número de iteraciones Newton-Raphson (default: 10)
        """
        self.solver = solver if solver is not None else EulerLagrangeSolver()
        self.variational_tol = variational_tol
        self.variational_max_iter = variational_max_iter
    
    def integrate(
        self,
        model: nn.Module,
        q_init: torch.Tensor,
        qdot_init: torch.Tensor,
        t_span: float,
        dt: float = 0.01,
        method: str = "euler"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integra una trayectoria desde condiciones iniciales.
        
        Args:
            model: Modelo que predice L(q, q̇)
            q_init: [batch, n_dof] o [n_dof] - posición inicial
            qdot_init: [batch, n_dof] o [n_dof] - velocidad inicial
            t_span: Tiempo total de integración
            dt: Paso de tiempo
            method: Método de integración ('euler' o 'variational')
            
        Returns:
            q_traj: [n_steps+1, ...] - trayectoria de posiciones
            qdot_traj: [n_steps+1, ...] - trayectoria de velocidades
        """
        if method == "euler":
            return self._integrate_euler(model, q_init, qdot_init, t_span, dt)
        elif method == "variational":
            return self._integrate_variational(model, q_init, qdot_init, t_span, dt)
        else:
            raise ValueError(f"Método '{method}' no implementado. Use 'euler' o 'variational'")
    
    def _integrate_euler(
        self,
        model: nn.Module,
        q_init: torch.Tensor,
        qdot_init: torch.Tensor,
        t_span: float,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integración con método de Euler explícito.
        
        q(t+dt) = q(t) + q̇(t) * dt
        q̇(t+dt) = q̇(t) + q̈(t) * dt
        """
        # Normalizar formas
        if q_init.dim() == 1:
            q_init = q_init.unsqueeze(0)  # [1, n_dof]
        if qdot_init.dim() == 1:
            qdot_init = qdot_init.unsqueeze(0)  # [1, n_dof]
        
        batch_size, n_dof = q_init.shape
        n_steps = int(t_span / dt)
        
        # Inicializar trayectorias
        q_traj = [q_init]
        qdot_traj = [qdot_init]
        
        # Estado actual
        q = q_init.clone().requires_grad_(True)
        qdot = qdot_init.clone().requires_grad_(True)
        
        for step in range(n_steps):
            # Calcular q̈ usando el modelo
            qddot = self.solver.compute_acceleration(model, q, qdot)
            
            # Integrar un paso (Euler)
            qdot_new = qdot + qddot * dt
            q_new = q + qdot_new * dt
            
            # Guardar (sin gradientes para eficiencia)
            q_traj.append(q_new.detach())
            qdot_traj.append(qdot_new.detach())
            
            # Actualizar estado (requerir gradientes para siguiente paso)
            q = q_new.detach().requires_grad_(True)
            qdot = qdot_new.detach().requires_grad_(True)
        
        # Concatenar trayectorias
        q_traj = torch.cat(q_traj, dim=0)  # [n_steps+1, batch, n_dof]
        qdot_traj = torch.cat(qdot_traj, dim=0)  # [n_steps+1, batch, n_dof]
        
        # Si era batch=1, quitar dimensión extra
        if batch_size == 1:
            q_traj = q_traj.squeeze(1)  # [n_steps+1, n_dof]
            qdot_traj = qdot_traj.squeeze(1)  # [n_steps+1, n_dof]
        
        return q_traj, qdot_traj
    
    def _discrete_lagrangian(
        self,
        model: nn.Module,
        q_k: torch.Tensor,
        q_kp1: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Calcula el Lagrangiano discreto L_d(q_k, q_{k+1}, dt).
        
        Usa la aproximación del punto medio:
        L_d(q_k, q_{k+1}, dt) ≈ dt · L((q_k + q_{k+1})/2, (q_{k+1} - q_k)/dt)
        
        Args:
            model: Modelo que predice L(q, q̇)
            q_k: [batch, n_dof] - posición en tiempo k
            q_kp1: [batch, n_dof] - posición en tiempo k+1
            dt: Paso de tiempo
            
        Returns:
            L_d: [batch, 1] - Lagrangiano discreto
        """
        # Punto medio para q
        q_mid = (q_k + q_kp1) / 2.0  # [batch, n_dof]
        
        # Velocidad discreta: q̇ ≈ (q_{k+1} - q_k) / dt
        qdot_mid = (q_kp1 - q_k) / dt  # [batch, n_dof]
        
        # Calcular L(q_mid, qdot_mid)
        x = torch.cat([q_mid, qdot_mid], dim=1)  # [batch, 2*n_dof]
        L = model(x)  # [batch, 1]
        
        # Lagrangiano discreto
        L_d = dt * L  # [batch, 1]
        
        return L_d
    
    def _discrete_euler_lagrange_residual(
        self,
        model: nn.Module,
        q_km1: torch.Tensor,
        q_k: torch.Tensor,
        q_kp1: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Calcula el residuo de la ecuación del Discrete Euler-Lagrange:
        
        R = D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1})
        
        donde D_1 es derivada respecto al primer argumento y D_2 respecto al segundo.
        
        Args:
            model: Modelo que predice L(q, q̇)
            q_km1: [batch, n_dof] - posición en tiempo k-1
            q_k: [batch, n_dof] - posición en tiempo k
            q_kp1: [batch, n_dof] - posición en tiempo k+1
            dt: Paso de tiempo
            
        Returns:
            residual: [batch, n_dof] - residuo de la ecuación DEL
        """
        batch_size, n_dof = q_k.shape
        
        # Calcular D_2 L_d(q_{k-1}, q_k): derivada respecto a q_k (segundo argumento)
        q_km1_grad = q_km1.clone().detach().requires_grad_(False)
        q_k_grad = q_k.clone().detach().requires_grad_(True)
        
        L_d_prev = self._discrete_lagrangian(model, q_km1_grad, q_k_grad, dt)
        dL_d_prev_dqk = torch.autograd.grad(
            outputs=L_d_prev.sum(),
            inputs=q_k_grad,
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, n_dof]
        
        # Calcular D_1 L_d(q_k, q_{k+1}): derivada respecto a q_k (primer argumento)
        # IMPORTANTE: q_{k+1} debe tener requires_grad=True para que el residual dependa de él
        # cuando se calcule el jacobiano respecto a q_{k+1}
        q_k_grad2 = q_k.clone().detach().requires_grad_(True)
        # q_kp1 ya debería tener requires_grad=True cuando se llama desde _solve_variational_step
        # pero lo aseguramos aquí también
        q_kp1_grad = q_kp1.clone().detach().requires_grad_(True)
        
        L_d_next = self._discrete_lagrangian(model, q_k_grad2, q_kp1_grad, dt)
        dL_d_next_dqk = torch.autograd.grad(
            outputs=L_d_next.sum(),
            inputs=q_k_grad2,
            create_graph=True,
            retain_graph=True
        )[0]  # [batch, n_dof]
        
        # Residuo: D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1})
        # Ahora el residual depende de q_{k+1} a través del grafo computacional de L_d_next
        residual = dL_d_prev_dqk + dL_d_next_dqk  # [batch, n_dof]
        
        return residual
    
    def _solve_variational_step(
        self,
        model: nn.Module,
        q_km1: torch.Tensor,
        q_k: torch.Tensor,
        dt: float,
        q_kp1_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Resuelve la ecuación del Discrete Euler-Lagrange para encontrar q_{k+1}.
        
        Ecuación: D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1}) = 0
        
        Usa Newton-Raphson para resolver la ecuación implícita.
        
        Args:
            model: Modelo que predice L(q, q̇)
            q_km1: [batch, n_dof] - posición en tiempo k-1
            q_k: [batch, n_dof] - posición en tiempo k
            dt: Paso de tiempo
            q_kp1_init: [batch, n_dof] - estimación inicial para q_{k+1} (opcional)
            
        Returns:
            q_kp1: [batch, n_dof] - posición en tiempo k+1
        """
        batch_size, n_dof = q_k.shape
        
        # Estimación inicial: usar extrapolación lineal si no se proporciona
        if q_kp1_init is None:
            # Extrapolación lineal: q_{k+1} ≈ q_k + (q_k - q_{k-1})
            q_kp1 = q_k + (q_k - q_km1)
        else:
            q_kp1 = q_kp1_init.clone()
        
        # Newton-Raphson
        for iteration in range(self.variational_max_iter):
            q_kp1 = q_kp1.clone().detach().requires_grad_(True)
            
            # Calcular residuo (necesita create_graph=True para calcular jacobiano)
            residual = self._discrete_euler_lagrange_residual(
                model, q_km1, q_k, q_kp1, dt
            )  # [batch, n_dof]
            
            # Verificar convergencia
            residual_norm = torch.norm(residual, dim=-1).max().item()
            if residual_norm < self.variational_tol:
                break
            
            # Calcular Jacobiano: ∂R/∂q_{k+1}
            # El residuo R = D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1})
            # Solo el término D_1 L_d(q_k, q_{k+1}) depende de q_{k+1}
            # Necesitamos calcular ∂(D_1 L_d(q_k, q_{k+1}))/∂q_{k+1}
            # Esto es equivalente a D_1 D_2 L_d(q_k, q_{k+1}) = ∂²L_d/(∂q_k ∂q_{k+1})
            
            # Recalcular el término que depende de q_{k+1} con q_{k+1} con gradientes
            q_k_fixed = q_k.clone().detach().requires_grad_(False)
            q_kp1_grad = q_kp1.clone().detach().requires_grad_(True)
            
            # Calcular D_1 L_d(q_k, q_{k+1}) = ∂L_d/∂q_k (con q_{k+1} variable)
            # Necesitamos q_k con gradientes para esto
            q_k_var = q_k.clone().detach().requires_grad_(True)
            L_d_next_jac_full = self._discrete_lagrangian(model, q_k_var, q_kp1_grad, dt)
            dL_d_next_dqk_jac = torch.autograd.grad(
                outputs=L_d_next_jac_full.sum(),
                inputs=q_k_var,
                create_graph=True,
                retain_graph=True
            )[0]  # [batch, n_dof] - D_1 L_d(q_k, q_{k+1})
            
            # Ahora calcular el jacobiano: ∂(D_1 L_d)/∂q_{k+1}
            jacobian = torch.zeros(batch_size, n_dof, n_dof, device=q_kp1.device, dtype=q_kp1.dtype)
            
            for i in range(n_dof):
                grad_i = torch.autograd.grad(
                    outputs=dL_d_next_dqk_jac[:, i].sum(),
                    inputs=q_kp1_grad,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=False
                )[0]  # [batch, n_dof]
                jacobian[:, i, :] = grad_i
            
            # Resolver sistema lineal: J · Δq = -R
            # Para cada muestra del batch
            q_kp1_new = q_kp1.clone()
            for b in range(batch_size):
                J_b = jacobian[b]  # [n_dof, n_dof]
                R_b = residual[b]  # [n_dof]
                
                # Agregar regularización para estabilidad
                J_reg = J_b + torch.eye(n_dof, device=J_b.device, dtype=J_b.dtype) * 1e-8
                
                # Resolver: Δq = -J^{-1} · R
                try:
                    delta_q = torch.linalg.solve(J_reg, -R_b)  # [n_dof]
                    q_kp1_new[b] = q_kp1[b].detach() + delta_q
                except:
                    # Si falla la inversión, usar paso pequeño en dirección del gradiente
                    q_kp1_new[b] = q_kp1[b].detach() - 0.1 * R_b
            
            q_kp1 = q_kp1_new.detach()
        
        return q_kp1
    
    def _integrate_variational(
        self,
        model: nn.Module,
        q_init: torch.Tensor,
        qdot_init: torch.Tensor,
        t_span: float,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integración usando método variacional puro (Discrete Euler-Lagrange).
        
        Preserva la energía y la estructura simpléctica del espacio de fases.
        Más lento que Euler pero más estable a largo plazo.
        
        Args:
            model: Modelo que predice L(q, q̇)
            q_init: [batch, n_dof] o [n_dof] - posición inicial
            qdot_init: [batch, n_dof] o [n_dof] - velocidad inicial
            t_span: Tiempo total de integración
            dt: Paso de tiempo
            
        Returns:
            q_traj: [n_steps+1, n_dof] - trayectoria de posiciones
            qdot_traj: [n_steps+1, n_dof] - trayectoria de velocidades
        """
        # Normalizar formas
        if q_init.dim() == 1:
            q_init = q_init.unsqueeze(0)  # [1, n_dof]
        if qdot_init.dim() == 1:
            qdot_init = qdot_init.unsqueeze(0)  # [1, n_dof]
        
        batch_size, n_dof = q_init.shape
        n_steps = int(t_span / dt)
        
        # Inicializar trayectorias
        q_traj = [q_init]
        qdot_traj = [qdot_init]
        
        # Para el primer paso, necesitamos q_{-1} para calcular q_1
        # Usamos extrapolación: q_{-1} ≈ q_0 - q̇_0 * dt
        q_km1 = q_init - qdot_init * dt  # [batch, n_dof]
        q_k = q_init.clone()
        
        for step in range(n_steps):
            # Resolver ecuación variacional para encontrar q_{k+1}
            q_kp1 = self._solve_variational_step(
                model, q_km1, q_k, dt
            )  # [batch, n_dof]
            
            # Calcular q̇_k usando el Lagrangiano discreto
            # q̇_k ≈ (q_{k+1} - q_k) / dt (aproximación de primer orden)
            # O mejor: usar el punto medio del Lagrangiano discreto
            qdot_k = (q_kp1 - q_k) / dt  # [batch, n_dof]
            
            # Guardar
            q_traj.append(q_kp1.detach())
            qdot_traj.append(qdot_k.detach())
            
            # Actualizar para siguiente paso
            q_km1 = q_k.detach()
            q_k = q_kp1.detach()
        
        # Concatenar trayectorias
        q_traj = torch.cat(q_traj, dim=0)  # [n_steps+1, batch, n_dof]
        qdot_traj = torch.cat(qdot_traj, dim=0)  # [n_steps+1, batch, n_dof]
        
        # Si era batch=1, quitar dimensión extra
        if batch_size == 1:
            q_traj = q_traj.squeeze(1)  # [n_steps+1, n_dof]
            qdot_traj = qdot_traj.squeeze(1)  # [n_steps+1, n_dof]
        
        return q_traj, qdot_traj
