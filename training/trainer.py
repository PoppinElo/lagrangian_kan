"""
Trainer especializado para entrenar modelos que aprenden Lagrangianos.

Maneja diferentes tipos de loss y proporciona métricas físicas útiles.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Callable, List
from ..losses.lagrangian_losses import (
    EulerLagrangeResidualLoss,
    AccelerationRegularizer,
    MassRegularizer,
    InteractionRegularizer,
)


class LagrangianTrainer:
    """
    Entrena un modelo para aprender L(q, q̇).
    
    Usa EulerLagrangeResidualLoss como único loss principal.
    Soporta regularizadores opcionales:
    - AccelerationRegularizer: Regulariza q̈ predicho vs q̈ verdadero
    - MassRegularizer: Regulariza la matriz de masa
    - InteractionRegularizer: Regulariza interacciones
    """
    
    def __init__(
        self,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            optimizer: Optimizador (se crea uno por defecto si None)
            device: Dispositivo (CPU o CUDA)
        """
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cpu')
    
    def train(
        self,
        model: nn.Module,
        train_data: Dict[str, torch.Tensor],
        num_epochs: int = 1000,
        lr: float = 0.001,
        print_every: int = 100,
        # Regularizadores opcionales (si weight=0 o None, no se usa)
        acceleration_weight: Optional[float] = None,
        mass_epsilon: float = 0.01,
        mass_weight: Optional[float] = None,
        interaction_weight: Optional[float] = None,
        **kwargs
    ) -> Dict[str, list]:
        """
        Entrena el modelo usando EulerLagrangeResidualLoss como loss principal.
        
        Los regularizadores se activan automáticamente si su peso es > 0.
        Si weight es None o 0, ese regularizador no se calcula (eficiencia).
        
        Args:
            model: Modelo que predice L(q, q̇)
            train_data: dict con keys: 'q', 'qdot', 'qddot' (requerido)
            num_epochs: Número de épocas
            lr: Learning rate
            print_every: Imprimir métricas cada N épocas
            acceleration_weight: Peso del AccelerationRegularizer (default: None, no se usa)
            mass_epsilon: Umbral mínimo para tr(M) (solo si mass_weight > 0)
            mass_weight: Peso del MassRegularizer (default: None, no se usa)
            interaction_weight: Peso del InteractionRegularizer (default: None, no se usa)
            **kwargs: Argumentos adicionales para el optimizador
            
        Returns:
            history: dict con historial de pérdidas
        """
        model = model.to(self.device)
        
        # Mover datos al dispositivo
        q = train_data['q'].to(self.device)
        qdot = train_data['qdot'].to(self.device)
        qddot = train_data.get('qddot', None)
        if qddot is None:
            raise ValueError("train_data debe contener 'qddot'")
        qddot = qddot.to(self.device)
        
        # Filtrar argumentos de regularizadores de kwargs (no deben ir al optimizador)
        # Estos argumentos están explícitos en la firma, pero los filtramos por seguridad
        regularizer_args = {
            'acceleration_weight', 'mass_epsilon', 'mass_weight', 
            'interaction_weight'
        }
        optimizer_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in regularizer_args}
        
        # Crear optimizador si no existe
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
        
        # Loss principal: Euler-Lagrange residual
        main_loss = EulerLagrangeResidualLoss()
        
        # Crear regularizadores solo si sus pesos son > 0
        regularizers = {}
        if acceleration_weight is not None and acceleration_weight > 0:
            regularizers['acceleration'] = AccelerationRegularizer(weight=acceleration_weight)
        if mass_weight is not None and mass_weight > 0:
            regularizers['mass'] = MassRegularizer(epsilon=mass_epsilon, weight=mass_weight)
        if interaction_weight is not None and interaction_weight > 0:
            regularizers['interaction'] = InteractionRegularizer(weight=interaction_weight)
        
        # Historial
        history = {'loss': [], 'main_loss': []}
        for reg_name in regularizers.keys():
            history[f'reg_{reg_name}'] = []
        
        # Loop de entrenamiento
        model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Loss principal
            main_loss_value = main_loss(model, q, qdot, qddot)
            total_loss = main_loss_value
            
            # Agregar regularizadores (solo si existen)
            reg_values = {}
            if 'acceleration' in regularizers:
                reg_values['acceleration'] = regularizers['acceleration'](model, q, qdot, qddot)
                total_loss = total_loss + reg_values['acceleration']
            if 'mass' in regularizers:
                reg_values['mass'] = regularizers['mass'](model, q, qdot)
                total_loss = total_loss + reg_values['mass']
            if 'interaction' in regularizers:
                reg_values['interaction'] = regularizers['interaction'](model, q, qdot)
                total_loss = total_loss + reg_values['interaction']
            
            total_loss.backward()
            self.optimizer.step()
            
            # Guardar historial
            history['loss'].append(total_loss.item())
            history['main_loss'].append(main_loss_value.item())
            for reg_name, reg_value in reg_values.items():
                history[f'reg_{reg_name}'].append(reg_value.item())
            
            if (epoch + 1) % print_every == 0 or epoch == 0:
                reg_str = ""
                if reg_values:
                    reg_str = " | " + " | ".join([f"{name}: {v.item():.6f}" 
                                                  for name, v in reg_values.items()])
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss.item():.6f} | "
                      f"Main: {main_loss_value.item():.6f}{reg_str}")
        
        return history
    
    def evaluate(
        self,
        model: nn.Module,
        test_data: Dict[str, torch.Tensor],
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de test.
        
        Args:
            model: Modelo entrenado
            test_data: dict con datos de test
            metrics: Lista de métricas a calcular
            
        Returns:
            metrics_dict: dict con valores de métricas
        """
        model = model.to(self.device)
        model.eval()
        
        # Mover datos al dispositivo
        q = test_data['q'].to(self.device)
        qdot = test_data['qdot'].to(self.device)
        qddot_true = test_data.get('qddot', None)
        if qddot_true is not None:
            qddot_true = qddot_true.to(self.device)
        L_true = test_data.get('L', None)
        if L_true is not None:
            L_true = L_true.to(self.device)
        
        metrics_dict = {}
        
        with torch.no_grad():
            # Métrica 1: Error en L (si está disponible) - usar función de métrica
            if L_true is not None:
                from ..utils.metrics import compute_lagrangian_error
                X = torch.cat([q, qdot], dim=1)
                L_error = compute_lagrangian_error(model, X, L_true)
                metrics_dict['L_MSE'] = L_error
            
            # Métrica 2: Error en q̈ (si está disponible) - usar función de métrica
            if qddot_true is not None:
                from ..utils.metrics import compute_acceleration_error
                accel_errors = compute_acceleration_error(model, q, qdot, qddot_true)
                metrics_dict['qddot_MSE'] = accel_errors['MSE']
                metrics_dict['qddot_MAE'] = accel_errors['MAE']
        
        return metrics_dict
