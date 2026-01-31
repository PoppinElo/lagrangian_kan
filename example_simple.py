"""
Ejemplo simple de uso del framework lagrangian_kan.

Este ejemplo muestra cómo:
1. Crear un sistema físico (oscilador armónico)
2. Generar datos de entrenamiento
3. Crear y entrenar un modelo KAN
4. Evaluar el modelo aprendido
"""

import torch
import sys
sys.path.append('/home/colmanok/Code/mini_kan')

from mini_kan import KAN, KANLayer
from mini_kan.basis import MonomialBasis

from lagrangian_kan import (
    HarmonicOscillator,
    LagrangianTrainer,
    EulerLagrangeSolver,
    TrajectoryIntegrator,
)


def main():
    print("=" * 70)
    print("EJEMPLO: Aprendizaje de Lagrangiano - Oscilador Armónico")
    print("=" * 70)
    
    # 1. Crear sistema físico
    print("\n1. Creando sistema físico (oscilador armónico 1D)...")
    system = HarmonicOscillator(n_dof=1, m=1.0, k=1.0)
    print(f"   Sistema: {system}")
    
    # 2. Generar datos de entrenamiento
    print("\n2. Generando datos de entrenamiento...")
    train_data = system.generate_trajectories(
        n_trajectories=50,
        n_points_per_traj=50,
        t_max=10.0,
        seed=42
    )
    print(f"   Datos generados:")
    print(f"     - q: {train_data['q'].shape}")
    print(f"     - qdot: {train_data['qdot'].shape}")
    print(f"     - qddot: {train_data['qddot'].shape}")
    print(f"     - L: {train_data['L'].shape}")
    
    # 3. Crear modelo KAN
    print("\n3. Creando modelo KAN...")
    basis = MonomialBasis(order=3)
    kan_layers = [
        KANLayer(in_dim=2, out_dim=5, basis=basis, bias=True),
        KANLayer(in_dim=5, out_dim=1, basis=basis, bias=True),
    ]
    model = KAN(layers=kan_layers)
    print(f"   Modelo creado: {sum(p.numel() for p in model.parameters())} parámetros")
    
    # 4. Entrenar modelo
    print("\n4. Entrenando modelo...")
    trainer = LagrangianTrainer()
    history = trainer.train(
        model=model,
        train_data=train_data,
        loss_type="euler_lagrange",
        num_epochs=500,
        lr=0.001,
        print_every=100
    )
    
    # 5. Evaluar modelo
    print("\n5. Evaluando modelo...")
    test_data = system.generate_trajectories(
        n_trajectories=10,
        n_points_per_traj=50,
        t_max=10.0,
        seed=123
    )
    metrics = trainer.evaluate(model, test_data)
    print("   Métricas:")
    for key, value in metrics.items():
        print(f"     - {key}: {value:.6f}")
    
    # 6. Integrar trayectoria usando el modelo aprendido
    print("\n6. Integrando trayectoria usando el Lagrangiano aprendido...")
    integrator = TrajectoryIntegrator()
    q_init = torch.tensor([[1.0]])
    qdot_init = torch.tensor([[0.0]])
    
    q_traj, qdot_traj = integrator.integrate(
        model=model,
        q_init=q_init,
        qdot_init=qdot_init,
        t_span=10.0,
        dt=0.01
    )
    print(f"   Trayectoria integrada: {q_traj.shape}")
    
    print("\nEjemplo completado exitosamente!")
    print("=" * 70)


if __name__ == "__main__":
    main()
