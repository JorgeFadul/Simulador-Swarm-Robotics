"""Debug: comparar observaciones de entrenamiento vs inferencia."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np
from stable_baselines3 import PPO
from src.core.config import Config
from src.rl.env_simple import SwarmEnvSimple


def debug_observation():
    cfg = Config()
    with open("mapas/mapa_rl_train.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for k in ("grid_size", "map_width", "map_height", "num_robots", "walls"):
        if k in data:
            setattr(cfg, k, data[k])
    for k in ("defecto_num_blobs", "defecto_prob_ortogonal",
              "defecto_prob_diagonal", "defecto_decaimiento",
              "defecto_semilla_global"):
        if k in data:
            setattr(cfg, k, data[k])
    cfg.cols = cfg.map_width // cfg.grid_size
    cfg.rows = cfg.map_height // cfg.grid_size

    # Entorno para entrenar
    env_train = SwarmEnvSimple(cfg=cfg, seed=42, reward_mode="coverage",
                                max_steps=2000, headless=True, physics_steps=50)
    obs_train, _ = env_train.reset()

    # Entorno para inferencia
    env_infer = SwarmEnvSimple(cfg=cfg, seed=42, reward_mode="coverage",
                                max_steps=2000, headless=True, physics_steps=50)
    obs_infer, _ = env_infer.reset()

    print("=== DEBUG OBSERVATION ===")
    print(f"obs_train shape: {obs_train.shape}")
    print(f"obs_infer shape: {obs_infer.shape}")
    print(f"obs_train == obs_infer: {np.allclose(obs_train, obs_infer)}")

    # Diferencias
    diff = np.abs(obs_train - obs_infer)
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")

    # Detalle de la observación
    n_cells = env_train.n_cells
    print(f"\n--- obs_train[:{n_cells}] (mapa de cobertura) ---")
    print(f"  Unique values: {np.unique(obs_train[:n_cells])}")
    print(f"  Sum: {obs_train[:n_cells].sum()}")

    print(f"\n--- obs_train[{n_cells}:{n_cells+4}] (posiciones robots) ---")
    print(f"  {obs_train[n_cells:n_cells+4]}")

    print(f"\n--- obs_train[{n_cells+4}] (paso normalizado) ---")
    print(f"  {obs_train[n_cells+4]}")

    # Simular un paso
    print("\n=== Simulando 1 paso ===")
    action_train = np.array([10, 20])  # Acción de prueba
    obs_train2, r_train, _, _, _ = env_train.step(action_train)
    obs_infer2, r_infer, _, _, _ = env_infer.step(action_train)

    print(f"obs_train after step == obs_infer after step: {np.allclose(obs_train2, obs_infer2)}")
    diff2 = np.abs(obs_train2 - obs_infer2)
    print(f"Max diff after step: {diff2.max()}")

    # Cargar modelo y predecir
    model_path = "models/ppo_swarm_v2/final"
    model = PPO.load(model_path)

    action_pred, _ = model.predict(obs_train, deterministic=True)
    print(f"\nModelo predice action: {action_pred}")

    # Verificar si la acción es consistente
    action_pred2, _ = model.predict(obs_infer, deterministic=True)
    print(f"Modelo predice action (infer): {action_pred2}")

    env_train.close()
    env_infer.close()


if __name__ == "__main__":
    debug_observation()
