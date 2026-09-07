"""Inferencia RL con debug detallado."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np
from stable_baselines3 import PPO

from src.core.config import Config
from src.rl.env_simple import SwarmEnvSimple


def run_inference_debug():
    # Cargar config
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

    # Cargar modelo
    model_path = "models/ppo_swarm_v2/final"
    model = PPO.load(model_path)
    print(f"Modelo cargado desde {model_path}")

    # Crear entorno
    env = SwarmEnvSimple(cfg=cfg, seed=42, reward_mode="coverage",
                         max_steps=2000, headless=True, physics_steps=50)
    obs, _ = env.reset()

    print(f"\nMapa: {cfg.cols}x{cfg.rows} = {cfg.cols*cfg.rows} celdas")
    print(f"Robots: {cfg.num_robots}")
    print(f"Celdas libres: {env._mundo.total_libres}")
    print(f"Posiciones robots: {[(rb.x, rb.y) for rb in env._robots]}")

    # Verificar grid
    m = env._mundo
    print(f"\nGrid (0=libre, 1=pared):")
    for r in range(m.rows):
        row_str = ""
        for c in range(m.cols):
            row_str += str(m.grid[r, c])
        print(f"  {row_str}")

    print(f"\nIniciando episodio de inferencia...")
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if step < 10 or step % 50 == 0:
            print(f"  paso {step}: action={action} reward={reward:.4f} "
                  f"insp={info['inspeccionadas']}/{info['total_libres']} "
                  f"robots_pos={[(round(rb.x,1), round(rb.y,1)) for rb in env._robots]}")

        if terminated or truncated:
            break

    print(f"\nEpisodio terminado en {step+1} pasos.")
    print(f"Cobertura final: {info['cobertura']:.3f}")
    print(f"Inspeccionadas: {info['inspeccionadas']}/{info['total_libres']}")

    env.close()


if __name__ == "__main__":
    run_inference_debug()
