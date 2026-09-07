"""Inferencia RL con entorno simplificado."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import yaml
from stable_baselines3 import PPO

from src.core.config import Config
from src.rl.env_simple import SwarmEnvSimple


def run_inference():
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
                         max_steps=2000, headless=True, physics_steps=10)
    obs, _ = env.reset()
    reward_total = 0.0

    print(f"\nMapa: {cfg.cols}x{cfg.rows} = {cfg.cols*cfg.rows} celdas")
    print(f"Robots: {cfg.num_robots}")
    print(f"Celdas libres: {env._mundo.total_libres}")
    print("\nIniciando episodio de inferencia...")

    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward

        if step % 200 == 0:
            print(f"  paso {step}: cobertura={info['cobertura']:.3f} "
                  f"insp={info['inspeccionadas']}/{info['total_libres']} "
                  f"rew={reward_total:.2f}")

        if terminated or truncated:
            break

    print(f"\nEpisodio terminado en {step+1} pasos.")
    print(f"Cobertura final: {info['cobertura']:.3f}")
    print(f"Inspeccionadas: {info['inspeccionadas']}/{info['total_libres']}")
    print(f"Recompensa acumulada: {reward_total:.2f}")

    env.close()


if __name__ == "__main__":
    run_inference()
