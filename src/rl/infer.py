"""Script de inferencia RL para el Simulador Swarm 2D (Fase 4).

Carga un modelo PPO entrenado y lo ejecuta en el simulador con UI.

Uso:
    python -m src.rl.infer --model models/ppo_swarm/final
    python -m src.rl.infer --model models/ppo_swarm/final --mapa mapas/mapa_test_1.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pygame

from stable_baselines3 import PPO

from src.core.config import Config
from src.core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED
from src.core.simulacion import Simulacion
from src.robots.robot import Robot
from src.rl.env import SwarmEnv


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia RL en el simulador 2D.")
    p.add_argument("--model", required=True,
                   help="Ruta al modelo PPO entrenado (sin .zip).")
    p.add_argument("--mapa", default=None,
                   help="Archivo YAML del escenario.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--reward", choices=["time", "energy"], default="time")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def load_cfg(args):
    cfg = Config()
    if args.mapa:
        import yaml
        with open(args.mapa, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k in ("grid_size", "map_width", "map_height", "num_robots", "walls"):
            if k in data:
                setattr(cfg, k, data[k])
        if data.get("defecto_blobs"):
            cfg.defecto_blobs = data["defecto_blobs"]
            cfg.prob_defecto = 0.0
        cfg.cols = cfg.map_width // cfg.grid_size
        cfg.rows = cfg.map_height // cfg.grid_size
    return cfg


def run_inference(args):
    cfg = load_cfg(args)
    model = PPO.load(args.model)
    print(f"Modelo cargado desde {args.model}")

    env = SwarmEnv(cfg=cfg, seed=args.seed,
                   reward_mode=args.reward,
                   max_steps=args.max_steps,
                   headless=True)
    obs, _ = env.reset()
    reward_total = 0.0

    print("Iniciando episodio de inferencia...")
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward

        if step % 100 == 0:
            print(f"  paso {step}: cobertura={info['cobertura']:.3f} "
                  f"insp={info['inspeccionadas']}/{info['total_libres']} "
                  f"energia={info['energia_total']:.2f}J "
                  f"rew={reward_total:.2f}")

        if terminated or truncated:
            break

    print(f"\nEpisodio terminado en {step+1} pasos.")
    print(f"Cobertura final: {info['cobertura']:.3f}")
    print(f"Inspeccionadas: {info['inspeccionadas']}/{info['total_libres']}")
    print(f"Energia total: {info['energia_total']:.2f}J")
    print(f"Recompensa acumulada: {reward_total:.2f}")

    env.close()

    # Si no es headless, lanzar el simulador visual con la politica RL
    if not args.headless:
        print("\nLanzando simulador visual con politica RL...")
        run_visual(args, cfg, model)


def run_visual(args, cfg, model):
    """Ejecuta el simulador visual controlado por la politica RL."""
    import pygame

    sim = Simulacion(cfg, seed=args.seed, headless=False)
    sim.poblar_robots(cfg.num_robots)
    sim.iniciar()

    env = SwarmEnv(cfg=cfg, seed=args.seed,
                   reward_mode=args.reward,
                   max_steps=args.max_steps,
                   headless=True)
    obs, _ = env.reset()

    reloj = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if sim.estado == "running":
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Aplicar la accion RL al simulador visual
            m = sim.mundo
            gs = cfg.grid_size
            for i, rb in enumerate(sim.mundo.robots):
                if i >= len(action):
                    break
                act = int(action[i])
                if act >= m.rows * m.cols:
                    continue
                r, c = divmod(act, m.cols)
                if m.grid[r, c] == 0 and not m.inspec[r, c]:
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    if rb.necesita_objetivo and not rb.cola_tareas:
                        rb.asignar_objetivo(cx, cy, m)

            sim.step()

        sim.render()
        reloj.tick(60)

        if terminated or truncated or sim.estado == "completado":
            print(f"Inferencia visual completada: {info}")
            running = False

    pygame.quit()


if __name__ == "__main__":
    run_inference(parse_args())
