"""Script de inferencia RL para el Simulador Swarm 2D (Fase 4).

Uso:
    # Inferencia headless de un episodio (metricas impresas)
    python -m src.rl.infer --model models/ppo_swarm/final --map-known

    # Inferencia con UI (simulador visual controlado por la politica RL)
    python -m src.rl.infer --model models/ppo_swarm/final --no-headless

    # Benchmark de multiples episodios + export CSV (para curva de aprendizaje)
    python -m src.rl.infer --model models/ppo_swarm/final --benchmark \
        --n-episodes 3 --seeds 7,42,123 --out results/rl_benchmark.csv

Carga un modelo PPO entrenado y lo ejecuta en el simulador.
"""

import argparse
import sys
import os
import csv
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

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
    p.add_argument("--max-steps", type=int, default=None,
                   help="Maximo de pasos por episodio.")
    p.add_argument("--reward", choices=["time", "energy"], default="time")
    p.add_argument("--headless", dest="headless", action="store_true",
                   default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Lanzar UI visual.")

    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--map-known", dest="map_known", action="store_true",
                      default=True)
    modo.add_argument("--map-unknown", dest="map_known", action="store_false",
                      help="Agente con mapa desconocido.")

    p.add_argument("--no-auto-assign", dest="auto_assign", action="store_false",
                   default=True,
                   help="Desactivar el fallback de auto-asignacion de celdas.")

    # Benchmark
    p.add_argument("--benchmark", action="store_true",
                   help="Modo benchmark: multiples episodios + CSV.")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--seeds", default="7,42,123",
                   help="Lista de seeds separadas por coma.")
    p.add_argument("--out", default=None,
                   help="Ruta del CSV de salida (benchmark).")
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
        for k in ("defecto_num_blobs", "defecto_prob_ortogonal",
                  "defecto_prob_diagonal", "defecto_decaimiento",
                  "defecto_semilla_global"):
            if k in data:
                setattr(cfg, k, data[k])
        cfg.cols = cfg.map_width // cfg.grid_size
        cfg.rows = cfg.map_height // cfg.grid_size
    return cfg


def run_episode(model, cfg, args, seed, max_steps=None, collect_track=False):
    """Ejecuta un episodio y devuelve metricas finales (y trayectoria opcional)."""
    if max_steps is None:
        max_steps = 4 * cfg.cols * cfg.rows if cfg.cols else 10000
    if args.max_steps:
        max_steps = args.max_steps

    env = SwarmEnv(cfg=cfg, seed=seed,
                   reward_mode=args.reward,
                   max_steps=max_steps,
                   headless=True,
                   map_known=args.map_known,
                   auto_assign=args.auto_assign)
    obs, _ = env.reset()
    reward_total = 0.0
    track = [] if collect_track else None

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        if collect_track:
            track.append({
                "paso": step,
                "cobertura": info["cobertura"],
                "inspeccionadas": info["inspeccionadas"],
                "energia": info["energia_total"],
                "colisiones_rr": info["colisiones_rr"],
            })
        if terminated or truncated:
            break

    env.close()
    return {
        "seed": seed,
        "pasos": step + 1,
        "max_steps": max_steps,
        "inspeccionadas": info["inspeccionadas"],
        "total_libres": info["total_libres"],
        "cobertura": info["cobertura"],
        "energia_total": info["energia_total"],
        "recompensa": reward_total,
        "colisiones_rr": info["colisiones_rr"],
    }, track


def run_benchmark(model, cfg, args):
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No hay seeds validas en --seeds")
    seeds = seeds[:args.n_episodes]

    os.makedirs("results", exist_ok=True)

    rows = []
    for seed in seeds:
        est, _ = run_episode(model, cfg, args, seed)
        print(f"  seed {seed}: cobertura={est['cobertura']:.3f} "
              f"pasos={est['pasos']} energia={est['energia_total']:.2f}J "
              f"rew={est['recompensa']:.2f}")
        rows.append(est)

    # Agregar columna de variante
    for r in rows:
        r["variante"] = "RL-known" if args.map_known else "RL-unknown"

    out = args.out or "results/rl_benchmark.csv"
    fieldnames = ["variante", "seed", "pasos", "max_steps", "inspeccionadas",
                  "total_libres", "cobertura", "energia_total", "recompensa",
                  "colisiones_rr"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nBenchmark CSV guardado en: {out}")

    # Promedios
    avg = {k: np.mean([r[k] for r in rows]) for k in
           ("pasos", "cobertura", "energia_total", "recompensa")}
    print(f"Promedio: cobertura={avg['cobertura']:.3f} "
          f"pasos={avg['pasos']:.1f} energia={avg['energia_total']:.2f}J "
          f"rew={avg['recompensa']:.2f}")
    return rows, avg


def run_inference(args):
    cfg = load_cfg(args)
    model = PPO.load(args.model)
    print(f"Modelo cargado desde {args.model}")

    if args.benchmark:
        run_benchmark(model, cfg, args)
        return

    est, _ = run_episode(model, cfg, args, args.seed,
                         collect_track=not args.headless)
    print(f"\nEpisodio terminado en {est['pasos']} pasos.")
    print(f"Cobertura final: {est['cobertura']:.3f}")
    print(f"Inspeccionadas: {est['inspeccionadas']}/{est['total_libres']}")
    print(f"Energia total: {est['energia_total']:.2f}J")
    print(f"Recompensa acumulada: {est['recompensa']:.2f}")

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
                   max_steps=args.max_steps or (4 * cfg.cols * cfg.rows),
                   headless=True,
                   map_known=args.map_known,
                   auto_assign=args.auto_assign)
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
