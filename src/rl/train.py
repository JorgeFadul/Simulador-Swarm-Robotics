"""Pipeline de entrenamiento RL para el Simulador Swarm 2D (Fase 4).

Uso:
    python -m src.rl.train --reward time --steps 500000
    python -m src.rl.train --reward energy --steps 1000000 --n-envs 4

Entrena un agente PPO con Stable-Baselines3 sobre el entorno SwarmEnv.
Soporta entrenamiento paralelo via SubprocVecEnv y guarda checkpoints
periodicos en `models/`.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.core.config import Config
from src.rl.env import SwarmEnv, make_env


def parse_args():
    p = argparse.ArgumentParser(description="Entrena un agente RL swarm 2D.")
    p.add_argument("--reward", choices=["time", "energy"], default="time",
                   help="Funcion de recompensa: 'time' o 'energy'.")
    p.add_argument("--steps", type=int, default=500000,
                   help="Pasos totales de entrenamiento.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Numero de entornos paralelos (SubprocVecEnv).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mapa", default=None,
                   help="Archivo YAML del escenario.")
    p.add_argument("--max-steps", type=int, default=10000,
                   help="Maximo de pasos por episodio.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Pasos por rollouts de PPO.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--model-name", default="ppo_swarm",
                   help="Nombre base para la carpeta del modelo.")
    p.add_argument("--resume", default=None,
                   help="Ruta a un modelo pre-entrenado para continuar.")
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


def train(args):
    cfg = load_cfg(args)
    os.makedirs("models", exist_ok=True)

    # --- Crear entornos ---
    env_fns = [make_env(cfg=cfg, seed=args.seed + i,
                        reward_mode=args.reward,
                        max_steps=args.max_steps,
                        headless=True)
               for i in range(args.n_envs)]

    if args.n_envs == 1:
        env = DummyVecEnv([env_fns[0]])
    else:
        env = SubprocVecEnv(env_fns)

    # --- Modelo PPO ---
    if args.resume:
        model = PPO.load(args.resume, env=env)
        print(f"Modelo cargado desde {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            seed=args.seed,
            device="auto",
            tensorboard_log="models/tb_logs",
        )

    # --- Callbacks ---
    ckpt_cb = CheckpointCallback(
        save_freq=max(10000, args.n_steps),
        save_path=f"models/{args.model_name}",
        name_prefix="ppo_swarm",
    )
    eval_env = DummyVecEnv([make_env(cfg=cfg, seed=args.seed + 9999,
                                     reward_mode=args.reward,
                                     max_steps=args.max_steps,
                                     headless=True)])
    eval_cb = EvalCallback(
        eval_env, n_eval_episodes=3,
        eval_freq=max(10000, args.n_steps),
        log_path=f"models/{args.model_name}/eval",
        deterministic=True,
    )
    callbacks = CallbackList([ckpt_cb, eval_cb])

    # --- Entrenar ---
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks)
    elapsed = time.time() - t0

    # --- Guardar modelo final ---
    final_path = f"models/{args.model_name}/final"
    model.save(final_path)
    env.close()
    eval_env.close()

    print(f"\nEntrenamiento completado en {elapsed:.1f}s.")
    print(f"Modelo guardado en: {final_path}.zip")


if __name__ == "__main__":
    train(parse_args())
