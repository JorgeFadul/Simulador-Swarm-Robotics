"""Entrenamiento RL rapido para el Simulador Swarm 2D.
Configuracion optimizada para CPU: mapa pequeno, physics_steps=5, 50K pasos.

Uso:
    python train_rl_rapido.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.core.config import Config
from src.rl.env import SwarmEnv, make_env


def train_fast():
    # Cargar config del mapa pequeno
    cfg = Config()
    import yaml
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

    print(f"Mapa: {cfg.cols}x{cfg.rows} = {cfg.cols*cfg.rows} celdas")
    print(f"Robots: {cfg.num_robots}")
    print(f"Modo: {cfg.modo_fase}")

    os.makedirs("models", exist_ok=True)

    # Entorno con physics_steps=5 para velocidad
    def make_monitored_env(seed=42):
        def _init():
            env = SwarmEnv(cfg=cfg, seed=seed, reward_mode="time",
                           max_steps=2000, headless=True, physics_steps=5)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_monitored_env(42)])
    eval_env = DummyVecEnv([make_monitored_env(43)])

    # Modelo PPO con hiperparametros para entrenamiento rapido
    model = PPO(
        "MlpPolicy", env,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        verbose=1,
        seed=42,
        device="auto",
        tensorboard_log=None,
    )

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=5000,
        save_path="models/ppo_swarm_fast",
        name_prefix="ppo_swarm",
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=2,
        eval_freq=5000,
        log_path="models/ppo_swarm_fast/eval",
        deterministic=True,
    )
    callbacks = CallbackList([ckpt_cb, eval_cb])

    # Entrenar 200K pasos
    t0 = time.time()
    print("\n--- Iniciando entrenamiento (200K pasos) ---\n")
    model.learn(total_timesteps=200000, callback=callbacks)
    elapsed = time.time() - t0

    # Guardar modelo final
    final_path = "models/ppo_swarm_fast/final"
    model.save(final_path)
    env.close()
    eval_env.close()

    print(f"\nEntrenamiento completado en {elapsed:.1f}s.")
    print(f"Modelo guardado en: {final_path}.zip")


if __name__ == "__main__":
    train_fast()
