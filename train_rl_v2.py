"""Entrenamiento RL con entorno simplificado.
Configuración optimizada para aprendizaje rápido.

Uso:
    python train_rl_v2.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.core.config import Config
from src.rl.env_simple import SwarmEnvSimple, make_env_simple


def train_v2():
    # Cargar config del mapa pequeño
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

    print(f"Mapa: {cfg.cols}x{cfg.rows} = {cfg.cols*cfg.rows} celdas")
    print(f"Robots: {cfg.num_robots}")
    print(f"Modo: {cfg.modo_fase}")

    os.makedirs("models", exist_ok=True)

    # Entorno simplificado
    def make_monitored_env(seed=42, reward_mode="coverage"):
        def _init():
            env = SwarmEnvSimple(cfg=cfg, seed=seed, reward_mode=reward_mode,
                                 max_steps=2000, headless=True, physics_steps=50)
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_monitored_env(42, "coverage")])
    eval_env = DummyVecEnv([make_monitored_env(43, "coverage")])

    # Modelo PPO con más exploración
    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Más exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
        device="auto",
        tensorboard_log=None,
    )

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=10000,
        save_path="models/ppo_swarm_v2",
        name_prefix="ppo_swarm",
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=3,
        eval_freq=10000,
        log_path="models/ppo_swarm_v2/eval",
        deterministic=True,
    )
    callbacks = CallbackList([ckpt_cb, eval_cb])

    # Entrenar 100K pasos
    t0 = time.time()
    print("\n--- Iniciando entrenamiento v2 (100K pasos) ---\n")
    model.learn(total_timesteps=100000, callback=callbacks)
    elapsed = time.time() - t0

    # Guardar modelo final
    final_path = "models/ppo_swarm_v2/final"
    model.save(final_path)
    env.close()
    eval_env.close()

    print(f"\nEntrenamiento completado en {elapsed:.1f}s.")
    print(f"Modelo guardado en: {final_path}.zip")


if __name__ == "__main__":
    train_v2()
