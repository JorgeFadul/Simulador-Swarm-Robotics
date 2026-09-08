"""Script de entrenamiento para RL Hibrido V3-DES (Descentralizado e Invariante).

Entrena una politica PPO descentralizada que funciona con cualquier numero de
robots N y cualquier tamaño de mapa R x C. Usa observacion egocentrica local,
radio de alerta por defectos y penalización por inactividad.

Uso:
    python train_rl_v3_des.py --timesteps 500000 --num-robots 10 --patch-size 15
"""

import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.core.config import Config
from src.rl.env import SwarmEnvDescentralizado


def main():
    parser = argparse.ArgumentParser(description="Entrenador RL V3-DES Descentralizado")
    parser.add_argument("--timesteps", type=int, default=300000, help="Pasos de entrenamiento total")
    parser.add_argument("--num-robots", type=int, default=10, help="Numero de robots para la simulacion de entrenamiento")
    parser.add_argument("--patch-size", type=int, default=15, help="Tamaño de la ventana local KxK")
    parser.add_argument("--radio-alerta", type=float, default=500.0, help="Radio de señal de alerta por celdas defectuosas (px)")
    parser.add_argument("--out", type=str, default="models/v3_des", help="Directorio de salida")
    parser.add_argument("--n-envs", type=int, default=0,
                        help="Numero de entornos paralelos (0 = auto: nucleos CPU)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    cfg.num_robots = args.num_robots

    n_envs = args.n_envs
    if n_envs <= 0:
        n_envs = max(1, multiprocessing.cpu_count())
    print(f" ========== Envs paralelos: {n_envs} ==========")

    def _make_env(seed):
        def _init():
            return SwarmEnvDescentralizado(
                cfg=cfg,
                seed=seed,
                patch_size=args.patch_size,
                radio_alerta_px=args.radio_alerta,
                auto_assign=True,
                headless=True
            )
        return _init

    env = DummyVecEnv([_make_env(100 + i) for i in range(n_envs)])

    print(f"=== Iniciando entrenamiento RL V3-DES (Descentralizado e Invariante) ===")
    print(f" - Espacio de observacion: {env.observation_space.shape}")
    print(f" - Espacio de accion: {env.action_space}")
    print(f" - Robots en entrenamiento: {args.num_robots}")
    print(f" - Patch size: {args.patch_size}x{args.patch_size}")
    print(f" - Radio de alerta: {args.radio_alerta} px")
    print(f" - Envs paralelos (SubprocVecEnv): {n_envs}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=None
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(args.out, "checkpoints"),
        name_prefix="ppo_v3_des"
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    final_path = os.path.join(args.out, "final_v3_des.zip")
    model.save(final_path)
    print(f"Modelo final guardado exitosamente en: {final_path}")


if __name__ == "__main__":
    main()
