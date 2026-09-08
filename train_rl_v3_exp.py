"""Script de entrenamiento para RL de Exploracion Pura Descentralizada (V3-EXP).

Objetivo: Descubrir el mapa (SCOUTED y paredes DISCOVERED) en el menor tiempo y gasto
energetico, con coordinacion cooperativa entre vecinos (señal de actividad).

Uso:
    python train_rl_v3_exp.py --timesteps 300000 --num-robots 10 --patch-size 15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.core.config import Config
from src.rl.env import SwarmEnvExploracionDescentralizado


def main():
    parser = argparse.ArgumentParser(description="Entrenador RL V3-EXP (Exploracion Pura Descentralizada)")
    parser.add_argument("--timesteps", type=int, default=300000, help="Pasos de entrenamiento total")
    parser.add_argument("--num-robots", type=int, default=10, help="Numero de robots para entrenamiento")
    parser.add_argument("--patch-size", type=int, default=15, help="Tamaño de la ventana local KxK")
    parser.add_argument("--radio-com", type=float, default=400.0, help="Radio de comunicacion entre robots (px)")
    parser.add_argument("--reward-mode", type=str, default="energy", choices=["time", "energy"], help="Modo de penalizacion")
    parser.add_argument("--out", type=str, default="models/v3_exp", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    cfg.num_robots = args.num_robots

    env = SwarmEnvExploracionDescentralizado(
        cfg=cfg,
        patch_size=args.patch_size,
        radio_comunicacion_px=args.radio_com,
        reward_mode=args.reward_mode,
        auto_assign=True
    )

    print(f"=== Iniciando entrenamiento RL V3-EXP (Exploracion Pura Descentralizada) ===")
    print(f" - Espacio de observacion: {env.observation_space.shape}")
    print(f" - Espacio de accion: {env.action_space}")
    print(f" - Robots en entrenamiento: {args.num_robots}")
    print(f" - Patch size: {args.patch_size}x{args.patch_size}")
    print(f" - Radio de comunicacion vecinal: {args.radio_com} px")
    print(f" - Modo de recompensa: {args.reward_mode}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=os.path.join(args.out, "tb_logs")
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(args.out, "checkpoints"),
        name_prefix="ppo_v3_exp"
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    final_path = os.path.join(args.out, "final_v3_exp.zip")
    model.save(final_path)
    print(f"Modelo final de exploracion pura guardado en: {final_path}")


if __name__ == "__main__":
    main()
