"""Script de entrenamiento para RL de Exploracion Pura Descentralizada (V3-EXP).

Objetivo: Descubrir el mapa (SCOUTED y paredes DISCOVERED) en el menor tiempo y gasto
energetico, con coordinacion cooperativa entre vecinos (senal de actividad) y un
contador de robots vecinos dentro de un radio de vecindad (rl_vecindad_radio_mult).

Uso:
    python train_rl_v3_exp.py --timesteps 5000000 --num-robots 10 --patch-size 15
"""

import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from src.core.config import Config
from src.rl.env import SwarmEnvExploracionDescentralizado


def _make_env(cfg, seed, patch_size, radio_com, reward_mode):
    """Fabrica de entorno a nivel de modulo (pickleable para spawn en Windows)."""
    def _init():
        return SwarmEnvExploracionDescentralizado(
            cfg=cfg, seed=seed, patch_size=patch_size,
            radio_comunicacion_px=radio_com, reward_mode=reward_mode,
            auto_assign=True, headless=True)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Entrenador RL V3-EXP (Exploracion Pura Descentralizada)")
    parser.add_argument("--timesteps", type=int, default=300000, help="Pasos de entrenamiento total")
    parser.add_argument("--num-robots", type=int, default=10, help="Numero de robots para entrenamiento")
    parser.add_argument("--patch-size", type=int, default=15, help="Tamano de la ventana local KxK")
    parser.add_argument("--radio-com", type=float, default=400.0, help="Radio de comunicacion entre robots (px)")
    parser.add_argument("--reward-mode", type=str, default="energy", choices=["time", "energy"], help="Modo de penalizacion")
    parser.add_argument("--out", type=str, default="models/v3_exp", help="Directorio de salida")
    parser.add_argument("--n-envs", type=int, default=0,
                        help="Numero de entornos paralelos (0 = auto: nucleos CPU)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    cfg.num_robots = args.num_robots

    n_envs = args.n_envs
    if n_envs <= 0:
        n_envs = max(1, multiprocessing.cpu_count())
    print(f" ========== Envs paralelos (SubprocVecEnv): {n_envs} ==========", flush=True)

    env_fns = [_make_env(cfg, 100 + i, args.patch_size, args.radio_com, args.reward_mode)
               for i in range(n_envs)]
    env = SubprocVecEnv([lambda e=e: Monitor(e()) for e in env_fns])

    print(f"=== Iniciando entrenamiento RL V3-EXP (Exploracion Pura Descentralizada) ===", flush=True)
    print(f" - Espacio de observacion: {env.observation_space.shape}", flush=True)
    print(f" - Espacio de accion: {env.action_space}", flush=True)
    print(f" - Robots en entrenamiento: {args.num_robots}", flush=True)
    print(f" - Patch size: {args.patch_size}x{args.patch_size}", flush=True)
    print(f" - Radio de comunicacion vecinal: {args.radio_com} px", flush=True)
    print(f" - Radio de vecindad (contador de vecinos): "
          f"{cfg.rl_vecindad_radio_px:.1f} px = "
          f"{cfg.rl_vecindad_radio_mult}x radio del robot", flush=True)
    print(f" - Modo de recompensa: {args.reward_mode}", flush=True)
    print(f" - Envs paralelos (SubprocVecEnv): {n_envs}", flush=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="auto",
        tensorboard_log=None
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(args.out, "checkpoints"),
        name_prefix="ppo_v3_exp"
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    final_path = os.path.join(args.out, "final_v3_exp.zip")
    model.save(final_path)
    print(f"Modelo final de exploracion pura guardado en: {final_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()