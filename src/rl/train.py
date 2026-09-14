"""Pipeline de entrenamiento RL para el Simulador Swarm 2D (Fase 4).

Uso:
    # Entrenamiento estandar de una sola etapa
    python -m src.rl.train --reward time --steps 500000 --map-known
    python -m src.rl.train --reward time --steps 500000 --map-unknown

    # Curriculum learning (3->5->7->10 robots, mapas crecientes)
    python -m src.rl.train --curriculum --map-known
    python -m src.rl.train --curriculum --map-unknown

    # Opciones utiles para debug
    python -m src.rl.train --curriculum --map-known --dry-run --steps 300
    python -m src.rl.train --num-robots 3 --grid 10 --steps 50000 --map-known --fps-physics 5

Entrena agentes PPO (Stable-Baselines3) sobre el entorno SwarmEnv, con dos
variantes de agente (mapa conocido / mapa desconocido) y curriculum learning.
"""

import argparse
import sys
import os
import time
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from src.core.config import Config
from src.rl.env import SwarmEnv, make_env


class TimeBasedLogCallback(BaseCallback):
    """Imprime metricas de entrenamiento cada N segundos."""
    def __init__(self, interval_s=300, verbose=0):
        super().__init__(verbose)
        self.interval_s = interval_s
        self._t0 = None
        self._last_print = None

    def _on_training_start(self):
        import time as _t
        self._t0 = _t.time()
        self._last_print = self._t0

    def _on_step(self) -> bool:
        import time as _t
        now = _t.time()
        if now - self._last_print >= self.interval_s:
            t = self.model
            n_steps = t.num_timesteps
            el = now - self._t0
            print(f"[V2-log] pasos={n_steps}  t_total={el:.0f}s  "
                  f"t/paso={el/max(1,n_steps):.3f}s  "
                  f"fps={n_steps/max(1,el):.1f}", flush=True)
            self._last_print = now
        return True


def parse_args():
    p = argparse.ArgumentParser(description="Entrena un agente RL swarm 2D.")
    p.add_argument("--reward", choices=["time", "energy"], default="time",
                   help="Funcion de recompensa: 'time' o 'energy'.")
    p.add_argument("--steps", type=int, default=500000,
                   help="Pasos totales de entrenamiento (si no se usa --curriculum).")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Numero de entornos paralelos (SubprocVecEnv).")
    p.add_argument("--seed", type=int, default=42)

    # Modo del agente: mapa conocido vs desconocido
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--map-known", dest="map_known", action="store_true",
                      default=True,
                      help="El agente conoce todo el mapa desde el reset (default).")
    modo.add_argument("--map-unknown", dest="map_known", action="store_false",
                      help="El agente descubre el mapa mientras inspecciona.")

    # Auto-asignacion fallback (anti-idle, compatible con env_simple).
    # Activo por defecto: garantiza que todo robot libre reciba la celda libre
    # mas cercana, lo que mostro mejorar drasticamente la convergencia.
    p.add_argument("--no-auto-assign", dest="auto_assign", action="store_false",
                   default=True,
                   help="Desactivar el fallback de auto-asignacion de celdas.")

    # Curriculum learning
    p.add_argument("--curriculum", action="store_true",
                   help="Usa el curriculum definido en config/default.yaml "
                        "(3->5->7->10 robots).")
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime el plan del curriculum y termina sin entrenar.")

    # Sobreescritura de configuracion para modo no-curriculum
    p.add_argument("--mapa", default=None,
                   help="Archivo YAML del escenario.")
    p.add_argument("--num-robots", type=int, default=None,
                   help="Sobreescribe el numero de robots del config.")
    p.add_argument("--grid", type=int, default=None,
                   help="Sobreescribe a un grid cuadrado N x N celdas "
                        "(mapa = N*grid_size).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Maximo de pasos por episodio (sobreescribe default).")
    p.add_argument("--fps-physics", type=int, default=20,
                   help="Sub-pasos de fisica por step RL.")

    # Hiperparametros PPO individuales
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=None)
    p.add_argument("--net-arch", default=None,
                   help="Arquitectura de la red, ej '[256,256]' o '[128,128]'.")

    p.add_argument("--model-name", default="ppo_swarm",
                   help="Nombre base para la carpeta del modelo.")
    p.add_argument("--resume", default=None,
                   help="Ruta a un modelo pre-entrenado para continuar.")
    return p.parse_args()


def _apply_override(cfg, key, value):
    if value is None:
        return
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


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
        if "rl_reward" in data and isinstance(data["rl_reward"], dict):
            for k, v in data["rl_reward"].items():
                cfg.rl_reward[k] = v
        cfg.cols = cfg.map_width // cfg.grid_size
        cfg.rows = cfg.map_height // cfg.grid_size

    # Overrides lineales
    _apply_override(cfg, "num_robots", args.num_robots)
    if args.grid:
        # `--grid N` pide un grid cuadrado N x N celdas: ajustar el tamano del
        # mapa para que map_width/grid_size == N (independiente del default).
        gsize = getattr(cfg, "grid_size", 20)
        cfg.grid_size = gsize
        cfg.map_width = gsize * args.grid
        cfg.map_height = gsize * args.grid
        cfg.cols = cfg.map_width // cfg.grid_size
        cfg.rows = cfg.map_height // cfg.grid_size

    return cfg


def _build_ppo_hyperparams(cfg, args):
    """Devuelve dict de hiperparametros PPO desde config o CLI."""
    hp = dict(cfg.rl_ppo)
    # Overrides CLI
    _apply_override(hp, "learning_rate", args.lr)
    _apply_override(hp, "n_steps", args.n_steps)
    _apply_override(hp, "batch_size", args.batch_size)
    _apply_override(hp, "n_epochs", args.n_epochs)
    _apply_override(hp, "gamma", args.gamma)
    _apply_override(hp, "gae_lambda", args.gae_lambda)
    _apply_override(hp, "clip_range", args.clip_range)
    _apply_override(hp, "ent_coef", args.ent_coef)
    _apply_override(hp, "vf_coef", args.vf_coef)
    if args.net_arch:
        try:
            import ast
            hp["net_arch"] = ast.literal_eval(args.net_arch)
        except Exception:
            raise SystemExit("--net-arch debe ser una lista, ej '[256,256]'")
    return hp


def build_policy_kwargs(net_arch):
    return {"net_arch": list(net_arch)}


def create_ppo(env, hp, seed, resume=None, policy_kwargs=None):
    if resume:
        model = PPO.load(resume, env=env)
        print(f"Modelo cargado desde {resume}")
        return model
    return PPO(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="auto",
        tensorboard_log=None,
    )


def _make_vec(cfg, args, n_envs, seed):
    env_fns = [make_env(cfg=copy.deepcopy(cfg), seed=seed + i,
                        reward_mode=args.reward,
                        max_steps=cfg.rl_max_steps,
                        headless=True,
                        physics_steps=args.fps_physics,
                        map_known=args.map_known,
                        auto_assign=args.auto_assign)
               for i in range(n_envs)]
    env = DummyVecEnv([lambda e=e: Monitor(e()) for e in env_fns])
    return env


def train_single(args, cfg, total_timesteps, out_root):
    """Entrena una etapa con la configuracion dada y guarda en out_root."""
    os.makedirs(out_root, exist_ok=True)

    hp = _build_ppo_hyperparams(cfg, args)
    policy_kwargs = build_policy_kwargs(hp.pop("net_arch"))

    env = _make_vec(cfg, args, args.n_envs, args.seed)
    eval_env = _make_vec(cfg, args, 1, args.seed + 9999)

    model = create_ppo(env, hp, args.seed, resume=args.resume,
                       policy_kwargs=policy_kwargs)

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=max(10000, hp["n_steps"]),
        save_path=out_root,
        name_prefix="ppo_swarm",
    )
    eval_cb = EvalCallback(
        eval_env, n_eval_episodes=3,
        eval_freq=max(10000, hp["n_steps"]),
        log_path=f"{out_root}/eval",
        deterministic=True,
    )
    log_cb = TimeBasedLogCallback(interval_s=300)
    callbacks = CallbackList([ckpt_cb, eval_cb, log_cb])

    print(f"\n=== Entrenando etapa: {cfg.num_robots} robots, "
          f"grid {cfg.rows}x{cfg.cols} "
          f"({'conocido' if args.map_known else 'desconocido'}) "
          f"| pasos={total_timesteps} | map_known={args.map_known} ===")
    print(f"    obs_dim={env.observation_space.shape[0]}  "
          f"act_dim={env.action_space}")

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    final_path = f"{out_root}/final"
    model.save(final_path)
    env.close()
    eval_env.close()

    print(f"\nEtapa completada en {elapsed:.1f}s.")
    print(f"Modelo guardado en: {final_path}.zip")
    return final_path


def run_curriculum(args):
    cfg = Config()
    stages = cfg.rl_curriculum
    if not stages:
        raise SystemExit("No se encontro 'rl.curriculum' en config/default.yaml")

    policy_kwargs = build_policy_kwargs(list(cfg.rl_ppo["net_arch"]))

    final_path = None
    metadata = {"stages": []}

    for idx, stage in enumerate(stages):
        stage_cfg = copy.deepcopy(cfg)
        # ---- Construir mapa a partir de dimensiones de la etapa ----
        stage_cfg.num_robots = stage["num_robots"]
        # grid fijo (celdas) del config
        stage_cfg.map_width = stage["map_width"]
        stage_cfg.map_height = stage["map_height"]
        # grid_size ya definido en config (20)
        stage_cfg.cols = stage_cfg.map_width // stage_cfg.grid_size
        stage_cfg.rows = stage_cfg.map_height // stage_cfg.grid_size
        stage_cfg.rl_max_steps = stage["max_steps"]
        duration = stage["duration_steps"]
        stage_cfg.walls = []

        out_dir = f"models/{args.model_name}/stage{idx}"
        hp = _build_ppo_hyperparams(stage_cfg, args)

        env = _make_vec(stage_cfg, args, args.n_envs, args.seed + idx * 10)
        eval_env = _make_vec(stage_cfg, args, 1, args.seed + 9000 + idx * 10)

        # Continuar la red de la etapa anterior (transferencia de pesos)
        resume = final_path if final_path else args.resume
        model = create_ppo(env, hp, args.seed + idx, resume=resume,
                           policy_kwargs=policy_kwargs)

        ckpt_cb = CheckpointCallback(
            save_freq=max(10000, hp["n_steps"]),
            save_path=out_dir, name_prefix="ppo_swarm",
        )
        eval_cb = EvalCallback(
            eval_env, n_eval_episodes=3,
            eval_freq=max(10000, hp["n_steps"]),
            log_path=f"{out_dir}/eval", deterministic=True,
        )
        callbacks = CallbackList([ckpt_cb, eval_cb])

        print(f"\n=== Curriculum etapa {idx}: {stage_cfg.num_robots} robots, "
              f"grid {stage_cfg.rows}x{stage_cfg.cols}, "
              f"{'conocido' if args.map_known else 'desconocido'}, "
              f"pasos={duration} ===")
        print(f"    obs_dim={env.observation_space.shape[0]}  "
              f"act_dim={env.action_space}")

        if args.dry_run:
            print("    [dry-run] se omite el entrenamiento de esta etapa.")
            env.close()
            eval_env.close()
            metadata["stages"].append({
                "idx": idx, "num_robots": stage_cfg.num_robots,
                "grid": [stage_cfg.rows, stage_cfg.cols],
                "steps": duration, "map_known": args.map_known,
            })
            continue

        t0 = time.time()
        model.learn(total_timesteps=duration, callback=callbacks)
        elapsed = time.time() - t0

        final_path = f"{out_dir}/final"
        model.save(final_path)

        metadata["stages"].append({
            "idx": idx, "num_robots": stage_cfg.num_robots,
            "grid": [stage_cfg.rows, stage_cfg.cols],
            "steps": duration, "map_known": args.map_known,
        })
        print(f"Etapa {idx} completada en {elapsed:.1f}s -> {final_path}.zip")
        env.close()
        eval_env.close()

    with open(f"models/{args.model_name}/curriculum_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nCurriculum terminado. Metadata en "
          f"models/{args.model_name}/curriculum_metadata.json")


def train(args):
    os.makedirs("models", exist_ok=True)

    if args.curriculum:
        run_curriculum(args)
        return

    cfg = load_cfg(args)
    if args.max_steps:
        cfg.rl_max_steps = args.max_steps
    else:
        # default razonable en funcion del tamano del grid
        cfg.rl_max_steps = 4 * cfg.cols * cfg.rows if cfg.cols else 10000

    out_root = f"models/{args.model_name}"
    train_single(args, cfg, args.steps, out_root)


if __name__ == "__main__":
    train(parse_args())
