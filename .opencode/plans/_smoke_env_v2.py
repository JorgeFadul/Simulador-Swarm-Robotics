"""Smoke test rapido del env V2: verificar recompensas con defectos."""
import os
import sys
os.chdir(r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d")
sys.path.insert(0, os.getcwd())

import yaml
from src.core.config import Config
from src.rl.env import SwarmEnv

cfg = Config()
with open("mapas/mapa_rl_v2.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
for k in ("grid_size", "map_width", "map_height", "num_robots"):
    setattr(cfg, k, data[k])
cfg.cols = cfg.map_width // cfg.grid_size
cfg.rows = cfg.map_height // cfg.grid_size
for k, v in data["rl_reward"].items():
    cfg.rl_reward[k] = v
print(f"cfg.rl_reward = {cfg.rl_reward}")
print(f"mapa {cfg.cols}x{cfg.rows}, robots={cfg.num_robots}")

env = SwarmEnv(cfg=cfg, seed=42, reward_mode="time",
               max_steps=10000, headless=True, physics_steps=5,
               map_known=True, auto_assign=True)
obs, info = env.reset()
print(f"obs.shape={obs.shape}, info={info}")

# Avanzo 5 steps
for i in range(5):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    print(f"step {i}: r={r:.4f}  nuevas={info['nuevas_inspeccionadas']}  "
          f"def_nuevas={info['nuevas_defectuosas']}  "
          f"def_total={info['defectuosas_total']}  "
          f"term={term}  trunc={trunc}")
print("OK env V2")