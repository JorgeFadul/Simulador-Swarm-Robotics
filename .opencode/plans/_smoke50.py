import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
from src.core.config import Config
from src.rl.env import SwarmEnv

cfg = Config(path="config/default.yaml")
path = "mapas/mapa_rl_50x50_10r.yaml"
with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
for k in ("grid_size", "map_width", "map_height", "num_robots", "walls"):
    if k in data:
        setattr(cfg, k, data[k])
cfg.cols = cfg.map_width // cfg.grid_size
cfg.rows = cfg.map_height // cfg.grid_size

print(f"grid {cfg.rows}x{cfg.cols} robots={cfg.num_robots}")
env = SwarmEnv(cfg=cfg, seed=1, headless=True, map_known=True, auto_assign=True, physics_steps=5)
print("obs_space:", env.observation_space.shape)
print("action_space:", env.action_space)
obs, _ = env.reset()
print("obs len:", len(obs))
obs_size = 2*cfg.rows*cfg.cols + cfg.num_robots*3 + 3
print("obs_size esperado:", obs_size)
N = cfg.num_robots
n_cells = cfg.rows*cfg.cols
print("action esperado:", [n_cells+1]*N)
for _ in range(50):
    action = np.random.randint(0, n_cells+1, size=N)
    obs, rew, term, trunc, info = env.step(action)
print("smoke OK, cobertura:", info["cobertura"], "rew:", rew)
env.close()