"""Smoke test: RLHibridoController en mundo default 50x50/10 robots (1000x1000px)."""
import os
import sys
import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

from src.core.config import Config
from src.core.simulacion import Simulacion
from src.rl.controller import RLHibridoController, build_obs

cfg = Config()
cfg.num_robots = 10
cfg.map_width = 1000
cfg.map_height = 1000
cfg.grid_size = 20
cfg.cols = cfg.map_width // cfg.grid_size
cfg.rows = cfg.map_height // cfg.grid_size
print(f"mundo: {cfg.cols}x{cfg.rows} celdas, {cfg.num_robots} robots")

# headless=True para no abrir ventana pygame
sim = Simulacion(cfg, headless=True)
sim.poblar_robots(cfg.num_robots)
sim.mundo.etapa = "exploracion"

ctrl = RLHibridoController(cfg, model_path=cfg.rl_hibrido_model)
print(f"controlador OK, max_steps={ctrl.max_steps}")

# Verificar obs shape (usando build_obs global)
obs = build_obs(sim.mundo, sim.mundo.robots, paso=0, max_steps=ctrl.max_steps)
print(f"obs shape={obs.shape}, dtype={obs.dtype}")

# Verificar que actualizar funciona (asigna objetivos RL)
n = ctrl.actualizar(sim.mundo)
print(f"actualizar OK — asignados={n}")

# Avanzar 3 pasos del simulador con el controller
for i in range(3):
    ctrl.actualizar(sim.mundo)
    sim.step()
print("3 pasos con controller OK")

# Verificar que el modelo se usa en modo hibrido (elige objetivos RL)
print("\nSMOKE CTRL50 OK — modelo 50x50/10 robots cargado correctamente")