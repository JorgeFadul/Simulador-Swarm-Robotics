"""Smoke test V2 controller."""
import os
import sys
os.chdir(r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d")
sys.path.insert(0, os.getcwd())

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
print(f"V2: {cfg.cols}x{cfg.rows} celdas, {cfg.num_robots} robots")

sim = Simulacion(cfg, headless=True)
sim.poblar_robots(cfg.num_robots)
sim.mundo.etapa = "exploracion"

ctrl = RLHibridoController(cfg, model_path="models/rl_v2/final")
print(f"modelo V2 cargado OK, max_steps={ctrl.max_steps}")

for _ in range(3):
    ctrl.actualizar(sim.mundo)
    sim.step()
print("3 pasos OK")
print("V2 SMOKE OK")