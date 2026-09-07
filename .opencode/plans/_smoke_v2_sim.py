"""Smoke test: cargar simulacion con RL Hibrido V2."""
import os
import sys
os.chdir(r"C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d")
sys.path.insert(0, os.getcwd())

from src.core.config import Config
from src.core.simulacion import Simulacion
from src.core.config import ESTRATEGIAS_HIBRIDO

print("ESTRATEGIAS_HIBRIDO:", ESTRATEGIAS_HIBRIDO)

cfg = Config()
cfg.estrategia_hibrido = "RL Hibrido V2"
cfg.modo_fase = "hibrido"
cfg._normalizar_modo()
cfg.num_robots = int(getattr(cfg, "rl_hibrido_v2_num_robots", 10))
print(f"modelo V2: {cfg.rl_hibrido_v2_model}")

sim = Simulacion(cfg, headless=True)
sim.poblar_robots(cfg.num_robots)
sim.mundo.etapa = "exploracion"
sim.iniciar()

print(f"modo={sim.etapa}, planner={type(sim.planner).__name__}")
print("3 pasos...")
for _ in range(3):
    sim.step()
print("OK V2 en simulacion")