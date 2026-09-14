"""Diagnostico: por que hybrid + mosca apenas inspecciona (cobertura ~9%)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np, yaml
from src.core.config import Config
from src.core.simulacion import Simulacion
from src.robots import robot as robot_mod

cont = {"intentos": 0, "ok": 0, "fuera": 0, "pared": 0, "ya": 0, "lejos": 0,
        "llegadas_forzar": 0, "llegadas_ok": 0}
orig = robot_mod.Robot._iniciar_inspeccion_celda


def wrap(self, mundo, r, c, forzar=False):
    gs = self.cfg.grid_size
    cont["intentos"] += 1
    if forzar and self.goal_mode and self.goal is not None and not self.goal_alcanzado:
        cont["llegadas_forzar"] += 1
    ok = orig(self, mundo, r, c, forzar=forzar)
    if not ok:
        if r < 0 or r >= mundo.rows or c < 0 or c >= mundo.cols:
            cont["fuera"] += 1
        elif mundo.grid[r, c] == 1:
            cont["pared"] += 1
        elif self.inspeccion_individual:
            ya = self.inspec_robot[r, c] != 0
            cont["ya"] += int(ya)
        else:
            ya = mundo.inspec[r, c]
            cont["ya"] += int(ya)
    elif forzar and self.goal_mode and self.goal is not None and not self.goal_alcanzado:
        cont["llegadas_ok"] += 1
    return ok


robot_mod.Robot._iniciar_inspeccion_celda = wrap

cfg = Config()
data = yaml.safe_load(open("mapas/mapa_rl_20x20_10r.yaml", encoding="utf-8"))
for k in ("map_width", "map_height", "grid_size", "num_robots"):
    if k in data:
        setattr(cfg, k, int(data[k]))
cfg.walls = data.get("walls", [])
cfg.cols = cfg.map_width // cfg.grid_size
cfg.rows = cfg.map_height // cfg.grid_size
cfg.modo_fase = "hibrido"
cfg.estrategia_hibrido = "Cerebro Mosca (Dopamina)"
cfg.sync_strategy()
cfg.exploration_goal = 99.0

sim = Simulacion(cfg, seed=11, headless=True)
sim.poblar_robots(cfg.num_robots)
sim.iniciar()

# Tambien medir estado "atascado en llegada"
atascados = {}
for i in range(2500):
    sim.step()
    for r in sim.mundo.robots:
        if (r.goal_mode and r.goal is not None and not r.goal_alcanzado
                and not r.inspeccionando and not r.ruta):
            dg = math.hypot(r.goal[0] - r.x, r.goal[1] - r.y)
            if dg <= r.goal_reached_radius:
                key = id(r)
                atascados[key] = atascados.get(key, 0) + 1

print("intentos percusion:", cont["intentos"])
print("  OK:", cont["ok"], "| fuera:", cont["fuera"], "| pared:", cont["pared"],
      "| ya_inspeccionada:", cont["ya"], "| lejos:", cont["lejos"])
print("llegadas(forzar=True):", cont["llegadas_forzar"], "-> ok:", cont["llegadas_ok"])
print("ticks robot 'atascado' cerca del goal (ruta vacia, sin inspeccionar):",
      sum(atascados.values()), "en", len(atascados), "robots")
print("celdas inspeccionadas:", int(sim.mundo._inspeccionadas),
      "de", sim.mundo.rows * sim.mundo.cols)
print("etapa:", sim.etapa)