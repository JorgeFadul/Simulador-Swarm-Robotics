"""Entrenador Headless para la estrategia Bioinspirada Cerebro de Mosca (Drosophila).

Ejecuta multiples simulaciones consecutivas en memoria (a maxima velocidad sin render)
donde los cerebros aprenden por plasticidad sinaptica modulada por dopamina.
Al final de cada run, la memoria del cerebro con mayor exito (mas baldosas y defectos
hallados) se transfiere y consolida para la siguiente generacion/run, permitiendo
una curva de aprendizaje continua y acelerada.

El entrenamiento corre SIEMPRE en modo "hibrido" (una sola etapa): los robots
inspeccionan al visitar, lo que activa la dopamina por descubrimiento y por
defecto. Con "dos_etapas" la inspeccion solo ocurre en la fase 2 y el cerebro
de mosca no recibe senal de aprendizaje durante la exploracion.

Cada run esta acotada por un tope de pasos Y/o de tiempo real: la primera con-
dicion que se alcance cierra esa run (generacion), consolida su mejor cerebro y
arranca la siguiente, evitando desperdiciar pasos en sesiones que se estancan.

Uso:
    python train_bio_fly.py --runs 50 --robots 10 --mapas mapas/mapa_test_1.yaml
    python train_bio_fly.py --runs 200 --mapas mapas/mapa_rl_20x20_10r.yaml,mapas/mapa_rl_30x30.yaml \
        --max-steps 4000 --kc 400 --mutacion 0.02 --max-segundos 300
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import Config
from src.core.mundo import Mundo
from src.robots.robot import Robot
from src.bio.fly_controller import FlyBrainSwarmController


def cargar_mapa_en_cfg(cfg: Config, mapa_path: str, robots_pedidos: int | None,
                       defectos_semilla: int | None, run_idx: int):
    """Superpone un mapa YAML sobre la configuracion base.

    Los mapas definen dimensiones, grilla, paredes, robots y propagacion de
    defectos. Si `robots_pedidos` se da (CLI), manda sobre el valor del mapa.
    """
    raw = {}
    if mapa_path and os.path.exists(mapa_path):
        with open(mapa_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg.map_width = int(raw.get("map_width", cfg.map_width))
    cfg.map_height = int(raw.get("map_height", cfg.map_height))
    cfg.grid_size = int(raw.get("grid_size", cfg.grid_size))
    cfg.walls = list(raw.get("walls", cfg.walls))
    cfg.defecto_num_blobs = int(raw.get("defecto_num_blobs", cfg.defecto_num_blobs))
    cfg.num_robots = (int(robots_pedidos) if robots_pedidos is not None
                      else int(raw.get("num_robots", cfg.num_robots)))
    cfg.cols = cfg.map_width // cfg.grid_size
    cfg.rows = cfg.map_height // cfg.grid_size

    if defectos_semilla is not None:
        # Reproducible y distinta en cada run: layout variado pero comparable.
        cfg.defecto_semilla_global = defectos_semilla + (run_idx * 1000 + 7)
    else:
        cfg.defecto_semilla_global = None


def ejecutar_run(run_idx: int, cfg: Config, weights_path: str, max_steps: int,
                 max_segundos: int, n_kc: int, mutacion: float, semilla: int,
                 mapa_path: str, robots_pedidos: int | None,
                 defectos_semilla: int | None) -> dict:
    rng = np.random.default_rng(semilla + run_idx)
    cargar_mapa_en_cfg(cfg, mapa_path, robots_pedidos, defectos_semilla, run_idx)

    mundo = Mundo(cfg, rng=rng)
    ctrl = FlyBrainSwarmController(
        cfg, weights_path=weights_path, rng=rng, n_kc=n_kc, mutacion=mutacion
    )

    # Poblar robots
    for _ in range(cfg.num_robots):
        x, y = mundo.posicion_libre_aleatoria(rng, cfg.robot_radius_px)
        if x is not None:
            r = Robot(x, y, cfg, rng)
            mundo.robots.append(r)

    paso = 0
    motivo = "completado"
    t0 = time.time()
    while paso < max_steps:
        # Corte por tiempo real: no desperdiciar pasos en una sesion lenta.
        if time.time() - t0 >= max_segundos:
            motivo = "limite_tiempo"
            break

        ctrl.actualizar(mundo)
        mundo.update(0.05)
        paso += 1

        # Condicion de exito: 99% inspeccionado
        if mundo.porcentaje_inspeccionado >= 0.99:
            break

    if paso >= max_steps:
        motivo = "limite_pasos"

    dt_real = time.time() - t0
    tiempo_simulado = paso * 0.05
    cobertura = mundo.porcentaje_inspeccionado * 100.0
    defectos_gt = int(mundo._defectuosas_gt)
    defectos_encontrados = int(mundo._defectuosas)

    # Consolidar la generacion aunque la run haya sido cortada: el mejor cerebro
    # hereda su memoria a la siguiente run con la mutacion configurada.
    ctrl.guardar_enjambre(weights_path)
    if ctrl.cerebros:
        dopamina_max = max(c.dopamina_acumulada for c in ctrl.cerebros.values())
        dopamina_media = float(np.mean([c.dopamina_acumulada for c in ctrl.cerebros.values()]))
    else:
        dopamina_max = dopamina_media = 0.0

    return {
        "run": run_idx + 1,
        "mapa": os.path.basename(mapa_path) if mapa_path else "default",
        "robots": cfg.num_robots,
        "pasos": paso,
        "motivo": motivo,
        "tiempo_simulado_s": round(tiempo_simulado, 1),
        "tiempo_real_s": round(dt_real, 2),
        "cobertura_pct": round(cobertura, 2),
        "defectos_gt": defectos_gt,
        "defectos": defectos_encontrados,
        "defectos_pct": round(100.0 * defectos_encontrados / defectos_gt, 2) if defectos_gt else 0.0,
        "dopamina_max": round(dopamina_max, 1),
        "dopamina_media": round(dopamina_media, 1),
    }


def escribir_csv(ruta: str, filas: list[dict]):
    if not filas:
        return
    campos = list(filas[0].keys())
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Headless Cerebro de Mosca (Drosophila)")
    parser.add_argument("--runs", type=int, default=30, help="Numero de simulaciones/generaciones consecutivas")
    parser.add_argument("--robots", type=int, default=None, help="Numero de robots (moscas); si se omite, usa el del mapa")
    parser.add_argument("--mapas", type=str, default="mapas/mapa_test_1.yaml",
                        help="Mapa o lista separada por comas de mapas YAML (se rotan entre runs)")
    parser.add_argument("--mapa", type=str, default=None, help="Alias de --mapas (retrocompatibilidad)")
    parser.add_argument("--out", type=str, default="models/bio/fly_swarm_latest.npz", help="Ruta para guardar pesos consolidados")
    parser.add_argument("--reporte", type=str, default=None,
                        help="CSV con la curva de aprendizaje por run (default: <out>.csv)")
    parser.add_argument("--max-steps", type=int, default=15000, help="Tope de pasos por run (dt=0.05s); al alcanzarlo, la run se corta")
    parser.add_argument("--max-segundos", type=int, default=300,
                        help="Tope de tiempo real por run; al alcanzarlo, la run se corta y se pasa a la siguiente generacion")
    parser.add_argument("--kc", type=int, default=1000, help="Numero de Kenyon Cells por cerebro (menor = mas rapido)")
    parser.add_argument("--mutacion", type=float, default=0.02,
                        help="Perturbacion relativa de los pesos heredados W_kc_mbon (0 = clones exactos, >0 = diversidad)")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla base de la secuencia de runs")
    parser.add_argument("--defectos-semilla", type=int, default=None,
                        help="Semilla base de la disposicion de defectos (variada por run); omitir = aleatoria")
    parser.add_argument("--reset-weights", action="store_true", help="Reiniciar pesos y empezar de cero")
    args = parser.parse_args()

    lista_mapas = [m.strip() for m in (args.mapa or args.mapas).split(",") if m.strip()]
    reporte = args.reporte or (os.path.splitext(args.out)[0] + ".csv")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.reset_weights and os.path.exists(args.out):
        os.remove(args.out)
        print("[-] Pesos previos borrados. Empezando de cero.")

    # Forzar el modo en el que el cerebro de mosca recibe dopamina:
    # solo con modo "hibrido" los robots inspeccionan al visitar (robot.py).
    cfg = Config()
    cfg.modo_fase = "hibrido"
    cfg.estrategia_hibrido = "Cerebro Mosca (Dopamina)"
    cfg.sync_strategy()
    cfg.num_robots = args.robots if args.robots is not None else cfg.num_robots

    print("=" * 78)
    print(" ENTRENAMIENTO BIOINSPIRADO HEADLESS: CEREBRO DE MOSCA (DOPAMINA)")
    print(f" Runs: {args.runs} | Mapa(s): {lista_mapas} | Robots: {cfg.num_robots} (mapa/CLI)")
    print(f" Modo: {cfg.modo_fase} | Estrategia: {cfg.estrategia_hibrido}")
    print(f" KC: {args.kc} | Mutacion: {args.mutacion} | Max pasos/run: {args.max_steps} | Max seg/run: {args.max_segundos}s")
    print(f" Semilla: {args.semilla} | Defectos semilla: {args.defectos_semilla or 'aleatoria'}")
    print(f" Memoria consolidada: {args.out}")
    print(f" Curva de aprendizaje: {reporte}")
    print("=" * 78)

    filas = []
    for i in range(args.runs):
        mapa_path = lista_mapas[i % len(lista_mapas)]
        res = ejecutar_run(i, cfg, args.out, args.max_steps, args.max_segundos,
                           args.kc, args.mutacion,
                           args.semilla, mapa_path, args.robots, args.defectos_semilla)
        filas.append(res)
        escribir_csv(reporte, filas)

        print(f"Run {res['run']:03d}/{args.runs:03d} | {res['mapa']:<22} | "
              f"Sim: {res['tiempo_simulado_s']:>6.0f}s (Real: {res['tiempo_real_s']:>6.2f}s) | "
              f"Motivo: {res['motivo']:<13} | "
              f"Cob: {res['cobertura_pct']:>5.1f}% | "
              f"Def: {res['defectos']:>3d}/{res['defectos_gt']:>3d} ({res['defectos_pct']:>4.1f}%) | "
              f"Dopamina max: {res['dopamina_max']:>+7.1f} | media: {res['dopamina_media']:>+7.1f}")

    print("=" * 78)
    cobros = [f["cobertura_pct"] for f in filas]
    dregio = [f["defectos_pct"] for f in filas]
    print(f"[+] Entrenamiento bioinspirado completado ({args.runs} runs).")
    print(f"[+] Cobertura media ({len(cobros)} runs): {np.mean(cobros):.1f}% | Ultimas 10: {np.mean(cobros[-10:]):.1f}%")
    if dregio:
        print(f"[+] Defectos detectados media: {np.mean(dregio):.1f}% | Ultimas 10: {np.mean(dregio[-10:]):.1f}%")
    print(f"[+] Curva de aprendizaje: {reporte}")
    print(f"[+] Pesos consolidados listos para cargar en la GUI: {args.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()