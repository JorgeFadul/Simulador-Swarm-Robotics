import argparse
import json
import sys

sys.path.insert(0, "src")

import yaml

from src.core.config import (
    Config, ESTRATEGIAS_EXPLORACION, ESTRATEGIAS_INSPECCION,
    ESTRATEGIAS_HIBRIDO, ESTRATEGIAS_VALIDAS,
)
from src.core.simulacion import Simulacion
from src.gui.stats_window import StatsWindow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulador Swarm 2D para la tesis de maestria. "
                    "Sin --headless abre la interfaz grafica (Pygame + Tkinter)."
    )
    parser.add_argument("--headless", action="store_true",
                        help="Corre la simulacion en memoria (sin ventanas) y "
                             "transmite el estado por stdout en JSONL.")
    parser.add_argument("--modo", choices=["exploracion", "inspeccion", "hibrido"],
                        default=None,
                        help="Modo de fase: exploracion, inspeccion o hibrido.")
    parser.add_argument("--estrategia-exploracion",
                        choices=ESTRATEGIAS_EXPLORACION, default=None,
                        help="Estrategia para el modo exploracion.")
    parser.add_argument("--estrategia-inspeccion",
                        choices=ESTRATEGIAS_INSPECCION, default=None,
                        help="Estrategia para el modo inspeccion.")
    parser.add_argument("--estrategia-hibrido",
                        choices=ESTRATEGIAS_HIBRIDO, default=None,
                        help="Estrategia para el modo hibrido.")
    # Compatibilidad: --estrategia viejo
    parser.add_argument("--estrategia", default=None,
                        help="(Deprecated) Estrategia legacy. Mapea al modo "
                             "correspondiente automaticamente.")
    parser.add_argument("--semilla", type=int, default=None,
                        help="Semilla del generador aleatorio (reproducibilidad).")
    parser.add_argument("--mapa", default=None,
                        help="Archivo YAML con el escenario.")
    parser.add_argument("--generar-mapas", action="store_true",
                        help="(Re)genera los 5 mapas de prueba en `mapas/` y termina.")
    return parser.parse_args()


def cargar_mapa(cfg, ruta):
    """Carga un escenario YAML opcional."""
    with open(ruta, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "grid_size" in data:
        cfg.grid_size = int(data["grid_size"])
    if "map_width" in data:
        cfg.map_width = int(data["map_width"])
    if "map_height" in data:
        cfg.map_height = int(data["map_height"])
    if "num_robots" in data:
        cfg.num_robots = int(data["num_robots"])
    if "walls" in data:
        cfg.walls = data["walls"]
    if "defecto_blobs" in data and data["defecto_blobs"]:
        cfg.defecto_blobs = data["defecto_blobs"]
        cfg.prob_defecto = 0.0
    cfg.cols = cfg.map_width // cfg.grid_size
    cfg.rows = cfg.map_height // cfg.grid_size


def correr_headless(cfg, seed):
    from src.core.telemetria import (
        construir_resumen, escribir_resumen_json, escribir_serie_csv,
        estado_simulacion, fila_serie,
    )

    sim = Simulacion(cfg, seed=seed, headless=True)
    sim.poblar_robots(max(1, cfg.num_robots))
    sim.iniciar()

    if not sim.mundo.robots or sim.estado != "running":
        print(json.dumps({"error": "no se pudieron colocar robots en el escenario"},
                         ensure_ascii=False))
        return 1

    serie = []
    motivo = "running"
    while sim.estado == "running" and sim.pasos_totales < cfg.headless_max_pasos:
        sim.step()
        linea = estado_simulacion(sim)
        print(json.dumps(linea, ensure_ascii=False), flush=True)
        serie.append(fila_serie(sim))

    linea = estado_simulacion(sim)
    print(json.dumps(linea, ensure_ascii=False), flush=True)
    serie.append(fila_serie(sim))

    if sim.estado == "completado":
        motivo = "completado"
    elif sim.pasos_totales >= cfg.headless_max_pasos:
        motivo = "limite_pasos"

    strat = cfg._estrategia_activa()
    base = f"headless_{cfg.modo_fase}_{strat.replace(' ', '_')}"
    base += f"_seed{seed}" if seed is not None else "_ns"
    ruta_json = base + ".json"
    ruta_csv = base + ".csv"

    resumen = construir_resumen(sim, semilla=seed, estrategia=strat,
                                resultado=motivo)
    resumen["modo_fase"] = cfg.modo_fase
    escribir_resumen_json(ruta_json, resumen)
    escribir_serie_csv(ruta_csv, serie)

    print(f"# fin: headless de {strat} (modo {cfg.modo_fase}) | motivo={motivo} | "
          f"tiempo={sim.tiempo_simulado:.1f}s | pasos={sim.pasos_totales}")
    print(f"# resumen: {ruta_json}")
    print(f"# serie:   {ruta_csv}")
    return 0


def correr_gui(cfg):
    import tkinter as tk
    import pygame

    sim = Simulacion(cfg)
    root = tk.Tk()
    root.withdraw()
    stats = StatsWindow(root, sim)

    reloj = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                sim.manejar_evento(event)

        if sim.estado == "running":
            sim.step()

        sim.render()

        sim.fps_actual = reloj.get_fps()
        stats.actualizar()
        root.update()

        reloj.tick(60)

    pygame.quit()
    root.destroy()


def main():
    args = parse_args()

    cfg = Config()
    cfg.cargar_desde_archivo("sim_config.txt")

    # Aplicar argumentos CLI
    if args.modo:
        cfg.modo_fase = args.modo
    if args.estrategia_exploracion:
        cfg.estrategia_exploracion = args.estrategia_exploracion
    if args.estrategia_inspeccion:
        cfg.estrategia_inspeccion = args.estrategia_inspeccion
    if args.estrategia_hibrido:
        cfg.estrategia_hibrido = args.estrategia_hibrido

    # Compatibilidad: --estrategia viejo
    if args.estrategia:
        old = args.estrategia
        if old in ESTRATEGIAS_EXPLORACION:
            cfg.modo_fase = "exploracion"
            cfg.estrategia_exploracion = old
        elif old in ESTRATEGIAS_INSPECCION:
            cfg.modo_fase = "inspeccion"
            cfg.estrategia_inspeccion = old
        elif old in ESTRATEGIAS_HIBRIDO:
            cfg.modo_fase = "hibrido"
            cfg.estrategia_hibrido = old

    cfg.sync_strategy()

    # Mantener disponibles los 5 mapas de prueba
    from src.utils.mapas_prueba import asegurar_mapas_prueba
    asegurar_mapas_prueba(cfg)

    if args.generar_mapas:
        print("Mapas de prueba garantizados en: mapas/")
        return 0

    if args.mapa:
        cargar_mapa(cfg, args.mapa)

    if args.headless:
        return correr_headless(cfg, args.semilla)

    correr_gui(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
