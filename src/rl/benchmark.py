"""Benchmark comparativo: estrategias heuristicas vs agentes RL.

Evalua todas las estrategias de inspeccion/exploracion del simulador contra
agentes RL (mapa conocido / mapa desconocido) sobre los mapas de benchmark,
y exporta un CSV de metricas + graficas para la tesis.

Uso:
    # RL-known
    python -m src.rl.benchmark --rl-known models/ppo_swarm/final.zip \
        --mapas "mapas/mapa_test_1.yaml,mapas/mapa_test_2.yaml" --seeds 7,42,123

    # RL-known + RL-unknown
    python -m src.rl.benchmark --rl-known models/rl_known/final.zip \
        --rl-unknown models/rl_unknown/final.zip \
        --mapas "mapas/mapa_test_*.yaml" --seeds 7,42,123

    # Solo heuristicas
    python -m src.rl.benchmark --mapas "mapas/mapa_test_1.yaml" --seeds 7,42

Metricas por episodio: cobertura (inspeccion), energia total, pasos, colisiones.
Se guarda: results/benchmark.csv, results/benchmark_summary.csv y graficas PNG.
"""

import argparse
import sys
import os
import csv
import glob
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.core.config import Config
from src.core.mundo import Mundo
from src.core.simulacion import Simulacion
from src.robots.robot import Robot
from src.rl.env import SwarmEnv
from src.rl.infer import load_cfg, run_episode


# Estrategias heuristicas soportadas (modo_fase + estrategias)
HEURISTICAS = {
    "RandomWalk":      {"modo": "hibrido", "exp": "Random Walk",
                        "ins": "Random Walk", "hib": "Random Walk Hibrido"},
    "Frontier":        {"modo": "dos_etapas", "exp": "Frontier Based",
                        "ins": "Voronoi", "hib": "Random Walk Hibrido"},
    "SampleBased":     {"modo": "dos_etapas", "exp": "Sample Based",
                        "ins": "Voronoi", "hib": "Random Walk Hibrido"},
    "Voronoi":         {"modo": "dos_etapas", "exp": "Boustrophedon",
                        "ins": "Voronoi", "hib": "Random Walk Hibrido"},
    "Boustrophedon":   {"modo": "dos_etapas", "exp": "Boustrophedon",
                        "ins": "Boustrophedon Inspeccion", "hib": "Random Walk Hibrido"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark heuristicas vs RL.")
    p.add_argument("--rl-known", default=None,
                   help="Ruta al modelo RL de mapa conocido.")
    p.add_argument("--rl-unknown", default=None,
                   help="Ruta al modelo RL de mapa desconocido.")
    p.add_argument("--mapas", default="mapas/mapa_test_1.yaml",
                   help="Glob o lista separada por coma de mapas YAML.")
    p.add_argument("--seeds", default="7,42,123",
                   help="Lista de seeds separadas por coma.")
    p.add_argument("--mapas-heuristicas", action="store_true",
                   help="Correr heuristicas ademas del RL.")
    p.add_argument("--only-heuristics", action="store_true",
                   help="Correr solo heuristicas (sin RL).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Limite de pasos por episodio (todas las estrategias).")
    p.add_argument("--timeout-s", type=float, default=600.0,
                   help="Timeout real por episodio heuristica (segundos).")
    p.add_argument("--out-dir", default="results",
                   help="Directorio de salida.")
    return p.parse_args()


def _expand_mapas(pattern):
    if "," in pattern:
        out = []
        for p in pattern.split(","):
            p = p.strip()
            if any(ch in p for ch in "*?"):
                out.extend(glob.glob(p))
            else:
                out.append(p)
        return sorted(out)
    if any(ch in pattern for ch in "*?"):
        return sorted(glob.glob(pattern))
    return [pattern]


def _respeta(recompensa=None):
    pass


def run_heuristica(map_path, estrategia_conf, seed, cfg_template=None,
                   max_pasos=None, timeout_s=600.0):
    """Ejecuta una estrategia heuristica headless y devuelve metricas."""
    cfg = load_cfg_args(map_path, estrategia_conf)
    mts = max_pasos or cfg.headless_max_pasos

    t0 = time.time()
    sim = Simulacion(cfg, seed=seed, headless=True)
    sim.poblar_robots(cfg.num_robots)
    sim.iniciar()

    pasos = 0
    while sim.estado == "running" and pasos < mts:
        sim.step()
        pasos += 1
        if time.time() - t0 > timeout_s:
            break

    m = sim.mundo
    total_libres = m.total_libres if m.total_libres > 0 else 1
    insp = int(m.inspec.sum())
    cobertura = insp / total_libres
    energia = sum(r.energia.energia_consumida_total for r in m.robots)

    # colisiones aproximadas (paredes acumuladas)
    colisiones = sum(getattr(r, "paredes_robot", 0) for r in m.robots)

    return {
        "variante": estrategia_conf["nombre"],
        "mapa": os.path.basename(map_path),
        "seed": seed,
        "pasos": pasos,
        "inspeccionadas": insp,
        "total_libres": total_libres,
        "cobertura": cobertura,
        "energia_total": energia,
        "colisiones_pared": colisiones,
        "colisiones_rr": 0,
        "recompensa": 0.0,
        "completado": 1 if cobertura >= 0.99 else 0,
        "timeout": 1 if cobertura < 0.99 and pasos >= mts else 0,
    }


def load_cfg_args(map_path, estrategia_conf):
    """Carga Config desde mapa YAML + aplica modo_fase/estrategia."""
    import yaml
    cfg = Config()
    with open(map_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for k in ("grid_size", "map_width", "map_height", "num_robots", "walls"):
        if k in data:
            setattr(cfg, k, data[k])
    for k in ("defecto_num_blobs", "defecto_prob_ortogonal",
              "defecto_prob_diagonal", "defecto_decaimiento",
              "defecto_semilla_global"):
        if k in data:
            setattr(cfg, k, data[k])
    cfg.cols = cfg.map_width // cfg.grid_size
    cfg.rows = cfg.map_height // cfg.grid_size
    cfg.modo_fase = estrategia_conf["modo"]
    cfg.estrategia_exploracion = estrategia_conf["exp"]
    cfg.estrategia_inspeccion = estrategia_conf["ins"]
    cfg.estrategia_hibrido = estrategia_conf["hib"]
    return cfg


def run_rl(map_path, model_path, seed, map_known, max_pasos=None):
    args = type("A", (), {
        "mapa": map_path, "reward": "time", "max_steps": max_pasos,
        "map_known": map_known, "auto_assign": True,
    })()
    cfg = load_cfg(args)
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    est, _ = run_episode(model, cfg, args, seed)
    est["variante"] = "RL-known" if map_known else "RL-unknown"
    est["mapa"] = os.path.basename(map_path)
    est["colisiones_pared"] = 0
    est["colisiones_rr"] = est.pop("colisiones_rr", 0)
    est["completado"] = 1 if est["cobertura"] >= 0.99 else 0
    est["timeout"] = 1 if est["cobertura"] < 0.99 else 0
    return est


def write_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmark.csv")
    fieldnames = ["variante", "mapa", "seed", "pasos", "inspeccionadas",
                  "total_libres", "cobertura", "energia_total",
                  "colisiones_pared", "colisiones_rr", "recompensa",
                  "completado", "timeout"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"CSV guardado: {path}")
    return path


def summarize(rows, out_dir):
    """Resumen por (variante, mapa): media y desviacion de las metricas."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmark_summary.csv")
    groups = {}
    for r in rows:
        key = (r["variante"], r["mapa"])
        groups.setdefault(key, []).append(r)

    rows_out = []
    for (variante, mapa), eps in sorted(groups.items()):
        def mean(k):
            return float(np.mean([e[k] for e in eps]))
        def std(k):
            return float(np.std([e[k] for e in eps]))
        rows_out.append({
            "variante": variante, "mapa": mapa, "n": len(eps),
            "cobertura_media": round(mean("cobertura"), 4),
            "cobertura_std": round(std("cobertura"), 4),
            "pasos_media": round(mean("pasos"), 1),
            "pasos_std": round(std("pasos"), 1),
            "energia_media": round(mean("energia_total"), 2),
            "energia_std": round(std("energia_total"), 2),
            "completados": int(sum(1 for e in eps if e["completado"])),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Resumen guardado: {path}")
    return rows_out


def plot(rows, out_dir):
    """Genera graficas (barras de pasos/energia, cobertura)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible - se omite graficacion.")
        return

    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(rows, out_dir)

    # Agrupar por mapa y variante
    mapas = sorted({s["mapa"] for s in summary})
    variantes = sorted({s["variante"] for s in summary})
    color = {v: c for v, c in zip(variantes,
        plt.cm.tab10(np.linspace(0, 1, max(len(variantes), 1))))}

    # 1) Barras: cobertura media por estrategia (agrupada por mapa)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(variantes), 1)
    for vi, v in enumerate(variantes):
        vals = []
        maps_at = []
        for m in mapas:
            row = next((s for s in summary if s["variante"] == v and s["mapa"] == m), None)
            if row:
                vals.append(row["cobertura_media"])
                maps_at.append(m)
            else:
                vals.append(0.0)
                maps_at.append(m)
        xpos = np.arange(len(mapas)) + vi * width
        ax.bar(xpos, vals, width, label=v, color=color[v])
    ax.set_xticks(np.arange(len(mapas)) + width * (len(variantes) - 1) / 2)
    ax.set_xticklabels([os.path.splitext(m)[0] for m in mapas], rotation=20)
    ax.set_ylabel("Cobertura inspeccion (media)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.99, color="red", ls="--", lw=0.8, label="meta")
    ax.legend()
    ax.set_title("Cobertura de inspeccion por estrategia y mapa")
    fig.tight_layout()
    out = os.path.join(out_dir, "benchmark_cobertura.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Grafica: {out}")

    # 2) Barras: pasos media (menos es mejor)
    fig, ax = plt.subplots(figsize=(10, 5))
    for vi, v in enumerate(variantes):
        vals = []
        for m in mapas:
            row = next((s for s in summary if s["variante"] == v and s["mapa"] == m), None)
            vals.append(row["pasos_media"] if row else 0.0)
        xpos = np.arange(len(mapas)) + vi * width
        ax.bar(xpos, vals, width, label=v, color=color[v])
    ax.set_xticks(np.arange(len(mapas)) + width * (len(variantes) - 1) / 2)
    ax.set_xticklabels([os.path.splitext(m)[0] for m in mapas], rotation=20)
    ax.set_ylabel("Pasos (media)")
    ax.legend()
    ax.set_title("Pasos usados por estrategia y mapa")
    fig.tight_layout()
    out = os.path.join(out_dir, "benchmark_pasos.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Grafica: {out}")

    # 3) Barras: energia media (menos es mejor)
    fig, ax = plt.subplots(figsize=(10, 5))
    for vi, v in enumerate(variantes):
        vals = []
        for m in mapas:
            row = next((s for s in summary if s["variante"] == v and s["mapa"] == m), None)
            vals.append(row["energia_media"] if row else 0.0)
        xpos = np.arange(len(mapas)) + vi * width
        ax.bar(xpos, vals, width, label=v, color=color[v])
    ax.set_xticks(np.arange(len(mapas)) + width * (len(variantes) - 1) / 2)
    ax.set_xticklabels([os.path.splitext(m)[0] for m in mapas], rotation=20)
    ax.set_ylabel("Energia total (J, media)")
    ax.legend()
    ax.set_title("Energia consumida por estrategia y mapa")
    fig.tight_layout()
    out = os.path.join(out_dir, "benchmark_energia.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Grafica: {out}")


def main():
    args = parse_args()
    mapas = _expand_mapas(args.mapas)
    if not mapas:
        raise SystemExit(f"No se encontraron mapas con patrón: {args.mapas}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Mapas: {mapas}")
    print(f"Seeds: {seeds}")
    print(f"RL-known: {args.rl_known} | RL-unknown: {args.rl_unknown}")

    if args.only_heuristics:
        run_heuristics = True
        run_rls = False
    else:
        run_heuristics = True
        run_rls = True

    rows = []

    # --- Heuristicas ---
    for nombre, conf in HEURISTICAS.items():
        for map_path in mapas:
            for seed in seeds:
                if not run_heuristics:
                    break
                conf2 = dict(conf)
                conf2["nombre"] = nombre
                est = run_heuristica(map_path, conf2, seed,
                                     max_pasos=args.max_steps,
                                     timeout_s=args.timeout_s)
                print(f"[HEUR] {nombre} | {os.path.basename(map_path)} "
                      f"| seed {seed} | cobertura={est['cobertura']:.2f} "
                      f"pasos={est['pasos']} energia={est['energia_total']:.1f}J")
                rows.append(est)

    # --- RL ---
    if run_rls:
        for model, known in ((args.rl_known, True), (args.rl_unknown, False)):
            if not model:
                continue
            for map_path in mapas:
                for seed in seeds:
                    est = run_rl(map_path, model, seed, known,
                                 max_pasos=args.max_steps)
                    print(f"[RL ] {est['variante']} | {os.path.basename(map_path)} "
                          f"| seed {seed} | cobertura={est['cobertura']:.2f} "
                          f"pasos={est['pasos']} energia={est['energia_total']:.1f}J")
                    rows.append(est)

    if not rows:
        print("Sin ejecuciones. Revisa argumentos.")
        return

    write_csv(rows, args.out_dir)
    summarize(rows, args.out_dir)
    plot(rows, args.out_dir)
    print("\nBenchmark completado.")


if __name__ == "__main__":
    main()
