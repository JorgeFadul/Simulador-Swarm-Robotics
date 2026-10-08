"""Meta-algoritmo GA NSGA-II offline (Parte A).

Estrategia FIJA (no codificada en el genoma): modo dos_etapas con
exploracion "Hybrid Based" + inspeccion "Voronoi", mapa sin paredes.

Objetivos (minimizar): f1 energia media [mWh], f2 tiempo medio [s],
f3 tasa de error e incobertura [%] = (100-cobertura) + 50*N_colisiones
+ P_infactible (1000 si cobertura<98% o tasa_validez<0.5).

Barre escenarios (mapa x robots) para minimos locales por escenario y un
frente global. Evaluacion paralela con multiprocessing.Pool.

Uso:
  python optimizar_ga.py --piloto --workers 4
  python optimizar_ga.py --poblacion 40 --generaciones 30 --reps 5 --workers 8
  python optimizar_ga.py --mapas 20 30 --robots 10 --poblacion 20 --generaciones 10
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from barrido_estrategias import (  # noqa: E402
    WALL_MAX_ABSOLUTO,
    cobertura_pct,
    energia_enjambre,
    timeout_sim_adaptativo,
)

from src.algorithms.genetic import (  # noqa: E402
    GEN_BOUNDS,
    GEN_DEFAULTS,
    GEN_NAMES,
    P_INFACTIBLE,
    Individual,
    NSGA2Optimizer,
    genome_to_dict,
)

FASE1_FIJA = "Hybrid Based"
FASE2_FIJA = "Voronoi"


def correr_genoma_una(args_tupla):
    """Worker picklable (Windows spawn): una sim headless con genoma inyectado.

    Tupla: (genoma_lista, mapa_n, robots_r, rep, max_pasos, t_sim_to,
    wall_max, umbral). Devuelve dict de metricas de la corrida.
    """
    (genoma_lista, n_celdas, n_robots, rep, max_pasos, t_sim_to,
     wall_max, umbral) = args_tupla
    import time as _time
    from src.core.config import Config, aplicar_genoma
    from src.core.simulacion import Simulacion

    cfg = Config()
    cfg.grid_size = 20
    cfg.map_width = n_celdas * 20
    cfg.map_height = n_celdas * 20
    cfg.cols = n_celdas
    cfg.rows = n_celdas
    cfg.num_robots = n_robots
    cfg.walls = []
    cfg.modo_fase = "dos_etapas"
    cfg.estrategia_exploracion = FASE1_FIJA
    cfg.estrategia_inspeccion = FASE2_FIJA
    cfg.sync_strategy()
    aplicar_genoma(cfg, genoma_lista)

    seed = 1000 + int(rep) * 17 + int(n_celdas) * 131 + int(n_robots) * 1013
    t0 = _time.time()
    sim = Simulacion(cfg, seed=seed, headless=True)
    sim.poblar_robots(max(1, cfg.num_robots))
    sim.iniciar()

    motivo = "running"
    if not sim.mundo.robots or sim.estado != "running":
        motivo = "sin_robots"
    else:
        while sim.estado == "running" and sim.pasos_totales < max_pasos:
            sim.step()
            if sim.tiempo_simulado > t_sim_to:
                motivo = "timeout_sim"
                break
            if _time.time() - t0 > wall_max:
                motivo = "timeout_wall"
                break
        if motivo == "running":
            motivo = "completado" if sim.estado == "completado" else "limite_pasos"

    m = sim.mundo
    libres = m.total_libres or 1
    vistas = m._scouteadas + m._visitadas
    e_total, _, _ = energia_enjambre(m)
    cob = cobertura_pct("hibrida", vistas, m._inspeccionadas, libres)
    try:
        ncol = int(m.colisiones_totales())
    except Exception:
        ncol = sum(int(getattr(r, "paredes_robot", 0)) for r in m.robots)
    valida = bool(motivo == "completado" and cob >= umbral * 100.0)
    return {
        "t_sim_s": round(sim.tiempo_simulado, 2),
        "energia_j": round(e_total, 3),
        "cobertura_pct": cob,
        "colisiones": ncol,
        "valida": valida,
        "motivo": motivo,
    }


def evaluar_poblacion(pop, n_celdas, n_robots, reps, max_pasos, kappa,
                      t_sim_min, t_sim_max, wall_max, umbral, workers):
    """Evalua cada individuo con `reps` semillas; asigna .objectives."""
    from multiprocessing import Pool
    t_sim_to = timeout_sim_adaptativo(n_celdas, n_robots, kappa=kappa,
                                      t_min=t_sim_min, t_max=t_sim_max)
    trabajos = []
    idx_map = []
    for gi, ind in enumerate(pop):
        g = [float(v) for v in ind.genome.ravel()]
        for rep in range(reps):
            trabajos.append((g, n_celdas, n_robots, rep, max_pasos,
                             t_sim_to, wall_max, umbral))
            idx_map.append(gi)
    with Pool(processes=min(workers, max(1, len(trabajos)))) as pool:
        filas = list(pool.map(correr_genoma_una, trabajos))
    por_ind = [[] for _ in pop]
    for gi, fila in zip(idx_map, filas):
        por_ind[gi].append(fila)
    for ind, filas_i in zip(pop, por_ind):
        validas = [f for f in filas_i if f["valida"]]
        tasa = len(validas) / max(1, len(filas_i))
        ind.n_eval_ok = len(validas)
        ind.tasa_validez = tasa
        if validas:
            e_mean = sum(f["energia_j"] for f in validas) / len(validas)
            t_mean = sum(f["t_sim_s"] for f in validas) / len(validas)
            cob_mean = sum(f["cobertura_pct"] for f in validas) / len(validas)
            ncol_mean = sum(f["colisiones"] for f in validas) / len(validas)
        else:
            e_mean = 0.0
            t_mean = t_sim_to
            cob_mean = sum(f["cobertura_pct"] for f in filas_i) / max(1, len(filas_i))
            ncol_mean = sum(f["colisiones"] for f in filas_i) / max(1, len(filas_i))
        f1 = e_mean / 3.6  # J -> mWh
        f2 = t_mean
        infactible = not (cob_mean >= 98.0 and tasa >= 0.5)
        f3 = (100.0 - cob_mean) + 50.0 * ncol_mean + (P_INFACTIBLE if infactible else 0.0)
        ind.objectives = (float(f1), float(f2), float(f3))
    return pop


def evolucionar_escenario(n_celdas, n_robots, args, out_dir, semilla):
    """Corre el GA completo para un escenario (mapa, robots)."""
    opt = NSGA2Optimizer(seed=semilla)
    pop = opt.initialize_population(args.poblacion)
    historial = []
    t0 = time.time()
    for gen in range(args.generaciones):
        evaluar_poblacion(pop, n_celdas, n_robots, args.reps, args.max_pasos,
                          args.kappa, args.t_sim_min, args.t_sim_max,
                          args.wall_max, args.umbral_cobertura, args.workers)
        fronts = opt.fast_non_dominated_sort(pop)
        for f in fronts:
            opt.calculate_crowding_distance(f)
        f1 = [ [float(v) for v in ind.objectives] for ind in fronts[0]]
        historial.append({
            "escenario": f"{n_celdas}x{n_celdas}_{n_robots}r",
            "generacion": gen + 1,
            "frente_size": len(fronts[0]),
            "mejor_f3": round(min(o[2] for o in f1), 3),
            "mejor_f1_mwh": round(min(o[0] for o in f1), 3),
            "mejor_f2_s": round(min(o[1] for o in f1), 2),
            "tiempo_s": round(time.time() - t0, 1),
        })
        h = historial[-1]
        print(f"  [mapa {n_celdas} robots {n_robots}] gen {gen+1}/{args.generaciones} "
              f"frente={h['frente_size']} mejor_f3={h['mejor_f3']} "
              f"f1={h['mejor_f1_mwh']}mWh f2={h['mejor_f2_s']}s", flush=True)
        # Descendencia Q_t
        hijos = []
        while len(hijos) < args.poblacion:
            p1 = opt.tournament_selection(pop)
            p2 = opt.tournament_selection(pop)
            c1, c2 = opt.crossover_sbx(p1, p2)
            hijos.append(opt.mutate_polynomial(c1))
            if len(hijos) < args.poblacion:
                hijos.append(opt.mutate_polynomial(c2))
        evaluar_poblacion(hijos, n_celdas, n_robots, args.reps, args.max_pasos,
                          args.kappa, args.t_sim_min, args.t_sim_max,
                          args.wall_max, args.umbral_cobertura, args.workers)
        pop = opt.environmental_selection(pop + hijos, args.poblacion)
    fronts = opt.fast_non_dominated_sort(pop)
    opt.calculate_crowding_distance(fronts[0])
    return fronts[0], historial, pop


def guardar_frente_csv(path, frente, n_celdas, n_robots):
    campos = list(GEN_NAMES) + ["mapa", "robots", "f1_mwh", "f2_s", "f3_error",
                                "rank", "crowding", "n_ok", "tasa_validez"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for ind in frente:
            fila = {n: round(float(v), 6)
                    for n, v in zip(GEN_NAMES, ind.genome.ravel())}
            fila.update({
                "mapa": n_celdas, "robots": n_robots,
                "f1_mwh": round(ind.objectives[0], 4),
                "f2_s": round(ind.objectives[1], 2),
                "f3_error": round(ind.objectives[2], 3),
                "rank": ind.rank,
                "crowding": (round(ind.crowding_distance, 4)
                             if ind.crowding_distance != float("inf") else "inf"),
                "n_ok": ind.n_eval_ok,
                "tasa_validez": round(ind.tasa_validez, 3),
            })
            w.writerow(fila)


def graficar_frente_3d(path, frente, titulo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"  (sin grafico 3D: {e}; instala matplotlib)")
        return
    xs = [i.objectives[1] for i in frente]
    ys = [i.objectives[0] for i in frente]
    zs = [i.objectives[2] for i in frente]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Energia [mWh]")
    ax.set_zlabel("Error [%]")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="GA NSGA-II offline (Parte A).")
    ap.add_argument("--poblacion", type=int, default=40)
    ap.add_argument("--generaciones", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mapas", type=int, nargs="*", default=[10, 20, 30, 40, 50])
    ap.add_argument("--robots", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--max-pasos", type=int, default=60000)
    ap.add_argument("--kappa", type=float, default=15.0)
    ap.add_argument("--t-sim-min", type=float, default=300.0)
    ap.add_argument("--t-sim-max", type=float, default=12000.0)
    ap.add_argument("--wall-max", type=float, default=3600.0)
    ap.add_argument("--umbral-cobertura", type=float, default=0.98)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--out", default="resultados_ga")
    ap.add_argument("--piloto", action="store_true",
                    help="Modo rapido: poblacion 10, generaciones 3, reps 2.")
    args = ap.parse_args()
    if args.piloto:
        args.poblacion = 10
        args.generaciones = 3
        args.reps = 2
        args.mapas = [20]
        args.robots = [10]

    args.wall_max = min(float(args.wall_max), WALL_MAX_ABSOLUTO)
    os.makedirs(args.out, exist_ok=True)
    print(f"GA NSGA-II | escenarios={len(args.mapas)*len(args.robots)} "
          f"pop={args.poblacion} gen={args.generaciones} reps={args.reps} "
          f"workers={args.workers} | estrategia fija {FASE1_FIJA}+{FASE2_FIJA}")

    t0 = time.time()
    frentes = {}
    historial_global = []
    for n in args.mapas:
        for r in args.robots:
            sem = args.semilla + n * 100 + r
            print(f"Escenario mapa {n}x{n} robots {r} ...", flush=True)
            frente, hist, _ = evolucionar_escenario(n, r, args, args.out, sem)
            frentes[(n, r)] = frente
            historial_global.extend(hist)
            tag = f"{n}x{n}_{r}r"
            guardar_frente_csv(os.path.join(args.out, f"frente_pareto_{tag}.csv"),
                               frente, n, r)
            graficar_frente_3d(os.path.join(args.out, f"frente_pareto_{tag}.png"),
                               frente, f"Frente Pareto GA {tag}")

    # Frente global: no-dominados entre todos los frentes locales.
    todos = [ind.copy() for frente in frentes.values() for ind in frente]
    if todos:
        frentes_glob = NSGA2Optimizer.fast_non_dominated_sort(todos)
        NSGA2Optimizer.calculate_crowding_distance(frentes_glob[0])
        frente_global = frentes_glob[0]
    else:
        frente_global = []

    def _san(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        try:
            import numpy as _np
            if isinstance(o, _np.floating):
                return float(o)
            if isinstance(o, _np.integer):
                return int(o)
        except Exception:
            pass
        return o

    # Mejor compromiso global: menor f3, desempate menor f1+f2 normalizado.
    mejor = None
    if frente_global:
        f1s = [i.objectives[0] for i in frente_global]
        f2s = [i.objectives[1] for i in frente_global]
        r1 = max(f1s) - min(f1s) or 1.0
        r2 = max(f2s) - min(f2s) or 1.0
        mejor = min(frente_global, key=lambda i: (
            i.objectives[2],
            (i.objectives[0] - min(f1s)) / r1 + (i.objectives[1] - min(f2s)) / r2))
        from src.core.config import genoma_a_yaml
        genoma_a_yaml(
            os.path.join("config", "ga_best_config.yaml"),
            mejor.genome,
            meta={"f1_mwh": mejor.objectives[0], "f2_s": mejor.objectives[1],
                  "f3_error": mejor.objectives[2],
                  "estrategia_exploracion": FASE1_FIJA,
                  "estrategia_inspeccion": FASE2_FIJA})

    with open(os.path.join(args.out, "historial.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["escenario", "generacion", "frente_size",
                                          "mejor_f3", "mejor_f1_mwh", "mejor_f2_s",
                                          "tiempo_s"])
        w.writeheader()
        w.writerows(historial_global)

    resumen = {
        "mejor_global": (_san({
            "genoma": genome_to_dict(mejor.genome),
            "f1_mwh": mejor.objectives[0], "f2_s": mejor.objectives[1],
            "f3_error": mejor.objectives[2]}) if mejor is not None else None),
        "frentes_por_escenario": {
            f"{n}x{n}_{r}r": len(fr) for (n, r), fr in frentes.items()},
        "hiperparametros": {"poblacion": args.poblacion,
                            "generaciones": args.generaciones,
                            "reps": args.reps, "mapas": args.mapas,
                            "robots": args.robots,
                            "bounds": {n: list(b) for n, b in zip(GEN_NAMES, GEN_BOUNDS)},
                            "defaults": {n: v for n, v in zip(GEN_NAMES, GEN_DEFAULTS)}},
    }
    with open(os.path.join(args.out, "mejores_ga.json"), "w",
              encoding="utf-8") as f:
        json.dump(_san(resumen), f, indent=2, ensure_ascii=False)
    print(f"Listo en {time.time()-t0:.1f}s. Salidas en {args.out}/ "
          f"| frente global: {len(frente_global)} puntos.")


if __name__ == "__main__":
    main()
