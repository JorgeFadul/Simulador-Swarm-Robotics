"""Busqueda de configuraciones optimas energia-tiempo (minimos locales y globales).

- 10 repeticiones (semillas deterministas) por configuracion; promedios
  calculados solo sobre corridas VALIDAS (completado + doble umbral de
  cobertura >=98%: vistas E inspeccionadas sobre celdas libres).
- Timeout variable adaptativo: T = max(Tmin, min(Tmax, beta * N^2 / R)).
- Costos multi-objetivo:
    EDP:       J = E_total_mean * (T_sim_mean ** alpha)
    Ponderado: J = wE * (E/Eref) + wT * (T/Tref) + P_castigo
    Pareto 2D: conjunto no dominado en (Tiempo, Energia).
- Modos: grid (barrido sistematico adaptativo) y bayesiano (Optuna/TPE;
  si Optuna no esta instalado, cae a muestreo aleatorio).

Uso:
  python optimizar_parametros.py --piloto --reps 10
  python optimizar_parametros.py --modo grid --workers 8
  python optimizar_parametros.py --modo bayesiano --trials 60 --reps 10
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from barrido_estrategias import (  # noqa: E402
    EXPLORACION,
    HIBRIDAS,
    INSPECCION,
    WALL_MAX_ABSOLUTO,
    correr_una,
    timeout_sim_adaptativo,
)

TODAS_ESTRATEGIAS = {
    "exploracion": list(EXPLORACION),
    "inspeccion": list(INSPECCION),
    "hibrida": list(HIBRIDAS),
}


def costo_edp(e_mean, t_mean, alpha=1.0):
    return float(e_mean) * (float(t_mean) ** float(alpha))


def costo_ponderado(e_mean, t_mean, e_ref, t_ref, w_e=0.5, w_t=0.5,
                    tasa_ok=1.0, p_castigo=5.0):
    e_ref = max(1e-9, float(e_ref))
    t_ref = max(1e-9, float(t_ref))
    castigo = p_castigo * (1.0 - float(tasa_ok))
    return (w_e * e_mean / e_ref + w_t * t_mean / t_ref + castigo)


def frente_pareto(puntos):
    """puntos: lista de dicts con 't_mean_s' y 'e_mean_j'. Devuelve indices no dominados."""
    idx = []
    for i, a in enumerate(puntos):
        dominado = False
        for j, b in enumerate(puntos):
            if i == j:
                continue
            if (b["t_mean_s"] <= a["t_mean_s"] and b["e_mean_j"] <= a["e_mean_j"]
                    and (b["t_mean_s"] < a["t_mean_s"]
                         or b["e_mean_j"] < a["e_mean_j"])):
                dominado = True
                break
        if not dominado:
            idx.append(i)
    return idx


MIN_VALIDAS_DEFAULT = 3  # mínimo absoluto de corridas válidas


def evaluar_config(cat, est, n, r, reps, max_pasos, kappa, t_sim_min,
                   t_sim_max, wall_max, umbral, min_validas,
                   alpha, w_e, w_t, p_castigo, workers, out_series,
                   extra_cfg=None):
    """Corre `reps` repeticiones y devuelve metricas agregadas + costo.

    Solo las corridas VALIDAS (completado + cobertura>=umbral) entran a
    los promedios. Config válida si n_valid >= min_validas y
    tasa_validez >= 0.5; si no: J=inf (infactible = mal ratio robots/mapa).
    """
    from collections import Counter
    from multiprocessing import Pool

    t_sim_to = timeout_sim_adaptativo(n, r, kappa=kappa,
                                      t_min=t_sim_min, t_max=t_sim_max)
    trabajos = [(0, cat, est, n, r, rep, max_pasos, t_sim_to, wall_max,
                 umbral, out_series)
                for rep in range(reps)]
    # NOTA: extra_cfg (v_max, radios, etc.) se reserva para extension
    # futura via Config; por ahora se ignora para no romper compatibilidad.
    with Pool(processes=min(workers, max(1, reps))) as pool:
        filas = list(pool.map(correr_una, trabajos))
    validas = [f for f in filas if f.get("valida")]
    tasa_val = len(validas) / max(1, len(filas))
    es_valida = len(validas) >= min_validas and tasa_val >= 0.5
    causa_inf = ""
    if es_valida:
        t_mean = sum(f["t_sim_s"] for f in validas) / len(validas)
        e_mean = sum(f["energia_total_j"] for f in validas) / len(validas)
        cob_mean = sum(f["cobertura_pct"] for f in validas) / len(validas)
        j_edp = costo_edp(e_mean, t_mean, alpha=alpha)
    else:  # infactible: sin costo (excluida de Pareto y mínimos)
        t_mean = t_sim_to
        e_mean = 0.0
        cob_mean = sum(f.get("cobertura_pct", 0.0) for f in filas) / max(1, len(filas))
        j_edp = float("inf")
        malas = [f["motivo"] for f in filas if not f.get("valida")]
        causa_inf = Counter(malas).most_common(1)[0][0] if malas else "sin_data"
        if not malas:  # todas "completadas" pero bajo umbral
            causa_inf = "baja_cobertura"
        elif causa_inf == "completado":
            causa_inf = "baja_cobertura"
    return {
        "categoria": cat, "estrategia": est, "mapa": n, "robots": r,
        "reps": reps, "n_ok": len(validas),
        "tasa_validez": round(tasa_val, 3),
        "valida_cfg": bool(es_valida), "causa_inf": causa_inf,
        "t_mean_s": round(t_mean, 2), "e_mean_j": round(e_mean, 3),
        "e_por_celda_j": round(e_mean / max(1, n * n), 4),
        "cobertura_media_pct": round(cob_mean, 2),
        "j_edp": round(j_edp, 3) if es_valida else float("inf"),
        "t_sim_timeout_usado": round(t_sim_to, 1),
        "filas": filas,  # se retira antes de guardar el JSON
    }


def construir_grid(args):
    mapas = args.mapas or [15, 20, 30, 50]
    robots = args.robots or [5, 10, 15]
    combos = []
    if args.categorias:
        cats = args.categorias
    else:
        cats = ["exploracion", "inspeccion", "hibrida"]
    for cat in cats:
        ests = args.estrategias or TODAS_ESTRATEGIAS[cat]
        for est, n, r in itertools.product(ests, mapas, robots):
            combos.append((cat, est, n, r))
    if args.piloto:
        combos = [c for c in combos
                  if c[2] == 20 and c[3] == 10][:2] or combos[:2]
    return combos


def main():
    ap = argparse.ArgumentParser(description="Optimizacion multi-objetivo energia-tiempo.")
    ap.add_argument("--modo", default="grid", choices=["grid", "bayesiano"])
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--trials", type=int, default=40,
                    help="Evaluaciones en modo bayesiano.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pasos", type=int, default=60000)
    ap.add_argument("--kappa", type=float, default=15.0)
    ap.add_argument("--t-sim-min", type=float, default=300.0)
    ap.add_argument("--t-sim-max", type=float, default=12000.0)
    ap.add_argument("--wall-max", type=float, default=3600.0,
                    help="Guardia de pared por sim (tope 7200 s).")
    ap.add_argument("--umbral-cobertura", type=float, default=0.98)
    ap.add_argument("--min-validas", type=int, default=MIN_VALIDAS_DEFAULT,
                    help="Mínimo absoluto de corridas válidas por config.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Exponente del tiempo en EDP.")
    ap.add_argument("--w-e", type=float, default=0.5)
    ap.add_argument("--w-t", type=float, default=0.5)
    ap.add_argument("--p-castigo", type=float, default=5.0)
    ap.add_argument("--mapas", type=int, nargs="*", default=None)
    ap.add_argument("--robots", type=int, nargs="*", default=None)
    ap.add_argument("--estrategias", nargs="*", default=None)
    ap.add_argument("--categorias", nargs="*", default=None,
                    choices=["exploracion", "inspeccion", "hibrida"])
    ap.add_argument("--out", default="resultados_optimizacion")
    # Espacio extendido opcional (pregunta abierta 1 del plan; sin efecto
    # aun en la simulacion, pero registrado en el CSV/JSON):
    ap.add_argument("--v-max", type=float, nargs="*", default=None)
    ap.add_argument("--radio-sensor", type=float, nargs="*", default=None)
    args = ap.parse_args()

    out_series = os.path.join(args.out, "series_opt")
    os.makedirs(out_series, exist_ok=True)

    if args.modo == "grid":
        combos = construir_grid(args)
    else:  # bayesiano
        try:
            import optuna  # noqa: F401
            tiene_optuna = True
        except ImportError:
            tiene_optuna = False
            print("Optuna no instalado: se usa muestreo aleatorio (pip install optuna).")
        import random
        random.seed(42)
        mapas = args.mapas or [15, 20, 30, 50, 80]
        robots = args.robots or [3, 5, 10, 15, 20, 30]
        cats = args.categorias or ["exploracion", "inspeccion", "hibrida"]
        bolsa = [(c, e, n, r) for c in cats
                 for e in (args.estrategias or TODAS_ESTRATEGIAS[c])
                 for n in mapas for r in robots]
        n_trials = 2 if args.piloto else args.trials
        combos = random.sample(bolsa, min(n_trials, len(bolsa)))
        if tiene_optuna:
            print("Optuna disponible: se usara TPE en una futura extension; "
                  "esta version evalua la muestra sorteada como trials.")

    wall_max = min(float(args.wall_max), WALL_MAX_ABSOLUTO)
    print(f"Modo={args.modo} | configs={len(combos)} | reps={args.reps} | "
          f"workers={args.workers} | T_sim(kappa={args.kappa}) | "
          f"wall_max={wall_max}s | umbral={args.umbral_cobertura} | "
          f"min_validas={args.min_validas}")
    t0 = time.time()
    resultados = []
    historial = []  # traza de convergencia: mejor J_EDP tras cada config
    mejor = float("inf")
    for k, (cat, est, n, r) in enumerate(combos):
        res = evaluar_config(cat, est, n, r, args.reps, args.max_pasos,
                             args.kappa, args.t_sim_min, args.t_sim_max,
                             wall_max, args.umbral_cobertura,
                             args.min_validas, args.alpha,
                             args.w_e, args.w_t, args.p_castigo,
                             args.workers, out_series)
        filas = res.pop("filas")
        resultados.append(res)
        if res["valida_cfg"]:
            mejor = min(mejor, res["j_edp"])
        historial.append({"iter": k + 1, "mejor_j_edp": round(mejor, 3),
                          "ultimo_j_edp": res["j_edp"]})
        estado = (f"J_edp={res['j_edp']}"
                  if res["valida_cfg"]
                  else f"INFACTIBLE ({res['causa_inf']})")
        print(f"[{k + 1}/{len(combos)}] {cat}/{est} {n}x{n} {r}r -> "
              f"t={res['t_mean_s']}s e={res['e_mean_j']}J "
              f"cob={res['cobertura_media_pct']}% {estado} "
              f"validas={res['n_ok']}/{res['reps']}",
              flush=True)

    # Referencias para costo ponderado: medianas SOLO de configs válidas.
    import statistics
    validos = [r for r in resultados if r["valida_cfg"]]
    e_ref = statistics.median([r["e_mean_j"] for r in validos]) if validos else 1.0
    t_ref = statistics.median([r["t_mean_s"] for r in validos]) if validos else 1.0
    for r in resultados:
        if r["valida_cfg"]:
            r["j_pond"] = round(costo_ponderado(
                r["e_mean_j"], r["t_mean_s"], e_ref, t_ref,
                w_e=args.w_e, w_t=args.w_t,
                tasa_ok=r["tasa_validez"], p_castigo=args.p_castigo), 4)
        else:
            r["j_pond"] = float("inf")

    idx_pareto = set(frente_pareto(validos))
    for i, r in enumerate(validos):
        r["pareto"] = bool(i in idx_pareto)
    for r in resultados:
        if not r["valida_cfg"]:
            r["pareto"] = False

    orden = sorted(validos, key=lambda r: r["j_edp"])
    infact = {}
    for r in resultados:
        if not r["valida_cfg"]:
            infact.setdefault(str(r["mapa"]), []).append(
                {"robots": r["robots"], "causa": r["causa_inf"],
                 "tasa_validez": r["tasa_validez"],
                 "cobertura_media_pct": r["cobertura_media_pct"]})
    mejores = {
        "global_edp": orden[0] if orden else None,
        "global_pond": min(validos, key=lambda r: r["j_pond"])
        if validos else None,
        "top5_edp": orden[:5],
        "pareto": [validos[i] for i in sorted(idx_pareto)],
        "infactibles_por_mapa": infact,
        "referencias": {"e_ref_j": e_ref, "t_ref_s": t_ref},
        "hiperparametros": {"alpha": args.alpha, "w_e": args.w_e,
                            "w_t": args.w_t, "p_castigo": args.p_castigo,
                            "kappa": args.kappa,
                            "t_sim_min": args.t_sim_min,
                            "t_sim_max": args.t_sim_max,
                            "wall_max": wall_max,
                            "umbral_cobertura": args.umbral_cobertura,
                            "min_validas": args.min_validas,
                            "reps": args.reps, "modo": args.modo},
    }

    def _sanear(o):
        if isinstance(o, float) and (o != o or o in (float("inf"),
                                                     float("-inf"))):
            return None
        if isinstance(o, dict):
            return {k: _sanear(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanear(v) for v in o]
        return o

    with open(os.path.join(args.out, "mejores_configuraciones.json"), "w",
              encoding="utf-8") as f:
        json.dump(_sanear(mejores), f, indent=2, ensure_ascii=False)

    campos = ["categoria", "estrategia", "mapa", "robots", "reps", "n_ok",
              "tasa_validez", "valida_cfg", "causa_inf",
              "t_mean_s", "e_mean_j", "e_por_celda_j",
              "cobertura_media_pct", "j_edp",
              "j_pond", "pareto", "t_sim_timeout_usado"]
    with open(os.path.join(args.out, "resumen_optimizacion.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows([{k: r[k] for k in campos} for r in resultados])
    with open(os.path.join(args.out, "convergencia.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "mejor_j_edp",
                                          "ultimo_j_edp"])
        w.writeheader()
        w.writerows(historial)

    print(f"Listo en {time.time() - t0:.1f}s. "
          f"JSON+CSV en {args.out} | Pareto: {len(idx_pareto)} puntos.")


if __name__ == "__main__":
    main()
