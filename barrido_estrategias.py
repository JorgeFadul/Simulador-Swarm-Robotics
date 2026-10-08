"""Barrido de estrategias headless en paralelo.

Diseno experimental:
  - Mapas vacios (sin paredes internas) de 20x20, 50x50 y 80x80 celdas.
  - Robots: 5, 10 y 15. Repeticiones: 10 por combinacion (semillas distintas).
  - Exploracion (modo dos_etapas, fase 1 varia, fase 2 fija Voronoi).
  - Inspeccion (modo dos_etapas, fase 1 fija Hybrid Based, fase 2 varia).
  - Hibridas (modo hibrido).

Guarda TODA la data por corrida:
  resultados_barrido/series/<id>.npz  (serie completa paso a paso)
  resultados_barrido/resumenes.csv    (una fila por corrida)

Uso:
  python barrido_estrategias.py --piloto
  python barrido_estrategias.py --workers 11 --max-pasos 60000 --timeout-s 900
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAPAS = (15, 20, 30, 50)
ROBOTS = (5, 10, 15)
REPS = 10

EXPLORACION = ("Random Walk", "Frontier Based", "Sample Based",
               "Hybrid Based", "Boustrophedon")
INSPECCION = ("Random Walk", "Voronoi", "Boustrophedon Inspeccion",
              "Cerebro Mosca (Dopamina)", "RL Inspeccion V3")
HIBRIDAS = ("Random Walk Hibrido", "RL Hibrido V3",
            "Cerebro Mosca (Dopamina)")

FASE2_FIJA_EXP = "Voronoi"
FASE1_FIJA_INS = "Hybrid Based"


def timeout_adaptativo(n_celdas, n_robots, beta=2.0, t_min=120.0,
                       t_max=1500.0):
    """Timeout de PARED variable adaptativo (legado: solo red de seguridad).

    T_timeout = max(T_min, min(T_max, beta * N_celdas^2 / N_robots))
    """
    n_robots = max(1, int(n_robots))
    t = beta * (float(n_celdas) ** 2) / n_robots
    return max(float(t_min), min(float(t_max), t))


WALL_MAX_ABSOLUTO = 7200.0  # tope de pared: 2 h (decisión del usuario)


def timeout_sim_adaptativo(n_celdas, n_robots, kappa=15.0,
                           t_min=300.0, t_max=12000.0):
    """Timeout en TIEMPO SIMULADO: crece con el mapa, decrece con robots.

    T_sim(N, R) = max(T_min, min(T_max, kappa * N^2 / R)).
    kappa=15 calibra con margen x3 los datos try1-try4
    (t_sim*R/N^2 observado: 3.7-5.3 en rango medio; anomalía
    30x30-5r en 1202 s queda cubierta con holgura).
    """
    n_robots = max(1, int(n_robots))
    t = kappa * (float(n_celdas) ** 2) / n_robots
    return max(float(t_min), min(float(t_max), t))


def cobertura_pct(categoria, vistas, inspeccionadas, libres):
    """Doble umbral: el mínimo entre exploración e inspección.

    Una corrida solo es válida si AMBAS coberturas superan el umbral:
    vistas (scouteadas+visitadas) e inspeccionadas, sobre celdas libres.
    Esto elimina artefactos tipo "completado con 1% inspeccionado".
    """
    libres = max(1, libres)
    vistas_pct = 100.0 * vistas / libres
    inspecc_pct = 100.0 * inspeccionadas / libres
    return round(min(vistas_pct, inspecc_pct), 2)


def energia_enjambre(mundo):
    """Energia total acumulada del enjambre y desglose."""
    robots = getattr(mundo, "robots", []) or []
    e_total = sum(r.energia.energia_consumida_total for r in robots)
    e_mov = sum(r.energia.consumida_movimiento for r in robots)
    e_ins = sum(r.energia.consumida_inspeccion for r in robots)
    return round(e_total, 3), round(e_mov, 3), round(e_ins, 3)


def construir_combinaciones(piloto=False, solo_mapa=None, solo_categoria=None):
    combos = []
    for n in MAPAS:
        if solo_mapa and n not in solo_mapa:
            continue
        for r in ROBOTS:
            for e in EXPLORACION:
                combos.append(("exploracion", e, n, r))
            for i in INSPECCION:
                combos.append(("inspeccion", i, n, r))
            for h in HIBRIDAS:
                combos.append(("hibrida", h, n, r))
    if solo_categoria:
        combos = [c for c in combos if c[0] in solo_categoria]
    if piloto:
        # 2 combos: mapa mediano, 10 robots, una exploracion y una inspeccion
        combos = [c for c in combos
                  if c[2] == 50 and c[3] == 10
                  and c[:2] in (("exploracion", "Frontier Based"),
                                ("inspeccion", "Voronoi"))]
    return combos


def correr_una(args_tupla):
    (idx, categoria, estrategia, n_celdas, n_robots, rep,
     max_pasos, t_sim_timeout, wall_max, umbral, out_series) = args_tupla
    from src.core.config import Config
    from src.core.simulacion import Simulacion

    import hashlib
    clave = f"{categoria}|{estrategia}|{n_celdas}|{n_robots}|{rep}"
    seed = 1000 + int(hashlib.md5(clave.encode()).hexdigest()[:6], 16)
    cfg = Config()
    cfg.grid_size = 20
    cfg.map_width = n_celdas * 20
    cfg.map_height = n_celdas * 20
    cfg.cols = n_celdas
    cfg.rows = n_celdas
    cfg.num_robots = n_robots
    cfg.walls = []
    if categoria == "exploracion":
        cfg.modo_fase = "dos_etapas"
        cfg.estrategia_exploracion = estrategia
        cfg.estrategia_inspeccion = FASE2_FIJA_EXP
    elif categoria == "inspeccion":
        cfg.modo_fase = "dos_etapas"
        cfg.estrategia_exploracion = FASE1_FIJA_INS
        cfg.estrategia_inspeccion = estrategia
    else:
        cfg.modo_fase = "hibrido"
        cfg.estrategia_hibrido = estrategia
    cfg.sync_strategy()

    t0 = time.time()
    sim = Simulacion(cfg, seed=seed, headless=True)
    sim.poblar_robots(max(1, cfg.num_robots))
    sim.iniciar()

    run_id = f"{categoria}_{estrategia}_{n_celdas}x{n_celdas}_{n_robots}r_rep{rep}"
    run_id = run_id.replace(" ", "_").replace("(", "").replace(")", "")

    serie = []
    motivo = "running"
    if not sim.mundo.robots or sim.estado != "running":
        motivo = "sin_robots"
    else:
        while sim.estado == "running" and sim.pasos_totales < max_pasos:
            sim.step()
            m = sim.mundo
            serie.append((sim.pasos_totales, round(sim.tiempo_simulado, 3),
                          m._scouteadas + m._visitadas, m._visitadas,
                          m._inspeccionadas, m._defectuosas))
            if sim.tiempo_simulado > t_sim_timeout:
                motivo = "timeout_sim"
                break
            if time.time() - t0 > wall_max:
                motivo = "timeout_wall"
                break
        if motivo == "running":
            motivo = "completado" if sim.estado == "completado" else "limite_pasos"

    wall_s = round(time.time() - t0, 1)
    m = sim.mundo
    libres = m.total_libres or 1
    vistas = m._scouteadas + m._visitadas
    e_total, e_mov, e_ins = energia_enjambre(m)
    denom = max(1, vistas)
    e_por_celda = round(e_total / denom, 4)

    vistas_pct = round(100.0 * vistas / libres, 2)
    cob = cobertura_pct(categoria, vistas, m._inspeccionadas, libres)
    valida = bool(motivo == "completado" and cob >= umbral * 100.0)

    import numpy as np
    arr = np.array(serie, dtype=np.float64)
    np.savez_compressed(
        os.path.join(out_series, run_id + ".npz"),
        serie=arr, estrategia=estrategia, categoria=categoria,
        mapa=n_celdas, robots=n_robots, rep=rep, seed=seed,
        motivo=motivo, libres=libres,
        energia_total=e_total, energia_mov=e_mov, energia_ins=e_ins,
        energia_por_celda=e_por_celda,
        cobertura_pct=cob, valida=valida,
        t_sim_timeout=t_sim_timeout)

    return {
        "id": run_id, "categoria": categoria, "estrategia": estrategia,
        "mapa": n_celdas, "robots": n_robots, "rep": rep, "seed": seed,
        "motivo": motivo, "pasos": sim.pasos_totales,
        "t_sim_s": round(sim.tiempo_simulado, 2), "wall_s": wall_s,
        "vistas": vistas, "vistas_pct": vistas_pct,
        "visitadas": m._visitadas,
        "inspeccionadas": m._inspeccionadas,
        "defectuosas": m._defectuosas,
        "energia_total_j": e_total, "energia_mov_j": e_mov,
        "energia_ins_j": e_ins, "energia_por_celda_j": e_por_celda,
        "cobertura_pct": cob, "valida": valida,
        "t_sim_timeout_usado": round(t_sim_timeout, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Barrido headless de estrategias.")
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--max-pasos", type=int, default=60000)
    ap.add_argument("--timeout-s", type=float, default=1500.0)
    ap.add_argument("--timeout-adaptativo", action="store_true", default=True,
                    help="Usa T_timeout(n_celdas, n_robots) en vez del fijo.")
    ap.add_argument("--no-timeout-adaptativo", dest="timeout_adaptativo",
                    action="store_false")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--t-min", type=float, default=120.0)
    ap.add_argument("--t-max", type=float, default=1500.0)
    ap.add_argument("--kappa", type=float, default=15.0,
                    help="Factor del timeout en tiempo simulado.")
    ap.add_argument("--t-sim-min", type=float, default=300.0)
    ap.add_argument("--t-sim-max", type=float, default=12000.0)
    ap.add_argument("--wall-max", type=float, default=3600.0,
                    help="Guardia de pared por sim (tope absoluto 7200 s).")
    ap.add_argument("--umbral-cobertura", type=float, default=0.98,
                    help="Fraccion de cobertura para corrida valida.")
    ap.add_argument("--checkpoint-cada", type=int, default=10,
                    help="Imprime progreso y vuelca CSV cada N sims.")
    ap.add_argument("--out", default="resultados_barrido")
    ap.add_argument("--solo-mapa", type=int, nargs="*", default=None,
                    help="Solo esos tamanos de mapa (20, 50, 80).")
    ap.add_argument("--solo-categoria", nargs="*", default=None,
                    choices=["exploracion", "inspeccion", "hibrida"])
    args = ap.parse_args()

    out_series = os.path.join(args.out, "series")
    os.makedirs(out_series, exist_ok=True)

    combos = construir_combinaciones(piloto=args.piloto,
                                     solo_mapa=args.solo_mapa,
                                     solo_categoria=args.solo_categoria)
    wall_max = min(float(args.wall_max), WALL_MAX_ABSOLUTO)
    trabajos = []
    for i, (cat, est, n, r) in enumerate(combos):
        t_sim = timeout_sim_adaptativo(n, r, kappa=args.kappa,
                                       t_min=args.t_sim_min,
                                       t_max=args.t_sim_max)
        for rep in range(REPS):
            run_id = (f"{cat}_{est}_{n}x{n}_{r}r_rep{rep}".replace(" ", "_")
                      .replace("(", "").replace(")", ""))
            # Resume: saltar corridas ya guardadas
            if os.path.exists(os.path.join(out_series, run_id + ".npz")):
                continue
            trabajos.append((i, cat, est, n, r, rep, args.max_pasos,
                             t_sim, wall_max, args.umbral_cobertura,
                             out_series))
    print(f"Corridas: {len(trabajos)} | workers: {args.workers} | "
          f"max_pasos: {args.max_pasos} | T_sim adaptativo "
          f"(kappa={args.kappa}, [{args.t_sim_min}, {args.t_sim_max}]s) | "
          f"wall_max={wall_max}s | umbral={args.umbral_cobertura}")

    from multiprocessing import Pool
    ruta = os.path.join(args.out, "resumenes.csv")
    campos = ["id", "categoria", "estrategia", "mapa", "robots", "rep",
              "seed", "motivo", "pasos", "t_sim_s", "wall_s", "vistas",
              "vistas_pct", "visitadas", "inspeccionadas", "defectuosas",
              "energia_total_j", "energia_mov_j", "energia_ins_j",
              "energia_por_celda_j", "cobertura_pct", "valida",
              "t_sim_timeout_usado"]
    # Checkpoint: cabecera al inicio; filas anexadas cada N sims.
    if not os.path.exists(ruta):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=campos).writeheader()

    t0 = time.time()
    hechas = 0
    with Pool(processes=args.workers) as pool:
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            for fila in pool.imap_unordered(correr_una, trabajos):
                w.writerow(fila)
                hechas += 1
                if hechas % args.checkpoint_cada == 0:
                    f.flush()
                    print(f"[{hechas}/{len(trabajos)}] "
                          f"{time.time() - t0:.0f}s | ultimo: {fila['id']} "
                          f"({fila['motivo']}, t_sim={fila['t_sim_s']}s)",
                          flush=True)
    print(f"Listo en {time.time() - t0:.1f}s: {hechas} corridas")

    # Reconstruir resumen completo desde los .npz (nada se pierde aunque
    # el proceso muera: la data por corrida siempre queda en series/).
    import numpy as np
    todas = []
    for p in sorted(os.listdir(out_series)):
        if not p.endswith(".npz"):
            continue
        d = np.load(os.path.join(out_series, p), allow_pickle=True)
        s = d["serie"]
        libres = int(d["libres"])
        if len(s):
            vistas = int(s[-1, 2])
            e_tot = float(d["energia_total"]) if "energia_total" in d else 0.0
            e_mov = float(d["energia_mov"]) if "energia_mov" in d else 0.0
            e_ins = float(d["energia_ins"]) if "energia_ins" in d else 0.0
            e_pc = (float(d["energia_por_celda"])
                    if "energia_por_celda" in d
                    else round(e_tot / max(1, vistas), 4))
            cat = str(d["categoria"])
            vistas_pct = round(100.0 * vistas / libres, 2)
            inspecc = int(s[-1, 4])
            if "cobertura_pct" in d:
                cob = float(d["cobertura_pct"])
                val = bool(d["valida"])
            else:  # npz legado: recalcular con umbral por defecto
                cob = cobertura_pct(cat, vistas, inspecc, libres)
                val = str(d["motivo"]) == "completado" and cob >= 98.0
            t_sim_to = (float(d["t_sim_timeout"])
                        if "t_sim_timeout" in d else 0.0)
            todas.append({
                "id": p[:-4], "categoria": cat,
                "estrategia": str(d["estrategia"]), "mapa": int(d["mapa"]),
                "robots": int(d["robots"]), "rep": int(d["rep"]),
                "seed": int(d["seed"]), "motivo": str(d["motivo"]),
                "pasos": len(s), "t_sim_s": round(float(s[-1, 1]), 2),
                "wall_s": "", "vistas": vistas,
                "vistas_pct": vistas_pct,
                "visitadas": int(s[-1, 3]),
                "inspeccionadas": inspecc,
                "defectuosas": int(s[-1, 5]),
                "energia_total_j": round(e_tot, 3),
                "energia_mov_j": round(e_mov, 3),
                "energia_ins_j": round(e_ins, 3),
                "energia_por_celda_j": e_pc,
                "cobertura_pct": cob, "valida": val,
                "t_sim_timeout_usado": round(t_sim_to, 1),
            })
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(todas)
    print(f"Resumen reconstruido: {ruta} ({len(todas)} corridas) | "
          f"series: {out_series}")


if __name__ == "__main__":
    main()
