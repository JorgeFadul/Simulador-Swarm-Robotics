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
     max_pasos, timeout_s, out_series) = args_tupla
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
            if time.time() - t0 > timeout_s:
                motivo = "timeout"
                break
        if motivo == "running":
            motivo = "completado" if sim.estado == "completado" else "limite_pasos"

    wall_s = round(time.time() - t0, 1)
    m = sim.mundo
    libres = m.total_libres or 1
    vistas = m._scouteadas + m._visitadas

    import numpy as np
    arr = np.array(serie, dtype=np.float64)
    np.savez_compressed(
        os.path.join(out_series, run_id + ".npz"),
        serie=arr, estrategia=estrategia, categoria=categoria,
        mapa=n_celdas, robots=n_robots, rep=rep, seed=seed,
        motivo=motivo, libres=libres)

    return {
        "id": run_id, "categoria": categoria, "estrategia": estrategia,
        "mapa": n_celdas, "robots": n_robots, "rep": rep, "seed": seed,
        "motivo": motivo, "pasos": sim.pasos_totales,
        "t_sim_s": round(sim.tiempo_simulado, 2), "wall_s": wall_s,
        "vistas": vistas, "vistas_pct": round(100.0 * vistas / libres, 2),
        "visitadas": m._visitadas,
        "inspeccionadas": m._inspeccionadas,
        "defectuosas": m._defectuosas,
    }


def main():
    ap = argparse.ArgumentParser(description="Barrido headless de estrategias.")
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--max-pasos", type=int, default=60000)
    ap.add_argument("--timeout-s", type=float, default=1500.0)
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
    trabajos = []
    for i, (cat, est, n, r) in enumerate(combos):
        for rep in range(REPS):
            run_id = (f"{cat}_{est}_{n}x{n}_{r}r_rep{rep}".replace(" ", "_")
                      .replace("(", "").replace(")", ""))
            # Resume: saltar corridas ya guardadas
            if os.path.exists(os.path.join(out_series, run_id + ".npz")):
                continue
            trabajos.append((i, cat, est, n, r, rep, args.max_pasos,
                             args.timeout_s, out_series))
    print(f"Corridas: {len(trabajos)} | workers: {args.workers} | "
          f"max_pasos: {args.max_pasos} | timeout: {args.timeout_s}s")

    from multiprocessing import Pool
    ruta = os.path.join(args.out, "resumenes.csv")
    campos = ["id", "categoria", "estrategia", "mapa", "robots", "rep",
              "seed", "motivo", "pasos", "t_sim_s", "wall_s", "vistas",
              "vistas_pct", "visitadas", "inspeccionadas", "defectuosas"]
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
            todas.append({
                "id": p[:-4], "categoria": str(d["categoria"]),
                "estrategia": str(d["estrategia"]), "mapa": int(d["mapa"]),
                "robots": int(d["robots"]), "rep": int(d["rep"]),
                "seed": int(d["seed"]), "motivo": str(d["motivo"]),
                "pasos": len(s), "t_sim_s": round(float(s[-1, 1]), 2),
                "wall_s": "", "vistas": vistas,
                "vistas_pct": round(100.0 * vistas / libres, 2),
                "visitadas": int(s[-1, 3]),
                "inspeccionadas": int(s[-1, 4]),
                "defectuosas": int(s[-1, 5]),
            })
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(todas)
    print(f"Resumen reconstruido: {ruta} ({len(todas)} corridas) | "
          f"series: {out_series}")


if __name__ == "__main__":
    main()
