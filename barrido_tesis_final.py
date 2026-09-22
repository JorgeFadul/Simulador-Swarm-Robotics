"""Barrido TESIS FINAL (diseno del usuario 2026-09-20).

Diseno:
  - Mapas vacios: 10,20,30,40,50 (de 10 en 10).
  - Robots: 5,10,15,20,25,30 (de 5 en 5).
  - Reps: 5 por combinacion.
  - dos_etapas: todas las combinaciones exploracion x inspeccion con
      EXPLORACION = (Hybrid Based, Boustrophedon)
      INSPECCION  = (Voronoi, Boustrophedon Inspeccion)  -> 4 combos
  - hibrido: Random Walk Hibrido, RL Hibrido V3, Cerebro Mosca (Dopamina) -> 3
  - Total grupos: 5*6*7=210, corridas: 1050.

Reglas de parada:
  - Timeout en TIEMPO SIMULADO proporcional al mapa: T=k*N^2/R (kappa, min, max).
  - Timeout de pared proporcional (red de seguridad por sim).
  - STALL: si pasan STALL_S (=120) segundos simulados sin nueva celda
    vista (scouteada+visitada) NI inspeccionada nueva, se detiene
    (motivo stall_120s). Equivale a "sin vista en exp / sin insp en insp-hibrida"
    unificado: cualquier progreso (vista o inspeccion) reinicia el contador.

Serie guardada por corrida (series/<id>.npz), columnas:
  paso, t_sim, vistas, visitadas, inspeccionadas, defectuosas, energia_total_J
Mas resumenes.csv (una fila por corrida).

Uso por partes (reanudable, salta .npz existentes):
  python barrido_tesis_final.py --solo-mapa 10 20 --workers 10
  python barrido_tesis_final.py --solo-robots 5 10 --solo-mapa 50 --workers 10
  python barrido_tesis_final.py --solo-grupo dos_etapas --workers 10
"""

import argparse
import csv
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAPAS = (10, 20, 30, 40, 50)
ROBOTS = (5, 10, 15, 20, 25, 30)
REPS = 5

EXPLORACION = ("Hybrid Based", "Boustrophedon")
INSPECCION = ("Voronoi", "Boustrophedon Inspeccion")
HIBRIDAS = ("Random Walk Hibrido", "RL Hibrido V3",
            "Cerebro Mosca (Dopamina)")

STALL_S = 120.0


def timeout_sim(N, R, kappa=8.0, t_min=200.0, t_max=12000.0):
    R = max(1, int(R))
    return max(float(t_min), min(float(t_max), kappa * float(N) ** 2 / R))


def wall_timeout(N, R, beta=2.0, w_min=120.0, w_max=1800.0):
    R = max(1, int(R))
    return max(float(w_min), min(float(w_max), beta * float(N) ** 2 / R))


def construir_trabajos(solo_mapa=None, solo_robots=None, solo_grupo=None):
    specs = []  # (grupo, exp, insp/hib)
    if solo_grupo is None or "dos_etapas" in solo_grupo:
        for e in EXPLORACION:
            for i in INSPECCION:
                specs.append(("dos_etapas", e, i))
    if solo_grupo is None or "hibrida" in solo_grupo:
        for h in HIBRIDAS:
            specs.append(("hibrida", None, h))
    trabajos_meta = []
    for n in MAPAS:
        if solo_mapa and n not in solo_mapa:
            continue
        for r in ROBOTS:
            if solo_robots and r not in solo_robots:
                continue
            for (g, e, s) in specs:
                for rep in range(REPS):
                    trabajos_meta.append((g, e, s, n, r, rep))
    return trabajos_meta


def correr_una(tupla):
    (grupo, exp, estr, n, r, rep, max_pasos, kappa, t_min, t_max,
     beta, w_min, w_max, stall_s, out_series) = tupla
    from src.core.config import Config
    from src.core.simulacion import Simulacion

    clave = f"{grupo}|{exp}|{estr}|{n}|{r}|{rep}"
    seed = 1000 + int(hashlib.md5(clave.encode()).hexdigest()[:6], 16)
    cfg = Config()
    cfg.grid_size = 20
    cfg.map_width = n * 20
    cfg.map_height = n * 20
    cfg.cols = n
    cfg.rows = n
    cfg.num_robots = r
    cfg.walls = []
    if grupo == "dos_etapas":
        cfg.modo_fase = "dos_etapas"
        cfg.estrategia_exploracion = exp
        cfg.estrategia_inspeccion = estr
    else:
        cfg.modo_fase = "hibrido"
        cfg.estrategia_hibrido = estr
    cfg.sync_strategy()

    t_sim_to = timeout_sim(n, r, kappa, t_min, t_max)
    w_to = wall_timeout(n, r, beta, w_min, w_max)

    if grupo == "dos_etapas":
        run_id = f"dos_etapas_{exp}+{estr}_{n}x{n}_{r}r_rep{rep}"
    else:
        run_id = f"hibrida_{estr}_{n}x{n}_{r}r_rep{rep}"
    run_id = run_id.replace(" ", "_").replace("(", "").replace(")", "")

    t0 = time.time()
    sim = Simulacion(cfg, seed=seed, headless=True)
    sim.poblar_robots(max(1, cfg.num_robots))
    sim.iniciar()

    serie = []
    motivo = "running"
    if not sim.mundo.robots or sim.estado != "running":
        motivo = "sin_robots"
    else:
        m = sim.mundo
        last_prog_t = 0.0
        last_vistas = m._scouteadas + m._visitadas
        last_insp = m._inspeccionadas
        # muestreo cada paso pero guardando energia cada paso (robots<=30, barato)
        while sim.estado == "running" and sim.pasos_totales < max_pasos:
            sim.step()
            m = sim.mundo
            vistas = m._scouteadas + m._visitadas
            insp = m._inspeccionadas
            if vistas > last_vistas or insp > last_insp:
                last_vistas, last_insp = vistas, insp
                last_prog_t = sim.tiempo_simulado
            e_tot = sum(rb.energia.energia_consumida_total for rb in m.robots)
            serie.append((sim.pasos_totales, round(sim.tiempo_simulado, 3),
                          vistas, m._visitadas, insp, m._defectuosas,
                          round(e_tot, 4)))
            if sim.tiempo_simulado - last_prog_t > stall_s:
                motivo = "stall_120s"
                break
            if sim.tiempo_simulado > t_sim_to:
                motivo = "timeout_sim"
                break
            if time.time() - t0 > w_to:
                motivo = "timeout_wall"
                break
        if motivo == "running":
            motivo = "completado" if sim.estado == "completado" else "limite_pasos"

    wall_s = round(time.time() - t0, 1)
    m = sim.mundo
    libres = m.total_libres or 1
    vistas = m._scouteadas + m._visitadas
    e_total = sum(rb.energia.energia_consumida_total for rb in m.robots)
    e_mov = sum(rb.energia.consumida_movimiento for rb in m.robots)
    e_ins = sum(rb.energia.consumida_inspeccion for rb in m.robots)

    import numpy as np
    arr = np.array(serie, dtype=np.float64)
    np.savez_compressed(
        os.path.join(out_series, run_id + ".npz"),
        serie=arr, grupo=grupo, exp=(exp or ""),
        estrategia=estr, mapa=n, robots=r, rep=rep, seed=seed,
        motivo=motivo, libres=libres,
        robots_efectivos=len(m.robots),
        energia_total=round(float(e_total), 3),
        energia_mov=round(float(e_mov), 3),
        energia_ins=round(float(e_ins), 3),
        t_sim_timeout=round(t_sim_to, 1), wall_timeout=round(w_to, 1))
    return {
        "id": run_id, "grupo": grupo, "exp": exp or "",
        "estrategia": estr, "mapa": n, "robots": r, "rep": rep,
        "seed": seed, "motivo": motivo, "pasos": sim.pasos_totales,
        "robots_efectivos": len(sim.mundo.robots),
        "t_sim_s": round(sim.tiempo_simulado, 2), "wall_s": wall_s,
        "vistas": vistas, "visitadas": m._visitadas,
        "inspeccionadas": m._inspeccionadas, "defectuosas": m._defectuosas,
        "energia_total_j": round(float(e_total), 3),
        "energia_mov_j": round(float(e_mov), 3),
        "energia_ins_j": round(float(e_ins), 3),
        "t_sim_timeout": round(t_sim_to, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-pasos", type=int, default=400000)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--t-min", type=float, default=200.0)
    ap.add_argument("--t-max", type=float, default=12000.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--w-min", type=float, default=120.0)
    ap.add_argument("--w-max", type=float, default=1800.0)
    ap.add_argument("--stall-s", type=float, default=STALL_S)
    ap.add_argument("--solo-mapa", type=int, nargs="*", default=None)
    ap.add_argument("--solo-robots", type=int, nargs="*", default=None)
    ap.add_argument("--solo-grupo", nargs="*", default=None,
                    choices=["dos_etapas", "hibrida"])
    ap.add_argument("--out", default="resultados_tesis_final")
    ap.add_argument("--checkpoint-cada", type=int, default=10)
    args = ap.parse_args()

    out_series = os.path.join(args.out, "series")
    os.makedirs(out_series, exist_ok=True)
    meta = construir_trabajos(args.solo_mapa, args.solo_robots,
                              args.solo_grupo)
    trabajos = []
    for (g, e, s, n, r, rep) in meta:
        if g == "dos_etapas":
            rid = f"dos_etapas_{e}+{s}_{n}x{n}_{r}r_rep{rep}"
        else:
            rid = f"hibrida_{s}_{n}x{n}_{r}r_rep{rep}"
        rid = rid.replace(" ", "_").replace("(", "").replace(")", "")
        if os.path.exists(os.path.join(out_series, rid + ".npz")):
            continue
        trabajos.append((g, e, s, n, r, rep, args.max_pasos, args.kappa,
                         args.t_min, args.t_max, args.beta, args.w_min,
                         args.w_max, args.stall_s, out_series))
    print(f"Pendientes: {len(trabajos)} (total diseno {len(meta)}) | "
          f"workers={args.workers} | Tsim=k*N^2/R k={args.kappa} "
          f"[{args.t_min},{args.t_max}] | wall beta={args.beta} "
          f"[{args.w_min},{args.w_max}] | stall={args.stall_s}s",
          flush=True)

    from multiprocessing import Pool
    ruta = os.path.join(args.out, "resumenes.csv")
    campos = ["id", "grupo", "exp", "estrategia", "mapa", "robots", "rep",
              "seed", "motivo", "pasos", "robots_efectivos",
              "seed", "motivo", "pasos", "t_sim_s", "wall_s", "vistas",
              "visitadas", "inspeccionadas", "defectuosas",
              "energia_total_j", "energia_mov_j", "energia_ins_j",
              "t_sim_timeout"]
    if not os.path.exists(ruta):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=campos).writeheader()
    # ids ya resumidos (no duplicar al reanudar)
    hechos_ids = set()
    try:
        with open(ruta, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hechos_ids.add(row["id"])
    except Exception:
        pass
    t0 = time.time()
    hechas = 0
    with Pool(processes=args.workers) as pool:
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            for fila in pool.imap_unordered(correr_una, trabajos):
                if fila["id"] not in hechos_ids:
                    w.writerow(fila)
                    hechos_ids.add(fila["id"])
                hechas += 1
                if hechas % args.checkpoint_cada == 0:
                    f.flush()
                    print(f"[{hechas}/{len(trabajos)}] {time.time()-t0:.0f}s | "
                          f"{fila['id']} ({fila['motivo']}, "
                          f"t_sim={fila['t_sim_s']}s)", flush=True)
    print(f"Listo en {time.time()-t0:.1f}s: {hechas} corridas")


if __name__ == "__main__":
    main()
