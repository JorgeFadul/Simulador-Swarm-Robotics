"""Graficas TESIS FINAL.

Metricas vs tiempo (todas en % salvo indicacion), promedio de 5 reps:
  1. %celdas_vistas (scouteadas+visitadas)/libres
  2. %celdas_visitadas
  3. %celdas_inspeccionadas
  4. velocidad de inspeccion: %insp/s (derivada, ventana suavizada) y celdas/s
  5. %inspeccionadas / energia (pct por julio) y celdas/J

Salidas en figs_tesis/:
  01_curvas/<grupo>/<config>/  curvas vs tiempo real (s) por n robots y por mapa
  02_resumen_mapa/             promedio sobre robots: metrica final vs mapa
  03_resumen_robots/           promedio sobre mapas: metrica final vs robots
  04_promedios/                barras generales + heatmaps + tabla CSV
  datos/                      curvas_promedio.csv, resumen_final.csv, promedios.csv

Uso: python graficar_tesis_final.py [--series resultados_tesis_final/series] [--out resultados_tesis_final/figs]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAU_N = 200  # puntos de la grilla temporal comun


def etiqueta(grupo, exp, estr):
    if grupo == "dos_etapas":
        return f"{exp}+{estr}"
    return estr


def cargar_series(d):
    runs = []
    for p in sorted(os.listdir(d)):
        if not p.endswith(".npz"):
            continue
        z = np.load(os.path.join(d, p), allow_pickle=True)
        s = z["serie"]
        if len(s) < 2:
            continue
        runs.append({
            "grupo": str(z["grupo"]), "exp": str(z["exp"]),
            "estrategia": str(z["estrategia"]),
            "mapa": int(z["mapa"]), "robots": int(z["robots"]),
            "rep": int(z["rep"]), "seed": int(z["seed"]),
            "motivo": str(z["motivo"]), "libres": int(z["libres"]),
            "key": f"{z['grupo']}|{z['exp']}|{z['estrategia']}|"
                   f"{int(z['mapa'])}|{int(z['robots'])}|{int(z['rep'])}",
            "t": s[:, 1].astype(float), "vistas": s[:, 2].astype(float),
            "visit": s[:, 3].astype(float), "insp": s[:, 4].astype(float),
            "defec": s[:, 5].astype(float),
            "e": s[:, 6].astype(float) if s.shape[1] > 6 else np.zeros(len(s)),
        })
    return runs


def metricas_run(r):
    L = max(1, r["libres"])
    out = {
        "p_vistas": r["vistas"] / L * 100.0,
        "p_visit": r["visit"] / L * 100.0,
        "p_insp": r["insp"] / L * 100.0,
    }
    dt = np.diff(r["t"])
    dt[dt <= 0] = 0.05
    d_insp = np.diff(r["insp"])
    vel_cells = np.concatenate([[0.0], d_insp / dt])
    vel_pct = vel_cells / L * 100.0
    # suavizado ventana 21
    k = np.ones(21) / 21.0
    out["vel_pct"] = np.convolve(vel_pct, k, mode="same")
    out["vel_cells"] = np.convolve(vel_cells, k, mode="same")
    e = np.maximum(r["e"], 1e-9)
    out["pct_por_J"] = out["p_insp"] / e
    out["cells_por_J"] = r["insp"] / e
    out["e_abs"] = r["e"].astype(float)
    return out


def promediar_grupo(runs):
    """Interpola cada corrida a grilla comun 0..t_max y promedia."""
    t_max = max(r["t"][-1] for r in runs)
    tg = np.linspace(0, t_max, TAU_N)
    acc = {}
    for r in runs:
        m = metricas_run(r)
        for k, v in m.items():
            acc.setdefault(k, []).append(np.interp(tg, r["t"], v))
    media = {k: float(np.mean(v, axis=0)[-1]) for k, v in acc.items()}
    curvas = {k: (np.mean(v, axis=0), np.std(v, axis=0)) for k, v in acc.items()}
    finales = {"p_vistas": np.mean([metricas_run(r)["p_vistas"][-1] for r in runs]),
               "p_visit": np.mean([metricas_run(r)["p_visit"][-1] for r in runs]),
               "p_insp": np.mean([metricas_run(r)["p_insp"][-1] for r in runs]),
               "vel_pct_max": np.mean([metricas_run(r)["vel_pct"].max() for r in runs]),
               "pct_J": np.mean([metricas_run(r)["pct_por_J"][-1] for r in runs]),
               "cells_J": np.mean([metricas_run(r)["cells_por_J"][-1] for r in runs]),
               "t_fin": np.mean([r["t"][-1] for r in runs]),
               "e_tot": np.mean([r["e"][-1] for r in runs])}
    return tg, curvas, finales, len(runs)


def slug(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-").replace("+", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="resultados_tesis_final/series")
    ap.add_argument("--out", default="resultados_tesis_final/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for p in os.listdir(args.out):
        r = os.path.join(args.out, p)
        if p.endswith(".png"):
            os.remove(r)
        elif os.path.isdir(r) and p != "datos":
            import shutil
            shutil.rmtree(r, ignore_errors=True)

    runs = cargar_series(args.series)
    print(f"Corridas cargadas: {len(runs)}")
    if not runs:
        return
    # agrupar por (grupo,exp,estrategia,mapa,robots)
    grupos = {}
    for r in runs:
        grupos.setdefault((r["grupo"], r["exp"], r["estrategia"],
                           r["mapa"], r["robots"]), []).append(r)
    prom = {}
    for k, v in grupos.items():
        prom[k] = promediar_grupo(v)

    METS = [("p_vistas", "% celdas vistas"),
            ("p_visit", "% celdas visitadas"),
            ("p_insp", "% celdas inspeccionadas"),
            ("vel_pct", "velocidad insp (%/s)"),
            ("pct_por_J", "%insp / energia (pct/J)"),
            ("e_abs", "energia total (J)")]
    YLAB = dict(METS)

    # 01 curvas: por config fija, lineas = robots (para cada mapa) y lineas = mapas (para cada robots)
    for (g, e, s, m, rb) in prom:
        pass
    configs = sorted({(k[0], k[1], k[2]) for k in prom})
    for (g, e, s) in configs:
        lab = etiqueta(g, e, s)
        mapas = sorted({k[3] for k in prom if k[:3] == (g, e, s)})
        robots = sorted({k[4] for k in prom if k[:3] == (g, e, s)})
        for m in mapas:
            fig, axes = plt.subplots(3, 2, figsize=(11, 12))
            for i, (ax, (mk, yl)) in enumerate(zip(axes.flat, METS)):
                for rb in robots:
                    key = (g, e, s, m, rb)
                    if key not in prom:
                        continue
                    tg, curvas, fin, n = prom[key]
                    media, std = curvas[mk]
                    ax.plot(tg, media, label=f"{rb}r")
                    ax.fill_between(tg, media - std, media + std, alpha=0.12)
                ax.set_xlabel("tiempo (s)")
                ax.set_ylabel(yl)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
                ax.text(0.5, -0.28, f"({chr(97 + i)})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=12, fontweight="bold")
            fig.suptitle(f"{lab} | mapa {m}x{m} (n={prom.get((g,e,s,m,robots[0]),(None,None,None,0))[3] if prom else 0} reps)", fontsize=10, fontweight="bold")
            fig.tight_layout()
            d = os.path.join(args.out, "01_curvas", g, slug(lab))
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, f"vs_robots_mapa{m}.png"), dpi=110)
            plt.close(fig)

    # 02/03 resumenes: metrica final vs mapa (promedio robots) y vs robots (promedio mapas)
    FINALES = [("p_vistas", "% vistas final"), ("p_visit", "% visitadas final"),
               ("p_insp", "% inspeccionadas final"),
               ("vel_pct_max", "vel. max (%/s)"),
               ("pct_J", "%insp/J final"), ("cells_J", "celdas/J final"),
               ("t_fin", "t fin (s)")]
    for eje, nombre, carpeta in (("mapa", "robots", "02_resumen_mapa"),
                                 ("robots", "mapa", "03_resumen_robots")):
        for fk, fl in FINALES:
            fig = plt.figure(figsize=(10, 5.5))
            for (g, e, s) in configs:
                lab = etiqueta(g, e, s)
                xs, ys = [], []
                vals = sorted({k[3] if eje == "mapa" else k[4] for k in prom if k[:3] == (g, e, s)})
                for v in vals:
                    sel = [prom[k][2][fk] for k in prom if k[:3] == (g, e, s) and (k[3] == v if eje == "mapa" else k[4] == v)]
                    if sel:
                        xs.append(v)
                        ys.append(float(np.mean(sel)))
                if xs:
                    plt.plot(xs, ys, marker="o", label=lab)
            plt.xlabel("mapa (celdas)" if eje == "mapa" else "n robots")
            plt.ylabel(fl)
            plt.title(f"{fl} vs {eje} (promedio sobre {nombre})", fontweight="bold")
            plt.legend(fontsize=7)
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            d = os.path.join(args.out, carpeta)
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, f"{fk}_vs_{eje}.png"), dpi=110)
            plt.close(fig)

    # 04b comparativa general de energia vs tiempo: todas las estrategias
    # del mismo grupo superpuestas (promedio sobre robots, banda = std).
    for g in sorted({c[0] for c in configs}):
        mapas_g = sorted({k[3] for k in prom if k[0] == g})
        for m in mapas_g:
            estrat = sorted({(k[1], k[2]) for k in prom
                             if k[0] == g and k[3] == m})
            fig = plt.figure(figsize=(10, 5.5))
            for (e, s) in estrat:
                curvas_rb = []
                for k in sorted(prom):
                    if k[:3] == (g, e, s) and k[3] == m:
                        tg, curvas, fin, n = prom[k]
                        curvas_rb.append((tg, curvas["e_abs"][0],
                                          curvas["e_abs"][1]))
                if not curvas_rb:
                    continue
                t_max = max(tg[-1] for tg, _, _ in curvas_rb)
                tgrid = np.linspace(0, t_max, TAU_N)
                apil = [np.interp(tgrid, tg, med) for tg, med, _ in curvas_rb]
                media = np.mean(apil, axis=0)
                std = np.std(apil, axis=0)
                lab = etiqueta(g, e, s)
                plt.plot(tgrid, media, label=lab)
                plt.fill_between(tgrid, media - std, media + std, alpha=0.12)
            plt.xlabel("tiempo (s)")
            plt.ylabel("energia total (J)")
            plt.title(f"Energia vs tiempo por estrategia | mapa {m}x{m} | {g}",
                      fontweight="bold")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            d = os.path.join(args.out, "05_comparacion_energia", g)
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, f"energia_vs_tiempo_mapa{m}.png"),
                        dpi=110)
            plt.close(fig)

    # 04 promedios: barras por estrategia (promedio global) + heatmap p_insp
    d04 = os.path.join(args.out, "04_promedios")
    os.makedirs(d04, exist_ok=True)
    for fk, fl in FINALES:
        labs = [etiqueta(g, e, s) for (g, e, s) in configs]
        vals = [float(np.mean([prom[k][2][fk] for k in prom if k[:3] == c])) for c in configs]
        fig = plt.figure(figsize=(11, 5))
        plt.bar(labs, vals)
        plt.ylabel(fl)
        plt.title(f"Promedio global por estrategia: {fl}", fontweight="bold")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(d04, f"global_{fk}.png"), dpi=110)
        plt.close(fig)

    def heatmap(fin_key, titulo, unidad, fname, fmt):
        """Heatmap por config: filas robots, cols mapas."""
        for (g, e, s) in configs:
            lab = etiqueta(g, e, s)
            mapas = sorted({k[3] for k in prom if k[:3] == (g, e, s)})
            robots = sorted({k[4] for k in prom if k[:3] == (g, e, s)})
            Z = np.full((len(robots), len(mapas)), np.nan)
            for i, rb in enumerate(robots):
                for j, m in enumerate(mapas):
                    key = (g, e, s, m, rb)
                    if key in prom:
                        Z[i, j] = prom[key][2][fin_key]
            fig = plt.figure(figsize=(1.6 * len(mapas) + 3, 1.0 * len(robots) + 2.5))
            plt.imshow(Z, aspect="auto", origin="upper")
            plt.colorbar(label=unidad)
            plt.xticks(range(len(mapas)), mapas)
            plt.yticks(range(len(robots)), robots)
            plt.xlabel("mapa")
            plt.ylabel("robots")
            for i in range(len(robots)):
                for j in range(len(mapas)):
                    if not np.isnan(Z[i, j]):
                        plt.text(j, i, fmt % Z[i, j], ha="center",
                                 va="center", fontsize=8, color="white")
            plt.title(f"{titulo}: {lab}", fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(d04, f"{fname}_{slug(lab)}.png"), dpi=110)
            plt.close(fig)

    heatmap("p_insp", "%inspeccionadas", "% insp", "heatmap_pinsp", "%.0f")
    heatmap("t_fin", "Tiempo final medio", "t (s)", "heatmap_tfin", "%.0f")
    heatmap("e_tot", "Energia total media", "E (J)", "heatmap_energia", "%.0f")

    # CSVs
    dd = os.path.join(args.out, "datos")
    os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, "resumen_final.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "exp", "estrategia", "etiqueta", "mapa", "robots",
                    "n", "p_vistas", "p_visit", "p_insp", "vel_max_pct_s",
                    "pct_por_J", "cells_por_J", "t_fin_s", "e_tot_J"])
        for k in sorted(prom):
            tg, cv, fin, n = prom[k]
            w.writerow([k[0], k[1], k[2], etiqueta(k[0], k[1], k[2]), k[3], k[4], n,
                        round(fin["p_vistas"], 2), round(fin["p_visit"], 2),
                        round(fin["p_insp"], 2), round(fin["vel_pct_max"], 4),
                        round(fin["pct_J"], 5), round(fin["cells_J"], 5),
                        round(fin["t_fin"], 1), round(fin["e_tot"], 2)])
    with open(os.path.join(dd, "promedios.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["etiqueta", "grupo", "p_vistas_m", "p_visit_m", "p_insp_m",
                    "vel_max_m", "pctJ_m", "cellsJ_m", "t_fin_m", "e_tot_m"])
        for (g, e, s) in configs:
            sel = [prom[k][2] for k in prom if k[:3] == (g, e, s)]
            w.writerow([etiqueta(g, e, s), g,
                        round(float(np.mean([v["p_vistas"] for v in sel])), 2),
                        round(float(np.mean([v["p_visit"] for v in sel])), 2),
                        round(float(np.mean([v["p_insp"] for v in sel])), 2),
                        round(float(np.mean([v["vel_pct_max"] for v in sel])), 4),
                        round(float(np.mean([v["pct_J"] for v in sel])), 5),
                        round(float(np.mean([v["cells_J"] for v in sel])), 5),
                        round(float(np.mean([v["t_fin"] for v in sel])), 1),
                        round(float(np.mean([v["e_tot"] for v in sel])), 2)])
    nfig = sum(1 for _dp, _dn, fn in os.walk(args.out) for p in fn if p.endswith(".png"))
    print(f"Figuras: {nfig} en {args.out}")


if __name__ == "__main__":
    main()
