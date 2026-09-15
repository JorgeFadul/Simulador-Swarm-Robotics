"""Graficas del barrido de estrategias.

Lee resultados_barrido/series/*.npz (data completa) y genera:
  - Curvas promedio normalizadas en % de tiempo de ejecucion:
    promedio punto a punto sobre grilla comun 0-100% (solo completadas).
  - Tiempos: robots vs tiempo, mapa vs tiempo, barras por estrategia.
  - 3D: metrica final vs robots vs tiempo, y vs mapa vs tiempo.
  - Tasas de completado por grupo.

Exploracion: vistas + visitadas. Inspeccion: inspeccionadas + defectuosas
(+ vistas de contexto). Hibridas: vistas + inspeccionadas + defectuosas.

Uso: python graficar_barrido.py [--out resultados_barrido/figs]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAU = np.linspace(0, 100, 101)

METRICAS = {
    "exploracion": [("vistas", 2, "tab:blue"),
                    ("visitadas", 3, "tab:orange")],
    "inspeccion": [("vistas", 2, "tab:blue"),
                   ("inspeccionadas", 4, "tab:green"),
                   ("defectuosas", 5, "tab:red")],
    "hibrida": [("vistas", 2, "tab:blue"),
                ("inspeccionadas", 4, "tab:green"),
                ("defectuosas", 5, "tab:red")],
}


def cargar_grupos(series_dir):
    """Agrupa corridas por (categoria, estrategia, mapa, robots)."""
    grupos = {}
    for p in sorted(os.listdir(series_dir)):
        if not p.endswith(".npz"):
            continue
        d = np.load(os.path.join(series_dir, p), allow_pickle=True)
        s = d["serie"]
        if len(s) == 0:
            continue
        key = (str(d["categoria"]), str(d["estrategia"]),
               int(d["mapa"]), int(d["robots"]))
        vals = {}
        for _cat in METRICAS:
            for nombre, idx, _ in METRICAS[_cat]:
                vals.setdefault(nombre, s[:, idx].astype(float))
        grupos.setdefault(key, []).append({
            "t": s[:, 1].astype(float),
            "vals": vals,
            "motivo": str(d["motivo"]),
            "libres": int(d["libres"]),
            "t_fin": float(s[-1, 1]),
        })
    return grupos


def promediar(grupo):
    """Normaliza cada corrida a % del t promedio del grupo e interpola a TAU.
    Solo corridas completadas. Devuelve dict con curvas promedio y stats."""
    comp = [r for r in grupo if r["motivo"] == "completado"]
    n_tot = len(grupo)
    if not comp:
        return None
    t_mean = float(np.mean([r["t_fin"] for r in comp]))
    curvas = {}
    cols = list(comp[0]["vals"].keys())
    for col in cols:
        acc = []
        for r in comp:
            tau = r["t"] / t_mean * 100.0 if t_mean > 0 else r["t"] * 0
            acc.append(np.interp(TAU, tau, r["vals"][col],
                                 left=r["vals"][col][0],
                                 right=r["vals"][col][-1]))
        curvas[col] = float(np.mean([r["vals"][col][-1] for r in comp])), \
            np.mean(acc, axis=0), np.std(acc, axis=0)
    return {"t_mean": t_mean, "t_std": float(np.std([r["t_fin"] for r in comp])),
            "n": len(comp), "n_tot": n_tot, "curvas": curvas,
            "libres": comp[0]["libres"]}


def _slug(s):
    return (s.replace(" ", "_").replace("(", "").replace(")", "")
             .replace("/", "-"))


def _outdir(base, *partes):
    ruta = os.path.join(base, *partes)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def fig_curvas(out, categoria, estrategia, mapa, por_robots):
    """01_curvas_por_robot: Curvas promedio de una estrategia, linea por nº robots."""
    n_met = len(METRICAS[categoria])
    fig, axes = plt.subplots(1, n_met, figsize=(6 * n_met, 4.5),
                             squeeze=False)
    for ax, (nombre, _, color) in zip(axes[0], METRICAS[categoria]):
        for r in sorted(por_robots):
            prom = por_robots[r]
            if prom is None:
                continue
            _, media, _ = prom["curvas"][nombre]
            ax.plot(TAU, media, label=f"{r} robots (t={prom['t_mean']:.0f}s)")
        ax.set_xlabel("% del tiempo total de ejecucion")
        ax.set_ylabel(f"Celdas {nombre}")
        ax.set_title(f"{nombre}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{estrategia} | Mapa {mapa}x{mapa} | Categoria: {categoria}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "01_curvas_por_robot", categoria, _slug(estrategia)),
        f"{categoria}_curvas_{_slug(estrategia)}_mapa{mapa}.png"),
        dpi=120)
    plt.close(fig)


def fig_general(out, categoria, mapa, metrica, prom_por_estrat, pct=False):
    """03/04_comparacion_estrategias: Todas las estrategias en un grafico vs % tiempo."""
    fig = plt.figure(figsize=(9, 5.5))
    for estrategia in sorted(prom_por_estrat):
        medias = []
        for r in sorted(prom_por_estrat[estrategia]):
            prom = prom_por_estrat[estrategia][r]
            if prom is not None:
                m = prom["curvas"][metrica][1].astype(float)
                if pct and prom["libres"]:
                    m = m / prom["libres"] * 100.0
                medias.append(m)
        if medias:
            plt.plot(TAU, np.mean(medias, axis=0), label=estrategia)
    xlabel = "% del tiempo total de ejecucion"
    ylabel = f"Celdas {metrica}" + (" (% del total libre)" if pct else "")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    titulo = (f"Comparacion de estrategias: {metrica}" +
              (" (%)" if pct else "") +
              f" | Mapa {mapa}x{mapa} | {categoria}")
    plt.title(titulo, fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    sub = os.path.join("04_comparacion_estrategias_pct" if pct
                       else "03_comparacion_estrategias", categoria)
    fig.savefig(os.path.join(
        _outdir(out, sub),
        f"{categoria}_comparativa_{metrica}{'_pct' if pct else ''}_"
        f"mapa{mapa}.png"), dpi=120)
    plt.close(fig)


def fig_porcentaje(out, categoria, estrategia, mapa, por_robots):
    """02_pct_por_robot: % de celdas cubiertas (sobre libres) vs % tiempo, por nº robots."""
    n_met = len(METRICAS[categoria])
    fig, axes = plt.subplots(1, n_met, figsize=(6 * n_met, 4.5),
                             squeeze=False)
    for ax, (nombre, _, _) in zip(axes[0], METRICAS[categoria]):
        for r in sorted(por_robots):
            prom = por_robots[r]
            if prom is None or not prom["libres"]:
                continue
            _, media, _ = prom["curvas"][nombre]
            ax.plot(TAU, np.asarray(media, dtype=float) / prom["libres"]
                    * 100.0, label=f"{r} robots (t={prom['t_mean']:.0f}s)")
        ax.set_xlabel("% del tiempo total de ejecucion")
        ax.set_ylabel(f"% {nombre} (sobre celdas libres)")
        ax.set_title(f"{nombre}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{estrategia} | Mapa {mapa}x{mapa} | Categoria: {categoria} | % sobre libres",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "02_pct_por_robot", categoria, _slug(estrategia)),
        f"{categoria}_pct_{_slug(estrategia)}_mapa{mapa}.png"),
        dpi=120)
    plt.close(fig)


def fig_normalizado(out, categoria, mapa, metrica, prom_por_estrat):
    """06_progreso_normalizado: Todas las estrategias, progreso de 0 a 100% del valor final vs % tiempo."""
    fig = plt.figure(figsize=(9, 5.5))
    for estrategia in sorted(prom_por_estrat):
        normadas = []
        for r in sorted(prom_por_estrat[estrategia]):
            prom = prom_por_estrat[estrategia][r]
            if prom is None:
                continue
            m = np.asarray(prom["curvas"][metrica][1], dtype=float)
            fin = m[-1] if m[-1] else 1.0
            normadas.append(m / fin * 100.0)
        if normadas:
            plt.plot(TAU, np.mean(normadas, axis=0), label=estrategia)
    plt.xlabel("% del tiempo total de ejecucion")
    plt.ylabel(f"{metrica} normalizada (% del valor final)")
    plt.title(f"Progreso normalizado: {metrica} | Mapa {mapa}x{mapa} | {categoria}",
              fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "06_progreso_normalizado", categoria),
        f"{categoria}_norm_{metrica}_mapa{mapa}.png"), dpi=120)
    plt.close(fig)


def fig_general_abs(out, categoria, mapa, metrica, prom_por_estrat):
    """05_comparacion_estrategias_tiempo_real: Todas las estrategias vs tiempo absoluto (segundos)."""
    fig = plt.figure(figsize=(10, 6))
    for estrategia in sorted(prom_por_estrat):
        proms_list = []
        for r in sorted(prom_por_estrat[estrategia]):
            prom = prom_por_estrat[estrategia][r]
            if prom is not None:
                proms_list.append(prom)
        if not proms_list:
            continue
        t_max_global = max(p["t_mean"] for p in proms_list)
        t_grid = np.linspace(0, t_max_global, 200)
        curvas_interp = []
        for prom in proms_list:
            _, media, _ = prom["curvas"][metrica]
            t_abs = TAU * prom["t_mean"] / 100.0
            curvas_interp.append(np.interp(t_grid, t_abs, media,
                                          left=media[0], right=media[-1]))
        media_final = np.mean(curvas_interp, axis=0)
        std_final = np.std(curvas_interp, axis=0)
        plt.plot(t_grid, media_final,
                 label=f"{estrategia} ({len(proms_list)} configs robots)")
        plt.fill_between(t_grid, media_final - std_final,
                         media_final + std_final, alpha=0.15)
    plt.xlabel("Tiempo de ejecucion (s)")
    plt.ylabel(f"Celdas {metrica}")
    plt.title(f"Comparacion de estrategias: {metrica} vs tiempo real | Mapa {mapa}x{mapa} | {categoria}",
              fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "05_comparacion_estrategias_tiempo_real", categoria),
        f"{categoria}_abs_{metrica}_mapa{mapa}.png"), dpi=120)
    plt.close(fig)


def fig_tabla_resumen(out, proms):
    """Genera CSV con resumen general: metrica final, tiempo, tasa completado."""
    rows = []
    for (cat, est, mapa, rob), p in sorted(proms.items()):
        row = {"categoria": cat, "estrategia": est, "mapa": mapa,
               "robots": rob, "n_completado": p["n"], "n_total": p["n_tot"],
               "tasa_completado": round(p["n"] / p["n_tot"], 3) if p["n_tot"] else 0,
               "t_mean_s": round(p["t_mean"], 1),
               "t_std_s": round(p["t_std"], 1),
               "libres": p["libres"]}
        for nombre, _, _ in METRICAS[cat]:
            fin, _, _ = p["curvas"][nombre]
            row[f"{nombre}_final"] = round(fin, 1)
            if p["libres"]:
                row[f"{nombre}_pct"] = round(fin / p["libres"] * 100, 1)
        rows.append(row)
    if not rows:
        return
    # Recopilar todos los campos posibles (distintas categorias = distintas metricas)
    campos = ["categoria", "estrategia", "mapa", "robots",
              "n_completado", "n_total", "tasa_completado",
              "t_mean_s", "t_std_s", "libres"]
    for r in rows:
        for k in r:
            if k not in campos:
                campos.append(k)
    ruta_csv = os.path.join(_outdir(out, "datos"), "resumen_general.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # Resumen por categoria+estrategia (promedio sobre mapas y robots)
    agg = {}
    for r in rows:
        k = (r["categoria"], r["estrategia"])
        agg.setdefault(k, []).append(r)
    ruta_agg = os.path.join(_outdir(out, "datos"), "resumen_por_estrategia.csv")
    with open(ruta_agg, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "estrategia", "n_grupos",
                     "tasa_completado_media", "t_mean_s_media",
                     "vistas_final_media", "vistas_pct_media",
                     "visitadas_final_media", "visitadas_pct_media",
                     "inspeccionadas_final_media", "inspeccionadas_pct_media",
                     "defectuosas_final_media", "defectuosas_pct_media"])
        for (cat, est), runs in sorted(agg.items()):
            n = len(runs)
            w.writerow([
                cat, est, n,
                round(np.mean([r["tasa_completado"] for r in runs]), 3),
                round(np.mean([r["t_mean_s"] for r in runs]), 1),
                round(np.mean([r.get("vistas_final", 0) for r in runs]), 1),
                round(np.mean([r.get("vistas_pct", 0) for r in runs]), 1),
                round(np.mean([r.get("visitadas_final", 0) for r in runs]), 1),
                round(np.mean([r.get("visitadas_pct", 0) for r in runs]), 1),
                round(np.mean([r.get("inspeccionadas_final", 0) for r in runs]), 1),
                round(np.mean([r.get("inspeccionadas_pct", 0) for r in runs]), 1),
                round(np.mean([r.get("defectuosas_final", 0) for r in runs]), 1),
                round(np.mean([r.get("defectuosas_pct", 0) for r in runs]), 1),
            ])
    print(f"  Tablas resumen: {ruta_csv}")
    print(f"  Resumen por estrategia: {ruta_agg}")


def fig_tiempo(out, categoria, proms, eje, nombre_eje):
    """07_tiempo_vs_robots_mapa: Tiempo medio de ejecucion vs robots o mapa, una linea por estrategia."""
    for otro in sorted({k[2 if eje == "robots" else 3] for k in proms
                        if k[0] == categoria}):
        fig = plt.figure(figsize=(8, 5))
        for est in sorted({k[1] for k in proms if k[0] == categoria}):
            xs, ys, es = [], [], []
            for (cat, e2, mapa, rob), prom in sorted(proms.items()):
                if cat != categoria or e2 != est:
                    continue
                fijo = rob if eje == "mapa" else mapa
                if fijo != otro:
                    continue
                xs.append(rob if eje == "robots" else mapa)
                ys.append(prom["t_mean"])
                es.append(prom["t_std"])
            if xs:
                plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                             label=f"{est}")
        titulo_otro = ("mapa" if eje == "robots" else "robots")
        plt.xlabel(nombre_eje)
        plt.ylabel("Tiempo medio de ejecucion (s)")
        plt.title(f"Tiempo vs {eje} | {titulo_otro}={otro} | {categoria}",
                  fontsize=11, fontweight="bold")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _outdir(out, "07_tiempo_vs_robots_mapa", categoria),
            f"{categoria}_tiempo_vs_{eje}_{titulo_otro}{otro}.png"),
            dpi=120)
        plt.close(fig)


def fig_barras(out, categoria, proms):
    """08_barras_tiempos: Barras comparativas de tiempos promedio por estrategia."""
    estrat = sorted({k[1] for k in proms if k[0] == categoria})
    medias = [np.mean([p["t_mean"] for k, p in proms.items()
                       if k[0] == categoria and k[1] == e]) for e in estrat]
    fig = plt.figure(figsize=(10, 5.5))
    plt.bar(estrat, medias)
    plt.ylabel("Tiempo medio de ejecucion (s)")
    plt.title(f"Tiempos promedio por estrategia | {categoria}",
              fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "08_barras_tiempos", categoria),
        f"{categoria}_barras_tiempos.png"), dpi=120)
    plt.close(fig)


def _superficie(ax, xs, ys, zs, xlabel):
    """Puntos + superficie suavizada interpolada que pasa por ellos."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    ax.scatter(xs, ys, zs, s=30, depthshade=True)
    try:
        from scipy.interpolate import griddata
        gx = np.linspace(xs.min(), xs.max(), 30)
        gy = np.linspace(ys.min(), ys.max(), 30)
        GX, GY = np.meshgrid(gx, gy)
        GZ = griddata(np.c_[xs, ys], zs, (GX, GY), method="cubic")
        if np.isnan(GZ).all():
            raise ValueError("cubic fallo")
        # Rellenar bordes NaN con linear para superficie continua
        nan = np.isnan(GZ)
        if nan.any():
            GZ[nan] = griddata(np.c_[xs, ys], zs, (GX, GY),
                               method="linear")[nan]
        ax.plot_surface(GX, GY, GZ, alpha=0.45, cmap="viridis",
                        edgecolor="none")
    except Exception:
        try:
            ax.plot_trisurf(xs, ys, zs, alpha=0.45, cmap="viridis")
        except Exception:
            pass
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tiempo (s)")
    return ax
def fig_3d(out, categoria, estrategia, proms, metrica):
    """09_superficies_3d: Superficies 3D de metrica final vs robots vs tiempo y vs mapa vs tiempo."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xs_r, ys_t, zs_m, xs_m = [], [], [], []
    for (cat, est, mapa, rob), prom in proms.items():
        if cat != categoria or est != estrategia:
            continue
        fin = prom["curvas"][metrica][0]
        xs_r.append(rob)
        xs_m.append(mapa)
        ys_t.append(prom["t_mean"])
        zs_m.append(fin)
    if not xs_r:
        return
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    _superficie(ax1, xs_r, ys_t, zs_m, "n_robots")
    ax1.set_zlabel(f"Celdas {metrica}")
    ax1.set_title(f"{metrica} vs n_robots vs tiempo")
    ax2 = fig.add_subplot(122, projection="3d")
    _superficie(ax2, xs_m, ys_t, zs_m, "Mapa (celdas)")
    ax2.set_zlabel(f"Celdas {metrica}")
    ax2.set_title(f"{metrica} vs tamano mapa vs tiempo")
    fig.suptitle(f"{estrategia} | {categoria} | Metrica: {metrica}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _outdir(out, "09_superficies_3d", categoria, _slug(estrategia)),
        f"{categoria}_3d_{metrica}_{_slug(estrategia)}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="resultados_barrido/series")
    ap.add_argument("--out", default="resultados_barrido/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Limpieza de figuras y carpetas de versiones anteriores
    for p in os.listdir(args.out):
        ruta = os.path.join(args.out, p)
        if p.endswith(".png"):
            os.remove(ruta)
        elif os.path.isdir(ruta) and p not in ("datos",):
            import shutil
            shutil.rmtree(ruta, ignore_errors=True)
    datos_dir = _outdir(args.out, "datos")

    grupos = cargar_grupos(args.series)
    proms = {}
    for key, runs in grupos.items():
        p = promediar(runs)
        if p is not None:
            proms[key] = p
    print(f"Grupos con data: {len(proms)}/{len(grupos)}")

    # Curvas CSV promedio (data derivada guardada)
    with open(os.path.join(datos_dir, "curvas_promedio.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "estrategia", "mapa", "robots", "tau",
                    "metrica", "media", "std"])
        for (cat, est, mapa, rob), p in sorted(proms.items()):
            for nombre, _, _ in METRICAS[cat]:
                _, media, std = p["curvas"][nombre]
                for i, tau in enumerate(TAU):
                    w.writerow([cat, est, mapa, rob, round(float(tau), 1),
                                nombre, round(float(media[i]), 2),
                                round(float(std[i]), 2)])

    # Tasa de completado
    with open(os.path.join(datos_dir, "completado.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "estrategia", "mapa", "robots",
                    "completadas", "total", "tasa"])
        for key, runs in sorted(grupos.items()):
            nc = sum(1 for r in runs if r["motivo"] == "completado")
            w.writerow([*key, nc, len(runs),
                        round(nc / len(runs), 3)])

    for categoria in METRICAS:
        estrat = sorted({k[1] for k in proms if k[0] == categoria})
        mapas = sorted({k[2] for k in proms if k[0] == categoria})
        # Curvas por estrategia/mapa + generales (+ % y normalizadas)
        for est in estrat:
            for mapa in mapas:
                por_robots = {k[3]: proms.get(k) for k in proms
                              if k[0] == categoria and k[1] == est
                              and k[2] == mapa}
                if any(por_robots.values()):
                    fig_curvas(args.out, categoria, est, mapa, por_robots)
                    fig_porcentaje(args.out, categoria, est, mapa,
                                   por_robots)
        for mapa in mapas:
            for nombre, _, _ in METRICAS[categoria]:
                pp = {}
                for est in estrat:
                    pp[est] = {k[3]: proms.get(k) for k in proms
                               if k[0] == categoria and k[1] == est
                               and k[2] == mapa}
                fig_general(args.out, categoria, mapa, nombre, pp)
                fig_general(args.out, categoria, mapa, nombre, pp,
                            pct=True)
                fig_normalizado(args.out, categoria, mapa, nombre, pp)
                fig_general_abs(args.out, categoria, mapa, nombre, pp)
        # Tiempos y barras
        fig_tiempo(args.out, categoria, proms, "robots", "n_robots")
        fig_tiempo(args.out, categoria, proms, "mapa", "mapa (celdas)")
        fig_barras(args.out, categoria, proms)
        # 3D: primera metrica de interes (vistas / inspeccionadas)
        met3d = {"exploracion": "vistas", "inspeccion": "inspeccionadas",
                 "hibrida": "vistas"}[categoria]
        for est in estrat:
            fig_3d(args.out, categoria, est, proms, met3d)
        if categoria in ("inspeccion", "hibrida"):
            for est in estrat:
                fig_3d(args.out, categoria, est, proms, "defectuosas")

    # Tablas resumen
    fig_tabla_resumen(args.out, proms)

    nfig = sum(1 for _dp, _dn, fn in os.walk(args.out) for p in fn
               if p.endswith(".png"))
    print(f"Figuras: {nfig} en {args.out}")
    for _dp, _dn, _fn in os.walk(args.out):
        nivel = os.path.relpath(_dp, args.out)
        n = sum(1 for p in _fn if p.endswith(".png"))
        if n:
            print(f"  {nivel}: {n}")


if __name__ == "__main__":
    main()
