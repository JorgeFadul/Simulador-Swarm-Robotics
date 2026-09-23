"""Laminas generalistas de la tesis (cuerpo del documento).

Colapsa las 30 combinaciones mapa x robots de cada estrategia con grupos
adimensionales:
  densidad  d   = R / N^2   (robots por celda)
  carga     c   = N^2 / R   (celdas por robot)
  tiempo    tau = t * R / N^2  (robot-segundos por celda)
  energia   e_c = E / L     (julios por celda libre)

Lee resultados_tesis_final/series/*.npz + resumenes.csv y escribe en
resultados_tesis_final/figs_generalista/ (10 laminas + datos/*.csv).

Uso: python graficar_generalista.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graficar_tesis_final import cargar_series, etiqueta, metricas_run, slug

SERIES = "resultados_tesis_final/series"
RESUMENES = "resultados_tesis_final/resumenes.csv"
OUT = "resultados_tesis_final/figs_generalista"

TAU_N = 200


def libres_de_mapa(n):
    return (n - 2) ** 2  # mapas vacios con borde de pared


def grupos_por_estrategia(runs):
    g = {}
    for r in runs:
        g.setdefault((r["grupo"], r["exp"], r["estrategia"]), []).append(r)
    return g


def curva_colapsada(grupo_runs, clave_metrica, norm_e=False):
    """Media entre grupos de la metrica vs tau (nan donde no hay cobertura)."""
    per_group = []
    for (g, e, s, m, rb), rs in _por_grupo(grupo_runs).items():
        n = m
        tg_max = 0.0
        curvas = []
        for r in rs:
            met = metricas_run(r)[clave_metrica]
            if norm_e:
                met = met / max(1, r["libres"])
            tau = r["t"] * rb / (n * n)
            curvas.append((tau, met))
            tg_max = max(tg_max, tau[-1])
        tg = np.linspace(0, tg_max, TAU_N)
        acc = [np.interp(tg, t, v) for t, v in curvas]
        per_group.append((tg, np.mean(acc, axis=0)))
    tau_max = max(tg[-1] for tg, _ in per_group)
    grid = np.linspace(0, tau_max, TAU_N)
    pila = []
    for tg, med in per_group:
        col = np.interp(grid, tg, med, left=np.nan, right=np.nan)
        col[grid <= tg[-1]] = np.interp(grid[grid <= tg[-1]], tg, med)
        pila.append(col)
    pila = np.array(pila)
    with np.errstate(all="ignore"):
        media = np.nanmedian(pila, axis=0)
        lo = np.nanpercentile(pila, 25, axis=0)
        hi = np.nanpercentile(pila, 75, axis=0)
    cob = np.sum(~np.isnan(pila), axis=0)
    return grid, media, lo, hi, cob


def _por_grupo(runs):
    g = {}
    for r in runs:
        g.setdefault((r["grupo"], r["exp"], r["estrategia"],
                      r["mapa"], r["robots"]), []).append(r)
    return g


def gt_cache(runs, dd):
    """GT de celdas defectuosas por corrida (reproducible desde la semilla).
    Cache en datos/defectos_gt.csv: id -> gt_total."""
    from src.core.config import Config
    from src.core.mundo import Mundo
    ruta = os.path.join(dd, "defectos_gt.csv")
    gt = {}
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gt[row["id"]] = int(row["gt_total"])
    faltan = 0
    for r in runs:
        if r["key"] in gt:
            continue
        n = r["mapa"]
        cfg = Config()
        cfg.grid_size = 20
        cfg.map_width = n * 20
        cfg.map_height = n * 20
        cfg.cols = n
        cfg.rows = n
        cfg.walls = []
        # Misma cadena que el barrido: seed md5 + Mundo(cfg, default_rng(seed))
        m = Mundo(cfg, rng=np.random.default_rng(r["seed"]))
        gt[r["key"]] = int(m._defectuosas_gt)
        faltan += 1
    if faltan:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "gt_total"])
            for k in sorted(gt):
                w.writerow([k, gt[k]])
    print(f"GT defectos: {len(gt)} corridas ({faltan} calculados)")
    return gt


def main():
    os.makedirs(OUT, exist_ok=True)
    for p in os.listdir(OUT):
        r = os.path.join(OUT, p)
        if p.endswith(".png"):
            os.remove(r)
        elif os.path.isdir(r) and p != "datos":
            import shutil
            shutil.rmtree(r, ignore_errors=True)
    dd = os.path.join(OUT, "datos")
    os.makedirs(dd, exist_ok=True)

    runs = cargar_series(SERIES)
    print(f"Corridas: {len(runs)}")
    gest = grupos_por_estrategia(runs)
    configs = sorted(gest)
    labs = {c: etiqueta(*c) for c in configs}
    colores = {c: plt.cm.tab10(i % 10) for i, c in enumerate(configs)}

    # ---- 1-2. Curvas colapsadas vs tau ----
    for met, titulo, unidad, fname in (
            ("p_insp", "% celdas inspeccionadas", "% insp",
             "colapso_insp_vs_tau"),
            ("e_abs", "Energia por celda libre", "J/celda",
             "colapso_energia_vs_tau")):
        norm = (met == "e_abs")
        fig = plt.figure(figsize=(10, 5.5))
        for c in configs:
            grid, med, lo, hi, cob = curva_colapsada(
                gest[c], met, norm_e=norm)
            ok = cob >= len(_por_grupo(gest[c])) // 2
            plt.plot(grid[ok], med[ok], label=labs[c],
                     color=colores[c])
            plt.fill_between(grid[ok], lo[ok], hi[ok], alpha=0.12,
                             color=colores[c])
        plt.xlabel("tau = t * R / N^2 (robot-s por celda)")
        plt.ylabel(unidad)
        plt.title(f"{titulo} vs tiempo adimensional (todas las configs)",
                  fontweight="bold")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname + ".png"), dpi=120)
        plt.close(fig)

    gt = gt_cache(runs, dd)

    # ---- Recall de defectos vs tau (mediana por estrategia) ----
    fig = plt.figure(figsize=(10, 5.5))
    t_recall = {}  # estrategia -> {umbral: mediana de t}
    for c in configs:
        curvas = []
        tmaxs = []
        for r in gest[c]:
            total = max(1, gt[r["key"]])
            rec = r["defec"] / total * 100.0
            tau = r["t"] * r["robots"] / (r["mapa"] ** 2)
            curvas.append((tau, rec))
            tmaxs.append(tau[-1])
            for u in (50, 80, 90):
                idx = np.argmax(rec >= u)
                t_recall.setdefault(c, {}).setdefault(u, []).append(
                    float(r["t"][idx]) if rec[idx] >= u else np.nan)
        grid = np.linspace(0, max(tmaxs), TAU_N)
        pila = np.array([np.interp(grid, t, v) for t, v in curvas])
        with np.errstate(all="ignore"):
            med = np.nanmedian(pila, axis=0)
            lo = np.nanpercentile(pila, 25, axis=0)
            hi = np.nanpercentile(pila, 75, axis=0)
        plt.plot(grid, med, label=labs[c], color=colores[c])
        plt.fill_between(grid, lo, hi, alpha=0.12, color=colores[c])
    plt.xlabel("tau = t * R / N^2 (robot-s por celda)")
    plt.ylabel("% defectos encontrados (recall)")
    plt.title("Recall de celdas defectuosas vs tiempo adimensional",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_defectos_vs_tau.png"), dpi=120)
    plt.close(fig)

    # ---- Tiempo mediano a recall 50/80/90% por estrategia ----
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = np.arange(len(configs))
    for j, u in enumerate((50, 80, 90)):
        ms, ls, hs = [], [], []
        for c in configs:
            v = np.array(t_recall[c][u])
            ok = v[~np.isnan(v)]
            ms.append(float(np.median(ok)) if len(ok) else np.nan)
            ls.append(float(np.percentile(ok, 25)) if len(ok) else np.nan)
            hs.append(float(np.percentile(ok, 75)) if len(ok) else np.nan)
        ms, ls, hs = map(np.array, (ms, ls, hs))
        ax.bar(xs + (j - 1) * 0.25, ms, 0.25, yerr=[ms - ls, hs - ms],
               capsize=3, label=f"recall {u}%")
    ax.set_xticks(xs, [labs[c] for c in configs], fontsize=9,
                  rotation=20, ha="right")
    ax.set_xlim(-0.9, len(xs) - 0.1)
    ax.set_ylabel("tiempo mediano (s, p25-p75)")
    ax.set_title("Velocidad de hallazgo: tiempo a recall 50/80/90%",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = float(np.nanmax([np.nanmedian(v) for vs in t_recall.values()
                            for v in vs.values()]))
    ax.set_ylim(0, 1.05 * ymax)
    ax.set_yticks(np.arange(0, 1.05 * ymax + 1, 100))
    for ln in ax.lines:  # barras de error visibles aun fuera del eje
        ln.set_clip_on(False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, left=0.13)
    fig.savefig(os.path.join(OUT, "tiempo_a_recall.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(dd, "tiempo_a_recall.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "recall_pct", "t_med_s", "t_p25_s",
                    "t_p75_s", "n_alcanzan", "n_total"])
        for c in configs:
            for u in (50, 80, 90):
                v = np.array(t_recall[c][u])
                ok = v[~np.isnan(v)]
                w.writerow([labs[c], u,
                            round(float(np.median(ok)), 1) if len(ok) else "",
                            round(float(np.percentile(ok, 25)), 1)
                            if len(ok) else "",
                            round(float(np.percentile(ok, 75)), 1)
                            if len(ok) else "",
                            len(ok), len(v)])

    # ---- Recall por grupo: media de reps de recall%(t) en grilla comun ----
    grupos = _por_grupo(runs)
    rec_grupo = {}  # (g,e,s,m,rb) -> (tgrid, media)
    for key, rs in grupos.items():
        t_max = max(r["t"][-1] for r in rs)
        tg = np.linspace(0, t_max, TAU_N)
        acc = []
        for r in rs:
            total = max(1, gt[r["key"]])
            acc.append(np.interp(tg, r["t"], r["defec"] / total * 100.0))
        rec_grupo[key] = (tg, np.mean(acc, axis=0))

    # ---- Resumen por estrategia: recall vs t normal, lineas = mapas ----
    for c in configs:
        mapas_c = sorted({k[3] for k in rec_grupo if k[:3] == c})
        fig = plt.figure(figsize=(10, 5.5))
        for m in mapas_c:
            curvas = [rec_grupo[k] for k in sorted(rec_grupo)
                      if k[:3] == c and k[3] == m]
            t_max = max(tg[-1] for tg, _ in curvas)
            grid = np.linspace(0, t_max, TAU_N)
            pila = np.array([np.interp(grid, tg, med) for tg, med in curvas])
            with np.errstate(all="ignore"):
                med = np.nanmedian(pila, axis=0)
                lo = np.nanpercentile(pila, 25, axis=0)
                hi = np.nanpercentile(pila, 75, axis=0)
            plt.plot(grid, med, label=f"mapa {m}x{m}")
            plt.fill_between(grid, lo, hi, alpha=0.12)
        plt.xlabel("tiempo (s)")
        plt.ylabel("% defectos encontrados (recall)")
        plt.title(f"Recall de defectos vs tiempo: {labs[c]}",
                  fontweight="bold")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        d = os.path.join(OUT, "recall_por_estrategia")
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"recall_{slug(labs[c])}.png"), dpi=120)
        plt.close(fig)

    # ---- Generalista facetada: 5 paneles (a)-(e), uno por mapa ----
    mapas_all = sorted({k[3] for k in rec_grupo})
    fig, axes = plt.subplots(3, 2, figsize=(12, 13),
                             sharey=True)
    axes = axes.flat
    for i, (ax, m) in enumerate(zip(axes, mapas_all)):
        for c in configs:
            curvas = [rec_grupo[k] for k in sorted(rec_grupo)
                      if k[:3] == c and k[3] == m]
            if not curvas:
                continue
            t_max = max(tg[-1] for tg, _ in curvas)
            grid = np.linspace(0, t_max, TAU_N)
            pila = np.array([np.interp(grid, tg, med) for tg, med in curvas])
            with np.errstate(all="ignore"):
                med = np.nanmedian(pila, axis=0)
            ax.plot(grid, med, label=labs[c], color=colores[c])
        ax.set_xlabel("tiempo (s)")
        ax.set_title(f"mapa {m}x{m}")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.24, f"({chr(97 + i)})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("% defectos (recall)")
    axes[0].legend(fontsize=7)
    axes[5].axis("off")
    fig.suptitle("Recall de defectos vs tiempo por mapa (todas)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_facet_mapa.png"), dpi=120)
    plt.close(fig)

    # ---- Escenario de referencia: 30x30, 15 robots ----
    fig = plt.figure(figsize=(10, 5.5))
    for c in configs:
        key = (c[0], c[1], c[2], 30, 15)
        if key not in rec_grupo:
            continue
        tg, med = rec_grupo[key]
        plt.plot(tg, med, label=labs[c], color=colores[c])
    plt.xlabel("tiempo (s)")
    plt.ylabel("% defectos encontrados (recall)")
    plt.title("Recall de defectos: escenario referencia 30x30, 15 robots",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_referencia_30x15.png"), dpi=120)
    plt.close(fig)

    res = {}
    if os.path.exists(RESUMENES):
        with open(RESUMENES, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["grupo"], row["exp"], row["estrategia"],
                       int(row["mapa"]), int(row["robots"]))
                res.setdefault(key, []).append(row)

    def finales_grupo(key):
        rows = res.get(key, [])
        if not rows:
            return None
        t = np.mean([float(r["t_sim_s"]) for r in rows])
        e = np.mean([float(r["energia_total_j"]) for r in rows])
        ins = np.mean([float(r["inspeccionadas"]) for r in rows])
        L = libres_de_mapa(key[3])
        return {"t": t, "e": e, "pinsp": ins / L * 100.0,
                "e_celda": e / L, "n": len(rows)}

    # ---- 3. Ley de escala: t_fin vs N^2/R (log-log + pendiente) ----
    fig = plt.figure(figsize=(10, 5.5))
    pendientes = {}
    for c in configs:
        xs, ys = [], []
        for key in sorted(res):
            if key[:3] != c:
                continue
            f = finales_grupo(key)
            if f:
                xs.append(key[3] ** 2 / key[4])
                ys.append(f["t"])
        xs, ys = np.array(xs), np.array(ys)
        plt.loglog(xs, ys, "o", ms=4, color=colores[c], alpha=0.7)
        if len(xs) > 2:
            b, a = np.polyfit(np.log10(xs), np.log10(ys), 1)
            pendientes[labs[c]] = round(float(b), 3)
            xg = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
            plt.loglog(xg, 10 ** a * xg ** b, "-", color=colores[c],
                       label=f"{labs[c]} (pend. {b:.2f})")
        else:
            plt.loglog([], [], "o", color=colores[c], label=labs[c])
    plt.xlabel("carga N^2 / R (celdas por robot)")
    plt.ylabel("t_fin (s)")
    plt.title("Ley de escala: tiempo final vs carga (log-log)",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ley_escala_tiempo.png"), dpi=120)
    plt.close(fig)

    # ---- 4. Eficiencia: J/celda vs densidad R/N^2 ----
    fig = plt.figure(figsize=(10, 5.5))
    for c in configs:
        xs, ys = [], []
        for key in sorted(res):
            if key[:3] != c:
                continue
            f = finales_grupo(key)
            if f:
                xs.append(key[4] / key[3] ** 2)
                ys.append(f["e_celda"])
        o = np.argsort(xs)
        plt.plot(np.array(xs)[o], np.array(ys)[o], "o-",
                 label=labs[c], color=colores[c])
    plt.xlabel("densidad R / N^2 (robots por celda)")
    plt.ylabel("J por celda libre")
    plt.title("Eficiencia energetica vs densidad de robots",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eficiencia_vs_densidad.png"), dpi=120)
    plt.close(fig)

    # ---- 5. Ranking global con dispersion ----
    agg = {}
    for c in configs:
        vals = {"pinsp": [], "t": [], "e_celda": []}
        for key in res:
            if key[:3] != c:
                continue
            f = finales_grupo(key)
            if f:
                vals["pinsp"].append(f["pinsp"])
                vals["t"].append(f["t"])
                vals["e_celda"].append(f["e_celda"])
        agg[c] = vals
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    for ax, (fk, xl) in zip(axes, [("pinsp", "% insp media"),
                                   ("t", "t_fin medio (s)"),
                                   ("e_celda", "J/celda medio")]):
        xs = [labs[c] for c in configs]
        ms = [float(np.mean(agg[c][fk])) for c in configs]
        es = [float(np.std(agg[c][fk])) for c in configs]
        y = np.arange(len(xs))
        ax.barh(y, ms, xerr=es, capsize=3,
                color=[colores[c] for c in configs])
        ax.set_yticks(y, xs, fontsize=9)
        ax.set_xlabel(xl)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.32, f"({chr(97 + i)})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12, fontweight="bold")
    fig.suptitle("Ranking global por estrategia (media +- std, 30 grupos)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ranking_global.png"), dpi=120)
    plt.close(fig)

    # ---- 6. Frontera de Pareto tiempo vs energia ----
    fig = plt.figure(figsize=(9, 6))
    for c in configs:
        tm = float(np.median([finales_grupo(k)["t"] for k in res
                              if k[:3] == c and finales_grupo(k)]))
        em = float(np.median([finales_grupo(k)["e_celda"] for k in res
                              if k[:3] == c and finales_grupo(k)]))
        plt.scatter([tm], [em], s=120, color=colores[c], label=labs[c])
        plt.annotate(labs[c], (tm, em), fontsize=8,
                     xytext=(6, 6), textcoords="offset points")
    plt.xlabel("t_fin mediano (s)")
    plt.ylabel("J/celda mediano")
    plt.title("Frontera de Pareto: tiempo vs energia por estrategia",
              fontweight="bold")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto_tiempo_energia.png"), dpi=120)
    plt.close(fig)

    # ---- 7-9. Heatmaps facetados (una lamina por metrica) ----
    for fk, titulo, unidad, fname, fmt in (
            ("pinsp", "%inspeccionadas", "% insp", "facet_pinsp", "%.0f"),
            ("t", "Tiempo final medio", "t (s)", "facet_tfin", "%.0f"),
            ("e", "Energia total media", "E (J)", "facet_energia", "%.0f")):
        mapas = sorted({k[3] for k in res})
        robots = sorted({k[4] for k in res})
        ncf = len(configs)
        ncols = 2
        nrows = int(np.ceil(ncf / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.6 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        todo = [finales_grupo(k)[fk] for k in res if finales_grupo(k)]
        vmax = float(np.max(todo)) if todo else 1.0
        vmin = float(np.min(todo)) if todo else 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        for ax, c in zip(axes.flat, configs):
            Z = np.full((len(robots), len(mapas)), np.nan)
            for i, rb in enumerate(robots):
                for j, m in enumerate(mapas):
                    key = (c[0], c[1], c[2], m, rb)
                    f = finales_grupo(key)
                    if f:
                        Z[i, j] = f[fk]
            im = ax.imshow(Z, aspect="auto", origin="upper",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(mapas)), mapas, fontsize=7)
            ax.set_yticks(range(len(robots)), robots, fontsize=7)
            ax.set_xlabel("mapa", fontsize=8)
            ax.set_ylabel("robots", fontsize=8)
            ax.set_title(labs[c], fontsize=9)
            for i in range(len(robots)):
                for j in range(len(mapas)):
                    if not np.isnan(Z[i, j]):
                        ax.text(j, i, fmt % Z[i, j], ha="center",
                                va="center", fontsize=6, color="white")
        for ax in axes.flat[len(configs):]:
            ax.axis("off")
        fig.suptitle(titulo, fontweight="bold")
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=unidad)
        fig.savefig(os.path.join(OUT, fname + ".png"), dpi=120)
        plt.close(fig)

    # ---- 10. Rendimientos decrecientes: ganancia marginal de +5 robots ----
    pasos = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    fig = plt.figure(figsize=(10, 5.5))
    for c in configs:
        xs, ys = [], []
        for a, b in pasos:
            gains = []
            for m in {k[3] for k in res if k[:3] == c}:
                fa = finales_grupo((c[0], c[1], c[2], m, a))
                fb = finales_grupo((c[0], c[1], c[2], m, b))
                if fa and fb and fa["t"] > 0:
                    gains.append(100.0 * (fa["t"] - fb["t"]) / fa["t"])
            if gains:
                xs.append(f"{a}->{b}")
                ys.append(float(np.mean(gains)))
        plt.plot(xs, ys, "o-", label=labs[c], color=colores[c])
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("incremento de dotacion (+5 robots)")
    plt.ylabel("reduccion media de t_fin (%)")
    plt.title("Rendimientos decrecientes: ganancia marginal por robot",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rendimientos_decrecientes.png"),
                dpi=120)
    plt.close(fig)

    # ---- Rendimientos por mapa: 2 cols x 3 filas, (f) = promedio ----
    pasos = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    mapas_rd = sorted({k[3] for k in res})
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharey=True)
    axes = axes.flat
    for i, m in enumerate(mapas_rd + ["prom"]):
        ax = axes[i]
        for c in configs:
            xs, ys = [], []
            for a, b in pasos:
                if m == "prom":
                    maps = mapas_rd
                else:
                    maps = [m]
                gains = []
                for mm in maps:
                    fa = finales_grupo((c[0], c[1], c[2], mm, a))
                    fb = finales_grupo((c[0], c[1], c[2], mm, b))
                    if fa and fb and fa["t"] > 0:
                        gains.append(100.0 * (fa["t"] - fb["t"]) / fa["t"])
                if gains:
                    xs.append(f"{a}->{b}")
                    ys.append(float(np.mean(gains)))
            ax.plot(xs, ys, "o-", label=labs[c], color=colores[c])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"mapa {m}x{m}" if m != "prom" else "promedio mapas")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.24, f"({chr(97 + i)})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("reduccion media de t_fin (%)")
    axes[0].legend(fontsize=7)
    axes[5].legend(fontsize=7)
    fig.suptitle("Rendimientos decrecientes por tamano de mapa",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rendimientos_por_mapa.png"), dpi=120)
    plt.close(fig)

    # ---- datos derivados ----
    with open(os.path.join(dd, "pendientes_escala.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "pendiente_loglog"])
        for k, v in sorted(pendientes.items()):
            w.writerow([k, v])

    nfig = sum(1 for _dp, _dn, fn in os.walk(OUT) for p in fn
               if p.endswith(".png"))
    print(f"Laminas: {nfig} en {OUT} | pendientes: {pendientes}")


if __name__ == "__main__":
    main()
