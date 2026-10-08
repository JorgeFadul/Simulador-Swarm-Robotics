"""Graficos de optimizacion multi-objetivo energia-tiempo.

Lee resultados_optimizacion/resumen_optimizacion.csv (+ convergencia.csv)
y genera:
  - Frentes de Pareto (Tiempo vs Energia), uno por mapa y uno global.
  - Mapas de calor 2D del costo J(N_robots, mapa) (curva en U).
  - Trazas de convergencia del optimizador (mejor J vs iteracion).

Uso: python graficar_optimizacion.py --in resultados_optimizacion
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _es_verdad(v):
    return str(v).lower() in ("true", "1", "si")


def leer_resumen(ruta):
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                es_val = _es_verdad(row.get("valida_cfg",
                                            row.get("valida", True)))
                filas.append({
                    "categoria": row["categoria"],
                    "estrategia": row["estrategia"],
                    "mapa": int(row["mapa"]),
                    "robots": int(row["robots"]),
                    "t_mean_s": float(row["t_mean_s"]),
                    "e_mean_j": float(row["e_mean_j"]),
                    "j_edp": float(row["j_edp"]),
                    "j_pond": float(row.get("j_pond", row["j_edp"])),
                    "pareto": _es_verdad(row.get("pareto", "")),
                    "tasa_ok": float(row.get("tasa_ok",
                                             row.get("tasa_validez", 1.0))),
                    "valida_cfg": es_val,
                    "causa_inf": row.get("causa_inf", ""),
                    "cobertura": float(row.get("cobertura_media_pct",
                                               row.get("cobertura_pct", 100.0))
                                       or 0.0),
                })
            except (KeyError, ValueError):
                continue
    return filas


def leer_convergencia(ruta):
    hist = []
    if not os.path.exists(ruta):
        return hist
    with open(ruta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hist.append((int(row["iter"]), float(row["mejor_j_edp"]),
                         float(row["ultimo_j_edp"])))
    return hist


def fig_pareto(out, filas, titulo, nombre):
    fig = plt.figure(figsize=(8, 5.5))
    val = [r for r in filas if r["valida_cfg"]]
    inv = [r for r in filas if not r["valida_cfg"]]
    plt.scatter([r["t_mean_s"] for r in val],
                [r["e_mean_j"] for r in val],
                c="tab:blue", alpha=0.6, label="Válidas")
    if inv:
        plt.scatter([r["t_mean_s"] for r in inv],
                    [r["e_mean_j"] for r in inv],
                    c="gray", marker="x", s=50, alpha=0.7,
                    label="Infactibles (<98% o timeout)")
    par = [r for r in val if r["pareto"]]
    if par:
        par = sorted(par, key=lambda r: r["t_mean_s"])
        plt.scatter([r["t_mean_s"] for r in par],
                    [r["e_mean_j"] for r in par],
                    c="tab:red", s=60, label="Frente de Pareto")
        plt.plot([r["t_mean_s"] for r in par],
                 [r["e_mean_j"] for r in par], "r--", alpha=0.6)
        for r in par:
            plt.annotate(f"{r['estrategia'][:12]} {r['robots']}r",
                         (r["t_mean_s"], r["e_mean_j"]), fontsize=7)
    plt.xlabel("Tiempo medio (s)")
    plt.ylabel("Energia total media (J)")
    plt.title(titulo, fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre), dpi=120)
    plt.close(fig)


def fig_heatmap(out, filas, campo="j_edp"):
    mapas = sorted({r["mapa"] for r in filas})
    robots = sorted({r["robots"] for r in filas})
    if not mapas or not robots:
        return
    mat = np.full((len(mapas), len(robots)), np.nan)
    for r in filas:
        if not r["valida_cfg"]:
            continue  # infactibles fuera del heatmap
        i, j = mapas.index(r["mapa"]), robots.index(r["robots"])
        v = r[campo]
        if np.isnan(mat[i, j]):
            mat[i, j] = v
        else:
            mat[i, j] = min(mat[i, j], v)  # mejor costo por celda
    fig = plt.figure(figsize=(8, 5.5))
    im = plt.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    plt.colorbar(im, label=f"Costo {campo} (menor = mejor)")
    plt.xticks(range(len(robots)), robots)
    plt.yticks(range(len(mapas)), mapas)
    plt.xlabel("N robots")
    plt.ylabel("Mapa (celdas)")
    plt.title(f"Mapa de calor del costo {campo} (curva en U: "
              "penaliza sobrepoblacion)", fontsize=11, fontweight="bold")
    for i in range(len(mapas)):
        for j in range(len(robots)):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.0f}", ha="center",
                         va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"heatmap_{campo}.png"), dpi=120)
    plt.close(fig)


def fig_convergencia(out, hist):
    if not hist:
        return
    it = [h[0] for h in hist]
    fig = plt.figure(figsize=(8, 5))
    plt.plot(it, [h[1] for h in hist], label="Mejor J_EDP acumulado")
    plt.scatter(it, [h[2] for h in hist], s=15, alpha=0.5,
                label="J_EDP por iteracion")
    plt.xlabel("Iteracion")
    plt.ylabel("Costo J (EDP)")
    plt.title("Convergencia del optimizador", fontsize=11,
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergencia.png"), dpi=120)
    plt.close(fig)


def fig_validez(out, filas):
    """Tasa de validez vs robots, una linea por mapa (frontera de
    factibilidad: a la izquierda faltan robots, a la derecha sobra pared)."""
    fig = plt.figure(figsize=(8, 5))
    for mapa in sorted({r["mapa"] for r in filas}):
        sub = sorted([r for r in filas if r["mapa"] == mapa],
                     key=lambda r: r["robots"])
        plt.plot([r["robots"] for r in sub],
                 [r["tasa_ok"] for r in sub],
                 marker="o", label=f"Mapa {mapa}x{mapa}")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.6,
                label="Umbral validez (0.5)")
    plt.xlabel("N robots")
    plt.ylabel("Tasa de validez (validas / reps)")
    plt.title("Frontera de factibilidad: validez vs robots por mapa",
              fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "validez_por_mapa.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="resultados_optimizacion")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.inp, "figs")
    os.makedirs(out, exist_ok=True)

    filas = leer_resumen(os.path.join(args.inp, "resumen_optimizacion.csv"))
    print(f"Configs: {len(filas)}")
    if not filas:
        print("Sin datos: ejecuta primero optimizar_parametros.py")
        return
    fig_pareto(out, filas, "Frente de Pareto global (Tiempo vs Energia)",
               "pareto_global.png")
    for mapa in sorted({r["mapa"] for r in filas}):
        sub = [r for r in filas if r["mapa"] == mapa]
        fig_pareto(out, sub, f"Frente de Pareto | Mapa {mapa}x{mapa}",
                   f"pareto_mapa{mapa}.png")
    fig_heatmap(out, filas, "j_edp")
    fig_heatmap(out, filas, "j_pond")
    fig_validez(out, filas)
    fig_convergencia(out, leer_convergencia(
        os.path.join(args.inp, "convergencia.csv")))
    nfig = sum(1 for p in os.listdir(out) if p.endswith(".png"))
    print(f"Figuras: {nfig} en {out}")


if __name__ == "__main__":
    main()
