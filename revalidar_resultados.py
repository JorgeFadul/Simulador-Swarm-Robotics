"""Revalida corridas ya guardadas (.npz) con la regla de validez actual.

Marca cada corrida como valida solo si motivo == 'completado' y la
cobertura de su categoria supera el umbral (defecto 98%):
  - exploracion: vistas / libres
  - inspeccion / hibrida: inspeccionadas / libres
Agrega por config con el mismo criterio del optimizador
(min_validas absoluto + tasa >= 0.5) e informa que minimos viejos
sobreviven.

Uso:
  python revalidar_resultados.py --in resultados_optimizacion_try3
  python revalidar_resultados.py --in resultados_optimizacion_try3b --npz series_opt
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from barrido_estrategias import cobertura_pct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--npz", default="series_opt",
                    help="Subcarpeta con los .npz (series_opt o series).")
    ap.add_argument("--umbral", type=float, default=0.98)
    ap.add_argument("--min-validas", type=int, default=3)
    args = ap.parse_args()

    grupos = {}
    for p in sorted(glob.glob(os.path.join(args.inp, args.npz, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
            _ = d["serie"]
        except Exception as e:
            print(f"  AVISO: {os.path.basename(p)} corrupto, se omite ({e})")
            continue
        key = (str(d["categoria"]), str(d["estrategia"]),
               int(d["mapa"]), int(d["robots"]))
        s = d["serie"]
        libres = int(d["libres"])
        motivo = str(d["motivo"])
        if len(s):
            t = round(float(s[-1, 1]), 2)
            vistas, inspecc = int(s[-1, 2]), int(s[-1, 4])
        else:
            t, vistas, inspecc = 0.0, 0, 0
        e = float(d["energia_total"]) if "energia_total" in d else 0.0
        if "cobertura_pct" in d and "valida" in d and motivo != "timeout":
            # npz nuevo (no legado): respetar su marca salvo recalculo
            cob = float(d["cobertura_pct"])
        else:
            cob = cobertura_pct(key[0], vistas, inspecc, libres)
        valida = bool(motivo == "completado" and cob >= args.umbral * 100.0)
        grupos.setdefault(key, []).append(
            {"motivo": motivo, "t": t, "e": round(e, 3), "cob": cob,
             "valida": valida})

    filas = []
    for (cat, est, n, r), runs in sorted(grupos.items(),
                                         key=lambda kv: (kv[0][2], kv[0][3])):
        val = [x for x in runs if x["valida"]]
        tasa = len(val) / len(runs)
        ok = len(val) >= args.min_validas and tasa >= 0.5
        if ok:
            tm = sum(x["t"] for x in val) / len(val)
            em = sum(x["e"] for x in val) / len(val)
            cm = sum(x["cob"] for x in val) / len(val)
            j = round(em * tm, 3)
            causa = ""
        else:
            tm = em = cm = 0.0
            j = float("inf")
            malas = [x["motivo"] for x in runs if not x["valida"]]
            causa = Counter(malas).most_common(1)[0][0] if malas else "sin_data"
            if causa == "completado":
                causa = "baja_cobertura"
        filas.append({"categoria": cat, "estrategia": est, "mapa": n,
                      "robots": r, "reps": len(runs), "n_validas": len(val),
                      "tasa_validez": round(tasa, 3),
                      "valida_cfg": ok, "causa_inf": causa,
                      "t_mean_s": round(tm, 2), "e_mean_j": round(em, 3),
                      "cobertura_media_pct": round(cm, 2),
                      "j_edp": j if ok else "inf"})

    ruta = os.path.join(args.inp, "revalidacion.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"Revalidación: {ruta} ({len(filas)} configs)")
    for x in filas:
        estado = (f"J={x['j_edp']}"
                  if x["valida_cfg"] else f"INFACTIBLE ({x['causa_inf']})")
        print(f"  {x['mapa']:>4}x{str(x['mapa']):<4} {x['robots']:>3}r "
              f"validas={x['n_validas']}/{x['reps']} "
              f"cob={x['cobertura_media_pct']}% {estado}")
    print("\nMínimos válidos por mapa:")
    for n in sorted({x["mapa"] for x in filas}):
        sub = [x for x in filas if x["mapa"] == n and x["valida_cfg"]]
        if sub:
            m = min(sub, key=lambda x: x["j_edp"])
            print(f"  {n}x{n}: {m['robots']}r J={m['j_edp']}")
        else:
            print(f"  {n}x{n}: SIN configs válidas")


if __name__ == "__main__":
    main()
