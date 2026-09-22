"""Reconstruye resultados_tesis_final/resumenes.csv desde los .npz (deduplicado)."""
import csv
import glob
from collections import Counter

import numpy as np

CAMPOS = ["id", "grupo", "exp", "estrategia", "mapa", "robots", "rep",
          "seed", "motivo", "pasos", "robots_efectivos", "t_sim_s",
          "wall_s", "vistas", "visitadas", "inspeccionadas", "defectuosas",
          "energia_total_j", "energia_mov_j", "energia_ins_j",
          "t_sim_timeout"]

rows = []
for p in sorted(glob.glob("resultados_tesis_final/series/*.npz")):
    z = np.load(p, allow_pickle=True)
    s = z["serie"]
    rid = p.split("\\")[-1][:-4]
    rows.append({
        "id": rid, "grupo": str(z["grupo"]), "exp": str(z["exp"]),
        "estrategia": str(z["estrategia"]), "mapa": int(z["mapa"]),
        "robots": int(z["robots"]), "rep": int(z["rep"]),
        "seed": int(z["seed"]), "motivo": str(z["motivo"]),
        "pasos": len(s),
        "robots_efectivos": (int(z["robots_efectivos"])
                             if "robots_efectivos" in z else ""),
        "t_sim_s": round(float(s[-1, 1]), 2) if len(s) else 0,
        "wall_s": "",
        "vistas": int(s[-1, 2]) if len(s) else 0,
        "visitadas": int(s[-1, 3]) if len(s) else 0,
        "inspeccionadas": int(s[-1, 4]) if len(s) else 0,
        "defectuosas": int(s[-1, 5]) if len(s) else 0,
        "energia_total_j": float(z["energia_total"]),
        "energia_mov_j": float(z["energia_mov"]),
        "energia_ins_j": float(z["energia_ins"]),
        "t_sim_timeout": float(z["t_sim_timeout"]),
    })

with open("resultados_tesis_final/resumenes.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CAMPOS)
    w.writeheader()
    w.writerows(rows)

print(len(rows), Counter(r["motivo"] for r in rows))
vor = [r for r in rows if r["estrategia"] == "Voronoi"]
print("Voronoi:", Counter(r["motivo"] for r in vor))

libs = {}
for p in glob.glob("resultados_tesis_final/series/*.npz"):
    z = np.load(p, allow_pickle=True)
    libs[p.split("\\")[-1][:-4]] = int(z["libres"])
acc = {}
for r in vor:
    L = libs[r["id"]]
    acc.setdefault((r["mapa"], r["motivo"]), []).append(
        100.0 * int(r["inspeccionadas"]) / L)
for k in sorted(acc, key=str):
    v = acc[k]
    print(k, "n=", len(v), "pinsp media=", round(sum(v) / len(v), 1),
          "min=", round(min(v), 1))

ef = [int(r["robots_efectivos"]) for r in rows
      if r["robots_efectivos"] != ""]
print("robots_efectivos registrados:", len(ef))
corto = [(r["id"], r["robots"], r["robots_efectivos"]) for r in rows
         if r["robots_efectivos"] != ""
         and int(r["robots_efectivos"]) < int(r["robots"])]
print("corridas con menos robots de los pedidos:", len(corto))
