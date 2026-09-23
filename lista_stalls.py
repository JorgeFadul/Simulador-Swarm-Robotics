"""Tabla de estancamientos por estrategia: n stalls y %insp al estancarse."""
import csv
import glob
from collections import defaultdict

import numpy as np

acc = defaultdict(list)  # etiqueta -> [(motivo, pinsp, mapa, robots)]
for p in sorted(glob.glob("resultados_tesis_final/series/*.npz")):
    z = np.load(p, allow_pickle=True)
    s = z["serie"]
    g, e, s_ = str(z["grupo"]), str(z["exp"]), str(z["estrategia"])
    lab = f"{e}+{s_}" if g == "dos_etapas" else s_
    L = int(z["libres"])
    acc[lab].append((str(z["motivo"]), 100.0 * float(s[-1, 4]) / L,
                     int(z["mapa"]), int(z["robots"])))

print(f"{'estrategia':45s} {'runs':>4s} {'stall':>5s} "
      f"{'pinsp_med':>9s} {'min':>6s} {'max':>6s}")
filas = []
for lab in sorted(acc):
    v = acc[lab]
    st = [x for x in v if x[0] == "stall_120s"]
    if st:
        ps = [x[1] for x in st]
        print(f"{lab:45s} {len(v):4d} {len(st):5d} "
              f"{sum(ps) / len(ps):9.1f} {min(ps):6.1f} {max(ps):6.1f}")
        filas.append((lab, len(v), len(st), round(sum(ps) / len(ps), 1),
                      round(min(ps), 1), round(max(ps), 1)))
    else:
        print(f"{lab:45s} {len(v):4d} {0:5d} {'-':>9s} {'-':>6s} {'-':>6s}")
        filas.append((lab, len(v), 0, "", "", ""))

with open("resultados_tesis_final/figs_generalista/datos/stalls.csv",
          "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["estrategia", "runs", "stalls", "pinsp_media",
                "pinsp_min", "pinsp_max"])
    w.writerows(filas)

# detalle por mapa para las que tienen stalls
print("\nDetalle stalls por mapa (n, pinsp media):")
det = defaultdict(list)
for lab in sorted(acc):
    for m, p, mp, rb in acc[lab]:
        if m == "stall_120s":
            det[(lab, mp)].append(p)
for k in sorted(det):
    v = det[k]
    print(f"  {k[0][:40]:40s} mapa {k[1]}: n={len(v):3d} "
          f"pinsp {sum(v) / len(v):.1f} (min {min(v):.1f})")
