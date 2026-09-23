"""Ground truth de celdas defectuosas por corrida (compartido).

El GT depende solo de la semilla: misma cadena que el barrido
(Config base + Mundo(cfg, default_rng(seed))). Se cachea en CSV.
"""

import csv
import os

import numpy as np


def gt_cache(runs, ruta_csv):
    """Devuelve dict key corrida -> total defectuosas GT. Cachea en CSV."""
    from src.core.config import Config
    from src.core.mundo import Mundo
    gt = {}
    if os.path.exists(ruta_csv):
        with open(ruta_csv, encoding="utf-8") as f:
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
        m = Mundo(cfg, rng=np.random.default_rng(r["seed"]))
        gt[r["key"]] = int(m._defectuosas_gt)
        faltan += 1
    if faltan:
        os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "gt_total"])
            for k in sorted(gt):
                w.writerow([k, gt[k]])
    print(f"GT defectos: {len(gt)} corridas ({faltan} calculados)")
    return gt
