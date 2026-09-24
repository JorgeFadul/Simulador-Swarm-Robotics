# Cambios tesis final — bitácora junto al código

Barrido: 5 mapas (10–50) × 6 dotaciones (5–30 robots) × 7 estrategias × 5 reps
= **1050 corridas** (`resultados_tesis_final/series/`, `resumenes.csv`).
Timeout sim proporcional `T=8·N²/R` en [200,12000]s + stall 120s sin progreso.

## 2026-09-23 — Métricas J/defecto + Bio-RL

- Toda métrica "energía/celda" ahora es **J por celda defectuosa encontrada**
  (`E_total/defectuosas`; en curvas vs tiempo se usa `E(t)/GT`).
  Archivos: `graficar_tesis_final.py` (`j_por_def`, `j_def`),
  `graficar_generalista.py` (`e_por_def`, colapso, eficiencia, ranking, Pareto).
- Display **"Cerebro Mosca (Dopamina)" → "Bio-RL"** en todas las figuras
  (`etiqueta()` en `graficar_tesis_final.py`). Solo etiqueta: claves y
  semillas intactas.
- `% defectos` = recall vs GT reproducible por semilla (`gt_defectos.py`,
  `resultados_tesis_final/datos/defectos_gt.csv`, validado 15/15).

### Numéricos globales por estrategia (media de 30 grupos)

| Estrategia | %insp | %def (recall) | vel def max (cel/s) | J/def | t_fin (s) | E tot (J) |
|---|---|---|---|---|---|---|
| Boustrophedon+Boustrophedon Insp | 92.08 | 92.22 | 8.48 | 1.137 | 325.3 | 554.0 |
| Boustrophedon+Voronoi | 92.90 | 96.68 | 8.20 | 1.212 | 503.0 | 564.9 |
| Hybrid Based+Boustrophedon Insp | 92.20 | 92.36 | 8.68 | 1.103 | 292.2 | 547.8 |
| Hybrid Based+Voronoi | 93.07 | 97.05 | 8.32 | 1.190 | 465.5 | 557.9 |
| Bio-RL | 97.90 | 98.02 | 8.11 | 1.142 | 305.8 | 570.0 |
| RL Hibrido V3 | 100.0 | 100.0 | 6.83 | 1.233 | 375.6 | 621.5 |
| Random Walk Hibrido | 62.04 | 66.03 | 9.48 | 1.554 | 623.4 | 535.2 |

### Pareto (mediana por estrategia: t_fin vs J/defecto)

| Estrategia | t_med (s) | J/def_med |
|---|---|---|
| Hybrid Based+Boustrophedon Insp | 210.3 | 0.994 |
| Boustrophedon+Boustrophedon Insp | 227.4 | 0.997 |
| Bio-RL | 209.0 | 1.082 |
| RL Hibrido V3 | 269.9 | 1.109 |
| Hybrid Based+Voronoi | 240.8 | 1.193 |
| Boustrophedon+Voronoi | 267.4 | 1.200 |
| Random Walk Hibrido | 426.7 | 1.414 |

Boustrophedon+Boustrophedon Insp domina (abajo-izquierda); Random Walk
dominado en ambos ejes.

### Óptimos de dotación (tasa defectos/s, `datos/optimo_robots.csv`)

- 10×10: óptimos bajos (Voronoi en 5r; Boust.Insp en 15r) — hacinamiento.
- 20×20: 10–30r según estrategia (Voronoi 10–25r).
- 30–50×50: casi todo en 30r (más robots = más tasa).

### Estancamientos (137/1050, todos Voronoi)

Boustrophedon+Voronoi 71 (94.4% insp media), Hybrid Based+Voronoi 66
(95.2%). Mecanismo único verificado: cola de inspección (1–3 celdas sin
alcanzar en 120s). Detalle: `figs_generalista/datos/stalls.csv`
(`lista_stalls.py`).

### Ley de escala t_fin vs N²/R (pendientes log-log)

Boust.Insp 0.71, Boust+Voronoi 0.50, Hybrid+Boust.Insp 0.68,
Hybrid+Voronoi 0.46, Bio-RL 0.54, RL 0.67, RW 0.85 (sublineal; con
truncamiento por timeouts/stalls — matizar en tesis).

## Fix Voronoi aplicado (código)

- `src/algorithms/inspeccion.py`: bloque 5b sin descartes silenciosos
  (huérfanas → raíz más cercana; cobertura POI 100%), gate padre-hijo
  liberable tras 15s sin inspecciones (`GATE_LIBERAR_S`).
- `src/core/simulacion.py`: `_stall_inspeccion()` no declara fin si Voronoi
  tiene POIs pendientes (antes: falsos `completado` al 84–92%).

## Estructura de figuras

- `resultados_tesis_final/figs_generalista/` — cuerpo (23 láminas).
- `resultados_tesis_final/figs/{04_promedios,datos}` — globales.
- `resultados_tesis_final/figs_apendice/` — detalle (01,02,03,05).
- Espejo en `LATEX TESIS v2/figures/resultados_tesis_final/`
  (`capitulo/` ↔ generalista, `anexo/01_curvas` ↔ apéndice).
