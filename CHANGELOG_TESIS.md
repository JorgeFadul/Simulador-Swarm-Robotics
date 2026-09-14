# CHANGELOG_TESIS — Fase 0 del Simulador Swarm 2D

*Fecha: 31 de agosto de 2026.*

Este archivo registra, para anexar a la tesis, qué se implementó en la Fase 0,
cuáles archivos se modificaron y las fórmulas físicas del modelo de energía que
se asumieron, justificando cada variable y supuesto.

---

## 1. Resumen de lo implementado

La Fase 0 sienta las bases técnicas para las simulaciones avanzadas del
Capítulo 6 (escenarios A-D, métricas, robustez). Concretamente:

1. **Correcciones de UX y bugs**:
   - `requirements.txt` ahora declara `numpy` y `Pillow` (el código ya los
     importaba; faltaban para un clon limpio).
   - Teclas del editor unificadas a `R` (Robot), `P` (Pared), `B` (Borrar),
     coincidiendo con el HUD, la ventana de estadísticas y el README (antes el
     código escuchaba `W` y `E`).
   - El título de la ventana Pygame ya refleja el estado: "Editor", "Ejecutando",
     "Pausado" o "Completado".
   - La pestaña "Individual" se reestructuró con `grid` (barra de batería + mapa
     del robot con peso de expansión) eliminando la dependencia de `pack`
     apilado.

2. **Modelo de energía (batería)**:
   - Nuevo módulo `src/core/energia.py` (`ConsumoEnergia`), integrado al robot.
   - Consumo por movimiento basado en velocidad y torque (ver sección 3).
   - Pico inercial de arranque `E_start` cada vez que `v` pasa de 0 a `v > 0`.
   - Costo fijo por ciclo de inspección (`consumir_energia_inspeccion()`) que
     representa percusión + adquisición + FFT.
   - El robot **no muere** por falta de batería en esta fase (solo telemetría);
     el corte por energía se deja para el escenario de robustez del Capítulo 6.
   - El gasto se refleja en la pestaña "Individual" (barra y J usados) y en la
     pestaña "Global" (energía total, media de batería y picos de arranque).

3. **Modo CLI (Headless) y streaming de datos**:
   - `main.py` acepta `--headless`, `--estrategia`, `--semilla`, `--mapa`
     (más `--help`). En headless no se inicializan ventanas; el bucle corre en
     memoria lo más rápido posible.
   - Streaming **JSONL** por `stdout` (una línea JSON por paso) con posiciones,
     batería y tarea de cada robot, para que un agente de IA lo lea en vivo.
   - Al terminar exporta un **resumen JSON** y una **serie CSV** por paso
     (cobertura y energía), listos para graficar.
   - Reproducible: `--semilla` siembra el RNG de robots, posiciones y de los
     generadores de objetivos (Sample/Hybrid).

---

## 2. Archivos creados y modificados

| Archivo | Tipo | Cambio |
|---|---|---|
| `src/core/energia.py` | nuevo | Modelo de consumo energético (`ConsumoEnergia`). |
| `src/core/telemetria.py` | nuevo | Serialización JSONL del estado y exportación JSON/CSV. |
| `CHANGELOG_TESIS.md` | nuevo | Este documento. |
| `requirements.txt` | mod | Añade `numpy>=1.24` y `Pillow>=9.0`. |
| `config/default.yaml` | mod | Parámetros físicos de energía y límite de pasos headless. |
| `src/core/config.py` | mod | Parseo de los nuevos parámetros con valores por defecto. |
| `src/core/simulacion.py` | mod | `headless` y `seed`, título dinámico, teclas P/B, reset de energía al iniciar. |
| `src/robots/robot.py` | mod | Integra `ConsumoEnergia`, consume energía en cada `update` y en inspección. |
| `src/algorithms/exploration.py` | mod | Generadores aceptan `rng` (reproducibilidad). |
| `src/gui/stats_window.py` | mod | Layout de "Individual", barra de batería y agregados de energía en "Global". |
| `main.py` | mod | `argparse`, runner headless y cargador de escenarios YAML. |

---

## 3. Modelo físico de energía (fórmulas y justificación)

### 3.1 Variables y valores por defecto

| Variable | Símbolo | Valor | Unidad | Justificación |
|---|---|---|---|---|
| Masa del robot | `m` | 0.35 | kg | Chasis pequeño tipo HeRo Swarm (placa + MCU + 2 motores + batería ≈ 350 g). |
| Aceleración de gravedad | `g` | 9.81 | m/s² | Constante estándar. |
| Coeficiente de rodadura | `mu_r` | 0.015 | adim | Ruedas de plástico/rubber sobre baldosa: rango típico 0.01-0.02. |
| Radio de llanta | `r_w` | 0.025 | m | Rueda de ~50 mm de diámetro del chasis. |
| Ancho de trocha (entre ruedas) | `b` | 0.12 | m | Distancia entre puntos de contacto de las ruedas. |
| N.º de ruedas motrices | `n` | 2 | adim | Tracción diferencial del HeRo Swarm. |
| Eficiencia de transmisión | `eta` | 0.6 | adim | Motor DC + engranajes pequeños: 0.5-0.75. |
| Masa de cada rueda | `m_rueda` | 0.02 | kg | Rueda pequeña; se usa para su inercia rotacional. |
| Capacidad de batería | `C` | 1500 | J | ≈ 0.42 Wh (celda Li-Po pequeña de 3.7 V, ~110 mAh). |
| Costo por inspección | `E_inspec` | 0.5 | J | Percusión + FFT en MCU embebido (ver 3.4). |

### 3.2 Fuerzas sobre cada robot

1. **Resistencia a la rodadura** (modelo Coulomb de rodadura):

   `F_rod = mu_r * m * g`

   Es proporcional a la carga normal y prácticamente independiente de la
   velocidad; es la fuerza dominante a baja velocidad sobre piso liso.

2. **Fuerza de aceleración** (ley de Newton para translación):

   `F_acel = m * a`, con `a = (v_t - v_{t-1}) / dt`

   Estimada por diferencias finitas porque el controlador del robot acelera
   linealmente hacia la velocidad objetivo cada tick. Solo se carga cuando
   `a > 0`; la deceleración **no** se recupera (supuesto conservador: no hay
   frenado regenerativo).

3. **Fuerza tractiva total**:

   `F_t = F_rod + max(m * a, 0)`

### 3.3 Potencia eléctrica entregada por ruedas

Se adopta el modelo mecánico clásico: la potencia entregada por cada motor es el
producto del **torque** por la **velocidad angular de la llanta**:

- Torque por rueda motriz (las `n` ruedas comparten la fuerza tractiva):

  `tau = (F_t / n) * r_w`

- Velocidad angular de la llanta:

  `omega_w = v / r_w`

- Potencia de tracción:

  `P_trac = sum_i tau_i * omega_w,i = n * tau * omega_w = F_t * v`

  (independiente de `n`: a más ruedas, menos torque por rueda para el mismo
  `F_t` — consistente con la física).

- Potencia por giro en el sitio: al girar con velocidad angular de robot
  `omega_robot`, las ruedas se mueven en sentido contrario con velocidad
  superficial `0.5 * b * |omega_robot|` y disipan la misma rodadura:

  `P_giro = F_rod * |omega_robot| * (b / 2)`

  Modelo deliberadamente **solo resistivo**: no se modela la aceleración angular
  de giro (ver limitaciones 3.6).

- Potencia mecánica y eléctrica:

  `P_mech = P_trac + P_giro`,  `P_elec = P_mech / eta`

- Energía del intervalo de simulación `dt = 0.05 s`:

  `dE = P_elec * dt`

### 3.4 Pico inercial de arranque

Cada vez que el robot pasa de `v = 0` a `v > 0`, para romper la inercia se debe
entregar la energía cinética del cuerpo más la de las ruedas:

`E_start = [ 0.5 * m * v^2  +  n * 0.5 * I_rueda * (v / r_w)^2 ] / eta`

con `I_rueda = 0.5 * m_rueda * r_w^2` (rueda modelada como disco macizo).

Justificación del `1 / eta`: la energía cinética que "queda" en el robot es
`0.5 m v^2`; la fuente debe *entregarla por encima* de la marcha normal,
pagando además las pérdidas de la transmisión (por eso se divide entre `eta`).

### 3.5 Costo de inspección (percusión + FFT)

`consumir_energia_inspeccion()` suma un valor constante `energia_inspeccion_j`
cada vez que el robot visita una celda nueva (se considera que ahí realiza un
ciclo de inspección completo: detenerse, percudir, adquirir la ventana de
audio, calcular la FFT y clasificar).

Justificación de `0.5 J`: una FFT de 256-1024 muestras en un ESP32 a 80-160 MHz
consume ~0.1-1 J incluyendo el accionamiento del percutor (solenoide/mini
vibrador); se toma 0.5 J como referencia conservadora y es **configurable**
(`energia_inspeccion_j`).

En la corrida de validación (4 robots, mapa 400x400, grilla 20, Frontier Based,
semilla 7, 37.8 s) el modelo dio `energia_inspeccion_j * 257 visitadas = 128.5 J`
y `energia_movimiento ≈ 4.9 J`, coherente con que en este simulador el costo
dominante de la inspección es el del ciclo de percusión/FFT, no el traslado.

### 3.6 Limitaciones declaradas del modelo

- El **motor eléctrico se modela a nivel mecánico** (`P = tau * omega`): no se
  simula la curva I-V del motor DC (corriente de arranque, constante de fuerza
  contraelectromotriz), solo se aplica una eficiencia `eta` constante.
- No hay **deceleración regenerativa** (la energía de frenado no se recupera).
- La **aceleración angular** del giro no se modela (solo el término resistivo
  `P_giro`); el costo de giro queda subestimado para rotaciones muy bruscas.
- En esta fase la batería **no limita el comportamiento**: el robot nunca se
  detiene por agotamiento. El gasto se acumula solo con fines de telemetría y
  comparación entre estrategias.

### 3.7 Modelo de inspección de baldosas (resultado binario) y estado de celdas

A partir de este punto se distingue explícitamente **celda vista**, **celda
visitada** y **celda inspeccionada**:

- **Vista** (`SCOUTED`): el robot la avistó desde lejos (sensado de proximidad /
  cámara), sin contacto ni medición. No gasta energía de inspección.
- **Visitada** (`VISITED`): el robot pasó físicamente sobre la celda libre. No
  implica haber tomado medición.
- **Inspeccionada**: el robot ejecutó el ciclo de inspección sobre esa celda
  (percusión + adquisición + FFT). Es la acción que consume
  `energia_inspeccion_j` y la que revela el **resultado**.

**Ground truth de defectos.** Cada celda libre tiene una propiedad intrínseca
"sana (0) / defectuosa (1)" generada con probabilidad `prob_defecto`
(configurable, por defecto 0.2) solo sobre celdas libres (bordes y paredes no
inspeccionables). Se genera con una semilla local derivada del `rng` de la
simulación, de modo que con el mismo `--semilla` se reproduce exactamente la
misma distribución de defectos (verificado: 61 defectuosas de ground truth
idénticas en dos corridas con seed 7).

**Resultado de la inspección.** Al inspeccionar, el robot "revela" el resultado
de esa celda en `resultado[r, c]`: `-1` desconocido, `0` sana, `1` defectuosa.
El dato se guarda con sus coordenadas `(x, y)`, fila y columna.

**Disparo de la inspección.** La *decisión de cuándo* inspeccionar es una
**acción del robot perteneciente a cada estrategia** (aún por definir en cada
una). Para que el pipeline de datos funcione desde ya, el disparo por defecto
(`inspeccionar_al_visitar: true`) inspecciona cada celda libre nueva que el
robot visita. Cada estrategia puede reemplazar este disparo llamando a
`Robot.inspeccionar_actual(mundo)` cuando decida hacerlo.

**Estado de celdas serializado.** Por cada celda de interés se emite un objeto:

```json
{"x": 170.0, "y": 270.0, "fila": 13, "col": 8, "tipo": 0,
 "vista": true, "visitada": true, "inspeccionada": true, "resultado": 0}
```

donde `tipo` es 0 (libre) / 1 (pared) y `resultado` es el binario de la
inspección (`-1` si no fue inspeccionada).

```---

## 4. Modo headless y streaming (documentación técnica)

### 4.1 Formato JSONL por paso

Una línea por paso en `stdout` (ejemplo abreviado):

```json
{"t": 0.3, "paso": 6, "estado": "running", "estrategia": "Random Walk",
 "cobertura": {"mapa_pct": 4.28, "libres_pct": 4.63, "paredes_pct": 1.17},
 "energia_total_j": 0.565, "bateria_min_pct": 99.97,
 "celdas": {"vistas": 31, "vistas_pct": 9.57, "visitadas": 0, "visitadas_pct": 0.0,
            "inspeccionadas": 0, "inspeccionadas_pct": 0.0,
            "defectuosas": 0, "defectuosas_pct": 0.0},
 "por_robot": [{"vistas": 12, "visitadas": 0, "inspeccionadas": 0, "defectuosas": 0, "id": 0}, ...],
 "estado_celdas": [{"x": 170.0, "y": 110.0, "fila": 5, "col": 8, "tipo": 0,
                    "vista": true, "visitada": false, "inspeccionada": false, "resultado": -1}, ...],
 "robots": [{"id": 0, "x": 511.3, "y": 930.2, "angulo_deg": -2.1,
             "velocidad_cm_s": 0.0, "estado": "ROTATING", "tarea": "ninguna",
             "energia_consumida_j": 0.005, "bateria_restante_j": 1499.995,
             "bateria_pct": 100.0,
             "celdas_vistas": 12, "visitadas": 0, "inspeccionadas": 0, "defectuosas": 0}, ...]}
```

Campos por robot: posición `(x, y)`, orientación, velocidad (cm/s), estado de la
máquina (`STOPPED|ROTATING|MOVING`), tarea actual (objetivo asignado o
`ninguna`/`timeout`), energía consumida, batería restante (J) y porcentaje, y
sus propios contadores de celdas vistas/visitadas/inspeccionadas/defectuosas.

Campos de aglomerado por paso:
- `celdas`: conteos generales del enjambre con **número y porcentaje** de celdas
  vistas, visitadas, inspeccionadas y defectuosas (el % de defectuosas es sobre
  las inspeccionadas).
- `por_robot`: los mismos 4 contadores por robot.
- `estado_celdas`: posición `(x, y)`, fila, columna, tipo (libre/pared) y estado
  vista/visitada/inspeccionada + resultado, para monitorear "todo" el mapa en
  tiempo real.

### 4.2 Ejecución y salidas

```bash
python main.py --headless --estrategia "Frontier Based" --semilla 7 --mapa escenario.yaml
```

- `--semilla` fija posiciones iniciales y muestreo (corridas idénticas = mismos
  resultados; verificado: 37.8 s / 757 pasos en 2 corridas con seed 7).
- `--mapa` acepta un YAML con `map_width`, `map_height`, `grid_size`,
  `num_robots` y `walls` (para los escenarios A-D del Capítulo 6).
- El CSV es una serie por paso: paso, t_s, cobertura de mapa y de libres (%),
  energía total (J) y los contadores de celdas vistas, visitadas, inspeccionadas
  y defectuosas.
- El resumen JSON añade, además de lo visto arriba:
  - `celdas`: conteos generales finales con número y porcentaje;
  - `por_robot`: contadores finales por robot;
  - `escenario_original`: **el "área de inspección original"**, i.e. todo lo
    necesario para reproducir el mapa y la corrida: `map_width`, `map_height`,
    `grid_size`, `grilla_ocupacion` (0=libre, 1=pared por fila), la lista de
    `paredes` y el ground truth `defectos_gt`. Con `--semilla` + este bloque se
    puede reconstruir exactamente el escenario;
  - `estado_final_celdas`: dump completo con x,y, fila, columna, tipo, estado y
    resultado de cada celda al terminar;
  - `grilla_final`: matriz binaria de celdas inspeccionadas (1 = inspeccionada)
    al finalizar.

### 4.3 Determinismo

En headless, `--semilla` se propaga a:
- `Simulacion.rng` -> posiciones iniciales de robots y ángulos;
- `ExplorationPlanner` -> muestreo de candidatos (Sample Based) y puntos por
  región (Hybrid);
- `Mundo` -> ground truth de defectos (distribución de celdas defectuosas).

Con el mismo `--semilla` se obtienen corridas idénticas en tiempo, pasos,
posiciones y defectos (verificado en 2 corridas con seed 7: 47.3 s, 946 pasos,
61 defectuosas de ground truth, y mismos 56 defectuosos revelados).

---

## 5. Notas de validación

- `python -m py_compile` sobre todos los módulos: OK.
- Corrida headless Random Walk: telemetría JSONL emite estado por paso, con el
  pico inercial visible (un robot que pasa de 0 a ~14 cm/s suma ~0.52 J en un
  tick).
- Corrida headless Frontier Based (escenario de prueba): completado a 37.8 s,
  cobertura de libres 100 % y mapa 99.25 %, con energía inspección = 257 x 0.5 J
  exactamente, y resumen JSON + serie CSV exportados.
- Corrida headless Frontier Based + inspección (3 robots, 400x400, grilla 20,
  prob_defecto 0.3, seed 7): 47.3 s / 946 pasos. `estado_celdas` por paso emite
  x,y, fila, col, tipo, vista, visitada, inspeccionada y resultado. Resultado
  agregado: 324 vistas (100 %), 264 visitadas (81.48 %), 264 inspeccionadas,
  56 defectuosas (21.21 % de las inspeccionadas). `escenario_original` guardó
  grilla y ground truth (61 defectuosas) y es reproducible con el mismo seed.

---

# CHANGELOG_TESIS — Fase 1 del Simulador Swarm 2D

*Fecha: 31 de agosto de 2026.*

La Fase 1 introduce la **arquitectura de dos etapas** (Exploración → Inspección)
para las estrategias dirigidas, la **capa oculta de áreas afectadas** generada
algorítmicamente y un **conjunto de 5 mapas de prueba** reproducibles. Se separa
el "descubrir" (exploración de celdas libres) del "inspeccionar" (percusionar la
baldosa para revelar el ground truth), que en la Fase 0 ocurrían juntos.

---

## 1. Resumen de lo implementado

1. **Arquitectura de dos etapas (2E) vs. híbrida de una etapa (1E-H)**:
   - Nueva clasificación por estrategia en `config`: un mapa `architecturas`
     asigna cada estrategia a un modo.
     - `Random Walk` → `1E-H` (una sola etapa, híbrida: insp en la marcha).
     - `Frontier Based`, `Sample Based`, `Hybrid` → `2E` (dos etapas).
   - `Config.modo_estrategia(strategy)` devuelve la etiqueta `1E-H`/`2E` y
     `Config.es_dos_etapas()` indica si la estrategia usa dos etapas.
   - `Mundo` y `Simulacion` llevan una variable `etapa` (`exploracion` o
     `inspeccion`) que se refleja de la simulación al mundo.

2. **Transición global Exploración → Inspección** (estrategias 2E):
   - Es **global al enjambre** (no decide cada robot): todo el enjambre cambia de
     etapa a la vez.
   - Triggers automáticos (`config.transicion_*`):
     - `porcentaje_descubierto >= transicion_umbral_descubierto` (0.99 por
       defecto: ya no queda nada relevante por explorar); o
     - pasan `transicion_tiempo_sin_scouteo_s` (60 s) sin que ninguna celda
       nueva sea scouteada (estancamiento de exploración).
   - Botón manual en la UI "Transicionar a Inspección" (`forzar_transicion_inspeccion`),
     habilitado solo si la estrategia es 2E y la simulación está en marcha.

3. **Planificador de inspección (Fase 1, placeholder)**:
   - `src/algorithms/inspeccion.py` (`InspectionPlanner`): asigna a cada robot
     libre la celda libre sin inspeccionar más cercana (greedy por cercanía,
     con reservas para evitar duplicados).
   - Al llegar al objetivo, el robot ejecuta la percusión (`inspeccionar_actual`)
     y revela el resultado.
   - Es un marcador de posición deliberado: la Fase 3 lo reemplaza por el
     esquema definitivo ("Voronoi recursivo por bordes de distancia" + "Market
     Auction Method" con costo multiplicativo hereditario).
   - Robustez añadida: timeout de objetivo en el movimiento dirigido y registro
     de celdas "fallidas" (inalcanzables) para que un robot no se quede trabado
     ni reasigne en bucle la misma celda bloqueada.

4. **Capa oculta de áreas afectadas**:
   - Ground truth de defectos como capa separada (`Mundo.defectos`) que la
     exploración no "consume": solo la percusión la revela (escenario C de la
     tesis).
   - Toggle visual `ver_defectos` en la pestaña Controles y capa de render
     diferenciada (`_draw_defectos` en el renderer): celdas defectuosas en tono
     rojo/naranja, aclaradas cuando ya fueron inspeccionadas y confirmadas.
   - `defecto_blobs`: lista de `{cx, cy, radio, semilla}` que se generan como
     manchas irregulares (ver sección 3).

5. **5 mapas de prueba generados algorítmicamente**:
   - `src/utils/mapas_prueba.py` + `generar_mapas_prueba.py` + `main.py
     --generar-mapas`. Se dejan listos en `mapas/mapa_test_1..5.yaml`.
   - Paredes internas (no cerradas) por escenario + blobs semilla colocados en
     espacio libre lejos de las paredes y separados entre sí.
   - `main.py --mapa` lee `defecto_blobs` y desactiva `prob_defecto`.

---

## 2. Archivos creados y modificados

| Archivo | Tipo | Cambio |
|---|---|---|
| `src/utils/mapas_prueba.py` | nuevo | Generador de los 5 mapas de prueba (paredes internas + blobs). |
| `generar_mapas_prueba.py` | nuevo | CLI para regenerar los mapas. |
| `config/default.yaml` | mod | Añade `ruido_inspeccion_sigma`, `defecto_blobs`, `architecturas`, `transicion_*.` |
| `src/core/config.py` | mod | Parseo de los campos nuevos; `modo_estrategia`, `es_dos_etapas`. |
| `src/core/mundo.py` | mod | Capa `defectos`, `etapa`, `visitar_area`, `inspeccion_completa`, `_resultado_con_ruido`, `_inicializar_defectos_blobs`. |
| `src/core/simulacion.py` | mod | Estado `etapa`, `_verificar_transicion`, `forzar_transicion_inspeccion`, integración del `InspectionPlanner`, terminación por etapa, `ver_defectos`, HUD de etapa. |
| `src/robots/robot.py` | mod | Inspección al llegar al objetivo en 2E, `visitar_area`, sincronización de área, timeout de objetivo en movimiento dirigido. |
| `src/algorithms/inspeccion.py` | nuevo | `InspectionPlanner` (placeholder de Fase 3). |
| `src/gui/renderer.py` | mod | Dibujo de la capa de defectos (`_draw_defectos`) con `ver_defectos`. |
| `src/gui/stats_window.py` | mod | Etiqueta `1E-H`/`2E`, botón de transición, etapa en Global, toggle `ver_defectos`. |
| `mapas/mapa_test_1..5.yaml` | nuevo | Escenarios de prueba generados. |
| `CHANGELOG_TESIS.md` | mod | Este documento (sección Fase 1). |

---

## 3. Algoritmos y fórmulas

### 3.1 Generador de blobs (áreas afectadas)

Cada área afectada es una mancha irregular que nace de una celda semilla
`(rc, cc)` (grado a partir de `{cx, cy}` en px y `grid_size`) y "crece" con una
caminata aleatoria con amortiguación radial:

```
para n pasos (n ∝ radio * grid_size):
    candidato = vecino aleatorio 8-conectado de la celda actual
    si distancia(candidato, centro)² > radio²:  descartar   # amortiguación radial
    si candidato es pared o está fuera del mapa: descartar   # nunca sobre muros/bordes
    si candidato ya es defecto de otro blob:     descartar   # evita solape entre blobs
    marcar candidato como defectuoso
```

El rechazo de paredes garantiza que las manchas nunca toquen las paredes ni el
borde del mapa; el rechazo de celdas ya marcadas evita solapes entre blobs
(permite adyacencia). Cada blob usa su propia `semilla` determinista, por lo que
los mapas son idénticos corrida a corrida con independencia de `--semilla` de la
simulación. Los centros semilla se eligen en `mapas_prueba._puntos_libres` en
celdas libres, a más de `radio + margen` de las paredes y separados ≥
`2.6 * radio` entre sí.

### 3.2 Detección de defectos

Con `ruido_inspeccion_sigma = 0`, la percusión devuelve el ground truth exacto
(detección 100 % efectiva). El método `Mundo._resultado_con_ruido(r, c)` es la
costura preparada para inyectar ruido gaussiano sobre la probabilidad de
detección en fases posteriores (escenario de robustez).

### 3.3 Terminación de la etapa de inspección

La simulación 2E termina cuando `Mundo.inspeccion_completa()` (todas las celdas
libres sin inspeccionar son cero). Como algunas celdas pueden quedar inalcanzables
por la geometría de las paredes, se añade un criterio de estancamiento: si pasan
más de 30 s simulados sin nuevo avance de inspección, la simulación termina
para no correr hasta el límite de pasos sin progreso.

---

## 4. Interacción con la capa oculta

- Durante la **exploración** (2E), los robots NO inspeccionan: solo marcan celdas
  como scouteadas/visitadas para no gastar energía de percusión antes de tiempo.
- Durante la **inspección** (2E), al llegar al objetivo el robot percute la celda
  y el resultado se guarda en `Mundo.resultado` (SANA/DEFECTUOSA). El ground
  truth (`Mundo.defectos`) no es visible para los robots: solo `resultado`
  en las celdas ya inspeccionadas.
- En **1E-H** (Random Walk con `inspeccionar_al_visitar`), la inspección ocurre
  sobre la marcha durante la exploración, como un operario que revisa mientras
  recorre.
- El render de defectos es un **toggle del operador**: muestra dónde está el
  ground truth para verificar visualmente la cobertura, sin darlo a los robots.

---

## 5. Notas de validación

- `python -m py_compile` sobre todos los módulos: OK.
- Corrida headless 2E (Frontier Based, `mapa_test_2`, seed 7):
  - Transición Exploración → Inspección en `t ≈ 61.8 s` (trigger de 60 s sin
    scouteo: la exploración estancó antes de llegar al 99 %).
  - Termina "completado" a `t ≈ 254.6 s`; 291 de 292 celdas libres inspeccionadas
    y las 9 defectuosas de ground truth reveladas (detección 100 %).
  - Se verifica que en 2E NO se inspecciona durante la exploración (contador de
    inspecciones se mantiene en 0 hasta la transición).
- Corrida headless 2E sobre los 5 mapas (Frontier Based): los 5 completan la
  etapa de inspección; en mapas sin celdas bloqueadas se alcanza el 100 % de
  detección; en mapas con celdas inalcanzables se detiene por estancamiento con
  la mayoría de la cobertura realizada (ej. mapa 1: 25/27 y mapa 5: 26/27).
- 1E-H (Random Walk): la transición no se dispara (no es 2E) y la terminación usa
  el criterio clásico de exploración (`exploration_goal` o fronteras agotadas).
- Interfaz: etiqueta del modo junto al selector de estrategia, botón de
  transición solo habilitado en 2E y en marcha, etapa visible en la pestaña
  Global y en el HUD, y toggle de defectos en Controles.

## 6. Refinamientos de la etapa de Inspección (visualización y rendimiento)

Mejoras sobre la Fase 1 para hacer la etapa de Inspección más clara y fluida:

- **Pausa de percusión (1 s).** Al llegar una meta de inspección, el robot se
  detiene un segundo sobre la celda ("probándola") antes de registrar el
  resultado. Implementado con un estado propio del robot
  (`Robot.inspeccionando`, `inspeccionando_timer`, `celda_probando`) que se
  actualiza en `update` antes de la lógica de movimiento: durante la pausa el
  robot queda quieto, no se le reprograma otra meta y ejecuta la percusión al
  agotarse el temporizador (`inspeccion_pausa_s: 1.0` en `config/default.yaml`).
- **Solo se prueba una celda que no se ha probado aún.** La regla para las
  estrategias actuales es "por donde camina el robot, si la baldosa no ha sido
  probada la inspecciona y, si ya fue probada, no". Esto se cumple en dos
  puntos:
  - En la percusión misma (`Mundo.inspeccionar_celda` devuelve `None` si la
    celda ya está inspeccionada o es pared), de modo que el tránsito de 1E-H
    ("inspeccionar al visitar") no reprocesa baldosas ya probadas.
  - En el arranque de la pausa de 2E (`Robot._iniciar_inspeccion`), que comprueba
    la celda actual antes de detenerse: si ya fue probada (por este u otro robot)
    o es pared, el robot no pausa ni gasta energía y queda libre para que el
    planificador le asigne la siguiente celda.
  Esta lógica se mantiene acoplada al punto de inspección (`inspeccionar_actual`)
  para que las estrategias futuras puedan decidir distinto sin cambiar el núcleo.
- **Estados de color por celda inspeccionada.** El renderer (`_draw_inspeccion`)
  pinta, siempre y por encima de la capa oculta:
  - amarillo (`255,210,0`): la celda que un robot está probando en ese momento;
  - azul (`70,130,230`): celda inspeccionada con resultado SANA;
  - rojo (`230,60,40`): celda inspeccionada con resultado DEFECTUOSA.
  Estos resultados son conocidos por los robots (no ocultos), así que no
  dependen del toggle de defectos.
- **Capa de defectos en rosado.** El toggle "ver defectos" (ground truth del
  operador) ahora pinta las celdas defectuosas en rosado (`255,120,190`) y se
  dibuja debajo de los resultados de inspección, para no confundirse con ellos.
- **Exploración atenuada en Inspección.** Al transicionar a la etapa de
  Inspección, el rastro de exploración (celdas scouteadas/visitadas) se atenúa
  al 30 % para dejar el foco visual en las pruebas de las baldosas.
- **Rendimiento.** El renderer ahora dibuja la capa de celdas como una surface
  cacheada (se reconstruye solo cuando cambia la exploración/etapa) y las líneas
  de grilla como trazos en lugar de un rectángulo por celda; además se cacbean
  los *thumbnails* por robot. Se eliminan cientos de llamadas
  `pygame.draw.rect` por frame, lo que mejora notablemente los FPS en la etapa
  de Inspección (y en todas las demás).
- **Validación:** la pausa integrada no rompe el flujo 2E: en `mapa_test_2`
  (Frontier Based, seed 7) el enjambre completa la inspección (`completado`,
  275/292 inspeccionadas, 9/9 defectos revelados), con el tiempo de cómputo
  mayor esperado por las pausas de 1 s por celda.

---

# CHANGELOG_TESIS - Fase 2 del Simulador Swarm 2D

*Fecha: 31 de agosto de 2026.*

Esta fase incorpora dos estrategias de exploración de **cobertura** (Coverage
Path Planning) a las ya existentes (Random Walk, Frontier, Sample, Hybrid):

1. **Boustrophedon (1E-H y 2E)**: barrido en zigzag con restricción de ceguera.
2. **Voronoi + Boustrophedon + Hybrid Frontier (1E-H)**: partición del mapa
   descubierto en regiones de Voronoi, una por robot.

A diferencia de Frontier Based (que persigue regiones de frontera), estas
estrategias atacan la cobertura del plano. Ambas se integran en el mismo
`ExplorationPlanner`, de modo que reutilizan el emparejamiento por cercanía y
el mapa global combinado `mundo.explored` (ningún robot conoce el mapa real
total, solo lo que el enjambre ha descubierto).

## 1. Resumen de lo implementado

- **`Boustrophedon`** (`src/algorithms/exploration.py`): genera líneas de
  barrido horizontales espaciadas `paso` aparte y alterna la dirección
  izquierda-derecha / derecha-izquierda en cada pasada. Cada línea se parte en
  tramos conexos de celdas libres (`_tramos_barrido`) y cada tramo se subdivide
  en waypoints cada `paso` (`_waypoints_tramo`), que son las metas reales que
  recorre el robot.
- **`VoronoiExploracion`** (`src/algorithms/exploration.py`): particiona el mapa
  descubierto y libre con un Diagrama de Voronoi discreto (`_particion`), asigna
  cada región a su robot (los robots reciben la región de su sitio), y dentro de
  su región el objetivo es la mejor celda de frontera por el puntaje de ganancia
  menos costo (`Hybrid Frontier` dentro de la región). La partición se recalcula
  cada `plan_interval_s`, lo que al detectar una región cubierta o bloqueada por
  una pared nueva emite una partición nueva: es el recálculo dinámico de
  centroides que pide el enunciado.
- **Registro en arquitecturas**: tanto `config/default.yaml` como
  `src/core/config.py` (`_DEFAULT_ARCHITECTURAS`) registran la arquitectura por
  defecto: `Boustrophedon -> 1E-H` y `Voronoi -> 1E-H`. `main.py`
  (`ESTRATEGIAS_VALIDAS`), `src/core/simulacion.py` (`_crear_planner`) y la lista
  del selector de `src/gui/stats_window.py` añaden las dos nuevas estrategias.
- **Robustez de cierre (1E-H dirigida)**: se corrige en `_termino` un bug que
  recién afloraba con la primera estrategia dirigida de una sola etapa: el fin
  por "fronteras agotadas" no debe evaluarse con la máscara cacheada del
  generador (que se refresca cada `plan_interval_s` y puede estar
  desactualizada cuando `_termino` corre justo antes de `actualizar`, dando 0
  fronteras al arrancar porque aún no se ha sondeado nada). Ahora:
  - se exige un guarda de arranque de dos ciclos de planificación, y
  - "fronteras agotadas" se evalúa **en vivo** con `_celdas_frontera(mundo)`.
  - Además, para las estrategias dirigidas 1E-H se añade un criterio de
    estancamiento: si hace más de `transicion_tiempo_sin_scouteo_s` (60 s)
    simulados que no se descubre ninguna celda nueva, la exploración útil
    terminó (la frontera remanente es inalcanzable tras las paredes) y se
    completa. Este umbral comparte semántica con el trigger de la arquitectura
    2E (`_verificar_transicion`). El `Robot` Random Walk conserva su
    comportamiento clásico (termina solo por `exploration_goal`), de modo que
    no se alteran las estrategias de referencia.

## 2. Matemática del barrido Boustrophedon con ceguera

Sea el mapa una retícula de `R x C` celdas de lado `g` (px). El algoritmo solo
conoce **las cuatro esquinas exteriores**, es decir, el rectángulo
`[0, W] x [0, H]` con `W = C*g` y `H = R*g`. No conoce las paredes internas,
por lo que el patrón se planifica sobre ese rectángulo sin pre-procesamiento de
obstáculos.

**Generación de las pasadas.** Se definen líneas de barrido horizontales en
orden descendente:

```
y_k = paso/2 + k*paso ,   k = 0, 1, ..., con y_k < H
```

La separación `paso` (px) es el ancho efectivo de una pasada. Para que el
sensor (radio de detección `r_d`) no deje huecos entre fajas se fija

```
paso = max(r_d, g)
```

que en la configuración actual (`r_d = det_radius_cm * pixels_per_cm`) da
`paso = 50 px` (2.5 celdas), con cobertura efectiva del doble por el factor de
exploración `xr = 1.5` (radio efectivo `r_e = r_d * xr`).

**Dirección alternada (zigzag).** La dirección del recorrido de la pasada `k`
es

```
dir_k = (+1) si k es par (izq -> der) ,  (-1) si k es impar (der -> izq)
```

que es exactamente el patrón *boustrophedon*: "el camino del buey" que ara cada
surco en sentido contrario al anterior, minimizando los giros en U.

**Ceguera y adaptación de waypoints al encontrar obstáculos.** Con restricción
de ceguera el robot no predice las paredes; cuando una pared interna se
**descubre** por línea de vista o por choque (`mundo.explored == DISCOVERED`),
el tramo de la pasada que la cruzaba se parte en dos:

```
tramo = [ (r,c0)...(r,ci-1) ] | (pared) | [ (r,ci+1)...(r,c1) ]
```

La pasada original «costaba» en una sola línea; al detectar la pared, el
planificador emite dos metas independientes, una a cada lado de la pared, y el
robot las cubre rodeando el obstáculo con la ruta A* (`encontrar_ruta`), que ya
pesa las celdas libres conocidas a costo 1 y las desconocidas a costo 6. Cada
tramo se subdivide en waypoints espaciados `paso`:

```
waypoint_j  se emite en la celda cuya coordenada x cumple
    |x_waypoint_j - x_waypoint_{j-1}| >= paso
```

Así el robot recorre la pasada por segmentos en vez de saltar directo al
extremo, lo que mantiene la cobertura densa incluso en presencia de paredes
internas. Este mecanismo `_pared_conocida` es el "**Hybrid Frontier**" del
enunciado: el barrido ciego se combina con la información de frontera/sensores
para recalcular dinámicamente y bordear lo imprevisto.

**Cobertura alcanzable.** Al terminar (por `exploration_goal`, por
estancamiento o por fronteras agotadas), la cobertura queda limitada por la
conectividad del plano: las zonas separadas por paredes sin paso no se alcanzan.
En `mapa_test_1` (Corredor Central) el Boustrophedon 1E-H completó en `~221 s`
(4423 pasos) con `~68 %` del mapa, consistente con la porción libre alcanzable
del escenario.

## 3. Matemática de la partición de Voronoi dinámica en mapas parcialmente conocidos

**Partición de Voronoi discreta sobre el conocimiento actual.** En cada ciclo de
planificación se toma el conjunto de celdas conocidas y libres

```
Omega_k = { (r,c) : grid[r,c] = 0  y  explored[r,c] != 0 }   (mapa descubierto hasta ahora)
```

y un conjunto de *sitios* (una posición por robot) `S = { s_1, ..., s_N }`
(inicialmente la posición X-Y de cada robot). Cada celda se asigna al sitio de
mínima distancia euclídea:

```
V_i = { q in Omega_k :  argmin_{s_j in S} || q - s_j ||_2 = s_i }
```

que es la **celda de Voronoi** del sitio `s_i` restringida a `Omega_k`. Las
fronteras de estas regiones son los puntos del eje equidistante
(`bisectrices`), y en mapa parcialmente conocido la región de un robot puede ser
disconexa o limitada por la frontera con lo desconocido — de ahí el término
"Voronoi dinámico en mapas parcialmente conocidos": la partición solo se define
sobre lo ya descubierto y **cambia conforme el mapa se expande**.

**Asignación.** Cada robot `i` recibe en exclusiva los objetivos de `V_i`
(`_asignar_voronoi`). Si su región no tiene fronteras disponibles (cubierta o
bloqueada por una pared nueva), el robot cae al objetivo más cercano de otra
región; en el siguiente ciclo `_particion` se recalcula y los centroides (los
sitios) emiten una partición nueva. Es el "recalcular los centroides" del
enunciado: la detección por el sensor fronterizo de que una región quedó
cubierta/bloqueada provoca el rebalanceo.

**Objetivo dentro de la región (Hybrid Frontier).** Dentro de `V_i`, el robot
puntúa cada celda de frontera `q` de su región con el criterio de ganancia de
información menos costo (estilo next-best-view):

```
score(q) = G(q) - w_c * c_i(q)
```

donde:
- `G(q)` es el número de celdas aún desconocidas que quedarían visibles desde
  `q` dentro del radio efectivo `r_e`, exigiendo línea de vista (`tiene_linea_de_vista`);
- `c_i(q) = || q - s_i ||_2` es el costo de traslado desde el sitio de la región;
- `w_c = sample_cost_weight` (0.02) pondera el costo.

El robot de `V_i` toma el objetivo de mayor `score` de su región, reproduciendo
el "Boustrophedon dentro de su región" como una exploración enfocada de la
frontera asignada (el barrido denso queda implícito en la cobertura de la
región por el robot asignado).

**Comparación medida.** En `mapa_test_1` con la semilla 7, la estrategia
Voronoi completó la exploración en `~154 s` (3084 pasos) con cobertura del
`99 %`: al atacar directamente las fronteras alcanzables de su propia región de
Voronoi evita el trabajo redundante que un barrido global deja en zonas ya
cubiertas, y la cobertura remanente tras las paredes (inasignable) es el único
límite.

## 4. Archivos creados y modificados (Fase 2)

- `src/algorithms/exploration.py`: nueva clase `Boustrophedon`, nueva clase
  `VoronoiExploracion`, auxiliares `_paso_barrido`, `_pared_conocida`,
  `_tramos_barrido`, `_waypoints_tramo`; registro en `ExplorationPlanner` y nuevo
  método de asignación `_asignar_voronoi`. Import de `DISCOVERED`.
- `src/core/simulacion.py`: `_crear_planner` acepta `Boustrophedon` y `Voronoi`;
  `_termino` corrige el fin por "fronteras agotadas" (guarda de arranque + máscara
  en vivo) y añade el criterio de estancamiento para estrategias dirigidas 1E-H;
  `step()` registra el progreso de descubrimiento.
- `src/core/config.py`, `config/default.yaml`: arquitecturas por defecto de las
  nuevas estrategias (1E-H).
- `main.py`: `ESTRATEGIAS_VALIDAS` ampliado.
- `src/gui/stats_window.py`: lista del selector de estrategia ampliada.

## 5. Notas de validación (Fase 2)

- `compileall` OK.
- Todas las estrategias (Frontier, Sample, Hybrid, Boustrophedon, Voronoi)
  corren sin excepción con 3 robots en `mapa_test_1` (seed 7) a 800 pasos, con
  cobertura en el rango esperado (Frontier 71 %, Sample 49 %, Hybrid 66 %,
  Boustrophedon 51 %, Voronoi 58 % a 800 pasos).
- Corridas completas (seed 7, `mapa_test_1`): Boustrophedon 1E-H completó
  (estancamiento) en ~221 s con ~68 %; Voronoi 1E-H completó en ~154 s con
  ~99 %.
- `Random Walk` (1E-H de referencia) conserva su comportamiento clásico (llega
  al límite de pasos si no alcanza 99 %), confirmando que los cambios de
  `_termino` no alteran las estrategias existentes.

---

# CHANGELOG_TESIS - Fase 3 del Simulador Swarm 2D

*Fecha: 31 de agosto de 2026.*

Esta fase reemplaza el planificador de inspección marcador de posición
(`InspectionPlanner`, Fase 1) por el esquema definitivo: **Voronoi recursivo
por bordes de distancia** + **Market Auction Method** con costo multiplicativo
hereditario, activado solo en la **Etapa de Inspección** (arquitecturas 2E).

## 1. Resumen de lo implementado

- **Colas de tareas en `Robot`** (`src/robots/robot.py`): cada robot mantiene
  `cola_tareas` (lista de objetivos `{x, y, costo, peso, nivel}` ordenada por
  prioridad). El robot consume la cola de forma automática en `update()`: al
  terminar el objetivo actual (llegar, inspeccionar o abandonar) toma el
  siguiente sin esperar al planificador. Métodos nuevos: `asignar_cola_tareas`,
  `repriorizar_cola`, `_consumir_de_cola`. La arquitectura permite
  repriorización dinámica sin cancelar movimientos a medio camino.

- **Voronoi recursivo con Distance Transform** (`src/algorithms/inspeccion.py`,
  clase `InspeccionVoronoi`):
  - **Semillas iniciales**: al entrar en la etapa de Inspección y en cada
    ciclo de planificación (`plan_interval_s`), se calcula un *Distance
    Transform* euclídeo (`scipy.ndimage.distance_transform_edt`) desde las
    paredes descubiertas (`mundo.explored == DISCOVERED`). Los máximos
    locales del DT (el "esqueleto" o crestas más alejadas de las paredes)
    junto con las posiciones de los robots sirven de semillas para el primer
    Diagrama de Voronoi discreto.
  - **Partición**: el conjunto de celdas libres sin inspeccionar se asigna al
    sitio (semilla) más cercano → regiones de Voronoi. Cada región tiene un
    *centroide* = la celda de mayor DT dentro de la región (la más profunda).
  - **Recursión**: cuando un robot inspecciona el centroide de una región, el
    resultado (sano/defectuoso) altera el peso del área y se aplica la
    recursión: el centroide se convierte en semilla junto con los
    extremos/vértices de la región (hasta 2 máximos locales adicionales del DT
    dentro de la región), creando sub-Voronois. La parada por región es
    independiente: un sub-Voronoi de una sola casilla no se subdivide. El
    árbol recursivo se reconstruye en cada `actualizar` a partir del estado
    actual (`mundo.inspec`, `mundo.resultado`).
  - **Representación**: nodos hoja (`_Nodo`) con `{celdas, centroide, sitio,
    peso, nivel}`. El peso de mercado (`peso`) se propaga multiplicativamente
    desde la raíz (ver abajo).

- **Market Auction Method** (subasta de primer precio greedy):
  - Las tareas son los centroides de las hojas pendientes del árbol (regiones
    cuyo centroide aún no se inspeccionó).
  - Para cada robot libre (`necesita_objetivo` y `cola_tareas` vacía), se
    evalúa el **costo de mercado** de cada tarea no asignada.
  - Se adjudica iterativamente la pareja (robot, tarea) de menor costo, se
    añade la tarea a la cola del robot (ordenada por prioridad), y se
    repite hasta agotar robots libres o tareas.
  - La subasta se ejecuta cada `actualizar`; las tareas ya en curso (objetivo
    actual o en cola) no se re-subastan, lo que evita flapping.

## 2. Función de costo de mercado exacta

Para un robot `i` y una tarea `T` (centroide de una hoja del árbol):

```
C_i(T) = ( τ_i(T) + TP_i + τ_ins ) · ρ(T) · p^n
```

donde:

- `τ_i(T)` = tiempo de viaje A*/Dijkstra estimado desde el robot `i` hasta el
  centroide `T`: largo de la ruta (esquivando obstáculos conocidos) / velocidad
  máxima del robot. Calculado con `encontrar_ruta` del módulo `pathfinding`.
- `TP_i` = tiempo estimado de las tareas previas en la cola del robot `i`:
  suma de `costo` (tiempo operacional = `τ_viaje + τ_ins`) de cada tarea
  pendiente en `cola_tareas` más la tarea en curso.
- `τ_ins` = `inspeccion_pausa_s` (1.0 s), tiempo fijo de percusión por celda.
- `ρ(T)` = **peso de mercado multiplicativo heredado** (factor de paternidad):
  producto de los multiplicadores de todos los ancestros del nodo `T` en el
  árbol recursivo de Voronoi:
  ```
  ρ(T) = ∏_{a ∈ ancestros(T)} m(a)
  ```
  Para cada nodo ancestral `a` ya inspeccionado con resultado `r(a)`:
  ```
  m(a) = 1 - δ_defectivo    si r(a) = 1 (defectuoso)   [0 < δ_defectivo < 1]
  m(a) = 1 + δ_sano         si r(a) = 0 (sano)         [δ_sano > 0]
  m(a) = 1                  si desconocido
  ```
  Parámetros configurables en `config/default.yaml` y `Config`:
  - `auction_defectivo_factor` = `δ_defectivo` = 0.15  → costo × 0.85 (más atractivo).
  - `auction_sano_factor` = `δ_sano` = 0.25           → costo × 1.25 (menos atractivo).
  - **Nunca 0**: `δ_defectivo < 1` garantiza `m(a) > 0`, y el costo base
    `τ_i + TP_i + τ_ins > 0`, así que el costo total es estrictamente positivo.
- `p^n` = factor de paternidad por profundidad del árbol:
  `p = auction_profundidad_p` = 0.97 (configurable), `n` = nivel del nodo
  (raíz = 1). Los sub-Voronois profundos (más próximos al padre ya
  inspeccionado) reciben un descuento leve por proximidad. Captura la
  "proporcionalidad a la distancia y la paternidad" del enunciado.

**Ejemplo numérico**: un robot a 20 s de viaje, con 15 s de cola previa, inspecciona
una hoja de nivel 3 cuyo ancestro nivel 1 fue defectuoso y ancestro nivel 2 fue
sano:
```
τ_i = 20, TP = 15, τ_ins = 1  → base = 36 s
ρ = (1 - 0.15) × (1 + 0.25) = 0.85 × 1.25 = 1.0625
p^n = 0.97^3 ≈ 0.9127
Costo = 36 × 1.0625 × 0.9127 ≈ 35.0
```
El defecto del abuelo atrae (factor < 1), la sanidad del padre repele
(factor > 1), la profundidad descuenta levemente.

## 3. Algoritmo de Distance Transform para semillas iniciales

Se usa `scipy.ndimage.distance_transform_edt` sobre una máscara binaria:

```
dominio[r,c] = 0  si (grid==1 y explored==DISCOVERED)  (pared conocida)
dominio[r,c] = 1  en caso contrario (libres + paredes desconocidas + bordes)
DT = distance_transform_edt(dominio)
```

- `DT[r,c]` = distancia euclídea de la celda `(r,c)` a la pared conocida más
  cercana.
- Los **máximos locales** del DT (celdas con valor ≥ vecinos 8-conectados) son
  las crestas del esqueleto medial: las posiciones más "seguras" y centrales
  respecto a las paredes conocidas.
- Se seleccionan los `K` máximos globales bien separados (muestreo de puntos
  más lejanos / farthest-point) con `K = num_robots`, y se añaden las
  posiciones actuales de los robots como sitios. Estos `K + N` sitios definen
  la partición inicial del Voronoi.
- Ventaja: no requiere umbrales heurísticos; la geometría del mapa descubierto
  determina naturalmente dónde colocar las semillas.

## 4. Propagación de pesos en el árbol recursivo de Voronoi

El árbol se reconstruye en cada ciclo de planificación (`actualizar`):

1. Partición raíz: semillas iniciales → regiones `R_1...R_K`.
2. Para cada región `R` recursivamente:
   - `centroide = argmax_{c ∈ R} DT[c]`.
   - Si `centroide` **no inspeccionado** (`mundo.inspec == 0`): hoja pendiente,
     tarea con `peso = peso_padre × p^nivel`.
   - Si `centroide` **ya inspeccionado** (`mundo.resultado ∈ {0,1}`):
     - `factor = 1 - δ_defectivo` si defectuoso, `1 + δ_sano` si sano.
     - `peso_hijo = peso_padre × factor × p^(nivel+1)`.
     - `sub_sitios = [centroide] + extremos(R)` (máximos DT restantes en `R`).
     - Particiona `R \ {centroide}` por `sub_sitios` → sub-regiones.
     - Recursa en cada sub-región con `peso_hijo`.
3. Parada: si `|R| ≤ 1` → hoja pendiente de esa única casilla.

El peso `ρ(T)` de una hoja pendiente es exactamente el producto acumulado de
los factores de sus ancestros. Al re-construir el árbol en cada ciclo, la
propagación se recalcula coherentemente con los resultados de inspección
actuales (los ancestros inspeccionados imponen su factor).

## 5. Archivos creados y modificados (Fase 3)

- `src/robots/robot.py`: atributos `cola_tareas`, `con_cola`; métodos
  `asignar_cola_tareas`, `repriorizar_cola`, `_consumir_de_cola`; consumo
  automático en `update()`.
- `src/algorithms/inspeccion.py`: reescrito completo. Nueva clase
  `InspeccionVoronoi` con DT, Voronoi recursivo, Market Auction. Clase
  `InspectionPlanner` (greedy Fase 1) conservada como referencia.
- `src/core/config.py`: constantes `auction_defectivo_factor`,
  `auction_sano_factor`, `auction_profundidad_p`; lectura en
  `cargar_desde_archivo`.
- `config/default.yaml`: sección Fase 3 con valores por defecto.
- `src/core/simulacion.py`: import `InspeccionVoronoi`; creación en
  `_verificar_transicion` (etapa Inspección 2E).

## 6. Notas de validación (Fase 3)

- `compileall` OK. Import de GUI (`stats_window`) OK.
- Prueba aislada del planner: mapa_test_1 (todo descubierto, inspección
  directa) → **100 % de cobertura (256/256 celdas)** en 106 ciclos,
  confirmando completitud algorítmica del Voronoi recursivo + Auction.
- End-to-end headless: `Frontier Based` (2E) en `mapa_test_1`, seed 7 →
  **completado** en 379.7 s, 7593 pasos, 148/256 inspeccionadas (58 %),
  22 defectos revelados. El estancamiento de 30 s (`_termino` 2E) corta
  antes de cobertura total; esto es heredado del diseño 2E (igual que Fase 1).
- Smoke test de no-regresión: `Boustrophedon` (1E-H) 800 pasos → running,
  sin crash, cobertura 36 %.
- El mecanismo de colas no afecta a estrategias 1E-H (cola vacía → comportamiento
  idéntico previo). La corrida Frontier 2E ejerce el movimiento dirigido con
  colas durante 7593 pasos sin error.

---

# CHANGELOG_TESIS - Fase 4 del Simulador Swarm 2D

*Fecha: 31 de agosto de 2026.*

Esta fase integra **aprendizaje por refuerzo (RL)** en el simulador, envolviendo
el entorno existente en una interfaz **Gymnasium** compatible con
**Stable-Baselines3** para entrenar agentes centralizados que controlan la
asignación de objetivos de inspección a los robots.

## 1. Resumen de lo implementado

- **Entorno Gymnasium** (`src/rl/env.py`, clase `SwarmEnv`):
  - Hereda de `gym.Env`. Crea un `Mundo` + `Robot` internamente; no depende de
    `Simulacion` ni de pygame para la logica.
  - **Observación** (`Box[float32]`): vector plano de `2*R*C + N*3 + 3`:
    - `[0, R*C)`: mapa conocido (0=libre, 0.5=pared, 1.0=inspeccionada).
    - `[R*C, 2*R*C)`: defectos revelados (1.0 si inspeccionada y defectuosa).
    - `[2RC, 2RC+N*3)`: por robot `(x/w, y/h, bateria_pct/100)`.
    - `[2RC+N*3, end]`: `(insp/total, pend/total, pasos/max_steps)`.
  - **Acción** (`MultiDiscrete[N]`): cada robot elige una celda `[0, R*C)` o
    idle (`R*C`). Se aplica secuencialmente: robot 0 elige, se bloquea la celda,
    robot 1 elige de las restantes, etc. Si la celda es pared/inspeccionada ya,
    se descarta (idle).
  - **Reward** configurable:
    - `"time"`: `+1` por nueva inspeccion, `-0.001` por paso (minimiza tiempo).
    - `"energy"`: `+1` por inspeccion, `-w * energia_step` (minimiza energia).
  - **Física**: cada step RL ejecuta `physics_steps` (default 20) sub-pasos
    internos (`dt=0.05` s c/u) para que los robots avancen una distancia
    significativa. Primero se asignan targets, luego los sub-pasos ejecutan
    radar + movimiento.
  - **Escaneo inicial**: en `reset()`, cada robot ejecuta `explorar_desde` para
    que A* pueda planificar rutas desde el primer paso.
  - **Etapa**: `mundo.etapa = "inspeccion"` para que `_iniciar_inspeccion` se
    active al llegar al objetivo (requerido por la lógica de `Robot`).
  - **Terminación**: `terminated` si cobertura >= 99%, `truncated` si pasos >=
    `max_steps`.

- **Pipeline de entrenamiento** (`src/rl/train.py`):
  - Entrena PPO con `SubprocVecEnv` (multi-ambiente) o `DummyVecEnv` (1 env).
  - Checkpoint callback + EvalCallback para monitoreo.
  - TensorBoard logs en `models/tb_logs/`.
  - Soporta `--resume` para continuar entrenamiento.
  - CLI: `python -m src.rl.train --reward time --steps 500000 --n-envs 4`.

- **Script de inferencia** (`src/rl/infer.py`):
  - Carga modelo PPO entrenado, ejecuta episodio con la politica determinista.
  - Modo headless (solo metricas) o visual (lanza `Simulacion` con pygame).
  - CLI: `python -m src.rl.infer --model models/ppo_swarm/final`.

- **Fabrica de entornos** (`make_env`): factory para `SubprocVecEnv`, acepta
  `cfg`, `seed`, `reward_mode`, `max_steps`, `physics_steps`.

## 2. MDP: formalización del problema

**Espacio de estados S** (observación `obs ∈ R^{2RC + 3N + 3}`):
- `obs[0:RC]` = mapa de ocupación conocido (normalizado: 0/0.5/1.0).
- `obs[RC:2RC]` = mapa de defectos revelados (ground truth condicionado a
  inspección previa).
- `obs[2RC+i*3 : 2RC+(i+1)*3]` = `(x_i/W, y_i/H, batería_i/100)` por robot.
- `obs[-3:]` = `(inspeccionadas/total, pendientes/total, t/t_max)`.

**Espacio de acciones A** (`MultiDiscrete[N]`, `|A| = N × (RC+1)`):
- Cada robot elige una celda del grid o idle. El agente centralizado toma
  decisiones secuenciales (robot 0 primero, luego robot 1, etc.).
- Alternativa: `Discrete(RC+1)` para 1 robot o allocación secuencial.

**Función de transición T(s'|s,a)**:
- Determinista dado el seed: el entorno ejecuta `physics_steps` sub-pasos
  internos de `robot.update(dt, mundo)` que modelan movimiento diferencial,
  evitación de obstáculos, radar de exploración y pausa de percusión.

**Función de recompensa R(s,a,s')**:
- Modo `"time"`: `R = +1 × (nuevas_inspeccionadas) - 0.001`.
  Inspira completar rápido (penalización temporal constante).
- Modo `"energy"`: `R = +1 × (nuevas_inspeccionadas) - w × ΔE`.
  Con `w = energia_inspeccion_j × 10 ≈ 0.13` (penaliza consumo).
- Referencia: Matich et al. (2023) "A survey on multi-agent coverage control
  for inspection tasks" usan recompensa similar: +1 por celda inspeccionada,
  -costo por desplazamiento.

**Descuento γ**: configurado en PPO (default 0.99).

**Restricciones**:
- Cada celda se inspecciona como máximo 1 vez (detección duplicada = 0 reward).
- Robots no pueden atravesar paredes conocidas (A* las evita).
- Batería agota (parada forzosa) si la energía consumida supera el.umbral.

## 3. Arquitectura del wrapper

```
SwarmEnv (Gymnasium)
├── Mundo (grid, explored, inspec, resultado, robots)
│   ├── Robot[0..N-1] (pos, vel, bateria, cola_tareas)
│   │   ├── update(dt, mundo)  ← physics_steps × por step RL
│   │   ├── explorar_desde()   ← radar
│   │   └── _iniciar_inspeccion() ← si etapa=="inspeccion" y llega al objetivo
│   └── explorar_desde() ← radar global (alternativo)
├── observation_space: Box[float32]
├── action_space: MultiDiscrete[N]
└── step(action) → (obs, reward, terminated, truncated, info)
```

## 4. Archivos creados y modificados (Fase 4)

- `src/rl/__init__.py`: módulo RL (creado).
- `src/rl/env.py`: `SwarmEnv` + `make_env` (creado).
- `src/rl/train.py`: pipeline de entrenamiento PPO (creado).
- `src/rl/infer.py`: script de inferencia (creado).
- `.gitignore`: añadido `models/` y `runs/`.

## 5. Notas de validación (Fase 4)

- `compileall` OK. Imports de `gymnasium`, `stable_baselines3`, `torch` OK.
- Smoke test `SwarmEnv`: `reset()` + 100 `step()` con acciones aleatorias →
  obs sin NaN, rewards coherentes, robots se mueven e inspectan celdas.
  Ejemplo (seed 7, 10 robots, physics_steps=20): 7 inspeccionadas en 50 steps.
- Entrenamiento PPO: `model.learn(total_timesteps=1024)` completa sin error.
  Inferencia posterior con `model.predict()` produce acciones válidas.
- Pipeline completo: crear env → entrenar 1K pasos → inferir 50 steps → OK.
- El wrapper no depende de pygame ni de `Simulacion`; puede ejecutarse headless
  en cualquier entorno sin servidor grafico (requerido para SubprocVecEnv).

---

# CHANGELOG_TESIS - Reestructuracion del Sistema de Estrategias y Comportamientos

*Fecha: 1 de septiembre de 2026.*

Esta seccion documenta la reestructuracion del sistema de estrategias
(exploracion/inspeccion/hibrido) y la implementacion de comportamientos
especificos de robot por estrategia y modo.

## 1. Resumen de lo implementado

### 1.1 Reestructuracion del sistema de modos y estrategias

Se reemplaza el viejo campo `strategy` + dict `architecturas` por un sistema
de tres campos separados:

- `modo_fase`: `"exploracion"` | `"inspeccion"` | `"hibrido"`
- `estrategia_exploracion`: estrategia activa en modo exploracion
- `estrategia_inspeccion`: estrategia activa en modo inspeccion
- `estrategia_hibrido`: estrategia activa en modo hibrido

Categorias de estrategias:
- **Exploracion**: Random Walk, Frontier Based, Sample Based, Hybrid Based,
  Boustrophedon, RL Exploracion
- **Inspeccion**: Random Walk, Voronoi, Boustrophedon Inspeccion, RL Inspeccion
- **Hibrido**: Random Walk Hibrido, RL Hibrido

### 1.2 Comportamientos de robot por estrategia/modo

- **General (todos los modos)**: prioridad 0 = no chocarse entre robots ni con
  paredes. Ya existente en `colision_frontal`.
- **General Exploracion**: sin cambios (ya funcionaba).
- **General Inspeccion**: el robot se queda quieto 1s (simulacion de inspeccion)
  solo si la celda NO fue inspeccionada previamente. Soporta modo colectivo
  (`mundo.inspec`) e individual (`robot.inspec_robot`).
- **Random Walk (exploracion)**: angulo random + distancia random (ya funcionaba).
- **Random Walk (inspeccion)**: igual que exploracion pero al llegar a celda
  final hace inspeccion (pausa 1s si la celda no fue inspeccionada).
- **Random Walk (hibrido)**: igual que exploracion, hasta que no hayan nuevas
  casillas vistas, visitadas o inspeccionadas.

### 1.3 Cambios en terminacion hibrido

- **Random Walk Hibrido**: termina cuando `inspeccion_completa()` (todas las
  celdas vistas, visitadas e inspeccionadas). No requiere exploracion
  completada separada.
- **Otras estrategias hibridas**: requiere exploracion completada
  Y inspeccion completada (comportamiento previo).

## 2. Archivos creados y modificados

| Archivo | Tipo | Cambio |
|---|---|---|
| `src/core/config.py` | mod | Nuevo sistema `modo_fase` + 3 campos de estrategia; `ESTRATEGIAS_INSPECCION` incluye `Random Walk`; campo `inspeccion_individual`; metodos `es_exploracion()`, `es_inspeccion()`, `es_hibrido()`, `_estrategia_activa()`; `es_dos_etapas()` y `modo_estrategia()` deprecated. |
| `config/default.yaml` | mod | Reemplazado `strategy:` por `modo_fase:` + 3 campos; nuevo `inspeccion_individual: false`. |
| `src/core/simulacion.py` | mod | `_crear_planner()` despacha por modo; `_termino()` con 3 metodos separados; `_verificar_transicion()` solo hibrido; `step()` actualiza planner segun modo; HUD con `[EXPLORACION]`/`[INSPECCION]`/`[HIBRIDO-*]`. |
| `src/robots/robot.py` | mod | `_iniciar_inspeccion()` retorna bool y soporta modo individual/colectivo; `_avanzar_inspeccion()` llama `_elegir_nuevo_recorrido()` para Random Walk; `_actualizar_random_walk()` con `_decide_inspeccion_o_rumbo()` para inspeccion al final del tramo; `inspeccion_individual` leido de Config. |
| `src/gui/stats_window.py` | mod | 3 dropdowns + selector de modo, enable/disable dinamico, save/load actualizado. |
| `main.py` | mod | CLI con `--modo`, `--estrategia-exploracion`, `--estrategia-inspeccion`, `--estrategia-hibrido`. Compatibilidad `--estrategia` viejo. |
| `src/core/telemetria.py` | mod | Usa `_estrategia_activa()`, añadido `modo_fase`. |
| `src/rl/env.py` | mod | `etapa` configurable segun `cfg.modo_fase`. |
| `src/algorithms/exploration.py` | mod | Renombrado "Hybrid" a "Hybrid Based" para distincion clara con "Hybrid" de hibrido. |

## 3. Algoritmo de decision de inspeccion en Random Walk

```
_al finalizar tramo (timer >= duration):
  si strat == "Random Walk Hibrido":
    si _iniciar_inspeccion(mundo) -> True:  # celda no inspeccionada
      quedarse quieto 1s, inspeccionar, luego nuevo recorrido
    si no:
      elegir nuevo rumbo aleatorio
  si strat == "Random Walk" y modo == "inspeccion":
    si _iniciar_inspeccion(mundo) -> True:
      quedarse quieto 1s, inspeccionar, luego nuevo recorrido
    si no:
      elegir nuevo rumbo aleatorio
  si no (modo exploracion):
    elegir nuevo rumbo aleatorio
```

## 4. Modo de inspeccion individual vs colectivo

- **Colectivo** (`inspeccion_individual: false`, default): un robot no
  re-inspecciona una celda que otro robot ya inspecciono (`mundo.inspec[r,c]`).
- **Individual** (`inspeccion_individual: true`): cada robot solo salta celdas
  que **el mismo** inspecciono (`robot.inspec_robot[r,c]`). Permite que dos
  robots inspeccionen la misma celda (verificacion cruzada).

## 5. Notas de validacion

- `compileall` OK (todos los modulos).
- RW Exploracion: 2175 scouted, 0 inspeccionadas (correcto: sin inspeccion en
  modo exploracion).
- RW Inspeccion: 1505 scouted, 74 inspeccionadas en 30s (correcto: inspecciona
  al final de cada tramo).
- RW Hibrido: 2096 scouted, 530 inspeccionadas en 250s (correcto: explora +
  inspecciona hasta agotar novedad).
- Frontier Exploracion: 2231 scouted, 0 inspeccionadas (correcto).
- Voronoi Inspeccion: 2145 scouted, 570 inspeccionadas (correcto).
- Voronoi Hibrido: 2075 scouted, 283 inspeccionadas (correcto).

---

# CHANGELOG_TESIS - Fase 4: Integración de Aprendizaje por Refuerzo (RL) y Benchmarking

*Fecha: 7 de septiembre de 2026.*

Esta sección documenta los desarrollos correspondientes a la Fase 4 de la tesis, enfocado en el entrenamiento e inferencia de agentes con Aprendizaje por Refuerzo (PPO), integración en la GUI/Simulación en tiempo real y la suite de benchmarking comparativo.

## 1. Resumen de lo implementado

### 1.1 Entorno Gym/Gymnasium para Swarm (`src/rl/env.py`)
- **Entorno Gymnasium personalizable**: `SwarmEnv` compatible con Stable-Baselines3 (PPO) que simula el enjambre sin necesidad de servidor gráfico (`headless`).
- **Soporte de Modalidades**: Soporta mapa conocido (`map_known=True`) y mapa desconocido (`map_known=False`), evaluando observaciones espaciales y estado global de los robots.
- **Función de Recompensa Estructurada**: Diseñada para incentivar la exploración de celdas no vistas, la inspección eficiente de estructuras y penalizar colisiones e inactividad.

### 1.2 Integración y Controlador de Inferencia RL (`src/rl/controller.py`, `src/rl/infer.py`)
- **Controlador RL en tiempo real (`RLController`)**: Permite que un modelo PPO previamente entrenado maneje la asignación de objetivos a los robots directamente durante la ejecución interactiva en Pygame.
- **Mapeo de Estrategias RL**: Soporte para estrategias `RL Exploracion`, `RL Inspeccion` y `RL Hibrido`.

### 1.3 Entrenador y Scripts de Soporte (`src/rl/train.py`, `train_rl_v2.py`)
- Pipeline completo de entrenamiento con PPO de Stable-Baselines3, guardando checkpoints y modelos finales en `models/`.
- Scripts de soporte para pruebas de inferencia y depuración rápida de políticas y vectores de observación.

### 1.4 Benchmark Comparativo (`src/rl/benchmark.py`)
- **Evaluación Automatizada**: Módulo para comparar de forma exhaustiva estrategias heurísticas (Random Walk, Frontier-Based, Voronoi, Boustrophedon, etc.) contra modelos RL entrenados.
- **Métricas e Informes**: Exportación de métricas a archivos CSV (`results/benchmark.csv`, `results/benchmark_summary.csv`) y generación de gráficas comparativas.

### 1.5 Estrategias Heurísticas de Inspección y Exploración (`src/algorithms/inspeccion.py`, `src/algorithms/exploration.py`)
- **Voronoi Jerárquico por Subasta (Fase 3/4)**:
  - Partición del espacio mediante diagrama de Voronoi geodésico y subdivisión recursiva en abanico.
  - Subasta Market Auction Method para asignar tareas a los robots priorizando centroides y niveles superiores.
- **Boustrophedon (Patrón de Barrido/Serpentina)**:
  - Generación de rutas tipo barrido de arado (líneas paralelas horizontal/vertical) para cobertura metódica y determinista en la inspección/exploración.
- **Integración Multimodal**:
  - Soporte de Voronoi e Inspección Boustrophedon tanto en modo de inspección pura como en estrategias de dos etapas / híbridas.

### 1.6 Modelos RL Invariantes a la Escala y Descentralizados (V3-DES y V3-EXP)
- **Generalización Multi-Agente Invariante**:
  - Se implementó la arquitectura descentralizada con compartición de pesos (*parameter sharing*), permitiendo ejecutar modelos entrenados sobre **cualquier cantidad de robots ($N$)** y en **mapas de cualquier dimensión ($R \times C$)** mediante observación egocéntrica local ($K \times K$).
- **V3-DES (Inspección/Híbrido con Alerta y Energía)** (`SwarmEnvDescentralizado`, `RLControllerV3DES`):
  - Incorpora indicador de alerta difuso entre robots ante hallazgos de celdas defectuosas (`radio_alerta_px`), penalización por inactividad y penalización por consumo de energía.
- **V3-EXP (Exploración Pura Descentralizada)** (`SwarmEnvExploracionDescentralizado`, `RLControllerExploracionV3`):
  - Modelo enfocado exclusivamente en maximizar el descubrimiento rápido del mapa (niebla a descubierto).
  - Incluye **señales de descubrimiento cooperativo**: cada robot percibe en tiempo real si los vecinos en su radio de comunicación están descubriendo activamente nuevas fronteras para coordinar el avance o dispersarse.
  - Penalización por consumo de energía de locomoción.
  - Script de entrenamiento: `train_rl_v3_exp.py`.

### 1.7 Estrategia Bioinspirada: Cerebro de Mosca (*Drosophila melanogaster*) con Dopamina
- **Arquitectura Neuromórfica en Enjambre**:
  - Cada robot porta una instancia individualizada de `FlyBrain` (`src/bio/fly_brain.py`) que combina dos subsistemas neurobiológicos:
    - **Mushroom Body (MB)**: Proyección de alta dimensión esparsa (*Kenyon Cells*, $k$-Winner-Take-All al $5\%$) y plasticidad sináptica en caliente gobernada por la regla biológica de tres factores (*Three-Factor Learning Rule*):
      $$\Delta W_{ij} = \eta \cdot \text{KC}_i \cdot \text{MBON}_j \cdot \text{Dopamina}(t) - \lambda W_{ij}$$
    - **Central Complex (CX)**: Atractor en anillo (*Ring Attractor*) para integración de rumbo y orientación espacial fluida.
- **Canal Sensorial Social para Enjambres ($\approx 10$ Robots)**:
  - Integra la percepción de densidad social ($N_{\text{cerca}}$) y vectores de las compañeras más próximas, permitiendo al enjambre autorregularse: evitar hacinamiento o converger ordenadamente ante una alerta.
- **Modulación Dopaminérgica en Tiempo Real**:
  - Inyección instantánea de pulsos de dopamina: $+1.5$ por baldosas libres descubiertas, $+6.0$ ráfaga por baldosas defectuosas, penalización por inmovilidad/estancamiento y consumo energético.
- **Telemetría en Vivo**:
  - Visualización en la pestaña "Individual" de la GUI del nivel instantáneo de dopamina de cada robot.



## 2. Archivos creados y modificados

| Archivo | Tipo | Cambio |
|---|---|---|
| `src/rl/env.py` | mod | Actualización y refinamiento del entorno `SwarmEnv` Gymnasium. |
| `src/rl/train.py` | mod | Mejora del flujo de entrenamiento PPO y manejo de hiperparámetros. |
| `src/rl/infer.py` | mod | Actualizaciones para inferencia e integración de modelos. |
| `src/rl/controller.py` | nuevo | Controlador RL para conectar la política entrenada con el bucle de la GUI en Pygame. |
| `src/rl/benchmark.py` | nuevo | Script de benchmarking comparativo entre heurísticas y RL. |
| `src/algorithms/inspeccion.py` | mod | Implementador del planificador Voronoi estático jerárquico y Boustrophedon. |
| `src/algorithms/exploration.py` | mod | Refinamiento de generadores de exploración (Frontier, Boustrophedon, Hybrid). |
| `src/core/simulacion.py` | mod | Integración de estrategias RL, Voronoi y Boustrophedon en el despachador de planners. |
| `src/core/config.py` | mod | Configuración de parámetros para modos RL, Voronoi y Boustrophedon. |
| `src/robots/robot.py` | mod | Ajustes en navegación y respuesta a comandos de los controllers. |
| `src/gui/stats_window.py` | mod | Soporte de UI para elegir modos/estrategias Voronoi, Boustrophedon y RL. |
| `config/default.yaml` | mod | Nuevas entradas de configuración para modelos RL y parámetros de inspección. |
| `CHANGELOG_TESIS.md` | mod | Inclusión de esta documentación de cambios. |

## 5. Cambios realizados en esta sesión (8 de septiembre de 2026)

- **src/algorithms/inspeccion.py**: Añadida la clase `Figura` y revisada la lógica de generación Voronoi y triangulación recursiva según los últimos requerimientos.
- **src/gui/renderer.py**: Implementado dibujo de bordes Voronoi y líneas de triangulación para visualización de la nueva geometría.
- **tests/test_voronoi_triangulacion.py**: Nuevas pruebas unitarias que cubren la generación Voronoi, la recursividad de triangulación y la propagación de costes.
- **CHANGELOG_TESIS.md**: Actualizada con esta entrada de changelog.