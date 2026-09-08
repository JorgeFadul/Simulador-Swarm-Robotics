# Documentación del Simulador Swarm 2D

*Documento vivo del proyecto. Última actualización: 31 de agosto de 2026.*

---

## 1. Propósito de este documento

Este archivo resume, con el mayor detalle posible, todo lo referente al proyecto **Simulador Swarm 2D**:
qué contiene, cómo funciona por dentro y por qué se tomó cada decisión de diseño, qué se ha
hecho hasta hoy (con su línea de tiempo), en qué etapa estamos y cuáles son los siguientes pasos
hacia el objetivo final. Sirve tanto para uso interno como para presentar el avance a asesores o
al comité de tesis.

---

## 2. Contexto académico del proyecto

El simulador es el **entorno de simulación computacional** de la tesis de maestría:

> **"Estrategia basada en Enjambre de Robots Inteligentes para Inspección No Destructiva de Baldosas"**
> Jorge Luis Fadul Flores — Maestría en Ciencias de la Ingeniería Mecánica (especialidad
> Automatización y Robótica) — Universidad Tecnológica de Panamá — Asesor: Dr. Humberto Rodriguez.

La tesis propone un enjambre de robots móviles que inspecciona pisos de baldosa por
**percusión acústica** (tap-testing) y clasifica el estado de adherencia de cada baldosa mediante
análisis espectral (FFT). El estudio es **teórico-computacional**: no hay experimentación física
del enjambre; la validación de la coordinación se hace en simulación.

El Capítulo 6 de la tesis (Simulación Computacional y Resultados del Enjambre) exige demostrar el
comportamiento colectivo del enjambre en escenarios controlados con métricas de cobertura,
precisión diagnóstica, eficiencia temporal y robustez. **Este simulador es la herramienta que
produce esos resultados**, y reemplaza la dependencia de Webots por una implementación propia
ligera (Pygame + Tkinter) con la misma lógica de exploración cooperativa por fronteras.

El simulador resuelve la capa de **exploración y cobertura espacial** del enjambre: sensado con
línea de vista, construcción de mapa incremental tipo SLAM por celdas, planificación cooperativa
de objetivos y planificación de movimiento. La capa de **detección acústica** (percusión, FFT,
clasificación sana/desprendida) se está vinculando a partir del trabajo acústico ya desarrollado
en otra rama del proyecto (scripts de análisis FFT y el paper de inspección acústica de baldosas).

---

## 3. Qué hay en el proyecto (estructura y archivos)

```
simulador-swarm-2d/
├── main.py                 # Punto de entrada: arranca Pygame + Tkinter y el bucle principal
├── requirements.txt        # Dependencias (ver nota de faltantes en la sección 8)
├── sim_config.txt          # Parámetros guardados por el usuario (se carga al iniciar)
├── README.md               # Documentación de usuario del proyecto
├── DOCUMENTACION_SIMULADOR_SWARM.md   # Este documento
├── config/
│   └── default.yaml        # Valores por defecto de arranque (mapa, grilla, robots, paredes)
├── assets/                 # (recursos no usados actualmente; carpeta reservada)
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py       # Carga la configuración (YAML + sim_config.txt) y conversiones cm/px
│   │   ├── mundo.py        # Mundo, grilla, paredes, línea de vista, sensores, colisiones
│   │   └── simulacion.py   # Orquesta la edición y el bucle de simulación, maneja eventos
│   ├── robots/
│   │   └── robot.py        # Robot: cinemática, dos modos de movimiento, mapa propio, intención
│   ├── algorithms/
│   │   ├── exploration.py  # Generadores de objetivos (frontier/sample/hybrid) + asignación de tareas
│   │   └── pathfinding.py  # A* ponderado sobre la grilla (motion planning)
│   ├── gui/
│   │   ├── renderer.py     # Dibuja mapa, robots, intención, fronteras y mini-mapas
│   │   └── stats_window.py # Ventana Tkinter: estadísticas, controles y configuración en caliente
│   └── utils/
│       └── __init__.py
└── tests/
    └── __init__.py         # Sin framework formal; la validación es por scripts headless
```

### 3.1 El papel de cada módulo (cómo y por qué)

#### `main.py` — arranque y bucle principal

1. Crea la configuración (`Config`) y le aplica los valores guardados en `sim_config.txt`
   (si el archivo existe).
2. Crea la simulación (`Simulacion`), que a su vez inicializa Pygame, la ventana del mapa,
   el `Mundo`, el `Renderer` y el estado de edición.
3. Crea un `Tk` raíz oculto y la ventana de estadísticas/control (`StatsWindow`).
4. Corre a 60 FPS el bucle: procesa eventos de Pygame, avanza la simulación solo si está en
   estado `running`, renderiza, y refresca la ventana Tkinter.

**Por qué así:** se usa un bucle manual (no un juego con clocks internos) para mantener juntas las
dos ventanas (Pygame y Tkinter) dentro de un solo proceso y un solo hilo. El paso de simulación es
de tiempo fijo (`dt=0.05 s`, 20 pasos por segundo), lo que hace los resultados **repetibles** y
comparables entre corridas e independientes del FPS real del monitor.

#### `src/core/config.py` — configuración en dos capas

La configuración se lee en dos capas:

- **Capa base** (`config/default.yaml`): todo lo que define al mundo y los valores estándar de
  comportamiento (tamaño de mapa, `pixels_per_cm`, FPS, número de robots, radios físicos,
  estrategia inicial, parámetros de planificación y las paredes iniciales).
- **Capa de usuario** (`sim_config.txt`): sobrescribe valores con formato `clave=valor` en texto
  plano. Se escribe con el botón "Guardar config" y se carga automáticamente al arrancar.

`Config` convierte automáticamente las unidades físicas (cm) a unidades de pantalla (px)
multiplicando por `pixels_per_cm`, y expone ambos: `robot_radius_cm` y `robot_radius_px`,
`detection_radius_cm` y `detection_radius_px`, `robot_max_speed_cm_s` y `robot_max_speed_px_s`.

**Por qué así:** separar el "mundo físico" (cm/s) del "mundo gráfico" (px/s) permite calibrar la
simulación contra robots reales del proyecto (el chasis HeRo Swarm, por ejemplo, con radios y
velocidades medibles) y conservar parámetros de corrida en corrida sin tocar el código. El archivo
plano `clave=valor` es simple de leer/editar a mano y sirve de registro de los últimos parámetros
usados.

#### `src/core/mundo.py` — el mundo y su percepción

Define las **dos matrices del mundo**:

- `grid`: verdad absoluta del terreno. `0` = celda libre, `1` = celda de pared. Esta matriz
  **no la conoce ningún robot**; es la referencia contra la que se mide el éxito.
- `explored`: mapa descubierto por el enjambre (el conocimiento acumulado). Cada celda puede ser:

| Valor | Nombre       | Significado                                                              |
|------:|--------------|--------------------------------------------------------------------------|
| 0     | Desconocida  | Nunca vista                                                              |
| 1     | `SCOUTED`    | Celda libre avistada por un sensor con línea de vista                    |
| 2     | `VISITED`    | Celda libre por la que un robot pasó físicamente                         |
| 3     | `DISCOVERED` | Celda de pared conocida (vista o chocada)                                |

Clases y funciones clave (cómo y por qué):

- **`Pared`**: rectángulo con rotación. Guarda sus cuatro esquinas (`cache_rect`) rotadas y
  trasladadas. La rotación permite paredes con ángulo distinto de 0 en la configuración YAML.
- **Rasterización de paredes** (`_rasterizar_pared`, `_punto_en_pared`): marca como pared en
  `grid` toda celda cuyo centro cae dentro del rectángulo rotado. Se hace a la creación del mundo.
  Se crean siempre **cuatro paredes perimetrales** (bordes del mapa): así el enjambre no puede
  "escapar" y la exploración tiene fronteras cerradas, algo necesario para que la terminación por
  "sin celdas frontera" tenga sentido.
- **Línea de vista** (`tiene_linea_de_vista`): recorre las celdas entre dos puntos con el
  algoritmo de **Bresenham** y devuelve `False` si alguna es pared. Este es el corazón del
  sensado realista: **ningún sensor atraviesa paredes**, no se "adivina" geometría.
- **Explorar** (`explorar_desde`): recorre las celdas dentro de un radio y marca `SCOUTED` (libre)
  o `DISCOVERED` (pared) solo si hay línea de vista. Devuelve cuántas celdas nuevas se vieron.
- **Detectar obstáculos** (`detectar_en_rango`): calcula un **vector de alejamiento** promedio,
  normalizado, que apunta hacia afuera de los obstáculos visibles (paredes con línea de vista y
  otros robots). Es la base de la evitación reactiva.
- **Colisión frontal** (`colision_frontal`): determina si el frente del robot choca si avanza en
  la dirección dada. Tiene un modo `solo_descubiertas=True` que solo considera paredes `DISCOVERED`
  (el robot no puede "sentir" paredes que no ha visto — consistente con el modelo de conocimiento
  limitado). Con otros robots solo cuenta colisión si el obstáculo viene de frente
  (producto punto del vector al otro robot con el vector de velocidad mayor que 0).
- **Descubrir pared al chocar** (`descubrir_pared_frontal`): cuando el robot choca físicamente, se
  marcan como `DISCOVERED` las paredes que están al frente y con línea de vista. Así el choque
  **aporta información real al mapa** (la pared deja de ser desconocida).
- **Marcar visitado** (`marcar_visitado`): promueve una celda `SCOUTED` a `VISITED` cuando el robot
  pasa físicamente por ella (o directamente de 0 a `VISITED` si era libre).
- **Criterios de fin** (`exploracion_completa`, `porcentaje_descubierto`): la exploración se
  completa cuando la fracción de celdas libres vistas y de paredes vistas supera un umbral, o — por
  decisión de la simulación — cuando no quedan celdas frontera (ver simulacion.py). Importante:
  **no hace falta visitar todas las celdas**, solo descubrir el mapa.

#### `src/core/simulacion.py` — la máquina de estados del programa

Estados: `edicion` -> `running` -> `paused` | `completado` -> (ESC vuelve a `edicion`).

- **Editor**: herramientas Robot / Pared / Borrar, clic izquierdo para colocar, clic derecho para
  borrar, arrastre para dibujar pared continua, evitar colocar robots dentro de paredes o de otros
  robots. (Nota: hay una discrepancia entre teclas documentadas y reales; ver sección 8.)
- **`iniciar()`**: resetea la exploración de todos los robots y del mundo, crea el planificador
  según la estrategia (`ExplorationPlanner` para Frontier/Sample/Hybrid, `None` para Random Walk)
  y pasa a `running`.
- **`poblar_robots(n)`**: coloca `n` robots en posiciones libres aleatorias sin solaparse.
- **`step()`**: por cada tick de 0.05 s: verifica terminación; actualiza el planificador si existe;
  actualiza el mundo (cada robot). Acumula `tiempo_simulado` y `pasos_totales`.
- **Terminación** (`_termino`), en este orden:
  1. si `porcentaje_descubierto >= exploration_goal`, termina;
  2. si el planificador existió y alguna vez hubo fronteras, y la máscara de fronteras actual no
     tiene celdas, termina (no hay más frontera por explorar).
- **`render()`**: delega al `Renderer` y dibuja el HUD (porcentaje descubierto y estrategia, o
  mensaje de COMPLETADO).

#### `src/algorithms/pathfinding.py` — planificación de movimiento (A* ponderado)

`encontrar_ruta(mundo, x1, y1, x2, y2)` devuelve una lista de waypoints en píxeles (centros de
celda) o lista vacía si no hay ruta.

El A* es **ponderado por conocimiento**:

- celda libre **ya explorada**: costo 1 (se prefiere el camino conocido);
- celda libre **desconocida**: costo 6 (se puede cruzar de forma optimista, como es estándar en
  exploración por fronteras, pero pagando más);
- **pared**: intransitable.

Heurística Manhattan (movimiento de 4-vecindad), cola con `heapq`.

**Por qué así:** en un mundo desconocido, a veces la única vía a una frontera pasa por terreno no
visto. Permitir cruzarlo "de forma optimista" con costo alto evita que los robots se queden sin
ruta por miedo a lo desconocido, pero mantiene la preferencia por caminos seguros ya explorados.

#### `src/algorithms/exploration.py` — generación de objetivos y asignación de tareas

Implementa las tres familias de *target generation* descritas en el estado del arte
(survey de Wang et al., *Multi-Robot System for Cooperative Exploration in Unknown Environments*):

- **`_celdas_frontera(mundo)`**: máscara de celdas frontera = celda libre y conocida adyacente a
  una celda desconocida en **8-direcciones** (incluye diagonales, usando `np.roll` sobre la máscara
  de desconocidas). Es la definición clásica de Yamauchi extendida: se detectan más fronteras y
  regiones con esquinas tocadas se cuentan como una sola.
- **`_agrupar_regiones(mask, min_size, max_size=None)`**: agrupa celdas frontera conexas por
  **8-vecindad** mediante búsqueda con pila; se descartan regiones menores a `frontier_min_size`
  (filtra ruido) y las regiones mayores a `frontier_max_size` se **dividen en chunks de a lo más
  `frontier_max_size` celdas** (flood-fill BFS), cada uno una entidad independiente.
- **`FrontierBased`** (Yamauchi): un objetivo por entidad = la **celda de la frontera más cercana
  al centroide medio** de la entidad (siempre navegable; una frontera de 1 celda se comporta como
  celda individual). El puntaje es el tamaño de la entidad. Rápido, pero genera objetivos que a
  veces traen poca información nueva.
- **`SampleBased`** (next-best-view, González-Baños): muestrea `sample_candidates` puntos de vista
  cerca de la frontera y los puntúa como `ganancia - sample_cost_weight * costo`, donde:
  - `ganancia = _ganancia()`: número de celdas desconocidas con línea de vista que quedarían
    visibles si el robot se parara allí (cuenta real de información esperada);
  - `costo = _costo_mas_cercano()`: distancia al robot más cercano.
  Devuelve los mejores hasta `sample_targets`.
- **`Hybrid`** (Charrow et al.): agrupa en regiones y, por cada región, muestrea
  `sample_per_cluster` puntos de vista, conservando el mejor de cada región. Combina la eficiencia
  del agrupamiento con la calidad del NBV.
- **`ExplorationPlanner`**: orquesta *target generation* + *task allocation* de forma
  **centralizada, usando el mapa combinado** (`mundo.explored`). Ningún robot conoce el mapa real;
  solo lo que el enjambre ha descubierto. Mantiene:
  - `_objetivos` / `_regiones`: lista de candidatos del último ciclo;
  - `_reservas`: diccionario `indice_robot -> celda_centroide` indicando qué región está reservada
    para qué robot;
  - `_no_go`: diccionario `celda -> tiempo_hasta` de objetivos temporalmente bloqueados
    (inalcanzables o ya alcanzados). Expiran tras `goal_timeout_s`;
  - `_alguna_vez_hubo_fronteras`: bandera usada para el criterio de terminación.
  
  Cada `plan_interval_s`: regenera las regiones de frontera y asigna tareas:
  
  - **Frontier/Hybrid — asignación dinámica por costo con reserva exclusiva
    (`_asignar_dinamico`)**: las celdas de frontera se agrupan en entidades
    (regiones 8-conectadas, divividas si superan `frontier_max_size`, por
    defecto `grid_size // 2`) y cada entidad tiene un único objetivo en la
    celda más cercana a su centroide. Cada ciclo se recalculan los costos de
    TODAS las entidades contra TODOS los robots con la posición actual:
    `costo = distancia_al_centroide / (celdas_de_la_frontera + 1)`
    (más celdas -> menor costo; más distancia -> mayor costo). La lista de
    prioridad ordena los pares por costo creciente y cada entidad se reserva
    **en exclusiva para un solo robot**.
  - **Reasignación en tránsito**: un robot ocupado conserva su frontera salvo
    que aparezca otra disponible con costo menor a
    `reasignar_umbral_costo` veces el actual (default 0.7, es decir ~30% más
    barata), lo que hace que atienda primero las fronteras cercanas que van
    descubriendo en el camino. Una frontera de 1 celda se comporta como celda
    individual. Si un robot queda libre y no hay fronteras libres, cae a la de
    menor costo aunque esté reservada (no queda parado).
  - **Sample Based — asignación por puntaje**: conserva su asignación por puntaje de ganancia,
    eligiendo el robot libre más cercano con preferencia de línea de vista.
  - Los objetivos inalcanzables (sin ruta A*) se marcan como `no_go` temporalmente.
  
  **Por qué reservar regiones:** evitar que varios robots vayan a la misma frontera (solapamiento
  de trayectorias). La reserva exclusiva es lo que produce la **distribución espacial** del
  enjambre; sin ella la estrategia degenera en todo el grupo yendo al mismo lugar. La priorización
  por `distancia / (celdas + 1)` balancea dos fuerzas contrarias: ir a la frontera **más cercana**
  (minimizar recorrido) y a la **más grande** (maximizar información nueva por viaje).

#### `src/robots/robot.py` — el robot

Cada robot tiene su **mapa propio** `explored_robot` (lo que él, individualmente, ha visto) y sus
contadores (`scouteadas_robot`, `visitadas_robot`, `paredes_robot`). Esto permite la pestaña
"Individual" de la ventana de estadísticas: ver qué sabe cada agente.

Hay **dos modos de movimiento**, y es la distinción más importante del proyecto:

1. **Random Walk** (solo cuando la estrategia es "Random Walk"):
   - Máquina de estados `STOPPED -> ROTATING -> MOVING`.
   - En cada tramo elige un ángulo y una duración aleatorios; rota a velocidad de giro limitada y
     luego avanza a velocidad crucero.
   - El temporizador del tramo **solo avanza en `MOVING`**: el tiempo de giro no reduce la
     duración del movimiento planificado.
   - Ante peligro frontal, frena y gira hacia la dirección libre (o nuevo recorrido).
2. **Movimiento dirigido** (Frontier/Sample/Hybrid): cuando el robot persigue un objetivo del
   planificador:
   - Es **continuo y tipo diferencial**: gira hacia el siguiente waypoint de la ruta **mientras
     avanza**. La velocidad lineal se escala con el alineamiento:
     `speed = max_speed * (dirigido_avance_min + (1 - dirigido_avance_min) * cos(diff))`,
     de modo que si está mal alineado avanza poco (casi pivota) pero no se detiene del todo.
   - No hay tramos con duración fija, no hay abandono por timeout, no hay la secuencia
     "girar-total-luego-avanzar" del random walk.
   - **Evitación reactiva** (`detectar_en_rango`): el rumbo al objetivo se mezcla con el vector de
     repulsión **solo cuando el obstáculo está delante** (medido con el producto punto entre la
     dirección al destino y el vector de repulsión). El peso de la mezcla depende de la urgencia:
       - colisión inminente y sin obstáculo aparente: se desvía 45 grados;
       - colisión inminente: peso `dirigido_peso_evasion_imminente` (si el obstáculo no está detrás);
       - obstáculo claramente al frente (`dot < -0.15`): peso `dirigido_peso_evasion_fuerte`;
       - obstáculo algo al frente (`dot < 0.3`): peso `dirigido_peso_evasion_suave`,
         que al ser 0.8 mantiene casi todo el rumbo al objetivo;
       - si el obstáculo está a un lado o detrás: se ignora y se mantiene el rumbo.
   - Si choca físicamente: descubre la pared frontal, **replanifica la ruta** (A* de nuevo). Sin
     ruta alternativa, marca el objetivo como alcanzado/timeout y espera.
   - **Sin objetivo**: se **detiene y espera** al planificador. No recurre al random walk. Esta
     es una decisión deliberada: en estrategias dirigidas el robot debe ser disciplinado y
     obedecer al planificador.
   - El timeout del objetivo (`asignar_objetivo`) se escala con la longitud de la ruta
     (`max(goal_timeout_s, largo / max_speed * 3)`): da margen proporcional a la distancia.

Propiedades de **intención** (`intent_tiempo_restante`, `intent_distancia`): estiman cuánto y
hacia dónde avanzará el robot según su plan actual — usadas para dibujar la línea de intención.

#### `src/gui/renderer.py` — renderizado

Dibuja en orden (capas): celdas exploradas (verde oscuro = scouteada, verde claro = visitada,
rojo = pared descubierta), máscara de fronteras (turquesa oscuro), grilla, paredes (polígonos
borde), objetivos (círculo y línea magenta por robot), **líneas de intención** (cian punteadas con
punta de flecha: longitud = `intent_distancia`, acotada a 260 px y truncada en la primera pared
`DISCOVERED`), robots (cuerpo, rumbo, vector de velocidad, radio de detección) y, en la esquina
inferior derecha, un panel semitransparente con los **mini-mapas individuales** de cada robot
(`explored_robot`, 50 px cada uno, 4 por fila).

**Por qué las capas de planificación son conmutables:** la pestaña Controles permite ocultar
fronteras, objetivos o intención en caliente. Esto sirve para generar capturas "limpias" para la
tesis (por ejemplo, mostrar solo la exploración sin interferencia de los overlays de planificación)
y para depurar.

#### `src/gui/stats_window.py` — ventana Tkinter

Cinco pestañas:

- **Individual**: desplegable de robots; gráfico de barras con sus tres métricas (pisadas,
  scouteadas, muros) y su mini-mapa de exploración escalado con Pillow.
- **Cobertura**: mapa global de cobertura (mismo esquema de colores).
- **Global**: texto con métricas: estrategia, tiempo simulado, FPS, pasos, celdas totales/libres/
  paredes, visitadas/scouteadas/muros descubiertos, coberturas parciales y totales, objetivos
  pendientes, robots, y el mensaje de COMPLETADO con el tiempo final.
- **Controles**: herramientas (radio botones sincronizados con el editor), tamaño de grilla editable
  (reconstruye el mundo), población (Poblar/Limpiar/Reset), parámetros editables del robot y
  sensores en vivo, casillas de capas de planificación, botones Iniciar/Pausar/Detener, guardar y
  cargar config, atajos de teclado.
- **Estrategias**: combo de estrategia (Random Walk, Frontier Based, Sample Based, Hybrid) y todos
  los parámetros del planificador y del movimiento dirigido. Todo aplicable en caliente y guardable.

Los parámetros se editan por `.trace_add("write", ...)` (se aplican al `Config` en cuanto se escribe
un número válido) y `_param_changed` reconvierte las unidades cm/px cuando corresponde.

**Por qué Tkinter además de Pygame:** Pygame no trae controles; Tkinter da widgets de formulario
gratis sin dependencias extra. Se unifican en un solo hilo para que los controles toquen objetos en
vivo sin candados de concurrencia.

---

## 4. Cómo corre la simulación (qué pasa en cada paso)

1. `Config` carga YAML + superpone `sim_config.txt`.
2. `Mundo` se construye: grilla, paredes perimetrales, paredes del YAML, robots del editor.
3. Usuario prepara el escenario en el editor (colocar robots y paredes, o Poblar).
4. Botón **Iniciar** -> `Simulacion.iniciar()`: resetea exploración global y de cada robot, crea el
   planificador según estrategia, pasa a `running`.
5. Cada tick (0.05 s de tiempo simulado):
   a. `ExplorationPlanner.actualizar()`: limpia `no_go` vencidos, captura objetivos alcanzados o
      con timeout y los marca como `no_go`, regenera objetivos cada `plan_interval_s` y asigna
      tareas a los robots libres.
   b. Cada `Robot.update(dt)`:
      - percibe obstáculos (`detectar_en_rango`) y colisión frontal (solo paredes `DISCOVERED`);
      - si tiene objetivo -> movimiento dirigido; si Random Walk -> su máquina de estados; si no ->
        espera;
      - integra la posición, verifica colisión real (con `mundo.colision_frontal` sin restricción
        de descubiertas, o sea contra todo el terreno y robots); si choca: frena, descubre la
        pared frontal y replanifica;
      - marca la celda actual como `VISITED`;
      - explora (`mundo.explorar_desde`) con radio = `detection_radius px * exploration_radius_multiplier`
        y actualiza también su mapa propio `explored_robot`.
6. Termina cuando se alcanza `exploration_goal` o no quedan fronteras.

---

## 5. Los parámetros configurables (qué hace cada uno y por qué)

| Clave | Unidad | Efecto |
|---|---|---|
| `robot_radius_cm` | cm | Radio físico del robot (radio de colisión) |
| `robot_max_speed_cm_s` | cm/s | Velocidad lineal máxima |
| `robot_turn_speed_rad_s` | rad/s | Velocidad de giro (limita cuán rápido rota) |
| `detection_radius_cm` | cm | Radio de detección de sensores (no confundir con el de exploración) |
| `exploration_radius_multiplier` | adim | Multiplica el radio de detección para el mapa SLAM: se explora más lejos de lo que "detecta" evasión |
| `random_walk_interval_min/max` | s | Duración mínima/máxima de cada tramo recto del random walk |
| `strategy` | — | Estrategia: `Random Walk`, `Frontier Based`, `Sample Based`, `Hybrid` |
| `exploration_goal` | 0-1 | Fracción del mapa a descubrir para declarar COMPLETADO |
| `plan_interval_s` | s | Intervalo entre ciclos de regeneración de objetivos y asignación |
| `goal_reached_radius_px` | px | Distancia bajo la cual un waypoint/objetivo se considera alcanzado |
| `goal_timeout_s` | s | Tiempo que dura el bloqueo `no_go` de una celda objetivo inalcanzable |
| `frontier_min_size` | celdas | Tamaño mínimo de región de frontera para generar un objetivo (filtra ruido) |
| `sample_candidates` | — | Candidatos muestreados por *Sample Based* por ciclo |
| `sample_per_cluster` | — | Puntos de vista muestreados por región en *Hybrid* |
| `sample_cost_weight` | adim | Peso del costo (distancia) en el puntaje `ganancia - peso*costo` |
| `sample_targets` | — | Máximo de objetivos devueltos por los generadores de muestreo |
| `dirigido_avance_min` | adim | Fracción mínima de velocidad lineal mientras el robot gira para alinearse (0.45 = en el peor desalineamiento avanza al 45% de la máxima) |
| `dirigido_peso_evasion_imminente` | adim | Peso del rumbo al objetivo ante colisión inminente |
| `dirigido_peso_evasion_fuerte` | adim | Peso del rumbo al objetivo con obstáculo claramente al frente |
| `dirigido_peso_evasion_suave` | adim | Peso del rumbo al objetivo con obstáculo algo al frente (0.8 = apenas una desviación leve) |
| `dirigido_umbral_giro_deg` | grados | Umbral de alineación con el waypoint para dar el giro por terminado |

En `config/default.yaml` además: dimensiones del mapa y grilla, `fps`, `num_robots`,
`pixels_per_cm`, `exploration_threshold` y la lista de **paredes iniciales** (rectángulos
`x,y,width,height,angle`).

---

## 6. Qué hemos hecho (historial de desarrollo)

La evolución del repositorio (git), en orden cronológico:

1. **2026-07-31 — Base funcional** (`1e8d918`): Simulador Swarm 2D con:
   - sensado SLAM por celda en tres estados y **línea de vista** (no se atraviesan paredes);
   - dos ventanas (Pygame para el mapa + Tkinter para estadísticas);
   - editor de escenarios (robots, paredes, tamaño de grilla);
   - configuración persistente (`sim_config.txt`) y visualización de la **intención de movimiento**.
   - Random Walk como única estrategia de movimiento (máquina de estados con giro limitado y
     colisión frontal).
2. **2026-07-31 — Documentación** (`7377e32`, `f392bbf`): README detallado del proyecto y corrección
   ortográfica a español con tildes y eñe (UTF-8).
3. **2026-08-05 — Exploración cooperativa multirrobot** (`e13e4d2`): el salto grande:
   - generadores de objetivos **Frontier Based, Sample Based (NBV) e Hybrid**;
   - **motion planning con A\*** ponderado sobre el grid;
   - **asignación de tareas** por planificador global (cercanía con reserva exclusiva);
   - **movimiento dirigido continuo** tipo diferencial con evitación reactiva de obstáculos;
   - robots sin objetivo se detienen y esperan al planificador (no hacen random walk);
   - terminación por meta de descubrimiento o por "sin fronteras".
4. **2026-08-05 — Capas de planificación conmutables** (`15c1a8f`): casillas en la pestaña
   Controles para mostrar/ocultar fronteras, objetivos e intención, sin detener la simulación.
5. **2026-08-24 — Última actualización** (`59378df`, el pull más reciente): configuración ampliada
   (meta de exploración, intervalo de planificación, radio de llegada, timeout), refinamiento de
   los algoritmos de exploración (reservas de región, `no_go`, timeout escalado a la ruta),
   parámetros del movimiento dirigido (avance mínimo, pesos de evasión, umbral de giro), ventana de
   estadísticas con la pestaña **Estrategias** (todo editable en caliente y guardable), y README
   actualizado.

Resumen del estado actual del código (agosto de 2026): ~1400 líneas de Python distribuidas en
`core` (config, mundo, simulacion), `robots`, `algorithms` (exploration, pathfinding) y `gui`
(renderer, stats_window).

---

## 7. Qué estamos haciendo ahora

La **capa de exploración y cobertura cooperativa está completa y operativa**. El trabajo actual
consiste en cerrar la brecha entre el simulador de exploración y los módulos de la tesis que aún
faltan por integrar, en concreto:

1. **Modelado de la superficie de inspección**: pasar de "explorar el mapa" a "inspeccionar
   baldosas": cada celda del grid representará una baldosa con un **estado de adherencia**
   (sana / desprendida), con densidad de defectos controlada por escenario.
2. **Modelado del sensor acústico de percusión** como función de probabilidad de detección
   parametrizada, tomando los rangos espectrales de la literatura (baldosas sanas ~3-5 kHz,
   desprendidas ~0.5-2 kHz, umbral de decisión ~2 kHz) y los resultados del trabajo acústico FFT
   ya desarrollado con el ESP32.
3. **Ciclo unitario de inspección**: el robot se detiene, percute, adquiere la ventana de audio,
   clasifica (FFT) y actualiza el estado de la celda, antes de seguir con la siguiente.
4. **Protocolo de verificación redundante** (doble verificación cooperativa): un robot en rol
   explorador marca una celda "sospechosa", y un segundo robot en rol verificador acude, repite la
   percusión de forma independiente y dictamina el estado final, mitigando falsos positivos.
5. **Reglas de comportamiento local + coordinación**: mantener la evitación reactiva local ya
   construida y decidir el esquema global de coordinación (los generadores de objetivos actuales,
   y eventualmente una variante PSO) para la **cobertura de inspección** (no solo exploración).

Esto alinea el simulador con el Capítulo 4 (diseño de estrategias, roles, protocolo de doble
chequeo) y el Capítulo 6 (escenarios y métricas) de la tesis.

---

## 8. Problemas conocidos y notas pendientes

- **Discrepancia de teclas del editor**: el código escucha `W` (pared) y `E` (borrar) en
  `simulacion.py`, pero el HUD y el README anuncian `P` y `B`. Hay que unificar: o se cambia el
  código a `P`/`B` o se corrige la documentación a `W`/`E`. También aparece una tipografía rara en
  una línea del renderer (`font.render("Indiv.", ...)` sin asignar el resultado, y la llamada a
  `pygame.surfarray.make_surface` puede fallar por los ejes si `numpy >= 2.x` por el `transpose`).
- **`requirements.txt` incompleto**: lista solo `pygame` y `pyyaml`, pero el código importa
  `numpy` (obligatorio) y `Pillow` (mini-mapas). Aunque el `.venv` local los tiene instalados, un
  clon limpio fallaría: hay que añadir `numpy` y `Pillow` a `requirements.txt`.
- **Título de la ventana de Pygame**: siempre dice "... - Editor", incluso en simulación; un
  detalle cosmético pendiente.
- **`assets/` está vacía**: carpeta reservada, sin uso actual.

---

## 9. Qué vamos a hacer (hoja de ruta hacia el Capítulo 6 de la tesis)

Orden lógico de trabajo, alineado con los objetivos específicos de la tesis:

### Fase A — Escenarios de evaluación (requisito del Cap. 6)
- Definir los escenarios descritos en la tesis:
  - **A**: área reducida, baja densidad de fallos, enjambre mínimo;
  - **B**: incremento de área y de densidad de defectos;
  - **C**: escalabilidad masiva (enjambre denso);
  - **D**: robustez (inyección de fallos / desconexión de robots durante la corrida).
- El simulador ya permite crear escenarios vía editor, YAML y parámetros; falta la capa de
  **definición de defectos** y la **ejecución por lotes** (batch headless).

### Fase B — Métricas (KPIs) del Cap. 6
Implementar la medición y exportación de:
- % cobertura superficial acumulada (curvas de convergencia en el tiempo);
- tasa de detección de defectos (recall);
- precisión diagnóstica del mapa final (incluido el efecto del doble chequeo);
- eficiencia temporal de inspección (tiempo hasta completar, comparable entre estrategias);
- coeficiente de redundancia (solapamiento de trayectorias / celdas inspeccionadas más de una vez);
- robustez ante pérdida de agentes (escenario D);
- comparativa **Random Walk vs Frontier vs Sample vs Hybrid** sobre las mismas métricas.

Salida esperada: datos en CSV por corrida + gráficas para el documento de tesis.

### Fase C — Análisis y capturas para la tesis
- Usar las capas conmutables (sección 5) para generar figuras limpias.
- Tabulación de resultados por escenario y análisis estadístico.
- Vinculación de los resultados de cobertura con el modelado acústico (sensor probabilístico
  parametrizado por la literatura) para producir el bloque de resultados completo del Capítulo 6.

### Fase D — Cierre documental
- Actualizar README y esta documentación al estado final.
- Asegurar que el repositorio contenga, además del código, los escenarios, las métricas y las
  figuras usadas en la tesis (reproducibilidad).

---

## 10. Cómo correr el proyecto

```bash
cd "C:\Users\jorge\OneDrive\Escritorio\Maestría CIM\TESIS\TESIS\simulador-swarm-2d"

# Solo la primera vez:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # corrección pendiente: añadir numpy y Pillow
pip install pillow

# Cada vez:
.venv\Scripts\activate
python main.py
```

Consejos de uso: prepara el escenario en la ventana del mapa (editor), luego en la pestaña
**Controles** pulsa "Iniciar"; en **Estrategias** cambia la estrategia y sus parámetros en caliente
y guarda con "Guardar config" para que queden fijados en `sim_config.txt`. En la pestaña **Global**
verás las métricas de cobertura en vivo; el estado COMPLETADO aparece cuando se alcanza
`exploration_goal` o se terminan las fronteras.

---

*Fin del documento.* Este archivo debe mantenerse actualizado conforme avance el proyecto;
la cadena de valor es: escenario -> simulación -> métricas -> resultados -> tesis.