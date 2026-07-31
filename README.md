# Simulador Swarm 2D

Simulador de exploración cooperativa para enjambres de robots móviles con sensado limitado (línea de vista). Pensado como herramienta de estudio y validación para la tesis de maestría en CIM (Control Industrial y Mecatrónica).

El sistema combina dos ventanas:

- **Ventana de mapa (Pygame):** renderiza el mundo, las celdas exploradas y los robots en tiempo real, e incluye un editor de escenarios.
- **Ventana de estadísticas y control (Tkinter):** muestra métricas de cobertura por robot y global, y permite configurar parámetros sin tocar código.

## Características

- **Exploración SLAM por celda en tres niveles:** cada celda del mapa puede estar `SCOUTED` (avistada), `VISITED` (pisada por un robot) o `DISCOVERED` (es una pared que el robot ha visto o chocado).
- **Sensado con línea de vista (LOS):** los sensores de los robots no atraviesan paredes. Una pared solo es `DISCOVERED` cuando un robot la ve o la choca físicamente (no se "adivina" la geometría).
- **Robot como máquina de estados:** `STOPPED`, `ROTATING` y `MOVING`, con velocidad de giro limitada y detección de colisión frontal (obstáculos y otros robots).
- **Evitación reactiva de obstáculos:** vector de alejamiento promedio sobre obstáculos visibles dentro del radio de detección.
- **Exploración individual y global:** cada robot mantiene su propio mapa `explored_robot`, y el mapa global agrega todas las percepciones.
- **Visualización de la intención de movimiento:** línea punteada cian con punta de flecha que indica hacia dónde y cuánto avanzará el robot según su plan actual.
- **Editor integrado:** colocar robots, dibujar/borrar paredes por celda y cambiar el tamaño de la grilla.
- **Configuración persistente:** los parámetros se cargan de `sim_config.txt` al iniciar (si el archivo no existe, se usan los valores por defecto de `config/default.yaml`).
- **Mini-mapas por robot** en pantalla, más mapas de exploración individuales y métricas de cobertura en la ventana de estadísticas.

## Requisitos

- Python 3.10 o superior.
- Paquetes listados en `requirements.txt`:

```
pygame>=2.5.0
pyyaml>=6.0
```

Adicionalmente la ventana de estadísticas usa `Pillow` (PIL) para escalar los mini-mapas.

## Instalación

Desde la carpeta del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# o: source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install pillow
```

## Ejecución

```bash
python main.py
```

Se abren las dos ventanas. Para salir, cierra la ventana de Pygame o presiona la X; la ventana de estadísticas se puede cerrar y reabrir sin romper el programa (quedará inactiva y el simulador seguirá corriendo).

## Controles

### Editor (ventana de mapa)

| Tecla / Acción      | Efecto                                |
|---------------------|---------------------------------------|
| `R`                 | Seleccionar herramienta Robot         |
| `P`                 | Seleccionar herramienta Pared         |
| `B`                 | Seleccionar herramienta Borrar        |
| Clic izquierdo      | Colocar robot o pared según herramienta |
| Clic derecho        | Borrar robot o pared                  |
| Arrastrar           | Dibujar pared continua                |
| `ESC`               | Salir del modo simulación y volver al editor |

### Ventana de estadísticas (pestañas)

- **Individual:** selecciona un robot del desplegable; muestra un gráfico de barras con sus métricas (celdas exploradas, visitadas, paredes vistas) y su mini-mapa de exploración.
- **Cobertura:** vista general de la cobertura de exploración.
- **Global:** tiempo simulado, FPS, pasos totales, porcentajes de cobertura y cantidad de robots.
- **Controles:** herramientas, tamaño de grilla, población (botón "Poblar" agrega robots en posiciones libres), parámetros editables del robot, selección de estrategia, botones de simular/pausar/detener y guardar/cargar configuración.

## Configuración

### Archivo de parámetros guardados: `sim_config.txt`

Formato `clave=valor`, una por línea. Se escribe con el botón "Guardar config" de la pestaña Controles y se carga automáticamente al arrancar el programa.

| Clave                             | Unidad      | Descripción                          |
|-----------------------------------|-------------|--------------------------------------|
| `robot_radius_cm`                 | cm          | Radio físico del robot               |
| `robot_max_speed_cm_s`            | cm/s        | Velocidad lineal máxima              |
| `robot_turn_speed_rad_s`          | rad/s       | Velocidad de giro                    |
| `detection_radius_cm`             | cm          | Radio de detección de sensores       |
| `exploration_radius_multiplier`   | (adim)      | Multiplicador del radio de exploración (mapa SLAM) |
| `random_walk_interval_min`        | s           | Duración mínima de cada tramo de movimiento |
| `random_walk_interval_max`        | s           | Duración máxima de cada tramo        |
| `strategy`                        | -           | Estrategia seleccionada              |

Las distancias físicas (cm) se convierten a píxeles según `pixels_per_cm`. Si el archivo no existe, falta o un valor no es numérico, se ignora y se conservan los valores por defecto.

### Archivo base: `config/default.yaml`

Contiene la configuración de arranque: tamaño de mapa y grilla, `pixels_per_cm`, FPS, número de robots por defecto, parámetros físicos del robot, umbral de exploración y paredes iniciales (lista de rectángulos rotables).

## Arquitectura

```
simulador-swarm-2d/
├── main.py                    # Punto de entrada, bucle principal (Pygame + Tkinter)
├── requirements.txt
├── sim_config.txt             # Parámetros guardados (se carga al iniciar)
├── config/
│   └── default.yaml           # Valores por defecto
├── src/
│   ├── core/
│   │   ├── config.py          # Config: carga de YAML y de sim_config.txt
│   │   ├── mundo.py           # Mundo, grilla, LOS, paredes, sensores, colisiones
│   │   └── simulacion.py      # Orquesta Pygame, editor y el bucle de simulación
│   ├── robots/
│   │   └── robot.py           # Máquina de estados, exploración propia, intención
│   ├── gui/
│   │   ├── renderer.py        # Dibujado del mapa, robots, intención y mini-mapas
│   │   └── stats_window.py    # Ventana Tkinter de estadísticas y controles
│   ├── algorithms/            # (reservado para estrategias de exploración)
│   └── utils/                 # (utilidades)
```

## Modelo de exploración

### Tres estados por celda

| Estado | Valor | Significado |
|--------|------:|-------------|
| `SCOUTED`   | 1 | Celda libre avistada por un sensor con línea de vista |
| `VISITED`   | 2 | Celda libre por la que un robot pasó físicamente |
| `DISCOVERED`| 3 | Celda de pared conocida (vista o chocada) |

### Percepción con línea de vista

- `tiene_linea_de_vista(x1,y1,x2,y2)` recorre la celda a celda con el algoritmo de Bresenham y devuelve `False` si hay una pared en el camino.
- El sensado (`explorar_desde`, `detectar_en_rango`) solo considera paredes con línea de vista: no se puede explorar a través de una pared.
- `detectar_en_rango` devuelve un vector de alejamiento promedio normalizado hacia afuera de los obstáculos visibles.

### Colisiones y descubrimiento de paredes

- La anticipación de colisión usa solo paredes `DISCOVERED` (`colision_frontal` con `solo_descubiertas=True`): el robot no puede "sentir" paredes que nunca ha visto.
- Cuando el robot choca físicamente, `descubrir_pared_frontal` marca la pared como `DISCOVERED` (agrega información real al mapa) y el robot elige un nuevo recorrido.

### Robot y movimiento

La máquina de estados (`STOPPED` -> `ROTATING` -> `MOVING`) rota hacia el ángulo objetivo a velocidad limitada y luego avanza a velocidad crucero con aceleración suave. El temporizador de cada tramo (`timer`) **solo avanza mientras el robot está en `MOVING`**, por lo que el tiempo de giro no reduce la duración del movimiento planificado.

### Intención de movimiento

Cada robot expone su plan actual:

- `intent_tiempo_restante = max(0, duration - timer)`
- `intent_distancia = target_speed * intent_tiempo_restante`

El renderizador dibuja una línea punteada cian con punta de flecha en la dirección de `target_angle`, con longitud acotada a la `intent_distancia`, limitada a 260 px y detenida en la primera pared `DISCOVERED`. Se dibuja debajo de los robots para que estos queden siempre visibles.

## Mini-mapas

El renderer muestra en la esquina inferior derecha un panel semitransparente con un mini-mapa de 50 px por robot (4 por fila, 1 px por celda). El mapa individual de cada robot (pestaña "Individual") se escala con Pillow.

## Pruebas

No hay framework de pruebas configurado; la validación se hace con scripts headless que ejercitan la lógica pura (config, mundo, robot) sin abrir ventanas:

- Carga de `sim_config.txt` y fallback a valores por defecto.
- Línea de vista bloqueada y despejada.
- Paredes detrás de paredes no descubiertas.
- Sensores con línea de vista.
- Colisión con `solo_descubiertas`.
- Descubrimiento de pared frontal tras choque.
- Ningún robot colocado dentro de una pared tras poblar 2000 veces.
- API de intención de movimiento y su renderizado.
- El temporizador de movimiento no avanza durante la rotación.

## Estado del proyecto

Simulador funcional con editor, sensado por línea de vista, exploración SLAM por celda, métricas por robot/global, persistencia de configuración y visualización de intención. La carpeta `src/algorithms/` está reservada para la implementación de estrategias de exploración más allá del random walk.
