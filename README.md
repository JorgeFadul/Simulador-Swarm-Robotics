# Simulador Swarm 2D

Simulador de exploracion cooperativa para enjambres de robots moviles con sensado limitado (linea de vista). Pensado como herramienta de estudio y validacion para la tesis de maestria en CIM (Control Industrial y Mecatronica).

El sistema combina dos ventanas:

- **Ventana de mapa (Pygame):** renderiza el mundo, las celdas exploradas y los robots en tiempo real, e incluye un editor de escenarios.
- **Ventana de estadisticas y control (Tkinter):** muestra metricas de cobertura por robot y global, y permite configurar parametros sin tocar codigo.

## Caracteristicas

- **Exploracion SLAM por celda en tres niveles:** cada celda del mapa puede estar `SCOUTED` (avistada), `VISITED` (pisada por un robot) o `DISCOVERED` (es una pared que el robot ha visto o chocado).
- **Sensado con linea de vista (LOS):** los sensores de los robots no atraviesan paredes. Una pared solo es `DISCOVERED` cuando un robot la ve o la choca fisicamente (no se "adivina" la geometria).
- **Robot como maquina de estados:** `STOPPED`, `ROTATING` y `MOVING`, con velocidad de giro limitada y deteccion de colision frontal (obstaculos y otros robots).
- **Evitacion reactiva de obstaculos:** vector de alejamiento promedio sobre obstaculos visibles dentro del radio de deteccion.
- **Exploracion individual y global:** cada robot mantiene su propio mapa `explored_robot`, y el mapa global agrega todas las percepciones.
- **Visualizacion de la intencion de movimiento:** linea punteada cian con punta de flecha que indica hacia donde y cuanto avanzara el robot segun su plan actual.
- **Editor integrado:** colocar robots, dibujar/borrar paredes por celda y cambiar el tamano de la grilla.
- **Configuracion persistente:** los parametros se cargan de `sim_config.txt` al iniciar (si el archivo no existe, se usan los valores por defecto de `config/default.yaml`).
- **Mini-mapas por robot** en pantalla, mas mapas de exploracion individuales y metricas de cobertura en la ventana de estadisticas.

## Requisitos

- Python 3.10 o superior.
- Paquetes listados en `requirements.txt`:

```
pygame>=2.5.0
pyyaml>=6.0
```

Adicionalmente la ventana de estadisticas usa `Pillow` (PIL) para escalar las mini-mapas.

## Instalacion

Desde la carpeta del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# o: source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install pillow
```

## Ejecucion

```bash
python main.py
```

Se abren las dos ventanas. Para salir, cierra la ventana de Pygame o presiona la X; la ventana de estadisticas se puede cerrar y reabrir sin romper el programa (quedara inactiva y el simulador seguira corriendo).

## Controles

### Editor (ventana de mapa)

| Tecla / Accion      | Efecto                                |
|---------------------|---------------------------------------|
| `R`                 | Seleccionar herramienta Robot         |
| `P`                 | Seleccionar herramienta Pared         |
| `B`                 | Seleccionar herramienta Borrar        |
| Clic izquierdo      | Colocar robot o pared segun herramienta |
| Clic derecho        | Borrar robot o pared                  |
| Arrastrar           | Dibujar pared continua                |
| `ESC`               | Salir del modo simulacion y volver al editor |

### Ventana de estadisticas (pestanas)

- **Individual:** selecciona un robot del desplegable; muestra un grafico de barras con sus metricas (celdas exploradas, visitadas, paredes vistas) y su mini-mapa de exploracion.
- **Cobertura:** vista general de la cobertura de exploracion.
- **Global:** tiempo simulado, FPS, pasos totales, porcentajes de cobertura y cantidad de robots.
- **Controles:** herramientas, tamano de grilla, poblacion (boton "Poblar" agrega robots en posiciones libres), parametros editables del robot, seleccion de estrategia, botones de simular/pausar/detener y guardar/cargar configuracion.

## Configuracion

### Archivo de parametros guardados: `sim_config.txt`

Formato `clave=valor`, una por linea. Se escribe con el boton "Guardar config" de la pestana Controles y se carga automaticamente al arrancar el programa.

| Clave                             | Unidad      | Descripcion                          |
|-----------------------------------|-------------|--------------------------------------|
| `robot_radius_cm`                 | cm          | Radio fisico del robot               |
| `robot_max_speed_cm_s`            | cm/s        | Velocidad lineal maxima              |
| `robot_turn_speed_rad_s`          | rad/s       | Velocidad de giro                    |
| `detection_radius_cm`             | cm          | Radio de deteccion de sensores       |
| `exploration_radius_multiplier`   | (adim)      | Multiplicador del radio de exploracion (mapa SLAM) |
| `random_walk_interval_min`        | s           | Duracion minima de cada tramo de movimiento |
| `random_walk_interval_max`        | s           | Duracion maxima de cada tramo        |
| `strategy`                        | -           | Estrategia seleccionada              |

Las distancias fisicas (cm) se convierten a pixeles segun `pixels_per_cm`. Si el archivo no existe, falta o un valor no es numerico, se ignora y se conservan los valores por defecto.

### Archivo base: `config/default.yaml`

Contiene la configuracion de arranque: tamano de mapa y grilla, `pixels_per_cm`, FPS, numero de robots por defecto, parametros fisicos del robot, umbral de exploracion y paredes iniciales (lista de rectangulos rotables).

## Arquitectura

```
simulador-swarm-2d/
├── main.py                    # Punto de entrada, bucle principal (Pygame + Tkinter)
├── requirements.txt
├── sim_config.txt             # Parametros guardados (se carga al iniciar)
├── config/
│   └── default.yaml           # Valores por defecto
├── src/
│   ├── core/
│   │   ├── config.py          # Config: carga de YAML y de sim_config.txt
│   │   ├── mundo.py           # Mundo, grilla, LOS, paredes, sensores, colisiones
│   │   └── simulacion.py      # Orquesta Pygame, editor y el bucle de simulacion
│   ├── robots/
│   │   └── robot.py           # Maquina de estados, exploracion propia, intencion
│   ├── gui/
│   │   ├── renderer.py        # Dibujado del mapa, robots, intencion y mini-mapas
│   │   └── stats_window.py    # Ventana Tkinter de estadisticas y controles
│   ├── algorithms/            # (reservado para estrategias de exploracion)
│   └── utils/                 # (utilidades)
```

## Modelo de exploracion

### Tres estados por celda

| Estado | Valor | Significado |
|--------|------:|-------------|
| `SCOUTED`   | 1 | Celda libre avistada por un sensor con linea de vista |
| `VISITED`   | 2 | Celda libre por la que un robot paso fisicamente |
| `DISCOVERED`| 3 | Celda de pared conocida (vista o chocada) |

### Percepcion con linea de vista

- `tiene_linea_de_vista(x1,y1,x2,y2)` recorre la celda a celda con el algoritmo de Bresenham y devuelve `False` si hay una pared en el camino.
- El sensado (`explorar_desde`, `detectar_en_rango`) solo considera paredes con linea de vista: no se puede explorar a traves de una pared.
- `detectar_en_rango` devuelve un vector de alejamiento promedio normalizado hacia afuera de los obstaculos visibles.

### Colisiones y descubrimiento de paredes

- La anticipacion de colision usa solo paredes `DISCOVERED` (`colision_frontal` con `solo_descubiertas=True`): el robot no puede "sentir" paredes que nunca ha visto.
- Cuando el robot choca fisicamente, `descubrir_pared_frontal` marca la pared como `DISCOVERED` (agrega informacion real al mapa) y el robot elige un nuevo recorrido.

### Robot y movimiento

La maquina de estados (`STOPPED` -> `ROTATING` -> `MOVING`) rota hacia el angulo objetivo a velocidad limitada y luego avanza a velocidad crucero con aceleracion suave. El temporizador de cada tramo (`timer`) **solo avanza mientras el robot esta en `MOVING`**, por lo que el tiempo de giro no reduce la duracion del movimiento planificado.

### Intencion de movimiento

Cada robot expone su plan actual:

- `intent_tiempo_restante = max(0, duration - timer)`
- `intent_distancia = target_speed * intent_tiempo_restante`

El renderizador dibuja una linea punteada cian con punta de flecha en la direccion de `target_angle`, con longitud acotada a la `intent_distancia`, limitada a 260 px y detenida en la primera pared `DISCOVERED`. Se dibuja debajo de los robots para que estos queden siempre visibles.

## Mini-mapas

El renderer muestra en la esquina inferior derecha un panel semi-transparente con un mini-mapa de 50 px por robot (4 por fila, 1 px por celda). El mapa individual de cada robot (pestana "Individual") se escala con Pillow.

## Pruebas

No hay framework de pruebas configurado; la validacion se hace con scripts headless que ejercitan la logica pura (config, mundo, robot) sin abrir ventanas:

- Carga de `sim_config.txt` y fallback a valores por defecto.
- Linea de vista bloqueada y despejada.
- Paredes detras de paredes no descubiertas.
- Sensores con linea de vista.
- Colision con `solo_descubiertas`.
- Descubrimiento de pared frontal tras choque.
- Ningun robot colocado dentro de una pared tras poblar 2000 veces.
- API de intencion de movimiento y su renderizado.
- El temporizador de movimiento no avanza durante la rotacion.

## Estado del proyecto

Simulador funcional con editor, sensado por linea de vista, exploracion SLAM por celda, metricas por robot/global, persistencia de configuracion y visualizacion de intencion. La carpeta `src/algorithms/` esta reservada para la implementacion de estrategias de exploracion mas alla del random walk.
