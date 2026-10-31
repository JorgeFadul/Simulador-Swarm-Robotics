# Simulador Swarm 2D

Simulador de exploración e inspección cooperativa para enjambres de robots móviles en entornos bidimensionales discretizados (grid 2D) con sensado limitado por línea de vista (LOS - *Line of Sight*). Desarrollado como plataforma experimental y de validación para la tesis de maestría en Automatización y Robótica / CIM (Control Industrial y Mecatrónica).

El sistema opera tanto en modo gráfico interactivo (doble ventana: Pygame para renderizado del mapa/robots/sensores y Tkinter para telemetría en tiempo real y ajuste dinámico de hiperparámetros) como en modo *headless* de alto rendimiento para experimentos masivos, barridos de parámetros, optimización genética y entrenamiento de agentes mediante Aprendizaje por Refuerzo (RL) y modelos bioinspirados (Cerebro de Mosca Drosophila / Plasticidad Dopaminérgica).

---

## Tabla de Contenidos

1. [Características Principales](#características-principales)
2. [Requisitos e Instalación](#requisitos-e-instalación)
3. [Uso Rápido del Simulador](#uso-rápido-del-simulador)
4. [Referencia Completa de Argumentos CLI (Headless y Parsing)](#referencia-completa-de-argumentos-cli-headless-y-parsing)
5. [Modos de Operación y Estrategias](#modos-de-operación-y-estrategias)
6. [Telemetría y Formato de Salida en Modo Headless](#telemetría-y-formato-de-salida-en-modo-headless)
7. [Entrenamiento de Modelos RL y Bioinspirados](#entrenamiento-de-modelos-rl-y-bioinspirados)
8. [Optimización Genética y Validación](#optimización-genética-y-validación)
9. [Estructura del Repositorio](#estructura-del-repositorio)

---

## Características Principales

- **SLAM y Descubrimiento en 3 Niveles:**
  - `SCOUTED (1)`: Celda libre avistada mediante sensores con línea de vista (LOS mediante algoritmo de Bresenham).
  - `VISITED (2)`: Celda física transitada por la huella de un robot.
  - `DISCOVERED (3)`: Pared perimetral o interna detectada por línea de vista o por colisión física.
- **Sensado Realista con Línea de Vista (LOS):** Los sensores no atraviesan obstáculos ni paredes. Las paredes ocultas detrás de otras estructuras no se "adivinan".
- **Inspección de Defectos y Ground Truth:** Generación probabilística reproducible de parches (*blobs*) de defectos estructurales en el suelo del escenario para evaluar métricas de *recall*, tasa de detección y eficiencia energética ($J/\text{defecto}$).
- **Modelo Energético Detallado:** Simulación de consumo dinámico por movimiento (aceleración, rotación, fricción lineal), consumo estático basal del sistema y consumo específico de inspección sensorial por celda.
- **Doble Modo de Simulación:**
  - **Dos Etapas (`dos_etapas`):** Fase 1 de Exploración total del mapa combinada + Fase 2 de Inspección exhaustiva de celdas libres (cobertura total de área).
  - **Híbrido (`hibrido`):** Una única fase donde los robots exploran e inspeccionan concurrentemente al transitar las celdas.
- **Múltiples Algoritmos de Coordinación:**
  - Clásicos / Heurísticos: *Random Walk*, *Frontier-Based* (Yamauchi con reserva exclusiva dinámica), *Sample-Based* (Next-Best-View / Charrow), *Hybrid-Based*, *Boustrophedon* (descomposición celular morse/cobertura en pasadas paralelas) y *Voronoi* (partición espacial balanceada con asignación dinámica de POIs).
  - Aprendizaje por Refuerzo (RL PPO): Modelos descentralizados e invariantes a la escala del enjambre (`RL Hibrido V3`, `RL Inspeccion V3`).
  - Bioinspirado: `Cerebro Mosca (Dopamina)` / `Bio-RL` (modelo olfativo/antiguo de Drosophila con células de Kenyon, neuronas MBON y modulación por dopamina de plasticidad sináptica).

---

## Requisitos e Instalación

### Requisitos

- Python 3.10 o 3.11.
- Tarjeta gráfica compatible (opcional para GUI; el modo *headless* y RL corren en CPU o con aceleración CUDA si PyTorch lo detecta).

### Instalación

1. Clonar el repositorio y acceder a la carpeta:
   ```bash
   git clone https://github.com/JorgeFadul/Tesis.git
   cd Tesis/simulador-swarm-2d
   ```

2. Crear y activar un entorno virtual:
   ```bash
   # En Windows
   python -m venv .venv
   .venv\Scripts\activate

   # En Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias del proyecto:
   ```bash
   pip install -r requirements.txt
   ```

*Las dependencias principales incluyen `pygame`, `pyyaml`, `numpy`, `Pillow`, `stable-baselines3`, `torch`, `scipy` y `matplotlib`.*

---

## Uso Rápido del Simulador

### 1. Interfaz Gráfica (GUI Interactiva)

Para iniciar el simulador con visualización en Pygame y ventana de control y estadísticas en Tkinter:

```bash
python main.py
```

- **Ventana Pygame:** Muestra el mapa, celdas exploradas (grises/blancas), paredes detectadas (negras/rojas), defectos (amarillos/rojos) y los robots con sus vectores de intención y sensores cian. Incluye editor de paredes (`P`), robots (`R`) y borrador (`B`).
- **Ventana Tkinter:** Pestañas de telemetría individual por robot, gráfico de barras, estado global, configuración de parámetros y selector en caliente de estrategias.

### 2. Generar Mapas de Prueba Predeterminados

El repositorio cuenta con un generador automático de 5 escenarios estándar con diferente complejidad geométrica:

```bash
python main.py --generar-mapas
```
Esto asegura la creación de los escenarios YAML en la carpeta `mapas/`.

---

## Referencia Completa de Argumentos CLI (Headless y Parsing)

El punto de entrada `main.py` incorpora un parser robusto de argumentos basado en `argparse`. La sintaxis general es:

```bash
python main.py [OPCIONES]
```

### Tabla Completa de Opciones CLI

| Argumento | Tipo / Opciones | Por defecto | Descripción |
|---|---|---|---|
| `-h`, `--help` | Flag | - | Muestra el menú de ayuda y los argumentos disponibles. |
| `--headless` | Flag | `False` | Ejecuta la simulación en memoria sin abrir interfaces gráficas (Pygame/Tkinter). Emite la telemetría en tiempo real por `stdout` en formato JSON Lines (JSONL) y genera reportes finales en `.json` y `.csv`. |
| `--modo` | `dos_etapas`, `hibrido` | Configuración / `dos_etapas` | Selecciona el paradigma de misión: dos fases separadas (exploración $\to$ inspección) o fase unificada continua. |
| `--estrategia-exploracion` | `Random Walk`, `Frontier Based`, `Sample Based`, `Hybrid Based`, `Boustrophedon` | `Frontier Based` | Estrategia activa durante la Fase 1 (Exploración) en modo `dos_etapas`. |
| `--estrategia-inspeccion` | `Random Walk`, `Voronoi`, `Boustrophedon Inspeccion`, `Cerebro Mosca (Dopamina)`, `RL Inspeccion V3` | `Voronoi` | Estrategia activa durante la Fase 2 (Inspección) en modo `dos_etapas`. |
| `--estrategia-hibrido` | `Random Walk Hibrido`, `RL Hibrido V3`, `Cerebro Mosca (Dopamina)` | `Random Walk Hibrido` | Estrategia de control para el modo `hibrido`. Si se especifica, activa automáticamente `--modo hibrido`. |
| `--mapa` | Ruta a archivo YAML | `None` (usa default) | Carga un escenario específico (dimensiones, grilla, paredes, distribución de defectos y dotación de robots). |
| `--semilla` | Entero (`int`) | `None` (aleatorio) | Fija la semilla pseudoaleatoria del generador global para garantizar reproducibilidad exacta en colocación de robots, ruido de sensores y navegación. |
| `--generar-mapas` | Flag | `False` | Reconstruye o asegura la existencia de los 5 mapas estándar en `mapas/` y finaliza la ejecución inmediatamente. |
| `--estrategia` | Cadena (*deprecated*) | `None` | Parámetro heredado por compatibilidad; mapea automáticamente estrategias antiguas o alias legados a su modo correspondiente. |

### Ejemplos de Comandos en Modo Headless

#### Ejemplo 1: Misión Dos Etapas clásica (Fronteras + Voronoi) con semilla fija
```bash
python main.py --headless --modo dos_etapas --estrategia-exploracion "Frontier Based" --estrategia-inspeccion "Voronoi" --semilla 42 --mapa mapas/mapa_test_1.yaml
```

#### Ejemplo 2: Misión Dos Etapas sistemática (Boustrophedon + Boustrophedon Inspección)
```bash
python main.py --headless --modo dos_etapas --estrategia-exploracion "Boustrophedon" --estrategia-inspeccion "Boustrophedon Inspeccion" --semilla 123 --mapa mapas/mapa_rl_30x30.yaml
```

#### Ejemplo 3: Misión Híbrida con Aprendizaje por Refuerzo Descentralizado (RL V3)
```bash
python main.py --headless --modo hibrido --estrategia-hibrido "RL Hibrido V3" --semilla 7 --mapa mapas/mapa_rl_20x20_10r.yaml
```

#### Ejemplo 4: Misión Híbrida Bioinspirada (Cerebro de Mosca / Dopamina)
```bash
python main.py --headless --modo hibrido --estrategia-hibrido "Cerebro Mosca (Dopamina)" --semilla 99 --mapa mapas/mapa_test_2.yaml
```

---

## Modos de Operación y Estrategias

### 1. Modo Dos Etapas (`--modo dos_etapas`)
- **Fase 1 (Exploración):** El enjambre busca maximizar el porcentaje de celdas libres avistadas (`SCOUTED`) y delimitar todas las paredes (`DISCOVERED`). Se utiliza un planificador global basado en celdas de frontera o pasadas de barrido.
- **Transición:** Cuando se alcanza la meta de cobertura (`exploration_goal`, típicamente 95% o 99%) o se agotan las fronteras accesibles, el simulador conmuta automáticamente a la Fase 2.
- **Fase 2 (Inspección):** Los robots deben transitar físicamente sobre las celdas descubiertas para catalogarlas como `VISITED` e inspeccionar si contienen defectos estructurales en el piso.

### 2. Modo Híbrido (`--modo hibrido`)
No existe separación de fases: los robots exploran territorio virgen al tiempo que inspeccionan minuciosamente cada celda pisada. Este paradigma es el idóneo para algoritmos de **Aprendizaje por Refuerzo** y modelos **Bioinspirados**, ya que las recompensas (o señales dopaminérgicas) por descubrimiento y detección de defectos se alimentan de forma inmediata.

---

## Telemetría y Formato de Salida en Modo Headless

Al invocar `main.py --headless`, la simulación imprime en tiempo real un flujo de eventos JSON por línea (`stdout`), lo que permite conectar el simulador a pipes, agentes externos o dashboards:

```json
{"t": 1.25, "paso": 25, "estado": "running", "estrategia": "Frontier Based", "modo_fase": "dos_etapas", "cobertura": {"mapa_pct": 14.2, "libres_pct": 18.5, "paredes_pct": 5.0}, "energia_total_j": 12.45, "bateria_min_pct": 98.2, "celdas": {"vistas": 150, "visitadas": 24, "inspeccionadas": 24, "defectuosas": 3}, "robots": [{"id": 0, "x": 120.5, "y": 85.0, "estado": "MOVING", "velocidad_cm_s": 15.0, "energia_consumida_j": 1.25, ...}]}
```

Al culminar la ejecución (por éxito o por límite de pasos `headless_max_pasos`), se generan automáticamente dos archivos en el directorio raíz:

1. **Resumen Global (`headless_<modo>_<estrategia>_seed<N>.json`):**
   - Contiene el resultado (`completado`, `limite_pasos`, `estancamiento`).
   - Tiempos de simulación, pasos totales, balance energético discriminado (movimiento, inspección, picos de aceleración).
   - Métricas de celdas y métricas individuales por robot.
   - Estado matricial final del mapa de inspección y celdas defectuosas.
2. **Serie Temporal (`headless_<modo>_<estrategia>_seed<N>.csv`):**
   - Archivo tabular paso a paso con las columnas: `paso`, `t_s`, `cobertura_mapa_pct`, `cobertura_libres_pct`, `energia_total_j`, `celdas_vistas`, `celdas_visitadas`, `celdas_inspeccionadas` y `celdas_defectuosas`.

---

## Entrenamiento de Modelos RL y Bioinspirados

El simulador incluye entornos vectorizados compatibles con Gymnasium / Stable-Baselines3 y controladores bioinspirados avanzados.

### 1. Entrenamiento RL Híbrido V3 Descentralizado (`train_rl_v3_des.py`)

Entrena una política neuronal con PPO (*Proximal Policy Optimization*) que es completamente invariante al número de robots $N$ y al tamaño del mapa mediante observaciones locales egocéntricas (parches $K \times K$), comunicación vecinal y señales de alerta por defectos:

```bash
python train_rl_v3_des.py [OPCIONES]
```

**Argumentos disponibles:**
- `--timesteps`: Total de pasos de interacción de entrenamiento (por defecto `300000`, recomendado `5000000` para convergencia óptima).
- `--num-robots`: Número de robots interactuando en el entorno de entrenamiento (por defecto `10`).
- `--patch-size`: Dimensión de la ventana local egocéntrica $K \times K$ (por defecto `15`).
- `--radio-alerta`: Radio de propagación en píxeles de la señal de alerta por celda defectuosa (por defecto `500.0`).
- `--n-envs`: Cantidad de entornos paralelos (`SubprocVecEnv`). Si es `0`, detecta y utiliza automáticamente todos los núcleos de CPU disponibles.
- `--out`: Directorio de guardado de los puntos de control (`checkpoints`) y modelo final (por defecto `models/v3_des`).

**Ejemplo de ejecución:**
```bash
python train_rl_v3_des.py --timesteps 1000000 --num-robots 10 --patch-size 15 --radio-alerta 500.0 --out models/v3_des
```

---

### 2. Entrenamiento RL de Exploración Pura V3 (`train_rl_v3_exp.py`)

Especializado en aprendizaje por refuerzo descentralizado enfocado en la maximización de la tasa de descubrimiento de celdas libres y paredes con minimización de consumo energético:

```bash
python train_rl_v3_exp.py [OPCIONES]
```

**Argumentos disponibles:**
- `--timesteps`: Pasos totales de entrenamiento (por defecto `300000`).
- `--num-robots`: Cantidad de robots en la simulación (por defecto `10`).
- `--patch-size`: Tamaño del parche de observación local (por defecto `15`).
- `--radio-com`: Radio de comunicación cooperativa entre robots en píxeles (por defecto `400.0`).
- `--reward-mode`: Función objetivo de penalización (`energy` o `time`; por defecto `energy`).
- `--n-envs`: Número de entornos paralelos (0 = auto por núcleos de CPU).
- `--out`: Carpeta de destino del modelo (por defecto `models/v3_exp`).

**Ejemplo de ejecución:**
```bash
python train_rl_v3_exp.py --timesteps 2000000 --num-robots 10 --reward-mode energy --out models/v3_exp
```

---

### 3. Entrenamiento Bioinspirado: Cerebro de Mosca / Dopamina (`train_bio_fly.py`)

Implementa un modelo del cuerpo pedunculado (*Mushroom Body*) de la mosca *Drosophila melanogaster*. Consta de neuronas de proyección (PN), un estrato de células de Kenyon (KC) con codificación dispersa (*sparse coding*) y neuronas de salida (MBON) cuya sinapsis se modifica mediante plasticidad hebbiana modulada por dopamina (recompensa por celdas nuevas y celdas defectuosas; aversión por colisiones).

El entrenamiento consolida en cada corrida la memoria del cerebro con mejor rendimiento evolutivo:

```bash
python train_bio_fly.py [OPCIONES]
```

**Argumentos disponibles:**
- `--runs`: Cantidad de simulaciones/generaciones consecutivas (por defecto `30`).
- `--robots`: Número de robots en el enjambre (si se omite, usa el valor definido en el mapa).
- `--mapas`: Ruta a un mapa YAML o lista separada por comas de mapas para alternar entre runs (por defecto `mapas/mapa_test_1.yaml`).
- `--kc`: Número de Kenyon Cells por cerebro (por defecto `1000`; valores menores aceleran el cómputo).
- `--mutacion`: Tasa de perturbación relativa en pesos heredados $W_{\text{KC}\to\text{MBON}}$ (por defecto `0.02`).
- `--max-steps`: Límite máximo de pasos por corrida antes de corte (por defecto `15000`).
- `--max-segundos`: Límite de tiempo de reloj real por corrida (por defecto `300` s).
- `--semilla`: Semilla base de la secuencia de generaciones (por defecto `42`).
- `--defectos-semilla`: Semilla base para la distribución de defectos.
- `--out`: Archivo NumPy `.npz` donde se guardan los pesos sinápticos consolidados (por defecto `models/bio/fly_swarm_latest.npz`).
- `--reporte`: Archivo CSV con la evolución y curva de aprendizaje (por defecto `<out>.csv`).
- `--reset-weights`: Flag para descartar pesos previos y reiniciar el aprendizaje desde cero.

**Ejemplo de ejecución:**
```bash
python train_bio_fly.py --runs 50 --robots 10 --mapas mapas/mapa_test_1.yaml,mapas/mapa_test_2.yaml --kc 500 --mutacion 0.02 --out models/bio/fly_swarm_latest.npz
```

---

## Optimización Genética y Validación

El proyecto cuenta con herramientas para ajustar hiperparámetros de las heurísticas de movimiento y generar los análisis comparativos de la tesis:

- **Optimización por Algoritmos Genéticos (`optimizar_ga.py`):**
  Optimiza los parámetros de navegación dirigida y evitación de obstáculos:
  ```bash
  python optimizar_ga.py --poblacion 16 --generaciones 10 --mapa mapas/mapa_test_1.yaml
  ```
- **Generación de Gráficas de Tesis (`graficar_tesis_final.py` y `graficar_generalista.py`):**
  Procesan los CSV de resultados para construir curvas de Pareto ($J/\text{defecto}$ vs. tiempo), curvas de cobertura temporal, leyes de escala $N^2/R$ y análisis de dotación óptima de robots.
- **Validación de Defectos Ground Truth (`gt_defectos.py`):**
  Calcula con precisión matemática el número exacto de celdas con defectos generadas para cada semilla y mapa, garantizando métricas de *recall* libres de sesgo.

---

## Estructura del Repositorio

```
simulador-swarm-2d/
├── main.py                     # Punto de entrada principal (GUI y Headless CLI)
├── requirements.txt            # Dependencias del proyecto
├── sim_config.txt              # Configuración persistente del simulador
├── train_rl_v3_des.py          # Script de entrenamiento RL PPO Descentralizado V3
├── train_rl_v3_exp.py          # Script de entrenamiento RL PPO Exploración V3
├── train_bio_fly.py            # Script de entrenamiento modelo bioinspirado Drosophila
├── optimizar_ga.py             # Optimizador genético de parámetros de navegación
├── gt_defectos.py              # Cálculo de ground truth de defectos por semilla
├── graficar_tesis_final.py     # Generador de gráficos de rendimiento y tesis
├── config/
│   └── default.yaml            # Configuración base predeterminada
├── mapas/                      # Escenarios de prueba YAML (10x10, 20x20, 30x30, 50x50...)
├── models/                     # Checkpoints y pesos de RL y modelos bioinspirados
│   ├── bio/                    # Pesos de Cerebro de Mosca (.npz)
│   ├── v3_des/                 # Políticas entrenadas RL V3 Descentralizado
│   └── v3_exp/                 # Políticas entrenadas RL V3 Exploración
├── src/
│   ├── algorithms/             # Heurísticas: Frontier, Sample, Voronoi, Boustrophedon, A*
│   ├── bio/                    # Red neuronal bioinspirada de Drosophila (cuerpo pedunculado)
│   ├── core/                   # Motor físico, mundo 2D, sensores LOS, telemetría y config
│   ├── gui/                    # Renderizador Pygame y ventana de control Tkinter
│   ├── rl/                     # Entornos Gym/Gymnasium vectorizados para RL
│   └── robots/                 # Cinemática diferencial, máquina de estados y batería
└── resultados_tesis_final/     # Resultados experimentales, series temporales y figuras
```

---

## Licencia y Créditos

Desarrollado por **Jorge Fadul** como parte de la Tesis de Maestría en CIM (Control Industrial y Mecatrónica). Distribuido bajo licencia libre para fines académicos y de investigación.
