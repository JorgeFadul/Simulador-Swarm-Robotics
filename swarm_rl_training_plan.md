# Plan Completo: Entrenamiento de Swarm de E-pucks con RL (Política Global Compartida)

**Instrucciones para el agente de IA:**

Este documento describe el procedimiento exacto para implementar un sistema de entrenamiento por refuerzo que controle múltiples robots e-puck en un entorno de simulación Python. Sigue cada fase en orden. Cada sección especifica qué código escribir, qué valores usar, qué verificar y qué errores evitar. Cuando hay ambigüedad, esta sección aclara el enfoque exacto.

**Objetivo del sistema:** Entrenar una única red neuronal que controle todos los robots del swarm para inspeccionar un grid de baldosas, minimizando el tiempo total y aplicando penalizaciones por duración.

---

## FASE 1 — Modelar el problema correctamente

### Tipo de Problema
- **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process) con **parameter sharing**.
- **Definición:** Hay N robots, pero **una sola red neuronal** (una sola política π).
- **Funcionamiento:** En cada paso, cada robot le pasa su observación local a esa misma red y recibe su acción.
- Se entrena una sola vez con experiencias combinadas de todos los robots.
- **Resultado:** Todos los robots obedecen la misma lógica, pero con percepciones diferentes.

### Ventajas del Parameter Sharing
- ✅ Escalable: agregar/quitar robots no requiere reentrenar.
- ✅ Consistencia: todos usan la misma estrategia global.
- ✅ Eficiencia de entrenamiento: más datos por actualizacion de red.
- ❌ Menos flexibilidad: todos los robots son homogéneos en comportamiento.

---

## FASE 2 — Diseñar el espacio de observación

### Requisito crítico
**Tamaño fijo e idéntico para todos los robots** — es obligatorio para parameter sharing.

### Vector de observación recomendado por robot

| Componente | Tamaño | Descripción | Normalización |
|-----------|--------|-------------|----------------|
| Posición (x, y) | 2 | Coordenadas respecto al grid | Dividir por tamaño_grid |
| Orientación | 2 | sin(θ), cos(θ) | Ya están en [-1, 1] |
| Sensores IR | 8 | Lectura de los 8 sensores de proximidad | Dividir por rango máximo |
| Mapa local 3×3 | 9 | Ventana de celdas: estado de cobertura (0/1) | Ya está en [0, 1] |
| Cobertura global | 1 | Porcentaje total cubierto por el swarm | Dividir por 100 |
| Tiempo transcurrido | 1 | Tiempo usado / tiempo máximo | Dividir por t_max |
| Posiciones relativas (K=2) | 4 | Distancia y ángulo a los 2 robots más cercanos | Normalizar |

**Tamaño total: 2 + 2 + 8 + 9 + 1 + 1 + 4 = 27 valores**

### Implementación

```python
def get_observation(robot_i, swarm_state, grid_state, time_elapsed, time_max):
    """
    Retorna un vector numpy de tamaño 27 (normalizado a [-1, 1] o [0, 1])
    
    Args:
        robot_i: índice del robot
        swarm_state: posiciones y orientaciones de todos los robots
        grid_state: matriz de cobertura (0 = no cubierto, 1 = cubierto)
        time_elapsed: tiempo actual en pasos
        time_max: tiempo máximo permitido
    
    Returns:
        np.array de tamaño 27
    """
    
    # 1. Posición normalizada
    x, y = swarm_state[robot_i]["pos"]
    obs_pos = np.array([x / grid_width, y / grid_height])
    
    # 2. Orientación (evita discontinuidad en 0°/360°)
    theta = swarm_state[robot_i]["angle"]
    obs_orient = np.array([np.sin(theta), np.cos(theta)])
    
    # 3. Sensores IR (8 valores)
    obs_ir = np.array(swarm_state[robot_i]["ir_sensors"]) / IR_MAX_RANGE
    
    # 4. Mapa local 3x3 (baldosas alrededor del robot)
    robot_cell_x = int(x / cell_size)
    robot_cell_y = int(y / cell_size)
    obs_local_map = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cx, cy = robot_cell_x + dx, robot_cell_y + dy
            if 0 <= cx < grid_width and 0 <= cy < grid_height:
                obs_local_map.append(float(grid_state[cx, cy]))
            else:
                obs_local_map.append(-1.0)  # fuera de límites = -1
    obs_local_map = np.array(obs_local_map)
    
    # 5. Cobertura global
    coverage_total = np.sum(grid_state) / grid_state.size
    obs_coverage = np.array([coverage_total])
    
    # 6. Tiempo transcurrido (normalizado)
    obs_time = np.array([time_elapsed / time_max])
    
    # 7. Posiciones relativas de los 2 robots más cercanos
    obs_relative_pos = []
    distances = []
    for j in range(len(swarm_state)):
        if j != robot_i:
            dx = swarm_state[j]["pos"][0] - x
            dy = swarm_state[j]["pos"][1] - y
            dist = np.sqrt(dx**2 + dy**2)
            distances.append((dist, j))
    
    distances.sort()
    for _ in range(2):  # 2 robots más cercanos
        if distances:
            dist, j = distances.pop(0)
            angle_to_robot = np.arctan2(swarm_state[j]["pos"][1] - y, 
                                        swarm_state[j]["pos"][0] - x)
            obs_relative_pos.append(dist / ROBOT_SENSOR_RANGE)
            obs_relative_pos.append(angle_to_robot / np.pi)  # normalizar ángulo
        else:
            obs_relative_pos.append(0.0)
            obs_relative_pos.append(0.0)
    
    obs_relative_pos = np.array(obs_relative_pos)
    
    # Concatenar todos
    observation = np.concatenate([
        obs_pos,           # 2
        obs_orient,        # 2
        obs_ir,            # 8
        obs_local_map,     # 9
        obs_coverage,      # 1
        obs_time,          # 1
        obs_relative_pos   # 4
    ])
    
    return observation  # tamaño total: 27
```

---

## FASE 3 — Diseñar el espacio de acción

### Acciones Discretas

**Definición del espacio de acción:**

```python
ACTION_SPACE = {
    0: "girar_izquierda",
    1: "girar_derecha",
    2: "avanzar_e_inspeccionar"
}

# Tamaño: 3 acciones
# Salida de la red: vector de probabilidades de tamaño 3 + Softmax
```

**Explicación detallada de cada acción:**

- **Acción 0 (girar_izquierda):** El robot rota sobre sí mismo 45 grados en sentido antihorario. La posición NO cambia, solo la orientación. Tiempo de ejecución: 1 paso de simulación.

- **Acción 1 (girar_derecha):** El robot rota sobre sí mismo 45 grados en sentido horario. La posición NO cambia, solo la orientación. Tiempo de ejecución: 1 paso de simulación.

- **Acción 2 (avanzar_e_inspeccionar):** El robot se mueve **exactamente una celda** en la dirección que mira (determinada por su orientación actual). La celda se marca como "inspeccionada". Si colisiona con un obstáculo o está fuera del grid, no se mueve pero el paso de simulación se cuenta igual. Tiempo de ejecución: 1 paso de simulación.

**Parámetros de movimiento (CRÍTICO):**

```python
MOVEMENT_PARAMS = {
    "turn_angle": np.pi / 4,           # 45 grados en radianes
    "cell_size": 1.0,                  # tamaño de una celda en metros
    "velocity_linear_constant": 0.25,  # velocidad lineal FIJA (m/s) - CRÍTICO
    "simulation_timestep": 1.0,        # duración de un paso en segundos
}

# EXPLICACIÓN DETALLADA DE CÓMO FUNCIONA EL MOVIMIENTO:
#
# 1. Los parámetros de velocidad vienen del e-puck físico.
#    velocity_linear_constant = 0.25 m/s (velocidad máxima típica del e-puck)
#
# 2. En cada paso de simulación (timestep = 1.0 s):
#    - Si ejecutas acción 2 (avanzar):
#    - desplazamiento = velocity_linear_constant * simulation_timestep
#    - desplazamiento = 0.25 m/s * 1.0 s = 0.25 metros
#
# 3. El robot se desplaza 0.25 metros en la dirección de su orientación.
#    - Esto es una fracción de una celda (que mide 1.0 metro)
#    - Para recorrer una celda completa, se necesitan 4 pasos seguidos
#
# 4. La detección de "qué celda estoy en" ocurre en step():
#    - cell_x = int(robot_position_x / cell_size)
#    - Cada vez que cambias de celda, se marca como inspeccionada
#    - Un robot puede estar "entre" dos celdas durante la transición
#
# 5. Esto es REALISTA porque:
#    - El robot no teletransporta una celda completa en un paso
#    - El movimiento es gradual y suave
#    - Permite esquivar obstáculos en tiempo real
#
# 6. AJUSTE DE PARÁMETROS si quieres cambiar velocidad:
#    Si velocity_linear_constant = 0.5 m/s (más rápido):
#      - desplazamiento = 0.5 m/s * 1.0 s = 0.5 metros
#      - Se necesitarían 2 pasos para una celda
#    Si velocity_linear_constant = 1.0 m/s (muy rápido):
#      - desplazamiento = 1.0 m/s * 1.0 s = 1.0 metros
#      - Se cruza una celda EXACTA por paso (más jerárquico)
#
# 7. RECOMENDACIÓN: Usa 0.25 m/s para darle tiempo a la red neuronal
#    de "ver" múltiples pasos de toma de decisiones.
#    Con 1.0 m/s, todo es muy rápido y la red apenas puede reaccionar.
```

**Implementación del movimiento:**

```python
def execute_action(robot_idx, action, swarm_state, velocity_linear, timestep):
    """
    Ejecuta una acción discreta sobre un robot.
    
    Args:
        robot_idx: índice del robot en swarm_state
        action: int (0=girar_izq, 1=girar_der, 2=avanzar)
        swarm_state: lista de dicts con info de robots
        velocity_linear: velocidad lineal constante (m/s) del robot
        timestep: duración del paso de simulación (s)
    
    Returns:
        None (modifica swarm_state in-place)
    """
    
    turn_angle = np.pi / 4  # 45 grados
    
    if action == 0:  # girar izquierda
        # Rotar orientación sin cambiar posición
        swarm_state[robot_idx]["angle"] += turn_angle
        swarm_state[robot_idx]["angle"] = swarm_state[robot_idx]["angle"] % (2 * np.pi)
    
    elif action == 1:  # girar derecha
        # Rotar orientación sin cambiar posición
        swarm_state[robot_idx]["angle"] -= turn_angle
        swarm_state[robot_idx]["angle"] = swarm_state[robot_idx]["angle"] % (2 * np.pi)
    
    elif action == 2:  # avanzar e inspeccionar
        # Calcular desplazamiento basado en velocidad constante
        displacement = velocity_linear * timestep
        
        # Calcular nueva posición en dirección de orientación
        angle = swarm_state[robot_idx]["angle"]
        dx = displacement * np.cos(angle)
        dy = displacement * np.sin(angle)
        
        swarm_state[robot_idx]["pos"][0] += dx
        swarm_state[robot_idx]["pos"][1] += dy
        
        # IMPORTANTE: la inspección de la celda ocurre en step(), no aquí
        # En step() se determina qué celda está el robot y se marca como inspeccionada
```

**Verificación de integridad:**

Antes de usar este espacio de acción, verifica:

1. ✓ La salida de la red neuronal tiene tamaño exacto 3 (una probabilidad por acción)
2. ✓ El vector de acciones muestreado siempre contiene valores en {0, 1, 2}
3. ✓ Cada acción es invertible o reversible (girar izquierda + girar derecha = no-op) ← esto es importante para evitar trampas
4. ✓ El movimiento mantiene al robot dentro de los límites del grid (verificado en step(), con clipping)

---

## FASE 4 — Diseñar la función de recompensa (Restricción de tiempo)

### Ecuación de recompensa por robot en cada paso

```
r_t(robot_i) = 
    - c_tiempo                                   # penalización por tiempo
    + c_baldosa * I(nuevas_baldosas_cubiertas)  # bonus por cobertura nueva
    - c_redundancia * I(baldosa_ya_cubierta)    # penalización por redundancia
    - c_colision * I(colisión)                  # penalización por colisión
    + BONUS_FINAL                                # al completar (ver abajo)
```

### Componentes detallados

#### 1. Penalización por tiempo
```python
c_tiempo = -0.01  # cada paso cuesta 0.01 de recompensa

# Si un episodio dura 1000 pasos sin completar:
# Penalización acumulada = -0.01 * 1000 = -10
# Esto castiga la duración total
```

**Intuición:** Más tiempo = más pasos = más recompensa negativa acumulada.

#### 2. Reward por cobertura nueva
```python
c_baldosa = 0.5

# Se aplica SOLO si el robot inspeccionó una celda que no estaba cubierta
# en este paso
if grid_state[cell_x, cell_y] == 0:  # no cubierta
    reward += c_baldosa
    grid_state[cell_x, cell_y] = 1   # marcar como cubierta
```

**Intuición:** Descubrir territorio nuevo vale la pena, pero no compensa quedarse 50 pasos inspeccionando el mismo lugar.

#### 3. Penalización por redundancia
```python
c_redundancia = -0.1

# Si el robot inspecciona una celda que YA estaba cubierta
if grid_state[cell_x, cell_y] == 1:  # ya cubierta
    reward -= c_redundancia
```

**Intuición:** Evita que todos los robots hagan lo mismo (colapso de comportamiento). Fuerza especialización.

#### 4. Penalización por colisión
```python
c_colision = -0.3

# Detectar si hay colisión con otro robot
if distance_to_nearest_robot < collision_threshold:
    reward -= c_colision
```

#### 5. Bonus final al completar la tarea

Se aplica **al último paso del episodio** cuando se completa la cobertura (o se alcanza timeout):

```python
if coverage_percentage >= COVERAGE_TARGET:  # ej. 95% o 100%
    time_ratio = time_elapsed / time_max_episode
    
    # Bonus inversamente proporcional al tiempo
    # Si completas en 100 pasos y max=1000: time_ratio=0.1
    # Si completas en 900 pasos y max=1000: time_ratio=0.9
    
    BONUS_FINAL = 50 * (1 - time_ratio)
    
    # Rápido: BONUS_FINAL = 50 * 0.9 = 45 ✓
    # Lento: BONUS_FINAL = 50 * 0.1 = 5
    
elif time_elapsed >= time_max_episode:  # timeout sin completar
    BONUS_FINAL = -50  # castigo fuerte por no completar
```

### Cálculo de recompensa final (mezcla individual + global)

Para evitar "comportamiento egoísta", mezcla recompensa individual con recompensa de swarm:

```python
def calculate_final_reward(robot_i, r_individual, swarm_metrics):
    """
    Args:
        robot_i: índice del robot
        r_individual: recompensa calculada por ese robot
        swarm_metrics: dict con métricas globales del swarm
    
    Returns:
        r_final: recompensa ajustada
    """
    
    # Recompensa global compartida por todos
    r_global = swarm_metrics["new_cells_this_step"] * 0.5 - 0.01
    
    # Mezcla: α = peso a recompensa individual
    alpha = 0.6  # 60% individual, 40% global
    
    r_final = alpha * r_individual + (1 - alpha) * r_global
    
    return r_final
```

**Ajuste de α:**
- α = 1.0: totalmente egoísta, los robots compiten.
- α = 0.5: equilibrio, cooperación moderada (RECOMENDADO).
- α = 0.0: puramente global, robots pueden no coordinarse bien.

Empieza con α = 0.6 y ajusta según comportamiento observado.

### Resumen de hiperparámetros de recompensa

```python
REWARD_CONFIG = {
    "c_tiempo": -0.01,           # por cada paso
    "c_baldosa": 0.5,            # por descubrir celda nueva
    "c_redundancia": -0.1,       # por inspeccionar celda ya cubierta
    "c_colision": -0.3,          # por colisión
    "bonus_completar_base": 50,  # bonus base al completar
    "castigo_timeout": -50,      # si no completa a tiempo
    "alpha_mix": 0.6,            # peso individual vs global
    "coverage_target": 0.95,     # % de cobertura para considerar "completado"
}
```

---

## FASE 5 — Envolver tu simulador en interfaz estándar

### Estructura mínima del entorno

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SwarmInspectionEnv(gym.Env):
    """
    Entorno Gymnasium para swarm de robots inspeccionando grid.
    
    Soporta múltiples robots con parameter sharing:
    - En cada step, aplica N acciones (una por robot)
    - Retorna N observaciones y N recompensas
    """
    
    def __init__(self, 
                 num_robots=5,
                 grid_width=10,
                 grid_height=10,
                 cell_size=1.0,
                 time_max=1000,
                 velocity_linear=0.25,
                 simulation_timestep=1.0,
                 seed=None):
        """
        Inicializa el ambiente de simulación.
        
        Args:
            num_robots: número de robots en el swarm
            grid_width: ancho del grid en celdas
            grid_height: alto del grid en celdas
            cell_size: tamaño de una celda en metros (típicamente 1.0)
            time_max: tiempo máximo del episodio en pasos de simulación
            velocity_linear: velocidad lineal CONSTANTE (m/s) - del e-puck
            simulation_timestep: duración de cada paso de simulación (segundos)
            seed: semilla para RNG
        """
        
        self.num_robots = num_robots
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.time_max = time_max
        self.velocity_linear = velocity_linear
        self.simulation_timestep = simulation_timestep
        self.current_time = 0
        self.grid_state = None  # matriz de cobertura
        self.swarm_state = None  # posiciones/orientaciones de robots
        
        # IMPORTANTE: un único espacio de observación/acción
        # (válido para todos los robots por parameter sharing)
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(27,),  # tamaño fijo de observación (Fase 2)
            dtype=np.float32
        )
        
        # Espacio de acciones: 3 acciones discretas
        self.action_space = spaces.Discrete(3)
        
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
    
    def reset(self, seed=None):
        """
        Reinicia el entorno completamente.
        
        Procedimiento:
        1. Reiniciar el generador de números aleatorios
        2. Reiniciar el contador de tiempo a 0
        3. Crear matriz de grid (todas las celdas sin inspeccionar)
        4. Posicionar aleatoriamente cada robot en el grid
        5. Calcular observación inicial para cada robot
        6. Retornar observaciones e info
        
        Returns:
            obs_list: lista de N observaciones (una por robot), tipo float32 numpy array
            info: dict vacío {}
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_time = 0
        
        # Crear grid vacío: todas las celdas sin inspeccionar (0 = no inspeccionado)
        self.grid_state = np.zeros((self.grid_width, self.grid_height), dtype=np.int32)
        
        # Inicializar estado del swarm
        self.swarm_state = []
        
        for i in range(self.num_robots):
            # Posición inicial aleatoria dentro del grid
            # Grid tiene tamaño (grid_width * cell_size) x (grid_height * cell_size)
            pos_x = np.random.uniform(0, self.grid_width * self.cell_size)
            pos_y = np.random.uniform(0, self.grid_height * self.cell_size)
            
            # Orientación inicial aleatoria (0 a 2π radianes)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Estructura de datos por robot
            self.swarm_state.append({
                "pos": np.array([pos_x, pos_y], dtype=np.float32),
                "angle": float(angle),
                "ir_sensors": np.zeros(8, dtype=np.float32),  # 8 sensores IR
                "has_inspected_current_cell": False,  # bandera de inspección
            })
        
        # Obtener observaciones iniciales para cada robot
        obs_list = []
        for i in range(self.num_robots):
            obs = self._get_observation(i)
            obs_list.append(obs)
        
        return obs_list, {}
    
    def step(self, actions_list):
        """
        Ejecuta un paso de simulación: aplica acciones, actualiza estado, calcula recompensas.
        
        Procedimiento paso a paso:
        
        1. Incrementar contador de tiempo
        2. Inicializar arrays de recompensas
        3. Aplicar acciones a cada robot (movimiento/rotación)
        4. Detectar inspecciones nuevas y redundancias
        5. Detectar colisiones entre robots
        6. Detectar límites de grid (clipping de posición)
        7. Mezclar recompensas individual + global
        8. Verificar condiciones de terminación
        9. Calcular nuevas observaciones
        10. Retornar todo
        
        Args:
            actions_list: lista de N acciones (int: 0=girar_izq, 1=girar_der, 2=avanzar)
                          Tipo: list[int], longitud N
        
        Returns:
            obs_list: lista de N nuevas observaciones (list[np.array(27,)])
            reward_list: lista de N recompensas por paso (list[float])
            terminated: bool, True si cobertura >= 95% o tiempo >= time_max
            truncated: bool, siempre False (no usamos truncation)
            info: dict con métricas: {"coverage", "time_elapsed", "new_cells", "terminated_reason"}
        """
        
        # 1. Incrementar tiempo
        self.current_time += 1
        
        # 2. Inicializar recompensas
        rewards_individual = np.zeros(self.num_robots, dtype=np.float32)
        
        # Penalización por tiempo: se aplica a cada paso
        time_penalty = -0.01  # c_tiempo
        rewards_individual += time_penalty
        
        # 3. Aplicar acciones y detectar inspecciones
        new_cells_this_step = 0
        cells_already_inspected = 0
        
        for i, action in enumerate(actions_list):
            # Ejecutar movimiento/rotación
            self._execute_action(i, action)
            
            # Detectar en qué celda está el robot después de moverse
            robot_x = self.swarm_state[i]["pos"][0]
            robot_y = self.swarm_state[i]["pos"][1]
            cell_x = int(robot_x / self.cell_size)
            cell_y = int(robot_y / self.cell_size)
            
            # IMPORTANTE: solo contar inspección si está dentro del grid
            if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                # Verificar si esta es una celda nueva
                if self.grid_state[cell_x, cell_y] == 0:
                    # PRIMERA VEZ que se inspecciona esta celda
                    self.grid_state[cell_x, cell_y] = 1
                    rewards_individual[i] += 0.5  # c_baldosa = +0.5
                    new_cells_this_step += 1
                else:
                    # Celda ya había sido inspeccionada
                    # Penalizar redundancia
                    rewards_individual[i] -= 0.1  # c_redundancia = -0.1
                    cells_already_inspected += 1
        
        # 4. Detectar colisiones entre robots
        collision_pairs = []
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(
                    self.swarm_state[i]["pos"] - self.swarm_state[j]["pos"]
                )
                collision_threshold = 0.5  # metros
                if dist < collision_threshold:
                    # Ambos robots reciben penalización
                    rewards_individual[i] -= 0.3  # c_colision = -0.3
                    rewards_individual[j] -= 0.3
                    collision_pairs.append((i, j))
        
        # 5. Clipping de posiciones (limitar al grid)
        for i in range(self.num_robots):
            self.swarm_state[i]["pos"][0] = np.clip(
                self.swarm_state[i]["pos"][0],
                0,
                self.grid_width * self.cell_size
            )
            self.swarm_state[i]["pos"][1] = np.clip(
                self.swarm_state[i]["pos"][1],
                0,
                self.grid_height * self.cell_size
            )
        
        # 6. Mezclar recompensas individual + global
        # Recompensa global: células nuevas descubiertas por todo el swarm
        r_global = new_cells_this_step * 0.5 - 0.01
        alpha = 0.6  # 60% individual, 40% global
        
        rewards_final = alpha * rewards_individual + (1 - alpha) * r_global
        
        # 7. Verificar condiciones de terminación
        coverage = np.sum(self.grid_state) / (self.grid_width * self.grid_height)
        terminated = False
        bonus_final = 0.0
        terminated_reason = None
        
        if coverage >= 0.95:  # 95% de cobertura alcanzado
            terminated = True
            # Bonus inversamente proporcional al tiempo
            time_ratio = self.current_time / self.time_max
            bonus_final = 50 * (1 - time_ratio)  # rápido: ~50, lento: ~0
            terminated_reason = "coverage"
        
        elif self.current_time >= self.time_max:  # timeout
            terminated = True
            bonus_final = -50.0  # castigo por no completar
            terminated_reason = "timeout"
        
        # Aplicar bonus final solo al terminar
        if terminated:
            rewards_final = rewards_final + bonus_final
        
        # 8. Calcular nuevas observaciones
        obs_list = []
        for i in range(self.num_robots):
            obs = self._get_observation(i)
            obs_list.append(obs)
        
        # 9. Preparar información de debug
        info = {
            "coverage": float(coverage),
            "time_elapsed": int(self.current_time),
            "time_max": int(self.time_max),
            "new_cells": int(new_cells_this_step),
            "cells_redundant": int(cells_already_inspected),
            "collisions": len(collision_pairs),
            "terminated_reason": terminated_reason,
        }
        
        # 10. Retornar
        return obs_list, rewards_final.tolist(), terminated, False, info
    
    def _execute_action(self, robot_idx, action):
        """
        Ejecuta una acción discreta sobre un robot específico.
        
        Acciones disponibles:
        - 0: Girar izquierda (contrareloj) 45°
        - 1: Girar derecha (reloj) 45°
        - 2: Avanzar e inspeccionar (mover una celda completa)
        
        Parámetros fijos:
        - turn_angle = 45° = π/4 radianes
        - velocity_linear = velocidad lineal constante (m/s)
        - simulation_timestep = 1.0 segundo
        - cell_size = 1.0 metro (tamaño de una celda)
        
        El movimiento es relativo a la orientación actual del robot.
        
        Args:
            robot_idx: índice del robot en swarm_state (0 a N-1)
            action: int (0=girar_izq, 1=girar_der, 2=avanzar)
        
        Returns:
            None (modifica swarm_state[robot_idx] in-place)
        """
        
        turn_angle = np.pi / 4  # 45 grados en radianes
        velocity_linear = self.velocity_linear  # m/s (parámetro del ambiente)
        simulation_timestep = self.simulation_timestep  # segundos (parámetro del ambiente)
        
        if action == 0:  # Girar izquierda
            # Incrementar ángulo (rotación contrareloj)
            self.swarm_state[robot_idx]["angle"] += turn_angle
            # Normalizar ángulo a [0, 2π)
            self.swarm_state[robot_idx]["angle"] = (
                self.swarm_state[robot_idx]["angle"] % (2 * np.pi)
            )
            # Posición NO cambia
        
        elif action == 1:  # Girar derecha
            # Decrementar ángulo (rotación reloj)
            self.swarm_state[robot_idx]["angle"] -= turn_angle
            # Normalizar ángulo a [0, 2π)
            self.swarm_state[robot_idx]["angle"] = (
                self.swarm_state[robot_idx]["angle"] % (2 * np.pi)
            )
            # Posición NO cambia
        
        elif action == 2:  # Avanzar e inspeccionar
            # Calcular desplazamiento basado en velocidad constante
            # desplazamiento = velocidad * tiempo
            displacement = velocity_linear * simulation_timestep
            
            # Dirección actual del robot
            angle = self.swarm_state[robot_idx]["angle"]
            
            # Calcular componentes de desplazamiento
            dx = displacement * np.cos(angle)
            dy = displacement * np.sin(angle)
            
            # Actualizar posición
            self.swarm_state[robot_idx]["pos"][0] += dx
            self.swarm_state[robot_idx]["pos"][1] += dy
            
            # Nota: dx ≈ 0.25 m, dy ≈ 0.25 m en este timestep
            # Para recorrer una celda completa (1.0 m) se necesitarán ~4 pasos
            # La detección de nueva inspección ocurre en step(), comparando
            # la celda actual del robot con el grid de cobertura
        
        # IMPORTANTE: No hacer clipping aquí
        # El clipping se hace en step() después de actualizar todos los robots
        # Esto evita que un robot "rebote" contra el límite antes de que otros se muevan
    
    def _get_observation(self, robot_idx):
        """
        Calcula observación para un robot específico.
        
        Retorna un vector de 27 valores normalizados, correspondiente a:
        - Posición (2): x, y normalizados
        - Orientación (2): sin(θ), cos(θ)
        - Sensores IR (8): lecturas de 8 sensores de proximidad
        - Mapa local 3×3 (9): estado de cobertura de celdas adyacentes
        - Cobertura global (1): porcentaje de grid cubierto
        - Tiempo (1): tiempo transcurrido normalizado
        - Posiciones relativas (4): distancia y ángulo a 2 robots más cercanos
        
        Total: 2+2+8+9+1+1+4 = 27 valores
        
        IMPORTANTE:
        - TODOS los valores deben estar en rango [-1, 1] o [0, 1]
        - Si un valor no existe (ej. no hay 2 robots cercanos), rellenar con 0.0
        - El orden DEBE ser exactamente el indicado (crítico para parameter sharing)
        - Retornar como np.array(27,) dtype=float32
        
        Args:
            robot_idx: índice del robot (0 a N-1)
        
        Returns:
            observation: np.array de shape (27,) dtype=float32
        """
        
        obs = np.zeros(27, dtype=np.float32)
        idx = 0
        
        # 1. Posición normalizada (2 valores)
        # Normalizar respecto a los límites del grid
        obs[idx] = self.swarm_state[robot_idx]["pos"][0] / (self.grid_width * self.cell_size)
        idx += 1
        obs[idx] = self.swarm_state[robot_idx]["pos"][1] / (self.grid_height * self.cell_size)
        idx += 1
        # Rango: [0, 1]
        
        # 2. Orientación: sin y cos (2 valores)
        # Evita discontinuidad en 0°/360°, está naturalmente en [-1, 1]
        theta = self.swarm_state[robot_idx]["angle"]
        obs[idx] = np.sin(theta)
        idx += 1
        obs[idx] = np.cos(theta)
        idx += 1
        # Rango: [-1, 1]
        
        # 3. Sensores IR (8 valores)
        # Lectura normalizada: ir_reading / MAX_RANGE
        ir_max_range = 10.0  # metros (rango máximo de sensores)
        for i in range(8):
            obs[idx] = self.swarm_state[robot_idx]["ir_sensors"][i] / ir_max_range
            idx += 1
        # Rango: [0, 1]
        
        # 4. Mapa local 3×3 (9 valores)
        # Ventana centrada en la posición actual del robot
        robot_cell_x = int(self.swarm_state[robot_idx]["pos"][0] / self.cell_size)
        robot_cell_y = int(self.swarm_state[robot_idx]["pos"][1] / self.cell_size)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_x = robot_cell_x + dx
                cell_y = robot_cell_y + dy
                
                # Verificar si está dentro del grid
                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    # 1 = ya inspeccionada, 0 = no inspeccionada
                    obs[idx] = float(self.grid_state[cell_x, cell_y])
                else:
                    # Fuera de límites: representa como -1 (distinguible de 0/1)
                    obs[idx] = -1.0
                
                idx += 1
        # Rango: [-1, 0, 1]
        
        # 5. Cobertura global (1 valor)
        # Porcentaje de baldosas ya cubiertas
        total_cells = self.grid_width * self.grid_height
        covered_cells = np.sum(self.grid_state)
        obs[idx] = covered_cells / total_cells
        idx += 1
        # Rango: [0, 1]
        
        # 6. Tiempo transcurrido normalizado (1 valor)
        obs[idx] = self.current_time / self.time_max
        idx += 1
        # Rango: [0, 1]
        
        # 7. Posiciones relativas de 2 robots más cercanos (4 valores)
        # Calcular distancias a todos los otros robots
        distances_to_robots = []
        for j in range(self.num_robots):
            if j != robot_idx:
                dx = self.swarm_state[j]["pos"][0] - self.swarm_state[robot_idx]["pos"][0]
                dy = self.swarm_state[j]["pos"][1] - self.swarm_state[robot_idx]["pos"][1]
                dist = np.sqrt(dx**2 + dy**2)
                angle_to_robot = np.arctan2(dy, dx)
                distances_to_robots.append((dist, angle_to_robot, j))
        
        # Ordenar por distancia
        distances_to_robots.sort(key=lambda x: x[0])
        
        # Tomar los 2 más cercanos
        robot_sensor_range = 10.0  # metros
        for k in range(2):  # 2 robots más cercanos
            if k < len(distances_to_robots):
                dist, angle_to_robot, _ = distances_to_robots[k]
                # Normalizar distancia
                obs[idx] = (dist / robot_sensor_range)
                idx += 1
                # Normalizar ángulo a [-1, 1] (dividir π)
                obs[idx] = (angle_to_robot / np.pi)
                idx += 1
            else:
                # No hay suficientes robots: rellenar con 0
                obs[idx] = 0.0
                idx += 1
                obs[idx] = 0.0
                idx += 1
        # Rango: [0, 1] para distancia, [-1, 1] para ángulo
        
        # VERIFICACIÓN: el índice final debe ser exactamente 27
        assert idx == 27, f"Error: observación tiene {idx} valores, esperados 27"
        
        return obs
    
    def render(self, mode='human'):
        """Visualización (opcional)."""
        pass
```

---

## FASE 5b — Detalles críticos de implementación del entorno

### Orden de operaciones en step()

**Es CRÍTICO que hagas esto en este ORDEN exacto**, de lo contrario los cálculos de recompensa serán incorrectos:

1. **Incrementar tiempo** (`self.current_time += 1`)
2. **Aplicar acciones** (mover/rotar robots)
3. **Detectar inspecciones** (después de moverse, ver qué celda ocupan)
4. **Detectar colisiones** (distancias entre robots)
5. **Clipping de límites** (contener robots dentro del grid)
6. **Calcular recompensas** (mezclar individual + global)
7. **Verificar terminación** (¿cobertura >= 95%? ¿timeout?)
8. **Calcular observaciones** (las nuevas observaciones después de todo)
9. **Retornar** todo

**RAZÓN:** Si calculas observaciones ANTES de mover los robots, verán el estado anterior. Si calculas recompensas ANTES de detectar inspecciones, obtendrás recompensas incorrectas.

### Detección de inspección de celdas

Cuando un robot se mueve, es posible que cruce MÚLTIPLES celdas en un solo paso (si velocity_linear es alta). Debes detectar **todas** las celdas que cruzó:

```python
# ✗ INCORRECTO: solo detectar la celda actual
cell_x = int(robot_pos_x / cell_size)
cell_y = int(robot_pos_y / cell_size)
if grid_state[cell_x, cell_y] == 0:
    # marcar
    
# ✓ CORRECTO: usar Bresenham o raycasting para detectar TODAS las celdas
# Pero para velocity_linear=0.25 m/s y cell_size=1.0 m,
# es improbable cruzar más de una celda por paso
# Así que es suficiente verificar la celda actual en cada paso
```

Para este proyecto con velocity_linear=0.25 m/s, la celda actual es suficiente.

### Manejo de límites del grid

```python
# En step(), DESPUÉS de aplicar todas las acciones:
for i in range(self.num_robots):
    self.swarm_state[i]["pos"][0] = np.clip(
        self.swarm_state[i]["pos"][0],
        0,
        self.grid_width * self.cell_size
    )
    self.swarm_state[i]["pos"][1] = np.clip(
        self.swarm_state[i]["pos"][1],
        0,
        self.grid_height * self.cell_size
    )

# IMPORTANTE:
# - No "penalices" el robot por chocar contra el límite
# - Solo hazlo "rebotar" o quedarse en el límite (clipping)
# - La penalización de colisión solo aplica para robot-robot, no robot-pared
```

### Cálculo de recompensa por redundancia

```python
# Cada celda inspeccionada debe ser contada UNA VEZ
# Si dos robots inspeccionan la misma celda en pasos diferentes:

if grid_state[cell_x, cell_y] == 0:  # nunca fue inspeccionada
    grid_state[cell_x, cell_y] = 1
    reward += 0.5  # c_baldosa
else:  # ya fue inspeccionada en un paso anterior
    reward -= 0.1  # c_redundancia

# El grid_state[cell_x, cell_y] se mantiene como 1 para siempre
# Esto evita contar dos veces
```

### Normalización de ángulos

Después de cualquier rotación, normaliza el ángulo a [0, 2π):

```python
angle = angle % (2 * np.pi)  # Siempre en [0, 2π)
if angle < 0:
    angle += 2 * np.pi
```

Esto es importante para evitar ángulos negativos o > 2π, que pueden causar problemas numéricos.

---

## FASE 6 — Arquitectura de la red neuronal (política compartida)

### Descripción general

```
┌─────────────────────────────┐
│    Input vector (27)        │
└────────────┬────────────────┘
             │
    ┌────────▼─────────┐
    │  Dense(128)      │
    │  + ReLU          │
    │  + LayerNorm     │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Dense(128)      │
    │  + ReLU          │
    │  + LayerNorm     │
    └────────┬─────────┘
             │
        ┌────┴────┐
        │          │
   ┌────▼────┐ ┌──▼───────┐
   │ Actor   │ │ Critic   │
   │ Head    │ │ Head     │
   └────┬────┘ └──┬───────┘
        │         │
   ┌────▼────┐ ┌──▼───────┐
   │Dense(4) │ │ Dense(1) │
   │Softmax  │ │           │
   └─────────┘ └───────────┘
        │         │
   Π(a|s)      V(s)
   (política)  (valor)
```

### Implementación con PyTorch

```python
import torch
import torch.nn as nn

class SharedActorCriticPolicy(nn.Module):
    """
    Red neuronal con arquitectura actor-crítico compartida.
    
    Características:
    - Capas compartidas: Dense(128) + Dense(128)
    - Actor head: Dense(3) → Softmax (3 acciones: girar_izq, girar_der, avanzar)
    - Critic head: Dense(1) → valor escalar
    - LayerNorm después de cada activación para estabilidad
    
    Válida para parameter sharing: misma red para todos los robots.
    """
    
    def __init__(self, obs_dim=27, action_dim=3, hidden_dim=128):
        super().__init__()
        
        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Cabeza de actor (política)
        # Retorna logits para las 3 acciones
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            # IMPORTANTE: Softmax se aplica en get_action_and_value, no aquí
        )
        
        # Cabeza de crítico (estimador de valor)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        """
        Forward pass de la red actor-crítico.
        
        Args:
            obs: tensor de shape (batch_size, 27) o (27,)
                 Observaciones para uno o múltiples robots
        
        Returns:
            action_logits: (batch_size, 3) - logits para las 3 acciones
            state_value: (batch_size,) - valores predichos de estado
        """
        
        # Si es observación individual, agregar batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Capas compartidas
        shared_features = self.shared(obs)
        
        # Políticas y valor
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_logits, state_value.squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        """
        Usado durante rollout e inferencia.
        
        Args:
            obs: observación(es)
            action: (opcional) acción para la que calcular log_prob
        
        Returns:
            action: acción muestreada (si action=None)
            log_prob: log-probabilidad de la acción
            value: valor de estado
            entropy: entropía de la distribución (para regularización)
        """
        
        logits, value = self.forward(obs)
        
        # Distribución de probabilidad
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Muestrear acción si no se proporciona
        if action is None:
            action = dist.sample()
        
        # Log-probabilidad de la acción
        log_prob = dist.log_prob(action)
        
        # Entropía (para fomentar exploración)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy
```

### Parámetros iniciales

```python
model = SharedActorCriticPolicy(
    obs_dim=27,        # tamaño observación
    action_dim=4,      # acciones discretas
    hidden_dim=128     # capas ocultas
)

# Optimizador Adam
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

---

## FASE 7 — Algoritmo PPO con Parameter Sharing

### Estructura general del entrenamiento

```
┌──────────────────────────────────────────┐
│      LOOP DE ENTRENAMIENTO PPO           │
└──────────────────────────────────────────┘
         │
         ├─→ [ROLLOUT COLLECTION]
         │   Ejecutar ambiente con política actual
         │   Guardar (obs, acción, reward, done) × N robots
         │   Duración: T pasos (ej. T=2048)
         │
         ├─→ [CALCULATE ADVANTAGES]
         │   Usar GAE para calcular ventajas (A_t)
         │   Calcular "returns" (G_t)
         │
         ├─→ [PPO UPDATES]
         │   Para K épocas (ej. K=10):
         │     - Mini-batches del buffer de rollout
         │     - Calcular pérdidas (clip, valor, entropía)
         │     - Backprop + optimizer.step()
         │
         └─→ Repetir
```

### 7.1 Recolección de experiencia (Rollout)

```python
def collect_rollout(env, policy, num_steps=2048):
    """
    Ejecuta el ambiente num_steps pasos.
    Guarda experiencia de todos los robots.
    
    Returns:
        buffer: dict con listas de estados, acciones, recompensas, etc.
    """
    
    obs_list, _ = env.reset()
    
    buffer = {
        "observations": [],      # lista de observaciones
        "actions": [],           # lista de acciones
        "log_probs": [],         # log-probabilidades
        "values": [],            # valores predichos
        "rewards": [],           # recompensas
        "dones": [],             # flags de terminación
        "advantages": [],        # se calcula después
        "returns": [],           # se calcula después
    }
    
    for step in range(num_steps):
        # Convertir observaciones a tensores
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
        
        # Forward pass en la red
        with torch.no_grad():
            actions, log_probs, values, _ = policy.get_action_and_value(obs_tensor)
        
        # Ejecutar acciones en ambiente
        actions_np = actions.cpu().numpy()
        obs_list, rewards, terminated, truncated, info = env.step(actions_np.tolist())
        
        # Guardar en buffer
        # IMPORTANTE: guardar observación de ANTES de la acción
        for i in range(len(obs_list)):
            buffer["observations"].append(obs_tensor[i].cpu().numpy())
            buffer["actions"].append(actions_np[i])
            buffer["log_probs"].append(log_probs[i].item())
            buffer["values"].append(values[i].item())
            buffer["rewards"].append(rewards[i])
            buffer["dones"].append(terminated)
        
        # Si episodio terminó, resetear ambiente
        if terminated:
            obs_list, _ = env.reset()
    
    return buffer
```

### 7.2 Calcular ventajas con GAE (Generalized Advantage Estimation)

```python
def calculate_advantages(buffer, gamma=0.99, lambda_gae=0.95):
    """
    Calcula ventajas A_t y retornos G_t usando GAE.
    
    Fórmula:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
        G_t = A_t + V(s_t)
    
    Args:
        buffer: dict con rewards, values, dones
        gamma: factor de descuento
        lambda_gae: parámetro de GAE
    
    Returns:
        buffer actualizado con "advantages" y "returns"
    """
    
    n_steps = len(buffer["rewards"])
    advantages = np.zeros(n_steps)
    gae = 0
    next_value = 0  # valor del siguiente estado (0 si terminal)
    
    # Recorrer hacia atrás
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = 0 if buffer["dones"][t] else 0
        else:
            next_value = buffer["values"][t + 1]
        
        # Temporal difference error
        delta = buffer["rewards"][t] + gamma * next_value - buffer["values"][t]
        
        # Accumulator de GAE
        gae = delta + gamma * lambda_gae * gae * (1 - buffer["dones"][t])
        advantages[t] = gae
    
    returns = advantages + np.array(buffer["values"])
    
    buffer["advantages"] = advantages.tolist()
    buffer["returns"] = returns.tolist()
    
    # Normalizar ventajas (estabilidad numérica)
    advantages_np = np.array(buffer["advantages"])
    buffer["advantages"] = (
        (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
    ).tolist()
    
    return buffer
```

### 7.3 Actualizar la red con pérdida PPO

```python
def ppo_update(policy, buffer, optimizer, 
               n_epochs=10, batch_size=64, clip_coef=0.2, 
               value_coef=0.5, entropy_coef=0.01):
    """
    Actualiza la red neuronal con PPO.
    
    Pérdida total:
        L = L_clip - c1 * L_valor + c2 * L_entropía
    
    Args:
        policy: red neuronal (SharedActorCriticPolicy)
        buffer: dict con observaciones, acciones, ventajas, etc.
        optimizer: torch optimizer (Adam)
        n_epochs: épocas de entrenamiento
        batch_size: tamaño de mini-batch
        clip_coef: coeficiente ε de clipping (típ. 0.2)
        value_coef: peso de pérdida de valor
        entropy_coef: peso de regularización de entropía
    """
    
    n_samples = len(buffer["observations"])
    old_log_probs = np.array(buffer["log_probs"])
    
    for epoch in range(n_epochs):
        # Mezclar índices
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Mini-batch
            obs_batch = torch.tensor(
                np.array([buffer["observations"][i] for i in batch_indices]),
                dtype=torch.float32
            )
            actions_batch = torch.tensor(
                np.array([buffer["actions"][i] for i in batch_indices]),
                dtype=torch.long
            )
            old_log_probs_batch = torch.tensor(
                old_log_probs[batch_indices],
                dtype=torch.float32
            )
            advantages_batch = torch.tensor(
                np.array([buffer["advantages"][i] for i in batch_indices]),
                dtype=torch.float32
            )
            returns_batch = torch.tensor(
                np.array([buffer["returns"][i] for i in batch_indices]),
                dtype=torch.float32
            )
            
            # Forward pass
            action_logits, values = policy.forward(obs_batch)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()
            
            # Ratio de probabilidades
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages_batch
            loss_clip = -torch.min(surr1, surr2).mean()
            
            # Pérdida de valor
            values_pred = values.squeeze(-1)
            loss_value = ((values_pred - returns_batch) ** 2).mean()
            
            # Pérdida de entropía (regularización)
            loss_entropy = -entropy.mean()
            
            # Pérdida total
            loss = loss_clip + value_coef * loss_value + entropy_coef * loss_entropy
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / (n_samples // batch_size)
        print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
```

### 7.4 Loop principal de entrenamiento

```python
def train_ppo_swarm(env, policy, optimizer, 
                    total_timesteps=100000,
                    rollout_steps=2048,
                    log_interval=10):
    """
    Loop principal de entrenamiento PPO.
    
    Args:
        env: ambiente (SwarmInspectionEnv)
        policy: red neuronal
        optimizer: optimizer PyTorch
        total_timesteps: duración total de entrenamiento
        rollout_steps: pasos por rollout
        log_interval: cada cuántos rollouts loguear
    """
    
    num_rollouts = total_timesteps // rollout_steps
    
    history = {
        "cumulative_reward": [],
        "coverage": [],
        "time_taken": [],
        "success_rate": [],
    }
    
    for rollout_idx in range(num_rollouts):
        # 1. Recolectar experiencia
        buffer = collect_rollout(env, policy, num_steps=rollout_steps)
        
        # 2. Calcular ventajas
        buffer = calculate_advantages(buffer, gamma=0.99, lambda_gae=0.95)
        
        # 3. Actualizar red
        ppo_update(
            policy, buffer, optimizer,
            n_epochs=10,
            batch_size=64,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01
        )
        
        # 4. Loguear métricas
        if (rollout_idx + 1) % log_interval == 0:
            avg_reward = np.mean([r for rewards in buffer["rewards"] 
                                 for r in rewards])
            print(f"Rollout {rollout_idx + 1}/{num_rollouts}: "
                  f"Avg Reward = {avg_reward:.4f}")
            
            # Guardar histórico (opcional: más adelante con TensorBoard)
            history["cumulative_reward"].append(avg_reward)
    
    return policy, history
```

### Hiperparámetros recomendados

```python
PPO_CONFIG = {
    # Entrenamiento
    "learning_rate": 3e-4,
    "n_epochs": 10,
    "batch_size": 64,
    
    # PPO clipping
    "clip_coef": 0.2,
    
    # Pérdidas
    "value_coef": 0.5,      # peso de pérdida de valor
    "entropy_coef": 0.01,   # peso de regularización
    
    # GAE
    "gamma": 0.99,          # factor de descuento
    "lambda_gae": 0.95,     # parámetro GAE
    
    # Rollout
    "rollout_steps": 2048,  # pasos por recolección
    "total_timesteps": 100000,  # total de pasos de entrenamiento
}
```

---

## FASE 8 — Curriculum Learning (Acelera convergencia)

### Estrategia de escalado progresivo

La dificultad debe aumentar **gradualmente** para que la política tenga puntos de apoyo intermedios.

#### Etapa 1: Exploración básica (1 robot, grid pequeño)

```python
ETAPA_1 = {
    "num_robots": 1,
    "grid_width": 5,
    "grid_height": 5,
    "num_obstacles": 0,
    "time_max": 200,
    "duration_steps": 10000,  # entrenar 10K pasos en esta etapa
}

# Objetivo: que 1 robot aprenda a cubrir sin chocar
# Métrica de éxito: > 90% cobertura en < 150 pasos promedio
```

#### Etapa 2: Introducir cooperación (2-3 robots, mismo grid)

```python
ETAPA_2 = {
    "num_robots": 3,
    "grid_width": 5,
    "grid_height": 5,
    "num_obstacles": 0,
    "time_max": 250,
    "duration_steps": 15000,
}

# Objetivo: robots aprenden a no pisarse
# Métrica: < 10% de redundancia (celdas revisadas múltiples veces)
```

#### Etapa 3: Grid más grande, swarm completo

```python
ETAPA_3 = {
    "num_robots": 5,
    "grid_width": 10,
    "grid_height": 10,
    "num_obstacles": 0,
    "time_max": 500,
    "duration_steps": 20000,
}

# Objetivo: coordinación en área más grande
# Métrica: > 95% en < 400 pasos
```

#### Etapa 4: Domain randomization (variabilidad, realismo)

```python
ETAPA_4 = {
    "num_robots": 5,
    "grid_width_range": (8, 12),      # tamaño variable
    "grid_height_range": (8, 12),
    "num_obstacles": (2, 5),          # obstáculos aleatorios
    "time_max_range": (400, 600),     # tiempo máximo variable
    "randomize_initial_positions": True,
    "sensor_noise": 0.05,             # ruido en sensores IR
    "duration_steps": 25000,
}

# Objetivo: robustez a variaciones del entorno
```

### Implementación del curriculum

```python
def run_curriculum_training(policy, optimizer, num_stages=4):
    """
    Ejecuta entrenamiento en múltiples etapas de dificultad.
    """
    
    stages_config = [ETAPA_1, ETAPA_2, ETAPA_3, ETAPA_4]
    
    for stage_idx, stage in enumerate(stages_config):
        print(f"\n{'='*50}")
        print(f"ETAPA {stage_idx + 1}/{num_stages}")
        print(f"{'='*50}\n")
        
        # Crear ambiente con config de esta etapa
        env = SwarmInspectionEnv(
            num_robots=stage["num_robots"],
            grid_width=stage.get("grid_width", 10),
            grid_height=stage.get("grid_height", 10),
        )
        
        # Entrenar
        policy, history = train_ppo_swarm(
            env,
            policy,
            optimizer,
            total_timesteps=stage["duration_steps"],
            rollout_steps=2048,
            log_interval=5,
        )
        
        # Evaluar si está listo para siguiente etapa
        avg_success = np.mean(history["success_rate"][-10:])
        if avg_success < 0.7:  # si < 70% de éxito, entrenar más
            print(f"Etapa {stage_idx + 1} no convergió aún ({avg_success:.1%}). "
                  f"Continuando...")
            # Entrenar más pasos en esta etapa
    
    return policy
```

---

## FASE 9 — Métricas de Monitoreo

### Herramientas

Usa **TensorBoard** para visualizar en tiempo real:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs/swarm_rl")

# En el loop de entrenamiento:
writer.add_scalar("reward/avg_episode", avg_reward, global_step)
writer.add_scalar("coverage/percent", coverage_pct, global_step)
writer.add_scalar("time/episode_length", episode_length, global_step)
writer.add_scalar("loss/policy", loss_clip, global_step)
writer.add_scalar("loss/value", loss_value, global_step)

writer.flush()
```

Lanza tensorboard:
```bash
tensorboard --logdir=./logs
```

### Métricas críticas a registrar

| Métrica | Descripción | Meta |
|---------|-------------|------|
| **Reward promedio** | Suma de recompensas por episodio | ↑ Debe crecer |
| **% Cobertura** | Porcentaje de baldosas cubiertas | ↑ Debe llegar a 95%+ |
| **Tiempo promedio** | Pasos hasta completar | ↓ Debe bajar |
| **Tasa de colisiones** | Colisiones por episodio | ↓ Debe bajar |
| **Redundancia** | % de celdas revisadas 2+ veces | ↓ Debe bajar |
| **Entropía de política** | Exploración vs explotación | ✓ Debe decrecer gradualmente |
| **Loss de política** | Magnitud de cambios de red | ✓ Debe estabilizarse |
| **Loss de valor** | Error en predicción de valor | ✓ Debe disminuir |

### Código de evaluación

```python
def evaluate_policy(env, policy, num_episodes=10, render=False):
    """
    Evalúa la política entrenada.
    
    Returns:
        resultados: dict con métricas agregadas
    """
    
    rewards_per_episode = []
    coverage_per_episode = []
    time_per_episode = []
    collisions_per_episode = []
    
    for ep in range(num_episodes):
        obs_list, _ = env.reset()
        episode_reward = 0
        episode_collisions = 0
        
        for step in range(env.time_max):
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
            
            with torch.no_grad():
                actions, _, _, _ = policy.get_action_and_value(obs_tensor)
            
            obs_list, rewards, terminated, truncated, info = env.step(
                actions.cpu().numpy().tolist()
            )
            
            episode_reward += np.sum(rewards)
            
            if terminated:
                break
        
        rewards_per_episode.append(episode_reward)
        coverage_per_episode.append(info["coverage"])
        time_per_episode.append(step)
        
        if render:
            print(f"Episode {ep}: Reward={episode_reward:.2f}, "
                  f"Coverage={info['coverage']:.1%}, Time={step}")
    
    return {
        "avg_reward": np.mean(rewards_per_episode),
        "avg_coverage": np.mean(coverage_per_episode),
        "avg_time": np.mean(time_per_episode),
        "success_rate": np.mean([c >= 0.95 for c in coverage_per_episode]),
    }
```

---

## FASE 10 — Errores comunes (y cómo evitarlos)

### ❌ Error 1: Reward scaling desbalanceado

**Síntoma:** Robot aprende a no moverse (penalización de tiempo es muy fuerte).

**Causa:** `c_baldosa < c_tiempo * expected_steps_per_cell`

**Solución:**
```python
# Si c_tiempo = -0.01 y tardas 10 pasos en cubrir una celda:
# Penalización acumulada = 0.01 * 10 = 0.1

# Debe cumplirse: c_baldosa >> 0.1
c_baldosa = 0.5  # > 0.1 ✓

# Test: compara reward de "cubrir algo" vs "no hacer nada durante 10 pasos"
reward_cover = 0.5 + (-0.01 * 10) = 0.4  # positivo ✓
reward_idle = -0.01 * 10 = -0.1          # negativo ✓
```

### ❌ Error 2: Observación inconsistente entre robots

**Síntoma:** Error en forward pass de red, dimensiones no coinciden.

**Causa:** Observaciones de distintos robots tienen tamaños diferentes (ej. uno ve 8 robots cercanos, otro solo 2).

**Solución:** **SIEMPRE** retornar observación con tamaño fijo, rellenar con ceros si no hay información:

```python
# ✓ Correcto
obs = np.zeros(27)  # tamaño FIJO
if nearest_robot_1_exists:
    obs[20:22] = [dist1, angle1]
else:
    obs[20:22] = [0.0, 0.0]  # rellenar

# ❌ Incorrecto
obs = []
if nearest_robot_1_exists:
    obs.extend([dist1, angle1])
# obs puede tener tamaño variable
```

### ❌ Error 3: Mezclar trayectorias en GAE

**Síntoma:** Valores extraños de ventaja, entrenamiento inestable.

**Causa:** Calcular GAE como si fuera una secuencia continua de todos los robots.

**Solución:** Calcular GAE **por robot independientemente**, usando `done` para truncar:

```python
# ✓ Correcto: truncar GAE en terminación
for t in reversed(range(n_steps)):
    if buffer["dones"][t]:
        gae = 0  # RESET en terminal state
    
    delta = buffer["rewards"][t] + gamma * next_value - buffer["values"][t]
    gae = delta + gamma * lambda_gae * gae
```

### ❌ Error 4: Episodios sin límite temporal en etapas tempranas

**Síntoma:** Entrenamiento muy lento, la red converge a políticas de "esperar".

**Causa:** Sin límite de tiempo, el robot puede aprender que lo peor es fracasar, así que se queda quieto indefinidamente.

**Solución:** Define `time_max` desde el inicio, ajustable por etapa:

```python
# Etapa 1: más tiempo para explorar
time_max = 200  # segundos simulados

# Etapa 4: más restrictivo
time_max = 400
```

### ❌ Error 5: Entropy coefficient muy alto o muy bajo

**Síntoma:**
- Muy alto (> 0.1): política casi aleatoria, no converge.
- Muy bajo (< 0.001): colapsa a política determinística, no explora.

**Solución:** Usar decay durante el entrenamiento:

```python
def get_entropy_coef(global_step, total_steps, start=0.01, end=0.0):
    """Decay exponencial de entropy coefficient."""
    return end + (start - end) * np.exp(-5 * global_step / total_steps)

entropy_coef = get_entropy_coef(rollout_idx, num_rollouts)
```

### ❌ Error 6: Reward compartido 100% (α=0)

**Síntoma:** Robots "perezosos", algunos dejan que otros hagan todo.

**Causa:** Sin incentivo individual, un robot puede "vaguear" sin costo.

**Solución:** Mantener α ≥ 0.5:

```python
alpha = 0.6  # 60% individual, 40% global
# Esto asegura que cada robot tiene incentivo personal de ser útil
```

---

## FASE 11 — Checklist de Implementación

### ✅ Paso 1: Diseño (sin código)
- [ ] Definir observación (lista de componentes, tamaño total)
- [ ] Definir acciones (discretas o continuas)
- [ ] Definir recompensa (fórmula completa, hiperparámetros)

### ✅ Paso 2: Ambiente base
- [ ] Implementar `reset()`
- [ ] Implementar `step(actions)`
- [ ] Probar con acciones aleatorias (verificar que outputs tengan sentido)

### ✅ Paso 3: Red neuronal
- [ ] Arquitectura actor-crítico
- [ ] Forward pass
- [ ] Get action & value

### ✅ Paso 4: PPO loop
- [ ] Rollout collection
- [ ] GAE calculation
- [ ] PPO update

### ✅ Paso 5: Entrenamiento Etapa 1
- [ ] 1 robot, grid 5×5, sin obstáculos
- [ ] Entrenar 10K pasos
- [ ] Validar: robot debe cubrir > 90% en < 150 pasos

### ✅ Paso 6: Etapas 2-4
- [ ] Aumentar complejidad gradualmente
- [ ] Evaluar métricas antes de avanzar
- [ ] Ajustar hiperparámetros según comportamiento

### ✅ Paso 7: Evaluación final
- [ ] Test en 10+ episodios con política fija (no explorando)
- [ ] Registrar: cobertura, tiempo, colisiones
- [ ] Comparar contra baseline (ej. exploración aleatoria)

---

## FASE 12 — Código de ejemplo (estructura completa)

### Script principal

```python
#!/usr/bin/env python3
"""
main.py - Entrenamiento de swarm RL con PPO
"""

import torch
import numpy as np
from datetime import datetime

# Importar clases del proyecto
from swarm_env import SwarmInspectionEnv
from policy_network import SharedActorCriticPolicy
from ppo_trainer import PPOTrainer
from curriculum import CurriculumLearning

def main():
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Crear política
    policy = SharedActorCriticPolicy(obs_dim=27, action_dim=4, hidden_dim=128)
    policy.to(DEVICE)
    
    # Crear optimizador
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Crear trainer
    trainer = PPOTrainer(
        policy=policy,
        optimizer=optimizer,
        device=DEVICE,
        config={
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lambda_gae": 0.95,
            "n_epochs": 10,
            "batch_size": 64,
        }
    )
    
    # Crear curriculum
    curriculum = CurriculumLearning()
    
    # Entrenar en todas las etapas
    for stage_idx in range(4):
        print(f"\n{'='*60}")
        print(f"Entrenando etapa {stage_idx + 1}")
        print(f"{'='*60}\n")
        
        env_config = curriculum.get_stage_config(stage_idx)
        env = SwarmInspectionEnv(**env_config)
        
        # Entrenar
        trainer.train(
            env=env,
            total_timesteps=env_config["duration_steps"],
            rollout_steps=2048,
            log_interval=5,
        )
        
        # Evaluar
        results = trainer.evaluate(env, num_episodes=10)
        print(f"\nResultados Etapa {stage_idx + 1}:")
        print(f"  Cobertura promedio: {results['avg_coverage']:.1%}")
        print(f"  Tiempo promedio: {results['avg_time']:.0f} pasos")
        print(f"  Tasa de éxito: {results['success_rate']:.1%}")
        
        # Guardar checkpoint
        checkpoint_name = f"policy_stage{stage_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(policy.state_dict(), checkpoint_name)
        print(f"\nModelo guardado: {checkpoint_name}")
    
    print("\n✓ Entrenamiento completado")
    
    # Evaluación final
    print("\nEvaluación final con 50 episodios...")
    env = SwarmInspectionEnv(num_robots=5, grid_width=10, grid_height=10)
    final_results = trainer.evaluate(env, num_episodes=50)
    
    print(f"\nResultados finales:")
    print(f"  Cobertura promedio: {final_results['avg_coverage']:.1%}")
    print(f"  Tiempo promedio: {final_results['avg_time']:.0f} pasos")
    print(f"  Tasa de éxito: {final_results['success_rate']:.1%}")

if __name__ == "__main__":
    main()
```

---

## FASE Extra — Verificación previa al entrenamiento

Antes de llamar a `train_ppo_swarm()`, ejecuta estos tests para validar que todo funciona:

### Test 1: Verificar que el entorno se inicializa correctamente

```python
def test_env_initialization():
    """Verifica que el entorno se crea sin errores."""
    env = SwarmInspectionEnv(
        num_robots=3,
        grid_width=5,
        grid_height=5,
        cell_size=1.0,
        time_max=100,
        velocity_linear=0.25,
        simulation_timestep=1.0,
    )
    
    obs_list, info = env.reset()
    
    assert len(obs_list) == 3, f"Esperados 3 obs, obtuvimos {len(obs_list)}"
    for obs in obs_list:
        assert obs.shape == (27,), f"Shape incorrecto: {obs.shape}"
        assert obs.dtype == np.float32, f"Dtype incorrecto: {obs.dtype}"
        assert np.all(obs >= -1.1) and np.all(obs <= 1.1), \
            f"Obs fuera de rango: min={obs.min()}, max={obs.max()}"
    
    print("✓ Test 1 pasado: Inicialización correcta")
```

### Test 2: Verificar que step() funciona con acciones aleatorias

```python
def test_random_actions():
    """Ejecuta 50 pasos con acciones aleatorias y verifica lógica."""
    env = SwarmInspectionEnv(num_robots=3, grid_width=5, grid_height=5)
    obs_list, _ = env.reset()
    
    for step_idx in range(50):
        # Acciones aleatorias
        actions = [np.random.randint(0, 3) for _ in range(3)]
        
        obs_list, rewards, terminated, truncated, info = env.step(actions)
        
        # Verificaciones
        assert len(obs_list) == 3
        assert len(rewards) == 3
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Recompensas deben ser números
        for r in rewards:
            assert isinstance(r, (int, float)), f"Reward inválida: {r}"
        
        # Cobertura debe estar en [0, 1]
        assert 0 <= info["coverage"] <= 1.0
        
        # Tiempo debe ser creciente
        assert info["time_elapsed"] == step_idx + 1
        
        if terminated:
            print(f"  Episodio terminó en paso {step_idx + 1}")
            print(f"  Razón: {info['terminated_reason']}")
            print(f"  Cobertura final: {info['coverage']:.1%}")
            break
    
    print("✓ Test 2 pasado: step() funciona correctamente")
```

### Test 3: Verificar que la red neuronal puede procesar observaciones

```python
def test_network_forward():
    """Verifica que la red neuronal acepta observaciones."""
    policy = SharedActorCriticPolicy(obs_dim=27, action_dim=3, hidden_dim=128)
    
    # Test individual
    obs_single = torch.randn(27)
    action, log_prob, value, entropy = policy.get_action_and_value(obs_single)
    
    assert action.shape == torch.Size([])  # escalar
    assert log_prob.shape == torch.Size([])
    assert value.shape == torch.Size([])
    
    # Test batch
    obs_batch = torch.randn(8, 27)  # 8 robots
    actions, log_probs, values, entropies = policy.get_action_and_value(obs_batch)
    
    assert actions.shape == torch.Size([8])
    assert log_probs.shape == torch.Size([8])
    assert values.shape == torch.Size([8])
    
    print("✓ Test 3 pasado: Red neuronal funciona correctamente")
```

### Test 4: Verificar recompensas (manual)

```python
def test_reward_logic():
    """Verifica que el cálculo de recompensas tiene sentido."""
    env = SwarmInspectionEnv(num_robots=1, grid_width=3, grid_height=3, time_max=100)
    obs_list, _ = env.reset()
    
    # Acción 2 = avanzar (debería cobrar +0.5 si es celda nueva, -0.01 por tiempo)
    obs_list, rewards, _, _, info = env.step([2])  # 1 robot, acción avanzar
    
    print(f"  Reward paso 1: {rewards[0]:.4f}")
    print(f"  New cells: {info['new_cells']}")
    print(f"  Coverage: {info['coverage']:.1%}")
    
    # En pasos siguientes, si el robot se queda quieto (acción 0/1 = girar):
    obs_list, rewards, _, _, info = env.step([1])  # girar derecha
    
    print(f"  Reward paso 2 (girar): {rewards[0]:.4f}")
    # Debe ser cerca de -0.01 (solo penalización de tiempo)
    
    print("✓ Test 4 pasado: Recompensas tienen sentido")
```

### Test 5: Verificar que parameter sharing funciona

```python
def test_parameter_sharing():
    """Verifica que los robots pueden usar la misma red."""
    policy = SharedActorCriticPolicy()
    env = SwarmInspectionEnv(num_robots=5)
    obs_list, _ = env.reset()
    
    # Convertir todas las observaciones a tensor
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
    
    # La red debe poder procesar todas simultáneamente
    with torch.no_grad():
        logits, values = policy.forward(obs_tensor)
    
    assert logits.shape == (5, 3), f"Shape incorrecto: {logits.shape}"
    assert values.shape == (5,), f"Shape de valores incorrecto: {values.shape}"
    
    print("✓ Test 5 pasado: Parameter sharing funciona")
```

### Test 6: Run rápido de entrenamiento

```python
def test_mini_training():
    """Realiza 100 pasos de entrenamiento mini para verificar estabilidad."""
    env = SwarmInspectionEnv(num_robots=2, grid_width=3, grid_height=3, time_max=50)
    policy = SharedActorCriticPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    print("  Corriendo mini-rollout...")
    for _ in range(10):  # 10 episodes
        obs_list, _ = env.reset()
        for _ in range(50):
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
            with torch.no_grad():
                actions, _, _, _ = policy.get_action_and_value(obs_tensor)
            obs_list, rewards, terminated, _, _ = env.step(actions.tolist())
            
            if terminated:
                break
    
    print("✓ Test 6 pasado: Entrenamiento mini ejecutable")
```

### Ejecución de todos los tests

```python
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EJECUTANDO TESTS PRE-ENTRENAMIENTO")
    print("="*60 + "\n")
    
    test_env_initialization()
    test_random_actions()
    test_network_forward()
    test_reward_logic()
    test_parameter_sharing()
    test_mini_training()
    
    print("\n" + "="*60)
    print("✓ TODOS LOS TESTS PASARON - LISTO PARA ENTRENAR")
    print("="*60 + "\n")
```

**Cómo ejecutar:**

```bash
python tests.py
```

Si algún test falla, **DETENTE AQUÍ y depura** antes de empezar el entrenamiento. Los tests están diseñados para encontrar errores simples que causan fallos mayores después.

---

## Resumen de flujo completo

```
1. DISEÑO
   ├─ Observación: 27 valores normalizados
   ├─ Acciones: 4 discretas
   └─ Recompensa: tiempo + cobertura + coordinación

2. IMPLEMENTACIÓN
   ├─ SwarmInspectionEnv (Fase 5)
   ├─ SharedActorCriticPolicy (Fase 6)
   ├─ PPO trainer (Fase 7)
   └─ Curriculum scheduler (Fase 8)

3. ENTRENAMIENTO
   ├─ Etapa 1: 1 robot, grid 5×5
   ├─ Etapa 2: 3 robots, grid 5×5
   ├─ Etapa 3: 5 robots, grid 10×10
   └─ Etapa 4: 5 robots, dominio randomizado

4. EVALUACIÓN
   ├─ Métricas: cobertura, tiempo, colisiones
   ├─ TensorBoard para monitoreo
   └─ Checkpoints cada etapa

5. DEPLOY
   ├─ Cargar policy entrenada
   ├─ Ejecutar en simulación
   └─ (Opcional) Transferir a robots reales
```

---

**¡Listo! Este plan te lleva de 0 a un swarm entrenado con RL, paso a paso, con detalles de código e intuiciones.**

