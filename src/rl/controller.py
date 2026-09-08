"""Controlador RL para el modo 'RL Hibrido' del simulador (Fase 4).

Integra un modelo PPO entrenado como estrategia hibrida: la politica decide
que celdas asignar como objetivos a los robots del mundo REAL de la
`Simulacion` (el mismo que se renderiza), de modo que el usuario puede ver la
politica RL actuando en tiempo real con la GUI.

A diferencia de `SwarmEnv` (que mantiene un mundo interno privado para
entrenar), este controlador opera sobre `mundo`/`robots` externos:
  1. Construye la observacion desde el mundo real (misma formula que SwarmEnv).
  2. `model.predict(obs)` -> acciones (idi de celda + idle por robot).
  3. Asigna los objetivos a los robots libres via `robot.asignar_objetivo()`.
  4. La funcion de la recompensa NO se usa (actua sobre el mundo ya vivido);
     `sim.step()` avanza la fisica normal y el renderer muestra el resultado.
"""

import os

import numpy as np

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - entorno sin SB3
    PPO = None

from ..core.mundo import DISCOVERED


def build_obs(mundo, robots, paso, max_steps, map_known=True):
    """Construye la observacion (misma formula que SwarmEnv._make_obs).

    `mundo`: instancia real de Mundo.
    `robots`: lista de robots del mundo (mundo.robots o una submuestra).
    """
    R = mundo.rows
    C = mundo.cols
    n = R * C
    N = len(robots)
    obs = np.zeros(n + N * 3 + 3, dtype=np.float32)

    if map_known:
        obs = np.zeros(2 * n + N * 3 + 3, dtype=np.float32)
        for r in range(R):
            for c in range(C):
                idx = r * C + c
                if mundo.grid[r, c] == 1:
                    obs[idx] = 0.5
                elif mundo.inspec[r, c]:
                    obs[idx] = 1.0
        for r in range(R):
            for c in range(C):
                idx = n + r * C + c
                if mundo.inspec[r, c] and mundo.resultado[r, c] == 1:
                    obs[idx] = 1.0
        base_r = 2 * n
    else:
        for r in range(R):
            for c in range(C):
                idx = r * C + c
                if mundo.inspec[r, c]:
                    obs[idx] = 0.75 if mundo.resultado[r, c] != 1 else 1.0
                elif mundo.grid[r, c] == 1:
                    obs[idx] = 0.25 if mundo.explored[r, c] == DISCOVERED else 0.5
                elif mundo.explored[r, c] == 0:
                    obs[idx] = 0.5
                else:
                    obs[idx] = 0.0
        base_r = n

    for i, rb in enumerate(robots[:N]):
        obs[base_r + i * 3] = rb.x / mundo.width_px
        obs[base_r + i * 3 + 1] = rb.y / mundo.height_px
        obs[base_r + i * 3 + 2] = rb.energia.porcentaje_bateria / 100.0

    base_g = base_r + N * 3
    total_libres = mundo.total_libres if mundo.total_libres > 0 else 1
    insp = int(mundo.inspec.sum())
    pend = max(0, total_libres - insp)
    obs[base_g] = insp / total_libres
    obs[base_g + 1] = pend / total_libres
    obs[base_g + 2] = min(paso / max_steps, 1.0)

    return obs


class RLHibridoController:
    """Controlador centralizado que asigna objetivos con una politica PPO.

    Se usa como 'planner' del modo hibrido: se llama a `actualizar(mundo)`
    cada ciclo de planificacion y asigna celdas a los robots que las necesitan.
    """

    def __init__(self, cfg, model_path, map_known=True, auto_assign=True,
                 gs_override=None, n_robots_override=None):
        self.cfg = cfg
        self.map_known = map_known
        self.auto_assign = auto_assign
        self.model = None
        self.model_path = model_path
        self._pasos = 0
        # Numero de robots/celdas que espera la red (debe coincidir con el
        # entorno de entrenamiento). Se fuerza con overrides si llegan.
        self._N_modelo = n_robots_override or cfg.num_robots
        self.max_steps = max(4 * cfg.rows * cfg.cols, 3600)
        self.load()

    def load(self):
        if PPO is None:
            raise RuntimeError("Stable-Baselines3 no esta disponible")
        if not os.path.exists(self.model_path + ".zip"):
            raise FileNotFoundError(f"Modelo RL no encontrado: {self.model_path}.zip")
        self.model = PPO.load(self.model_path)
        self._pasos = 0

    def _celdas_validas(self, mundo, gs):
        return [(r, c) for r in range(mundo.rows)
                for c in range(mundo.cols)
                if mundo.grid[r, c] == 0 and not mundo.inspec[r, c]]

    def _nearest_uninspected(self, rb, mundo, gs):
        rx, ry = rb.x, rb.y
        best = None
        best_d = float("inf")
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                if mundo.grid[r, c] == 1 or mundo.inspec[r, c]:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                d = (cx - rx) ** 2 + (cy - ry) ** 2
                if d < best_d:
                    best_d = d
                    best = (r, c)
        return best

    def actualizar(self, mundo):
        """Asigna objetivos a robots libres usando la politica RL.

        Devuelve el numero de objetivos asignados en este ciclo.
        """
        gs = self.cfg.grid_size
        robots = mundo.robots
        # Solo los robots del mundo (para obs y accion).
        obs = build_obs(mundo, robots, self._pasos, self.max_steps,
                        self.map_known)
        action, _ = self.model.predict(obs, deterministic=True)
        self._pasos += 1

        n_cells = mundo.rows * mundo.cols
        celdas_asignadas = set()
        n_asignados = 0

        for i, rb in enumerate(robots):
            if i >= len(action):
                break
            # Robot ocupado: no reasignar
            if rb.inspeccionando:
                continue
            if rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                continue

            act = int(action[i])
            target = None
            if act < n_cells:
                r, c = divmod(act, mundo.cols)
                if (r, c) not in celdas_asignadas and mundo.grid[r, c] == 0 \
                        and not mundo.inspec[r, c]:
                    target = (r, c)

            if target is None and self.auto_assign:
                target = self._nearest_uninspected(rb, mundo, gs)

            if target is None:
                continue
            r, c = target
            if (r, c) in celdas_asignadas:
                continue
            if mundo.grid[r, c] == 1 or mundo.inspec[r, c]:
                continue
            if not rb.necesita_objetivo or rb.cola_tareas:
                continue
            rb.asignar_objetivo(c * gs + gs / 2, r * gs + gs / 2, mundo)
            celdas_asignadas.add((r, c))
            n_asignados += 1

        return n_asignados


class RLControllerV3DES:
    """Controlador RL Descentralizado e Invariante V3-DES.
    
    Permite aplicar una politica descentralizada entrenada en cualquier enjambre (N robots)
    y cualquier dimension de mapa en tiempo real durante la simulacion.
    """

    def __init__(self, model_path=None, auto_assign=True, patch_size=15,
                 max_neighbors=3, radio_alerta_px=500.0):
        self.auto_assign = auto_assign
        self.patch_size = patch_size
        self.max_neighbors = max_neighbors
        self.radio_alerta_px = radio_alerta_px
        self.model = None
        self._ultimas_defectuosas = []
        self._ult_pos_robot = {}   # {id(robot): (x, y)}
        self._tiempo_sin_mov = {}  # {id(robot): segundos sin moverse}
        self.umbral_quieto_s = 2.0

        if model_path and os.path.exists(model_path):
            if PPO is None:
                raise RuntimeError("stable_baselines3 no instalado")
            self.model = PPO.load(model_path)

    def actualizar(self, mundo):
        robots = mundo.robots
        if not robots or self.model is None:
            return 0

        gs = mundo.cfg.grid_size
        half_k = self.patch_size // 2
        n_patch = self.patch_size * self.patch_size
        n_asignados = 0

        # Actualizar tracking de quietud fisica: cada ciclo del controlador, si
        # un robot no cambio de posicion acumula tiempo sin moverse; si se movio
        # se resetea. (El dt estimado no se conoce aqui, por lo que se usa un
        # incremento unitario por ciclo para alimentar la senal de observacion.)
        for rb in robots:
            pid = id(rb)
            ult = self._ult_pos_robot.get(pid)
            if ult is None:
                self._ult_pos_robot[pid] = (rb.x, rb.y)
                self._tiempo_sin_mov[pid] = 0.0
            elif (abs(rb.x - ult[0]) < 1e-3 and abs(rb.y - ult[1]) < 1e-3):
                self._tiempo_sin_mov[pid] = self._tiempo_sin_mov.get(pid, 0.0) + 1.0
            else:
                self._tiempo_sin_mov[pid] = 0.0
            self._ult_pos_robot[pid] = (rb.x, rb.y)

        # Actualizar memoria de celdas defectuosas
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                if mundo.inspec[r, c] and mundo.resultado[r, c] == 1:
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    if (cx, cy) not in self._ultimas_defectuosas:
                        self._ultimas_defectuosas.append((cx, cy))

        for rb in robots:
            if rb.inspeccionando or (rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado):
                continue

            r_center = int(rb.y // gs)
            c_center = int(rb.x // gs)

            # Construir obs egocentrica para este robot
            obs = np.zeros(n_patch + 1 + (self.max_neighbors * 3) + 2, dtype=np.float32)

            idx = 0
            for dr in range(-half_k, half_k + 1):
                for dc in range(-half_k, half_k + 1):
                    r = r_center + dr
                    c = c_center + dc
                    if 0 <= r < mundo.rows and 0 <= c < mundo.cols:
                        if mundo.inspec[r, c]:
                            val = 0.75 if mundo.resultado[r, c] != 1 else 1.0
                        elif mundo.grid[r, c] == 1:
                            val = 0.25 if mundo.explored[r, c] == DISCOVERED else 0.5
                        elif mundo.explored[r, c] == 0:
                            val = 0.5
                        else:
                            val = 0.0
                    else:
                        val = 0.25
                    obs[idx] = val
                    idx += 1

            # Alerta
            if self._ultimas_defectuosas:
                val_max = 0.0
                for dx, dy in self._ultimas_defectuosas[-5:]:
                    dist = np.hypot(dx - rb.x, dy - rb.y)
                    if dist < self.radio_alerta_px:
                        v = 1.0 - (dist / self.radio_alerta_px)
                        if v > val_max:
                            val_max = v
                obs[n_patch] = float(val_max)

            # Vecinos
            base_n = n_patch + 1
            vecinos = []
            for other in robots:
                if other is rb:
                    continue
                d = np.hypot(other.x - rb.x, other.y - rb.y)
                vecinos.append((d, other.x - rb.x, other.y - rb.y))
            vecinos.sort(key=lambda item: item[0])

            for i in range(self.max_neighbors):
                if i < len(vecinos):
                    d, rel_x, rel_y = vecinos[i]
                    obs[base_n + i * 3] = np.clip(rel_x / mundo.width_px, -1.0, 1.0)
                    obs[base_n + i * 3 + 1] = np.clip(rel_y / mundo.height_px, -1.0, 1.0)
                    obs[base_n + i * 3 + 2] = np.clip(d / self.radio_alerta_px, 0.0, 1.0)

            base_st = base_n + self.max_neighbors * 3
            obs[base_st] = rb.energia.porcentaje_bateria / 100.0
            # Senal de inactividad fisica, coherente con el entrenamiento:
            # tiempo sin moverse normalizado a umbral_quieto_s * 3.
            t_sin_mov = self._tiempo_sin_mov.get(id(rb), 0.0)
            obs[base_st + 1] = float(np.clip(t_sin_mov / max(self.umbral_quieto_s * 3, 1e-6), 0.0, 1.0))

            action, _ = self.model.predict(obs, deterministic=True)
            act = int(action)

            if act < n_patch:
                dr = (act // self.patch_size) - half_k
                dc = (act % self.patch_size) - half_k
                tr, tc = r_center + dr, c_center + dc
                if (0 <= tr < mundo.rows and 0 <= tc < mundo.cols and mundo.grid[tr, tc] == 0
                        and not mundo.inspec[tr, tc]):
                    rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, mundo)
                    n_asignados += 1
                elif self.auto_assign:
                    # Fallback: la accion predicha fue invalida (pared, celda ya
                    # inspeccionada, fuera de rango) -> reasignar la celda libre
                    # no inspeccionada mas cercana para que el robot no se quede
                    # quieto en la simulacion.
                    if self._auto_assign_local(rb, r_center, c_center, mundo):
                        n_asignados += 1
            elif self.auto_assign:
                # La accion fue Idle (act >= n_patch): nunca se permite descansar.
                # Se reasigna un objetivo para que el robot siga trabajando.
                if self._auto_assign_local(rb, r_center, c_center, mundo):
                    n_asignados += 1

        return n_asignados

    def _auto_assign_local(self, rb, r0, c0, mundo):
        """Asigna la celda libre no inspeccionada mas cercana en la zona.

        Devuelve True si asigno un objetivo nuevo, False si no habia celdas
        disponibles (el robot espera, no se le puede asignar nada)."""
        gs = mundo.cfg.grid_size
        celdas = []
        for rr in range(mundo.rows):
            for cc in range(mundo.cols):
                if mundo.grid[rr, cc] == 0 and not mundo.inspec[rr, cc]:
                    d_mh = abs(rr - r0) + abs(cc - c0)
                    celdas.append((d_mh, rr, cc))
        if not celdas:
            return False
        celdas.sort()
        _, tr, tc = celdas[0]
        # No dejar al robot quieto en su misma celda: si la mas cercana es la
        # que ya ocupa, tomar la siguiente si existe.
        if tr == r0 and tc == c0 and len(celdas) > 1:
            _, tr, tc = celdas[1]
        rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, mundo)
        return True


class RLControllerExploracionV3:

    """Controlador RL Descentralizado de Exploracion Pura (V3-EXP).

    Permite usar un modelo PPO entrenado para descubrimiento rapido del mapa,
    evaluando la ventana local KxK y las señales de descubrimiento cooperativo
    de los robots vecinos. Acepta cualquier cantidad N de robots y cualquier tamaño de mapa.
    """

    def __init__(self, model_path=None, auto_assign=True, patch_size=15,
                 max_neighbors=3, radio_comunicacion_px=400.0):
        self.auto_assign = auto_assign
        self.patch_size = patch_size
        self.max_neighbors = max_neighbors
        self.radio_comunicacion_px = radio_comunicacion_px
        self.model = None
        self._descubrimiento_previo = {}

        if model_path and os.path.exists(model_path):
            if PPO is None:
                raise RuntimeError("stable_baselines3 no instalado")
            self.model = PPO.load(model_path)

    def actualizar(self, mundo):
        robots = mundo.robots
        if not robots or self.model is None:
            return 0

        gs = mundo.cfg.grid_size
        half_k = self.patch_size // 2
        n_patch = self.patch_size * self.patch_size
        n_asignados = 0

        # Calcular actividad reciente de descubrimiento de cada robot
        actividad = {}
        for rb in robots:
            actual = rb.scouteadas_robot + rb.paredes_robot
            prev = self._descubrimiento_previo.get(id(rb), actual)
            actividad[id(rb)] = max(0, actual - prev)
            self._descubrimiento_previo[id(rb)] = actual

        for rb in robots:
            if rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                continue

            r_center = int(rb.y // gs)
            c_center = int(rb.x // gs)

            obs = np.zeros(n_patch + (self.max_neighbors * 4) + 2, dtype=np.float32)

            # 1. Parche local de exploracion
            idx = 0
            for dr in range(-half_k, half_k + 1):
                for dc in range(-half_k, half_k + 1):
                    r = r_center + dr
                    c = c_center + dc
                    if 0 <= r < mundo.rows and 0 <= c < mundo.cols:
                        if mundo.grid[r, c] == 1:
                            val = 0.8 if mundo.explored[r, c] == DISCOVERED else 0.5
                        elif mundo.explored[r, c] == 0:
                            val = 0.5  # Niebla / frontera
                        else:
                            val = 0.0  # Descubierta
                    else:
                        val = 1.0  # Fuera de limites
                    obs[idx] = val
                    idx += 1

            # 2. Vecinos y señal de descubrimiento cooperativo
            base_n = n_patch
            vecinos = []
            for other in robots:
                if other is rb:
                    continue
                d = np.hypot(other.x - rb.x, other.y - rb.y)
                vecinos.append((d, other.x - rb.x, other.y - rb.y, actividad.get(id(other), 0)))
            vecinos.sort(key=lambda item: item[0])

            for i in range(self.max_neighbors):
                if i < len(vecinos):
                    d, rel_x, rel_y, act_v = vecinos[i]
                    obs[base_n + i * 4] = np.clip(rel_x / mundo.width_px, -1.0, 1.0)
                    obs[base_n + i * 4 + 1] = np.clip(rel_y / mundo.height_px, -1.0, 1.0)
                    obs[base_n + i * 4 + 2] = np.clip(d / self.radio_comunicacion_px, 0.0, 1.0)
                    obs[base_n + i * 4 + 3] = np.clip(act_v / 10.0, 0.0, 1.0)

            # 3. Estado propio
            base_st = base_n + self.max_neighbors * 4
            obs[base_st] = rb.energia.porcentaje_bateria / 100.0
            obs[base_st + 1] = np.clip(actividad.get(id(rb), 0) / 10.0, 0.0, 1.0)

            action, _ = self.model.predict(obs, deterministic=True)
            act = int(action)

            if act < n_patch:
                dr = (act // self.patch_size) - half_k
                dc = (act % self.patch_size) - half_k
                tr, tc = r_center + dr, c_center + dc
                if 0 <= tr < mundo.rows and 0 <= tc < mundo.cols and mundo.grid[tr, tc] == 0:
                    rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, mundo)
                    n_asignados += 1
            elif self.auto_assign:
                # Fallback: frontera mas cercana
                self._auto_assign_frontera_local(rb, r_center, c_center, mundo)

        return n_asignados

    def _auto_assign_frontera_local(self, rb, r0, c0, mundo):
        gs = mundo.cfg.grid_size
        fronteras = []
        for dr in range(-8, 9):
            for dc in range(-8, 9):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < mundo.rows and 0 <= c < mundo.cols and mundo.grid[r, c] == 0:
                    if mundo.explored[r, c] == 0:
                        fronteras.append((abs(dr) + abs(dc), r, c))
        if fronteras:
            fronteras.sort()
            _, tr, tc = fronteras[0]
            rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, mundo)


