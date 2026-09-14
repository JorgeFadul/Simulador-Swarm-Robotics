"""Entorno Gymnasium compatible con el Simulador Swarm 2D (RL).

Envuelve `Mundo` + `Robot` en una interfaz estandar de RL para entrenar
agentes con PPO (Stable-Baselines3).  El agente centralizado decide que
celdas asignar como objetivos a los robots; la planificacion de ruta (A*) y
el movimiento diferencial se manejan por el simulador existente.

Dos modos de observacion:
  * map_known=True   : el agente conoce TODO el mapa desde el reset.
                       Inspeccion directa de las celdas (exploracion no necesaria).
  * map_known=False  : el agente descubre el mapa con su radar mientras
                       decide que inspeccionar. Aprende a explorar + inspeccionar.

Espacio de observacion (Box, float32):
  map_known=True  -> 2*R*C + N*3 + 3
      [0 .. R*C)          mapa de ocupacion: 0=libre, 0.5=pared, 1=inspeccionada
      [R*C .. 2*R*C)      mapa de defectos revelados: 1.0=insp+defectuosa, 0=resto
      [2RC .. 2RC+N*3)    por robot: x/w, y/h, bateria_pct/100
      [2RC+N*3 ..  end)   insp/total, pend/total, paso/max

  map_known=False -> R*C + N*3 + 3
      [0 .. R*C)          mapa integrado por celda:
                          0.00 = libre descubierto
                          0.25 = pared conocida (DISCOVERED)
                          0.50 = no descubierto (libre o pared sin explorar)
                          0.75 = inspeccionada sana
                          1.00 = inspeccionada defectuosa
      [R*C .. R*C+N*3)    por robot: x/w, y/h, bateria_pct/100
      [R*C+N*3 .. end)    insp/total, pend/total, paso/max

Espacio de accion (MultiDiscrete[N]): cada robot elige una celda [0, R*C)
o idle (R*C).  Se aplica secuencialmente: robot 0 elige, se bloquea, robot 1
elige de las restantes, etc.  Si la celda es pared/inspeccionada/ya asignada,
la accion se descarta (idle).  Con `auto_assign=True`, en su lugar se asigna
la celda libre no inspeccionada mas cercana (fallback anti-idle).

Funcion de recompensa (configurable): +1 por celda nueva inspeccionada,
-0.1 redundancia, -0.3 colision robot-robot, -0.1 colision pared,
-0.001 por paso; mezcla individual(60%) + global(40%) + bonus terminal (+50 si
completa rapido, -50 si timeout).  Ver `cfg.rl_reward`.
"""

import gymnasium as gym
import numpy as np
import math

from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED


class SwarmEnv(gym.Env):
    """Entorno Gymnasium para el Simulador Swarm 2D."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg=None, seed=None, reward_mode="time",
                 max_steps=10000, headless=True, physics_steps=20,
                 map_known=True, auto_assign=False):
        super().__init__()
        self.cfg = cfg or Config()
        self.seed_base = seed
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.headless = headless
        self.physics_steps = physics_steps
        self.map_known = map_known
        self.auto_assign = auto_assign

        R = self.cfg.rows
        C = self.cfg.cols
        self.R, self.C = R, C
        self.n_cells = R * C
        self.N = self.cfg.num_robots

        # --- Espacio de observacion ---
        if map_known:
            obs_size = 2 * self.n_cells + self.N * 3 + 3
        else:
            obs_size = self.n_cells + self.N * 3 + 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # --- Espacio de accion (multi-discrete: cada robot elige R*C+1) ---
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_cells + 1] * self.N
        )

        # Estado interno
        self._mundo = None
        self._robots = []
        self._pasos = 0
        self._inspeccionadas_prev = 0
        self._defectuosas_prev = 0
        self._energia_total_prev = 0.0
        self._colision_pared_prev = 0
        self._rng = None

    # ------------------------------------------------------------------
    # Observacion
    # ------------------------------------------------------------------
    def _make_obs(self):
        m = self._mundo
        R, C = self.R, self.C
        n = self.n_cells
        N = self.N

        if self.map_known:
            obs = np.zeros(2 * n + N * 3 + 3, dtype=np.float32)
            # Canal 0: mapa de ocupacion (libre=0, pared=0.5, inspeccionada=1)
            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    if m.grid[r, c] == 1:
                        obs[idx] = 0.5
                    elif m.inspec[r, c]:
                        obs[idx] = 1.0
            # Canal 1: defectos revelados (1.0 si inspeccionada y defectuosa)
            for r in range(R):
                for c in range(C):
                    idx = n + r * C + c
                    if m.inspec[r, c] and m.resultado[r, c] == 1:
                        obs[idx] = 1.0
            base_r = 2 * n
        else:
            obs = np.zeros(n + N * 3 + 3, dtype=np.float32)
            # Mapa integrado por celda (con memoria de descubrimiento + resultado)
            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    if m.inspec[r, c]:
                        obs[idx] = 0.75 if m.resultado[r, c] != 1 else 1.0
                    elif m.grid[r, c] == 1:
                        obs[idx] = 0.25 if m.explored[r, c] == DISCOVERED else 0.5
                    elif m.explored[r, c] == 0:
                        obs[idx] = 0.5  # libre pero aun no descubierto
                    else:
                        obs[idx] = 0.0  # libre y descubierto
            base_r = n

        # Datos por robot
        for i, rb in enumerate(self._robots):
            if i >= N:
                break
            obs[base_r + i * 3] = rb.x / m.width_px
            obs[base_r + i * 3 + 1] = rb.y / m.height_px
            obs[base_r + i * 3 + 2] = rb.energia.porcentaje_bateria / 100.0

        # Resumen global
        base_g = base_r + N * 3
        total_libres = m.total_libres if m.total_libres > 0 else 1
        insp = int(m.inspec.sum())
        pend = max(0, total_libres - insp)
        obs[base_g] = insp / total_libres
        obs[base_g + 1] = pend / total_libres
        obs[base_g + 2] = min(self._pasos / self.max_steps, 1.0)

        return obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.seed_base is not None:
            self._rng = np.random.default_rng(self.seed_base)
        else:
            self._rng = np.random.default_rng()

        self._mundo = Mundo(self.cfg, self._rng)
        # RL siempre en modo inspeccion para poder inspeccionar celdas
        self._mundo.etapa = "inspeccion"

        self._robots = []
        for _ in range(self.N):
            x, y = self._mundo.posicion_libre_aleatoria(self._rng,
                                                         self.cfg.robot_radius_px)
            if x is None:
                break
            from ..robots.robot import Robot
            rb = Robot(x, y, self.cfg, self._rng)
            self._robots.append(rb)
        self._mundo.robots = list(self._robots)

        # Escaneo inicial: cada robot descubre su entorno antes del primer paso
        for rb in self._robots:
            exp_radius = rb.detection_radius * self.cfg.exploration_radius_multiplier
            s, p = self._mundo.explorar_desde(rb.x, rb.y, exp_radius)
            rb.scouteadas_robot += s
            rb.paredes_robot += p
            rb._explorar_desde_propio(self._mundo, exp_radius)

        self._pasos = 0
        self._inspeccionadas_prev = int(self._mundo.inspec.sum())
        self._defectuosas_prev = int(
            sum(1 for r in range(self._mundo.rows)
                for c in range(self._mundo.cols)
                if self._mundo.inspec[r, c] and self._mundo.resultado[r, c] == 1))
        self._energia_total_prev = sum(
            rb.energia.energia_consumida_total for rb in self._robots
        )
        self._colision_pared_prev = sum(
            getattr(rb, "paredes_robot", 0) for rb in self._robots
        )

        return self._make_obs(), {}

    # ------------------------------------------------------------------
    # Helpers de asignacion
    # ------------------------------------------------------------------
    def _find_nearest_uninspected(self, rb, m):
        """Celda libre no inspeccionada mas cercana al robot."""
        gs = self.cfg.grid_size
        rx, ry = rb.x, rb.y
        best_dist = float("inf")
        best_cell = None
        for r in range(self.R):
            for c in range(self.C):
                if m.grid[r, c] == 1 or m.inspec[r, c]:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                dist = (cx - rx) ** 2 + (cy - ry) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_cell = (r, c)
        return best_cell

    def _asignar_a_robot(self, rb, target_cell, m, celdas_asignadas):
        """Asigna una celda a un robot libre. Devuelve True si se asigno."""
        if target_cell is None:
            return False
        r, c = target_cell
        if (r, c) in celdas_asignadas:
            return False
        if m.grid[r, c] == 1 or m.inspec[r, c]:
            return False
        if not rb.necesita_objetivo or rb.cola_tareas:
            return False
        gs = self.cfg.grid_size
        cx = c * gs + gs / 2
        cy = r * gs + gs / 2
        rb.asignar_objetivo(cx, cy, m)
        celdas_asignadas.add((r, c))
        return True

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        m = self._mundo
        R, C = self.R, self.C
        n = self.n_cells
        gs = self.cfg.grid_size

        # --- Decodificar acciones: asignar targets a robots libres ---
        celdas_asignadas = set()
        for i, rb in enumerate(self._robots):
            if i >= len(action):
                break
            # Solo asignar si el robot esta libre
            if rb.inspeccionando:
                continue
            if rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                continue

            act = int(action[i])
            target_cell = None
            if act < n:
                r, c = divmod(act, C)
                if (r, c) not in celdas_asignadas and m.grid[r, c] == 0 \
                        and not m.inspec[r, c]:
                    target_cell = (r, c)

            if target_cell is None and self.auto_assign:
                target_cell = self._find_nearest_uninspected(rb, m)

            self._asignar_a_robot(rb, target_cell, m, celdas_asignadas)

        # --- Simular multiples sub-pasos de fisica por step RL ---
        dt = 0.05
        colisiones_rr = 0
        colisiones_pared = 0
        for _sub in range(self.physics_steps):
            # Radar: cada robot escanea su entorno
            for rb in self._robots:
                exp_radius = rb.detection_radius * self.cfg.exploration_radius_multiplier
                s, p = m.explorar_desde(rb.x, rb.y, exp_radius)
                rb.scouteadas_robot += s
                rb.paredes_robot += p
                rb._explorar_desde_propio(m, exp_radius)
            # Movimiento y deteccion de colisiones
            for rb in self._robots:
                before_x, before_y = rb.x, rb.y
                rb.update(dt, m)
                moved = abs(rb.x - before_x) > 1e-6 or abs(rb.y - before_y) > 1e-6
                # Colision con pared: robot intento moverse pero quedo parado
                # (y no esta inspeccionando ni rotando solo)
                if moved is False and rb.goal_mode and not rb.goal_alcanzado:
                    pass

        # Deteccion precisa de colisiones post-fisica:
        # robot-robot: distancia menor que suma de radios
        rr_thresh = 2 * self.cfg.robot_radius_px * 0.9
        for i in range(len(self._robots)):
            for j in range(i + 1, len(self._robots)):
                dx = self._robots[i].x - self._robots[j].x
                dy = self._robots[i].y - self._robots[j].y
                if (dx * dx + dy * dy) < rr_thresh * rr_thresh:
                    colisiones_rr += 1

        # Colisiones con paredes: delta del contador acumulado del robot
        # (proxy de paredes discriminadas nuevas en este step).
        col_pared_actual = sum(getattr(rb, "paredes_robot", 0) for rb in self._robots)
        colisiones_pared = col_pared_actual - self._colision_pared_prev
        self._colision_pared_prev = col_pared_actual

        # --- Reasignar robots que completaron su objetivo (auto_assign) ---
        if self.auto_assign:
            for rb in self._robots:
                if not rb.inspeccionando and (rb.goal is None or rb.goal_alcanzado):
                    nc = self._find_nearest_uninspected(rb, m)
                    if nc is not None:
                        gs2 = self.cfg.grid_size
                        r, c = nc
                        rb.asignar_objetivo(
                            c * gs2 + gs2 / 2, r * gs2 + gs2 / 2, m)

        self._pasos += 1

        # --- Calcular recompensa ---
        insp_actual = int(m.inspec.sum())
        nuevas = insp_actual - self._inspeccionadas_prev
        self._inspeccionadas_prev = insp_actual

        defectuosas_actual = int(
            sum(1 for r in range(m.rows)
                for c in range(m.cols)
                if m.inspec[r, c] and m.resultado[r, c] == 1))
        nuevas_defectuosas = defectuosas_actual - self._defectuosas_prev
        self._defectuosas_prev = defectuosas_actual

        energia_actual = sum(rb.energia.energia_consumida_total
                             for rb in self._robots)
        energia_step = energia_actual - self._energia_total_prev
        self._energia_total_prev = energia_actual

        reward = self._compute_reward(nuevas, nuevas_defectuosas, energia_step,
                                      colisiones_rr, colisiones_pared)

        # --- Condicion de terminacion ---
        total_libres = m.total_libres if m.total_libres > 0 else 1
        cobertura = insp_actual / total_libres
        terminated = cobertura >= 0.99
        truncated = self._pasos >= self.max_steps

        info = {
            "inspeccionadas": insp_actual,
            "total_libres": total_libres,
            "cobertura": cobertura,
            "pasos": self._pasos,
            "energia_total": energia_actual,
            "nuevas_inspeccionadas": nuevas,
            "nuevas_defectuosas": nuevas_defectuosas,
            "defectuosas_total": defectuosas_actual,
            "colisiones_rr": colisiones_rr,
            "colisiones_pared": colisiones_pared,
        }

        return self._make_obs(), reward, terminated, truncated, info

    def _compute_reward(self, nuevas, nuevas_defectuosas, energia_step,
                        colisiones_rr, colisiones_pared):
        rl = self.cfg.rl_reward

        # Componente individual
        r_ind = 0.0
        r_ind += nuevas * rl["reward_nueva"]
        r_ind += rl["reward_tiempo"]
        r_ind += colisiones_rr * rl["reward_colision_rr"]
        r_ind += min(colisiones_pared, 50) * rl["reward_colision_pared"]
        r_ind += nuevas_defectuosas * rl.get("reward_defecto", 0.0)

        if self.reward_mode == "energy":
            w = self.cfg.energia_inspeccion_j * 10
            r_ind -= w * energia_step
        elif self.reward_mode == "time":
            # penalizacion temporal ya incluida en reward_tiempo
            pass

        # Componente global (compartida)
        r_global = (nuevas * rl["reward_nueva_global"]
                    + rl["reward_tiempo_global"]
                    + nuevas_defectuosas * rl.get("reward_defecto_global", 0.0))

        alpha = rl["alpha"]
        r_final = alpha * r_ind + (1 - alpha) * r_global

        # Bonus terminal
        total_libres = self._mundo.total_libres if self._mundo.total_libres > 0 else 1
        insp = int(self._mundo.inspec.sum())
        cobertura = insp / total_libres
        if cobertura >= 0.99:
            time_ratio = self._pasos / self.max_steps
            r_final += rl["bonus_completar"] * (1 - time_ratio)
        elif self._pasos >= self.max_steps:
            r_final += rl["castigo_timeout"]

        return float(r_final)

    def render(self):
        pass  # la renderizacion se hace via simulacion.py

    def close(self):
        pass


class SwarmEnvDescentralizado(gym.Env):
    """Entorno Gymnasium Descentralizado e Invariante V3-DES.
    
    Cada robot opera de forma independiente usando una observacion local (patch KxK),
    un indicador de alerta difuso por celdas defectuosas en su radio de comunicacion,
    vector de vecinos cercanos y penalizacion por inactividad.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg=None, seed=None, patch_size=15, max_neighbors=3,
                 radio_alerta_px=500.0, max_steps=10000, headless=True,
                 physics_steps=20, auto_assign=True, umbral_inactividad_pasos=30):
        super().__init__()
        self.cfg = cfg or Config()
        self.seed_base = seed
        self.patch_size = patch_size   # K x K local patch
        self.max_neighbors = max_neighbors
        self.radio_alerta_px = radio_alerta_px
        self.max_steps = max_steps
        self.headless = headless
        self.physics_steps = physics_steps
        self.auto_assign = auto_assign
        self.umbral_inactividad_pasos = umbral_inactividad_pasos

        # Radio de vecindad para el contador de robots vecinos: variable que
        # multiplica el radio del robot (default 5x -> 3.5cm * 5 = 17.5cm).
        self.radio_vecindad_px = getattr(self.cfg, "rl_vecindad_radio_px",
                                         self.cfg.robot_radius_px * 5.0)

        # Tamaño fijo del parche y vectores
        self.n_patch = patch_size * patch_size
        # Obs: patch_size^2 + 1 (alerta) + max_neighbors*3 (vecinos) + 2 (bateria, inactividad)
        #      + 1 (contador de robots vecinos dentro del radio de vecindad)
        self.obs_dim = self.n_patch + 1 + (self.max_neighbors * 3) + 2 + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Accion discreta local: una casilla dentro de la grilla KxK (sin Idle).
        # Se elimina la opcion de "quedarse quieto": el robot siempre elige un
        # desplazamiento dentro de la ventana, evitando el ocio aprendido.
        self.action_space = gym.spaces.Discrete(self.n_patch)

        self._mundo = None
        self._robots = []
        self._pasos = 0
        self._current_robot_idx = 0
        self._rng = None
        self._pasos_sin_defecto = {}  # {robot_id: pasos}
        self._ultimas_defectuosas = []  # [(x, y, tiempo)]
        self.reward_mode = "time"

        # Control de inactividad fisica (se quedo quieto): posicion ultima y
        # segundos acumulados sin que el robot cambie de posicion.
        self._ult_pos_robot = {}    # {id(robot): (x, y)}
        self._tiempo_sin_mov = {}   # {id(robot): segundos sin moverse}
        self.umbral_quieto_s = 2.0  # segundos permitidos quieto antes de penalizar
        self.penalizacion_quieto = 0.8  # penalizacion base por paso quieto

        # Rachas de celdas defectuosas consecutivas por robot, para incentivar
        # que sigan buscando defectos seguidos con un multiplicador creciente.
        self._racha_defectos = {}       # {id(robot): numero de defectos consecutivos}
        self.recompensa_defecto_base = 10.0   # recompensa alta por celda defectuosa
        self.factor_racha_defectos = 1.0      # multiplicador por cada defecto consecutivo

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self.seed_base)

        from ..core.mundo import Mundo
        from ..robots.robot import Robot

        self._mundo = Mundo(self.cfg)
        self._robots = []
        self._pasos = 0
        self._current_robot_idx = 0
        self._ultimas_defectuosas = []

        # Poblar robots
        for _ in range(self.cfg.num_robots):
            x, y = self._mundo.posicion_libre_aleatoria(self._rng, self.cfg.robot_radius_px)
            if x is not None:
                r = Robot(x, y, self.cfg, self._rng)
                self._robots.append(r)
                self._mundo.robots.append(r)

        self._pasos_sin_defecto = {id(r): 0 for r in self._robots}
        self._ult_pos_robot = {id(r): (r.x, r.y) for r in self._robots}
        self._tiempo_sin_mov = {id(r): 0.0 for r in self._robots}
        self._racha_defectos = {id(r): 0 for r in self._robots}
        return self._make_obs_single(0), {}

    def _get_alerta_val(self, rx, ry):
        """Calcula el valor del indicador difuso de alerta de defectos [0, 1]."""
        if not self._ultimas_defectuosas:
            return 0.0
        val_max = 0.0
        for dx, dy in self._ultimas_defectuosas[-5:]:
            dist = np.hypot(dx - rx, dy - ry)
            if dist < self.radio_alerta_px:
                val = 1.0 - (dist / self.radio_alerta_px)
                if val > val_max:
                    val_max = val
        return float(val_max)

    def _make_obs_single(self, robot_idx):
        """Construye el vector de observacion local egocentrico para un robot."""
        m = self._mundo
        rb = self._robots[robot_idx]
        gs = self.cfg.grid_size
        r_center = int(rb.y // gs)
        c_center = int(rb.x // gs)
        half_k = self.patch_size // 2

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Parche local KxK
        idx = 0
        for dr in range(-half_k, half_k + 1):
            for dc in range(-half_k, half_k + 1):
                r = r_center + dr
                c = c_center + dc
                if 0 <= r < m.rows and 0 <= c < m.cols:
                    if m.inspec[r, c]:
                        val = 0.75 if m.resultado[r, c] != 1 else 1.0
                    elif m.grid[r, c] == 1:
                        val = 0.25 if m.explored[r, c] == DISCOVERED else 0.5
                    elif m.explored[r, c] == 0:
                        val = 0.5
                    else:
                        val = 0.0
                else:
                    val = 0.25  # Fuera del mapa como pared
                obs[idx] = val
                idx += 1

        # 2. Indicador de alerta de defectos
        obs[self.n_patch] = self._get_alerta_val(rb.x, rb.y)

        # 3. Vecinos mas cercanos (hasta max_neighbors)
        base_n = self.n_patch + 1
        vecinos = []
        for other in self._robots:
            if other is rb:
                continue
            d = np.hypot(other.x - rb.x, other.y - rb.y)
            vecinos.append((d, other.x - rb.x, other.y - rb.y))
        vecinos.sort(key=lambda item: item[0])

        for i in range(self.max_neighbors):
            if i < len(vecinos):
                d, rel_x, rel_y = vecinos[i]
                obs[base_n + i * 3] = np.clip(rel_x / m.width_px, -1.0, 1.0)
                obs[base_n + i * 3 + 1] = np.clip(rel_y / m.height_px, -1.0, 1.0)
                obs[base_n + i * 3 + 2] = np.clip(d / self.radio_alerta_px, 0.0, 1.0)

        # 4. Estado propio
        base_st = base_n + self.max_neighbors * 3
        obs[base_st] = rb.energia.porcentaje_bateria / 100.0
        # Senal de inactividad FISICA (normalizado): cuanto tiempo lleva el robot
        # sin moverse. Permite que la politica aprenda a evitar quedarse quieto.
        t_sin_mov = self._tiempo_sin_mov.get(id(rb), 0.0)
        obs[base_st + 1] = np.clip(t_sin_mov / max(self.umbral_quieto_s * 3, 1e-6), 0.0, 1.0)

        # 5. Contador de robots vecinos dentro del radio de vecindad (normalizado):
        #    cuantos robots hay dentro de rl_vecindad_radio_mult * robot_radius_px.
        #    El robot es consciente de cuantos pares tiene alrededor.
        vecinos_radio = 0
        for other in self._robots:
            if other is rb:
                continue
            if np.hypot(other.x - rb.x, other.y - rb.y) <= self.radio_vecindad_px:
                vecinos_radio += 1
        max_vecinos = max(1, len(self._robots) - 1)
        obs[self.obs_dim - 1] = min(vecinos_radio / max_vecinos, 1.0)

        return obs

    def step(self, action):
        m = self._mundo
        rb = self._robots[self._current_robot_idx]
        gs = self.cfg.grid_size
        r_center = int(rb.y // gs)
        c_center = int(rb.x // gs)
        half_k = self.patch_size // 2

        # Estado base del robot actual para medir el trabajo real del paso
        inspec_base = rb.inspeccionadas_robot
        def_base = rb.defectuosas_robot
        energia_base = rb.energia.energia_consumida_total

        # Convertir accion discreta a coordenada (r, c)
        act = int(action)
        if act < self.n_patch:
            dr = (act // self.patch_size) - half_k
            dc = (act % self.patch_size) - half_k
            tr, tc = r_center + dr, c_center + dc
            if (0 <= tr < m.rows and 0 <= tc < m.cols and m.grid[tr, tc] == 0
                    and not m.inspec[tr, tc]):
                tx = tc * gs + gs / 2
                ty = tr * gs + gs / 2
                rb.asignar_objetivo(tx, ty, m)
            elif self.auto_assign:
                # Fallback: asignar celda libre no inspeccionada mas cercana en la zona
                self._auto_assign_local(rb, r_center, c_center)

        # Avanzar al siguiente robot o avanzar paso de fisica
        self._current_robot_idx += 1
        if self._current_robot_idx >= len(self._robots):
            self._current_robot_idx = 0
            for _ in range(self.physics_steps):
                m.update(0.05)
            self._pasos += 1

            # Medir inactividad fisica (robot quieto): cada ciclo de fisica
            # avanza physics_steps*0.05 segundos. Si un robot no cambio de
            # posicion respecto de su ultimo registro, acumula ese tiempo;
            # si se movio, se resetea el contador.
            dt_fisica = self.physics_steps * 0.05
            for r in self._robots:
                pid = id(r)
                ult = self._ult_pos_robot.get(pid)
                if ult is None:
                    self._ult_pos_robot[pid] = (r.x, r.y)
                    self._tiempo_sin_mov[pid] = 0.0
                    continue
                if (abs(r.x - ult[0]) < 1e-3 and abs(r.y - ult[1]) < 1e-3):
                    self._tiempo_sin_mov[pid] = self._tiempo_sin_mov.get(pid, 0.0) + dt_fisica
                else:
                    self._tiempo_sin_mov[pid] = 0.0
                self._ult_pos_robot[pid] = (r.x, r.y)

        # Registrar hallazgos de defectos
        for r in self._robots:
            if r.defectuosas_robot > 0:
                pass  # Los hallazgos actualizan las ultimas defectuosas
        for r_i in range(m.rows):
            for c_i in range(m.cols):
                if m.inspec[r_i, c_i] and m.resultado[r_i, c_i] == 1:
                    cx = c_i * gs + gs / 2
                    cy = r_i * gs + gs / 2
                    if (cx, cy) not in [(x, y) for x, y, _ in self._ultimas_defectuosas]:
                        self._ultimas_defectuosas.append((cx, cy, self._pasos))

        # Recompensa para el robot actual, con las metricas REALES del paso:
        # cuantas celdas inspecciono, cuantas defectuosas encontro y cuanta
        # energia gasto. Sin esto el robot nunca recibiria el feedback positivo
        # por trabajar (ver Solucion 4).
        nuevas_inspec = rb.inspeccionadas_robot - inspec_base
        nuevas_def = rb.defectuosas_robot - def_base
        energia_delta = rb.energia.energia_consumida_total - energia_base
        reward = self._compute_reward_single(rb, energia_delta=energia_delta,
                                             nuevas_inspec=nuevas_inspec,
                                             nuevas_def=nuevas_def)

        total_libres = m.total_libres if m.total_libres > 0 else 1
        cobertura = int(m.inspec.sum()) / total_libres
        terminated = cobertura >= 0.99
        truncated = self._pasos >= self.max_steps

        info = {
            "cobertura": cobertura,
            "pasos": self._pasos,
            "robot_idx": self._current_robot_idx,
        }

        return self._make_obs_single(self._current_robot_idx), reward, terminated, truncated, info

    def _auto_assign_local(self, rb, r0, c0):
        m = self._mundo
        gs = self.cfg.grid_size
        libres = []
        for dr in range(-8, 9):
            for dc in range(-8, 9):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < m.rows and 0 <= c < m.cols and m.grid[r, c] == 0 and not m.inspec[r, c]:
                    libres.append((abs(dr) + abs(dc), r, c))
        if libres:
            libres.sort()
            _, tr, tc = libres[0]
            rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, m)

    def _compute_reward_single(self, rb, energia_delta=0.0, nuevas_inspec=0, nuevas_def=0):
        reward = -0.01  # Penalizacion base por paso de tiempo

        # 1. Rewards por hallazgos e inspeccion
        reward += nuevas_inspec * 1.0

        # 1b. Recompensa ALTA por celdas defectuosas con MULTIPLICADOR por racha.
        #     Cada celda defectuosa consecutiva encontrada por el mismo robot
        #     incrementa su racha y el premio crece: recompensa = base * racha.
        #     Incentiva activamente a seguir buscando celdas defectuosas seguidas.
        pid = id(rb)
        if nuevas_def > 0:
            racha = self._racha_defectos.get(pid, 0) + nuevas_def
            self._racha_defectos[pid] = racha
            mult = 1.0 + self.factor_racha_defectos * (racha - 1)
            reward += self.recompensa_defecto_base * nuevas_def * mult
        else:
            # No encontro defectos en este paso: se rompe la racha.
            self._racha_defectos[pid] = 0

        # 2. Penalizacion por consumo energetico (Joules)
        if self.reward_mode == "energy":
            reward -= 0.005 * energia_delta

        # 3. Penalizacion por inactividad sin detectar celdas defectuosas
        if nuevas_def > 0:
            self._pasos_sin_defecto[pid] = 0
        else:
            pasos_sin = self._pasos_sin_defecto.get(pid, 0) + 1
            self._pasos_sin_defecto[pid] = pasos_sin
            if pasos_sin > self.umbral_inactividad_pasos:
                reward -= 0.05 * (pasos_sin - self.umbral_inactividad_pasos) / 10.0

        # 4. Penalizacion fuerte por inactividad FISICA (robot quieto mas de
        #    umbral_quieto_s segundos): se le resta una penalizacion creciente
        #    con el tiempo que lleva sin cambiar de posicion. Esto evita que la
        #    politica aprenda a elegir la accion Idle o a quedarse parado.
        t_sin_mov = self._tiempo_sin_mov.get(pid, 0.0)
        if t_sin_mov > self.umbral_quieto_s:
            exceso = t_sin_mov - self.umbral_quieto_s
            reward -= self.penalizacion_quieto * (1.0 + 0.5 * exceso)

        # 5. Bonus por desplazamiento fisico (Movement Bonus): si el robot se
        #    movio en este paso (cambio de posicion), se premia con una senal
        #    positiva moderada por mantener actividad. Refuerza el contrario de
        #    quedarse quieto y suaviza el costo energetico de estar en marcha.
        ult = self._ult_pos_robot.get(pid)
        if ult is not None:
            desp = math.hypot(rb.x - ult[0], rb.y - ult[1])
            if desp > 1e-3:
                reward += 0.02 * min(desp / float(self.cfg.grid_size), 1.0)

        return float(reward)


class SwarmEnvExploracionDescentralizado(gym.Env):
    """Entorno Gymnasium Descentralizado de Exploracion Pura (V3-EXP).

    El objetivo exclusivo es descubrir el mapa en el menor tiempo y gasto de energia.
    No realiza inspeccion. Cada robot observa un parche local KxK, la actividad de
    descubrimiento de sus vecinos mas cercanos y su estado propio.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg=None, seed=None, patch_size=15, max_neighbors=3,
                 radio_comunicacion_px=400.0, max_steps=10000, headless=True,
                 physics_steps=20, auto_assign=True, reward_mode="time"):
        super().__init__()
        self.cfg = cfg or Config()
        self.seed_base = seed
        self.patch_size = patch_size
        self.max_neighbors = max_neighbors
        self.radio_comunicacion_px = radio_comunicacion_px
        self.max_steps = max_steps
        self.headless = headless
        self.physics_steps = physics_steps
        self.auto_assign = auto_assign
        self.reward_mode = reward_mode

        # Radio de vecindad para el contador de robots vecinos: variable que
        # multiplica el radio del robot (default 5x -> 3.5cm * 5 = 17.5cm).
        self.radio_vecindad_px = getattr(self.cfg, "rl_vecindad_radio_px",
                                         self.cfg.robot_radius_px * 5.0)

        self.n_patch = patch_size * patch_size
        # Obs: patch_size^2 + (max_neighbors * 4) [dx, dy, dist, act_descubrimiento]
        #      + 2 [bateria, act_propia] + 1 [contador de vecinos en radio]
        self.obs_dim = self.n_patch + (self.max_neighbors * 4) + 2 + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Accion discreta local dentro del parche KxK o Idle (K^2)
        self.action_space = gym.spaces.Discrete(self.n_patch + 1)

        self._mundo = None
        self._robots = []
        self._pasos = 0
        self._current_robot_idx = 0
        self._rng = None
        self._descubrimiento_reciente = {}  # {robot_id: celdas_descubiertas_ult_pasos}
        self._energia_prev = {}

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self.seed_base)

        from ..core.mundo import Mundo
        from ..robots.robot import Robot

        self._mundo = Mundo(self.cfg)
        self._robots = []
        self._pasos = 0
        self._current_robot_idx = 0

        # Poblar robots
        for _ in range(self.cfg.num_robots):
            x, y = self._mundo.posicion_libre_aleatoria(self._rng, self.cfg.robot_radius_px)
            if x is not None:
                r = Robot(x, y, self.cfg, self._rng)
                self._robots.append(r)
                self._mundo.robots.append(r)

        self._descubrimiento_reciente = {id(r): 0 for r in self._robots}
        self._energia_prev = {id(r): r.energia.energia_consumida_total for r in self._robots}

        return self._make_obs_single(0), {}


    def _make_obs_single(self, robot_idx):
        m = self._mundo
        rb = self._robots[robot_idx]
        gs = self.cfg.grid_size
        r_center = int(rb.y // gs)
        c_center = int(rb.x // gs)
        half_k = self.patch_size // 2

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Parche local de exploracion KxK
        idx = 0
        for dr in range(-half_k, half_k + 1):
            for dc in range(-half_k, half_k + 1):
                r = r_center + dr
                c = c_center + dc
                if 0 <= r < m.rows and 0 <= c < m.cols:
                    if m.grid[r, c] == 1:
                        val = 0.8 if m.explored[r, c] == DISCOVERED else 0.5
                    elif m.explored[r, c] == 0:
                        val = 0.5  # Niebla / no descubierto aun
                    else:
                        val = 0.0  # Libre y ya descubierto
                else:
                    val = 1.0  # Limite exterior
                obs[idx] = val
                idx += 1

        # 2. Vecinos mas cercanos con señal de descubrimiento cooperativo
        base_n = self.n_patch
        vecinos = []
        for other in self._robots:
            if other is rb:
                continue
            d = np.hypot(other.x - rb.x, other.y - rb.y)
            act_vecino = self._descubrimiento_reciente.get(id(other), 0)
            vecinos.append((d, other.x - rb.x, other.y - rb.y, act_vecino))
        vecinos.sort(key=lambda item: item[0])

        for i in range(self.max_neighbors):
            if i < len(vecinos):
                d, rel_x, rel_y, act_v = vecinos[i]
                obs[base_n + i * 4] = np.clip(rel_x / m.width_px, -1.0, 1.0)
                obs[base_n + i * 4 + 1] = np.clip(rel_y / m.height_px, -1.0, 1.0)
                obs[base_n + i * 4 + 2] = np.clip(d / self.radio_comunicacion_px, 0.0, 1.0)
                # Señal continua de actividad de descubrimiento reciente del vecino [0, 1]
                obs[base_n + i * 4 + 3] = np.clip(act_v / 10.0, 0.0, 1.0)

        # 3. Estado propio
        base_st = base_n + self.max_neighbors * 4
        obs[base_st] = rb.energia.porcentaje_bateria / 100.0
        act_propia = self._descubrimiento_reciente.get(id(rb), 0)
        obs[base_st + 1] = np.clip(act_propia / 10.0, 0.0, 1.0)

        # 4. Contador de robots vecinos dentro del radio de vecindad (normalizado):
        #    cuantos robots hay dentro de rl_vecindad_radio_mult * robot_radius_px.
        vecinos_radio = 0
        for other in self._robots:
            if other is rb:
                continue
            if np.hypot(other.x - rb.x, other.y - rb.y) <= self.radio_vecindad_px:
                vecinos_radio += 1
        max_vecinos = max(1, len(self._robots) - 1)
        obs[self.obs_dim - 1] = min(vecinos_radio / max_vecinos, 1.0)

        return obs

    def step(self, action):
        m = self._mundo
        rb = self._robots[self._current_robot_idx]
        gs = self.cfg.grid_size
        r_center = int(rb.y // gs)
        c_center = int(rb.x // gs)
        half_k = self.patch_size // 2

        act = int(action)
        if act < self.n_patch:
            dr = (act // self.patch_size) - half_k
            dc = (act % self.patch_size) - half_k
            tr, tc = r_center + dr, c_center + dc
            if 0 <= tr < m.rows and 0 <= tc < m.cols and m.grid[tr, tc] == 0:
                tx = tc * gs + gs / 2
                ty = tr * gs + gs / 2
                rb.asignar_objetivo(tx, ty, m)
            elif self.auto_assign:
                self._auto_assign_frontera_local(rb, r_center, c_center)

        # Contadores de descubrimiento previos al avance
        scout_prev = rb.scouteadas_robot
        par_prev = rb.paredes_robot

        self._current_robot_idx += 1
        if self._current_robot_idx >= len(self._robots):
            self._current_robot_idx = 0
            for _ in range(self.physics_steps):
                m.update(0.05)
            self._pasos += 1

        # Celdas nuevas descubiertas por este robot
        nuevas_descubiertas = (rb.scouteadas_robot - scout_prev) + (rb.paredes_robot - par_prev)
        self._descubrimiento_reciente[id(rb)] = nuevas_descubiertas

        # Delta de energia
        e_actual = rb.energia.energia_consumida_total
        e_prev = self._energia_prev.get(id(rb), e_actual)
        e_delta = max(0.0, e_actual - e_prev)
        self._energia_prev[id(rb)] = e_actual

        reward = self._compute_reward_exploracion(rb, nuevas_descubiertas, e_delta)


        # Terminacion por meta de exploracion (99% descubierto) o tiempo maximo
        pct_desc = m.porcentaje_descubierto
        terminated = pct_desc >= self.cfg.exploration_goal
        truncated = self._pasos >= self.max_steps

        info = {
            "porcentaje_descubierto": pct_desc,
            "pasos": self._pasos,
            "robot_idx": self._current_robot_idx,
            "nuevas_descubiertas": nuevas_descubiertas,
        }

        return self._make_obs_single(self._current_robot_idx), reward, terminated, truncated, info

    def _auto_assign_frontera_local(self, rb, r0, c0):
        """Asigna como objetivo la frontera no descubierta mas cercana en la vecindad."""
        m = self._mundo
        gs = self.cfg.grid_size
        fronteras = []
        for dr in range(-8, 9):
            for dc in range(-8, 9):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < m.rows and 0 <= c < m.cols and m.grid[r, c] == 0:
                    if m.explored[r, c] == 0:  # Celda inexplorada adyacente
                        fronteras.append((abs(dr) + abs(dc), r, c))
        if fronteras:
            fronteras.sort()
            _, tr, tc = fronteras[0]
            rb.asignar_objetivo(tc * gs + gs / 2, tr * gs + gs / 2, m)

    def _compute_reward_exploracion(self, rb, nuevas_desc, energia_delta):
        reward = -0.01  # Penalizacion base de tiempo

        # 1. Recompensa principal por descubrir celdas nuevas
        reward += nuevas_desc * 1.5

        # 2. Cooperacion: Recompensa si los vecinos cercanos estan descubriendo
        vecinos_activos = 0
        for other in self._robots:
            if other is rb:
                continue
            if np.hypot(other.x - rb.x, other.y - rb.y) < self.radio_comunicacion_px:
                vecinos_activos += self._descubrimiento_reciente.get(id(other), 0)
        reward += min(vecinos_activos * 0.2, 2.0)

        # 3. Penalizacion energetica si aplica
        if self.reward_mode == "energy":
            reward -= 0.005 * energia_delta

        return float(reward)


def make_env(cfg=None, seed=None, reward_mode="time",
             max_steps=10000, headless=True, physics_steps=20,
             map_known=True, auto_assign=False):
    """Fabrica de entornos para vectorizar con SubprocVecEnv."""
    def _init():
        env = SwarmEnv(cfg=cfg, seed=seed, reward_mode=reward_mode,
                       max_steps=max_steps, headless=headless,
                       physics_steps=physics_steps,
                       map_known=map_known, auto_assign=auto_assign)
        return env
    return _init

