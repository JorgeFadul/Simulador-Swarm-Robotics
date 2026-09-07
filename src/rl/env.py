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
