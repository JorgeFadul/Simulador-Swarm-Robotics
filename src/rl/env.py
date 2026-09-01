"""Entorno Gymnasium compatible con el Simulador Swarm 2D (Fase 4).

Envuelve `Mundo` + `Simulacion` en una interfaz estandar de RL para entrenar
agentes con PPO/SAC via Stable-Baselines3.  El agente centralizado decide que
celdas asignar como objetivos a los robots; la planificacion de ruta (A*) y el
movimiento diferencial se manejan por el simulador existente.

Espacio de observacion (Box, float32):
    [0  .. R*C-1]     mapa conocido (libre=0, pared=1, inspeccionada=2, normalizado/2)
    [R*C .. 2*R*C-1]  mapa de defectos revelados (ground truth si visible, 0 si no)
    [2RC .. 2RC+N*3]   por robot: x_normalizado, y_normalizado, bateria_pct
    [2RC+N*3 .. end]   celdas_inspeccionadas_total (1), celdas_pendientes (1),
                       pasos_normalizados (1)

Espacio de accion (Discrete, Multi-Agent):
    N acciones por robot = R*C + 1 (idle).  Se aplica secuencialmente: robot 0
    elige, se bloquea la celda, robot 1 elige de las restantes, etc.  Si la
    celda elegida ya fue asignada o no es valida, se consume como idle.

Funcion de recompensa (configurable via `reward_mode`):
    "time"   - Minimiza tiempo total: +1 por nueva inspeccion, -0.001 por paso.
    "energy" - Minimiza energia: +1 por inspeccion, -w*energia_consumida_step.
"""

import gymnasium as gym
import numpy as np

from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED


class SwarmEnv(gym.Env):
    """Entorno Gymnasium para el Simulador Swarm 2D."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg=None, seed=None, reward_mode="time",
                 max_steps=10000, headless=True, physics_steps=20):
        super().__init__()
        self.cfg = cfg or Config()
        self.seed_base = seed
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.headless = headless
        self.physics_steps = physics_steps  # sub-pasos de fisica por step RL

        R = self.cfg.rows
        C = self.cfg.cols
        self.R, self.C = R, C
        self.n_cells = R * C
        self.N = self.cfg.num_robots

        # --- Espacio de observacion ---
        obs_size = 2 * self.n_cells + self.N * 3 + 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # --- Espacio de accion (multi-discrete: N robots, cada uno elige R*C+1) ---
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_cells + 1] * self.N
        )

        # Estado interno
        self._mundo = None
        self._robots = []
        self._pasos = 0
        self._inspeccionadas_prev = 0
        self._energia_total_prev = 0.0
        self._rng = None

    def _make_obs(self):
        """Construye el vector de observacion."""
        m = self._mundo
        R, C = self.R, self.C
        n = self.n_cells

        obs = np.zeros(2 * n + self.N * 3 + 3, dtype=np.float32)

        # Canal 0: mapa conocido (0=libre, 0.5=pared, 1=inspeccionada)
        for r in range(R):
            for c in range(C):
                idx = r * C + c
                if m.grid[r, c] == 1:
                    obs[idx] = 0.5
                elif m.inspec[r, c]:
                    obs[idx] = 1.0

        # Canal 1: defectos revelados (solo si la celda fue inspeccionada)
        for r in range(R):
            for c in range(C):
                idx = n + r * C + c
                if m.inspec[r, c] and m.resultado[r, c] == 1:
                    obs[idx] = 1.0

        # Datos por robot
        base = 2 * n
        for i, rb in enumerate(self._robots):
            if i >= self.N:
                break
            obs[base + i * 3] = rb.x / m.width_px
            obs[base + i * 3 + 1] = rb.y / m.height_px
            obs[base + i * 3 + 2] = rb.energia.porcentaje_bateria / 100.0

        # Resumen global
        base_r = 2 * n + self.N * 3
        total_libres = m.total_libres if m.total_libres > 0 else 1
        insp = int(m.inspec.sum())
        pend = max(0, total_libres - insp)
        obs[base_r] = insp / total_libres
        obs[base_r + 1] = pend / total_libres
        obs[base_r + 2] = min(self._pasos / self.max_steps, 1.0)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.seed_base is not None:
            self._rng = np.random.default_rng(self.seed_base)
        else:
            self._rng = np.random.default_rng()

        self._mundo = Mundo(self.cfg, self._rng)
        # Etapa segun el modo de fase del cfg
        if self.cfg.modo_fase == "inspeccion":
            self._mundo.etapa = "inspeccion"
        elif self.cfg.modo_fase == "hibrido":
            self._mundo.etapa = "inspeccion"  # hibrido: inspeccion habilitada
        else:
            self._mundo.etapa = "exploracion"

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
        self._energia_total_prev = sum(
            rb.energia.energia_consumida_total for rb in self._robots
        )

        return self._make_obs(), {}

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
            act = int(action[i])
            if act >= n:
                continue  # idle
            r, c = divmod(act, C)
            if (r, c) in celdas_asignadas:
                continue
            if m.grid[r, c] == 1 or m.inspec[r, c]:
                continue
            cx = c * gs + gs / 2
            cy = r * gs + gs / 2
            if rb.necesita_objetivo and not rb.cola_tareas:
                rb.asignar_objetivo(cx, cy, m)
                celdas_asignadas.add((r, c))

        # --- Simular multiples sub-pasos de fisica por step RL ---
        dt = 0.05
        for _sub in range(self.physics_steps):
            # Radar: cada robot escanea su entorno
            for rb in self._robots:
                exp_radius = rb.detection_radius * self.cfg.exploration_radius_multiplier
                s, p = m.explorar_desde(rb.x, rb.y, exp_radius)
                rb.scouteadas_robot += s
                rb.paredes_robot += p
                rb._explorar_desde_propio(m, exp_radius)
            # Movimiento
            for rb in self._robots:
                rb.update(dt, m)

        self._pasos += 1

        # --- Calcular recompensa ---
        insp_actual = int(m.inspec.sum())
        nuevas = insp_actual - self._inspeccionadas_prev
        self._inspeccionadas_prev = insp_actual

        energia_actual = sum(rb.energia.energia_consumida_total
                             for rb in self._robots)
        energia_step = energia_actual - self._energia_total_prev
        self._energia_total_prev = energia_actual

        reward = self._compute_reward(nuevas, energia_step)

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
        }

        return self._make_obs(), reward, terminated, truncated, info

    def _compute_reward(self, nuevas, energia_step):
        if self.reward_mode == "time":
            # +1 por nueva inspeccion, -0.001 por paso para minimizar tiempo
            return nuevas * 1.0 - 0.001
        elif self.reward_mode == "energy":
            # +1 por inspeccion, penalizar energia consumida
            w = self.cfg.energia_inspeccion_j * 10
            return nuevas * 1.0 - w * energia_step
        else:
            return nuevas * 1.0

    def render(self):
        pass  # la renderizacion se hace via simulacion.py

    def close(self):
        pass


def make_env(cfg=None, seed=None, reward_mode="time",
             max_steps=10000, headless=True, physics_steps=20):
    """Fabrica de entornos para vectorizar con SubprocVecEnv."""
    def _init():
        env = SwarmEnv(cfg=cfg, seed=seed, reward_mode=reward_mode,
                       max_steps=max_steps, headless=headless,
                       physics_steps=physics_steps)
        return env
    return _init
