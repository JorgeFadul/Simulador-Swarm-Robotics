"""Entorno RL simplificado para el Simulador Swarm 2D.
Version mejorada v3: robots idle se auto-asignan a celda mas cercana no inspeccionada.

Uso:
    python -m src.rl.env_simple
"""

import gymnasium as gym
import numpy as np

from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED


class SwarmEnvSimple(gym.Env):
    """Entorno Gymnasium simplificado para entrenamiento RL."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg=None, seed=None, reward_mode="coverage",
                 max_steps=2000, headless=True, physics_steps=50):
        super().__init__()
        self.cfg = cfg or Config()
        self.seed_base = seed
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.headless = headless
        self.physics_steps = physics_steps

        R = self.cfg.rows
        C = self.cfg.cols
        self.R, self.C = R, C
        self.n_cells = R * C
        self.N = self.cfg.num_robots

        # Observacion:
        # - Mapa de cobertura (n celdas): 0=libre, 0.5=pared, 1=inspeccionada
        # - Posicion de cada robot (N*2): x, y normalizados
        # - Estado de cada robot (N): 0=idle, 0.5=moviendo, 1=inspeccionando
        # - Paso normalizado (1)
        obs_size = self.n_cells + self.N * 2 + self.N + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Accion: cada robot elige una celda (0..n_cells-1) o idle (n_cells)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_cells + 1] * self.N
        )

        self._mundo = None
        self._robots = []
        self._pasos = 0
        self._inspeccionadas_prev = 0
        self._rng = None

    def _find_nearest_uninspected(self, rb, m):
        """Encuentra la celda libre no inspeccionada mas cercana al robot."""
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

    def _make_obs(self):
        """Construye vector de observacion compacto."""
        m = self._mundo
        R, C = self.R, self.C
        n = self.n_cells

        obs = np.zeros(n + self.N * 2 + self.N + 1, dtype=np.float32)

        # Mapa de cobertura
        for r in range(R):
            for c in range(C):
                idx = r * C + c
                if m.grid[r, c] == 1:
                    obs[idx] = 0.5
                elif m.inspec[r, c]:
                    obs[idx] = 1.0

        # Posiciones de robots
        base = n
        for i, rb in enumerate(self._robots):
            if i >= self.N:
                break
            obs[base + i * 2] = rb.x / m.width_px
            obs[base + i * 2 + 1] = rb.y / m.height_px

        # Estado de robots
        state_base = base + self.N * 2
        for i, rb in enumerate(self._robots):
            if i >= self.N:
                break
            if rb.inspeccionando:
                obs[state_base + i] = 1.0
            elif rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                obs[state_base + i] = 0.5
            else:
                obs[state_base + i] = 0.0

        # Paso normalizado
        obs[state_base + self.N] = min(self._pasos / self.max_steps, 1.0)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.seed_base is not None:
            self._rng = np.random.default_rng(self.seed_base)
        else:
            self._rng = np.random.default_rng()

        self._mundo = Mundo(self.cfg, self._rng)
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

        # Escaneo inicial
        for rb in self._robots:
            exp_radius = rb.detection_radius * self.cfg.exploration_radius_multiplier
            s, p = self._mundo.explorar_desde(rb.x, rb.y, exp_radius)
            rb.scouteadas_robot += s
            rb.paredes_robot += p
            rb._explorar_desde_propio(self._mundo, exp_radius)

        self._pasos = 0
        self._inspeccionadas_prev = int(self._mundo.inspec.sum())

        return self._make_obs(), {}

    def step(self, action):
        m = self._mundo
        R, C = self.R, self.C
        n = self.n_cells
        gs = self.cfg.grid_size

        # Asignar objetivos a robots segun la accion del agente
        celdas_asignadas = set()
        for i, rb in enumerate(self._robots):
            if i >= len(action):
                break
            # Solo asignar si el robot esta idle (sin objetivo activo)
            if rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                continue
            if rb.inspeccionando:
                continue

            act = int(action[i])
            target_cell = None
            if act < n:
                r, c = divmod(act, C)
                if (r, c) not in celdas_asignadas and m.grid[r, c] == 0 and not m.inspec[r, c]:
                    target_cell = (r, c)

            # Si la accion del agente no es valida, auto-asignar mas cercana
            if target_cell is None:
                target_cell = self._find_nearest_uninspected(rb, m)

            if target_cell is not None:
                r, c = target_cell
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                rb.asignar_objetivo(cx, cy, m)
                celdas_asignadas.add((r, c))

        # Simular fisica
        dt = 0.05
        for _sub in range(self.physics_steps):
            for rb in self._robots:
                exp_radius = rb.detection_radius * self.cfg.exploration_radius_multiplier
                s, p = m.explorar_desde(rb.x, rb.y, exp_radius)
                rb.scouteadas_robot += s
                rb.paredes_robot += p
                rb._explorar_desde_propio(m, exp_radius)
            for rb in self._robots:
                rb.update(dt, m)

        # Reasignar robots que completaron su objetivo
        for rb in self._robots:
            if not rb.inspeccionando and (rb.goal is None or rb.goal_alcanzado):
                nc = self._find_nearest_uninspected(rb, m)
                if nc is not None:
                    r, c = nc
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    rb.asignar_objetivo(cx, cy, m)

        self._pasos += 1

        # Calcular recompensa
        insp_actual = int(m.inspec.sum())
        nuevas = insp_actual - self._inspeccionadas_prev
        self._inspeccionadas_prev = insp_actual

        reward = self._compute_reward(nuevas, insp_actual, n)

        # Terminacion
        total_libres = m.total_libres if m.total_libres > 0 else 1
        cobertura = insp_actual / total_libres
        terminated = cobertura >= 0.99
        truncated = self._pasos >= self.max_steps

        info = {
            "inspeccionadas": insp_actual,
            "total_libres": total_libres,
            "cobertura": cobertura,
            "pasos": self._pasos,
            "nuevas_inspeccionadas": nuevas,
        }

        return self._make_obs(), reward, terminated, truncated, info

    def _compute_reward(self, nuevas, insp_actual, n_cells):
        if self.reward_mode == "coverage":
            return nuevas * 2.0
        elif self.reward_mode == "time":
            return nuevas * 1.0 - 0.005
        else:
            return nuevas * 1.0

    def render(self):
        pass

    def close(self):
        pass


def make_env_simple(cfg=None, seed=None, reward_mode="coverage",
                    max_steps=2000, headless=True, physics_steps=50):
    """Fabrica de entornos para vectorizar."""
    def _init():
        env = SwarmEnvSimple(cfg=cfg, seed=seed, reward_mode=reward_mode,
                             max_steps=max_steps, headless=headless,
                             physics_steps=physics_steps)
        return env
    return _init
