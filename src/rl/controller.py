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

        # Reasignacion auto (robots que quedaron libres tras completar)
        if self.auto_assign:
            for rb in robots:
                if not rb.inspeccionando and (rb.goal is None or rb.goal_alcanzado):
                    nc = self._nearest_uninspected(rb, mundo, gs)
                    if nc is not None:
                        r, c = nc
                        if (r, c) in celdas_asignadas:
                            continue
                        rb.asignar_objetivo(c * gs + gs / 2, r * gs + gs / 2, mundo)
                        celdas_asignadas.add((r, c))
                        n_asignados += 1

        return n_asignados
