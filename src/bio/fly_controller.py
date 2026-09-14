"""Controlador de enjambre basado en cerebros de mosca (Drosophila melanogaster).

Cada robot del enjambre porta una instancia individual de `FlyBrain` con su propio
Mushroom Body, Central Complex y plasticidad dopaminergica en tiempo real.
Integra percepcion social (densidad de otras moscas y distancias) para coordinar
alrededor de 10 robots sin colisiones ni hacinamiento.
"""

import math
import numpy as np

from .fly_brain import FlyBrain
from ..core.mundo import DISCOVERED


class FlyBrainSwarmController:
    """Controlador bioinspirado para el enjambre de robots."""

    def __init__(self, cfg, patch_size: int = 15, radio_social_px: float = 250.0,
                 radio_alerta_px: float = 500.0, rng: np.random.Generator = None):
        self.cfg = cfg
        self.patch_size = patch_size
        self.half_k = patch_size // 2
        self.radio_social_px = radio_social_px
        self.radio_alerta_px = radio_alerta_px
        self.rng = rng or np.random.default_rng(123)

        # Dimension del vector sensorial:
        # Patch local KxK (225) + Canal Social [densidad, dx1, dy1, d1, dx2, dy2, d2] (7)
        # + Alerta Defecto (1) + Bateria (1) + Inercia (1) = 235 floats
        self.n_inputs = (patch_size * patch_size) + 7 + 1 + 1 + 1

        self.cerebros: dict[int, FlyBrain] = {}
        self._ult_pos: dict[int, tuple[float, float]] = {}
        self._tiempo_quieto: dict[int, float] = {}
        self._ultimas_defectuosas: list[tuple[float, float]] = []

    def _obtener_cerebro(self, robot) -> FlyBrain:
        rid = id(robot)
        if rid not in self.cerebros:
            # Semilla individual para cada cerebro
            sub_rng = np.random.default_rng(self.rng.integers(0, 1000000))
            self.cerebros[rid] = FlyBrain(n_inputs=self.n_inputs, n_kc=1000,
                                          n_direcciones=8, rng=sub_rng)
            self._ult_pos[rid] = (robot.x, robot.y)
            self._tiempo_quieto[rid] = 0.0
        return self.cerebros[rid]

    def _construir_percepcion(self, robot, mundo) -> np.ndarray:
        """Construye el vector sensorial egocentrico incluyendo el canal social."""
        m = mundo
        gs = self.cfg.grid_size
        r_c = int(robot.y // gs)
        c_c = int(robot.x // gs)

        x = np.zeros(self.n_inputs, dtype=np.float32)

        # 1. Parche local KxK
        idx = 0
        for dr in range(-self.half_k, self.half_k + 1):
            for dc in range(-self.half_k, self.half_k + 1):
                r = r_c + dr
                c = c_c + dc
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
                    val = 0.25  # Limites exteriores
                x[idx] = val
                idx += 1

        # 2. Canal Social (compañeras moscas cerca)
        vecinos = []
        for other in m.robots:
            if other is robot:
                continue
            dist = math.hypot(other.x - robot.x, other.y - robot.y)
            vecinos.append((dist, other.x - robot.x, other.y - robot.y))
        vecinos.sort(key=lambda item: item[0])

        n_total_otros = max(1, len(m.robots) - 1)
        # Densidad social: proporcion de moscas dentro del radio social
        cercanas = [v for v in vecinos if v[0] <= self.radio_social_px]
        densidad_social = len(cercanas) / n_total_otros
        x[idx] = float(np.clip(densidad_social, 0.0, 1.0))
        idx += 1

        # Vectores de las 2 compañeras mas proximas
        for i in range(2):
            if i < len(vecinos):
                d, dx, dy = vecinos[i]
                x[idx] = float(np.clip(dx / m.width_px, -1.0, 1.0))
                x[idx + 1] = float(np.clip(dy / m.height_px, -1.0, 1.0))
                x[idx + 2] = float(np.clip(d / self.radio_social_px, 0.0, 1.0))
            idx += 3

        # 3. Alerta por defectos
        val_alerta = 0.0
        if self._ultimas_defectuosas:
            for dx, dy in self._ultimas_defectuosas[-5:]:
                dist = math.hypot(dx - robot.x, dy - robot.y)
                if dist < self.radio_alerta_px:
                    v = 1.0 - (dist / self.radio_alerta_px)
                    if v > val_alerta:
                        val_alerta = v
        x[idx] = float(val_alerta)
        idx += 1

        # 4. Bateria y estado
        x[idx] = float(robot.energia.porcentaje_bateria / 100.0)
        idx += 1
        t_q = self._tiempo_quieto.get(id(robot), 0.0)
        x[idx] = float(np.clip(t_q / 5.0, 0.0, 1.0))

        return x

    def actualizar(self, mundo):
        """Ciclo de actualizacion llamado por Simulacion en cada tick."""
        m = mundo
        gs = self.cfg.grid_size
        n_asignados = 0

        # Memoria global de celdas defectuosas
        for r in range(m.rows):
            for c in range(m.cols):
                if m.inspec[r, c] and m.resultado[r, c] == 1:
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    if (cx, cy) not in self._ultimas_defectuosas:
                        self._ultimas_defectuosas.append((cx, cy))

        dt = 0.05

        for robot in m.robots:
            rid = id(robot)
            cerebro = self._obtener_cerebro(robot)

            # Monitorear movimiento
            ult = self._ult_pos.get(rid, (robot.x, robot.y))
            desplazamiento = math.hypot(robot.x - ult[0], robot.y - ult[1])
            if desplazamiento < 0.2:
                self._tiempo_quieto[rid] = self._tiempo_quieto.get(rid, 0.0) + dt
            else:
                self._tiempo_quieto[rid] = 0.0
            self._ult_pos[rid] = (robot.x, robot.y)

            # 1. Calcular Dopamina generada en este paso
            dopamina = -0.01  # Costo base de tiempo

            # Recompensas por descubrimientos e inspeccion
            if hasattr(robot, "_nuevas_inspec_tick") and robot._nuevas_inspec_tick:
                dopamina += 1.5
                robot._nuevas_inspec_tick = False

            if hasattr(robot, "_defecto_hallado_tick") and robot._defecto_hallado_tick:
                dopamina += 6.0  # Ráfaga dopaminérgica mayor
                robot._defecto_hallado_tick = False

            # Castigo por estancamiento / estar quieto
            if self._tiempo_quieto[rid] > 1.0:
                dopamina -= 0.1 * min(self._tiempo_quieto[rid], 5.0)

            # Inyectar la dopamina y modular plasticidad
            cerebro.inyectar_recompensa(dopamina)

            # 2. Si el robot necesita nuevo objetivo, el cerebro decide
            if robot.inspeccionando:
                continue

            if robot.goal is None or robot.goal_alcanzado or len(robot.ruta) == 0:
                sensorial = self._construir_percepcion(robot, mundo)
                target_angle, speed_factor = cerebro.procesar(
                    sensorial, robot.angle, robot.omega, dt
                )

                # Proyectar el rumbo hacia un waypoint a una distancia razonable (ej. 3-4 celdas)
                dist_avance = gs * 3.5 * speed_factor
                tx = robot.x + math.cos(target_angle) * dist_avance
                ty = robot.y + math.sin(target_angle) * dist_avance

                # Clampear dentro del mapa libre
                tx = max(gs * 1.5, min(m.width_px - gs * 1.5, tx))
                ty = max(gs * 1.5, min(m.height_px - gs * 1.5, ty))

                tr = int(ty // gs)
                tc = int(tx // gs)
                if 0 <= tr < m.rows and 0 <= tc < m.cols and m.grid[tr, tc] == 0:
                    robot.asignar_objetivo(tx, ty, m)
                    n_asignados += 1
                else:
                    # Fallback suave hacia celda libre mas cercana en esa direccion
                    self._auto_assign_bio(robot, tr, tc, m)

        return n_asignados

    def _auto_assign_bio(self, robot, r0, c0, mundo):
        gs = self.cfg.grid_size
        for dr in (0, 1, -1, 2, -2):
            for dc in (0, 1, -1, 2, -2):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < mundo.rows and 0 <= c < mundo.cols and mundo.grid[r, c] == 0 and not mundo.inspec[r, c]:
                    robot.asignar_objetivo(c * gs + gs / 2, r * gs + gs / 2, mundo)
                    return
