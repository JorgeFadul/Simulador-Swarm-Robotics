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
                 radio_alerta_px: float = 500.0, weights_path: str = "models/bio/fly_swarm_latest.npz",
                 rng: np.random.Generator = None, n_kc: int = 1000, mutacion: float = 0.0):
        self.cfg = cfg
        self.patch_size = patch_size
        self.half_k = patch_size // 2
        self.radio_social_px = radio_social_px
        self.radio_alerta_px = radio_alerta_px
        self.weights_path = weights_path
        self.rng = rng or np.random.default_rng(123)
        self.n_kc = n_kc
        self.mutacion = mutacion

        # Dimension del vector sensorial:
        # Patch local KxK (225) + Canal Social [densidad, dx1, dy1, d1, dx2, dy2, d2] (7)
        # + Alerta Defecto (1) + Bateria (1) + Inercia (1) = 235 floats
        self.n_inputs = (patch_size * patch_size) + 7 + 1 + 1 + 1

        self.cerebros: dict[int, FlyBrain] = {}
        self._ult_pos: dict[int, tuple[float, float]] = {}
        self._tiempo_quieto: dict[int, float] = {}
        self._dopamina_pendiente: dict[int, float] = {}
        self._ultimas_defectuosas: list[tuple[float, float]] = []
        self._set_defectuosas: set[tuple[float, float]] = set()
        # Racha de defectos consecutivos (multiplica recompensa por 2^racha)
        self._racha_defectos: dict[int, int] = {}
        # Tiempo (en segundos sim) desde que el robot encontró su último defecto
        self._pasos_sin_defecto: dict[int, float] = {}

    def _obtener_cerebro(self, robot) -> FlyBrain:
        rid = id(robot)
        if rid not in self.cerebros:
            # Semilla individual para cada cerebro
            sub_rng = np.random.default_rng(self.rng.integers(0, 1000000))
            cerebro = FlyBrain(n_inputs=self.n_inputs, n_kc=self.n_kc,
                               n_direcciones=8, rng=sub_rng)
            # Intentar cargar pesos consolidados si existen
            import os
            if self.weights_path and os.path.exists(self.weights_path):
                try:
                    cerebro.cargar_pesos(self.weights_path, mutacion=self.mutacion)
                except Exception:
                    pass
            self.cerebros[rid] = cerebro
            self._ult_pos[rid] = (robot.x, robot.y)
            self._tiempo_quieto[rid] = 0.0
        return self.cerebros[rid]

    def _inyectar_pendiente(self, rid: int, cerebro: FlyBrain):
        """Libera la dopamina acumulada durante la trayectoria hacia el cerebro.

        Con credit assignment por trayectoria: los eventos (baldosas nuevas,
        defectos, atascos) que ocurren mientras la mosca sigue un rumbo se
        acumulan y se inyectan en el MOMENTO de elegir el siguiente rumbo,
        reforzando la percepcion que origino esa decision (last_kc_act) en vez
        de una activacion arbitraria del tick.
        """
        pend = self._dopamina_pendiente.get(rid, 0.0)
        if pend != 0.0:
            cerebro.inyectar_recompensa(pend)
            self._dopamina_pendiente[rid] = 0.0

    def _acumular_dopamina(self, rid: int, dopamina: float):
        self._dopamina_pendiente[rid] = self._dopamina_pendiente.get(rid, 0.0) + dopamina

    def drenar_pendientes(self):
        """Al terminar la run, libera la dopamina que quedo sin rumbos nuevos."""
        for rid, cerebro in self.cerebros.items():
            self._inyectar_pendiente(rid, cerebro)

    def guardar_enjambre(self, ruta: str = None):
        """Consolida y guarda el mejor cerebro o el promedio del enjambre en disco."""
        import os
        ruta = ruta or self.weights_path
        if not self.cerebros or not ruta:
            return
        self.drenar_pendientes()
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        # Seleccionar el cerebro con mayor dopamina acumulada (el mas exitoso de la corrida)
        mejor_cerebro = max(self.cerebros.values(), key=lambda c: c.dopamina_acumulada)
        mejor_cerebro.guardar_pesos(ruta)


    def _construir_percepcion(self, robot, mundo) -> np.ndarray:
        """Construye el vector sensorial egocentrico incluyendo el canal social."""
        m = mundo
        gs = self.cfg.grid_size
        r_c = int(robot.y // gs)
        c_c = int(robot.x // gs)

        x = np.zeros(self.n_inputs, dtype=np.float32)

        # 1. Parche local KxK (vectorizado sobre el mundo)
        capa = np.full((m.rows, m.cols), 0.5, dtype=np.float32)
        capa[(m.grid == 0) & (m.explored != 0)] = 0.0
        paredes = m.grid == 1
        capa[paredes] = np.where(m.explored[paredes] == DISCOVERED, 0.25, 0.5)
        inspec = m.inspec == 1
        capa[inspec] = np.where(m.resultado[inspec] == 1, 1.0, 0.75)

        # Almohadilla con el valor de borde exterior (0.25 igual que en el bucle)
        pad = self.half_k
        pad_r = 2 * pad
        capa_pad = np.full((m.rows + pad_r, m.cols + pad_r), 0.25, dtype=np.float32)
        capa_pad[pad:pad + m.rows, pad:pad + m.cols] = capa
        patch = capa_pad[r_c:r_c + self.patch_size, c_c:c_c + self.patch_size].ravel()
        x[:patch.size] = patch
        idx = patch.size

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
        objetivos_ocupados = self._celdas_objetivo_activas(m)

        # Memoria global de celdas defectuosas (vectorizado)
        for r, c in np.argwhere((m.inspec == 1) & (m.resultado == 1)):
            cx = c * gs + gs / 2
            cy = r * gs + gs / 2
            if (float(cx), float(cy)) not in self._set_defectuosas:
                self._ultimas_defectuosas.append((float(cx), float(cy)))
                self._set_defectuosas.add((float(cx), float(cy)))

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

            # Acumular tiempo sin encontrar defecto (penalización creciente)
            self._pasos_sin_defecto[rid] = self._pasos_sin_defecto.get(rid, 0.0) + dt

            # Recompensas por descubrimientos e inspeccion
            if hasattr(robot, "_nuevas_inspec_tick") and robot._nuevas_inspec_tick:
                dopamina += 1.5
                robot._nuevas_inspec_tick = False

            if hasattr(robot, "_defecto_hallado_tick") and robot._defecto_hallado_tick:
                # Racha: si el último defecto fue hace <10s, la racha sigue
                tiempo_desde_ult = self._pasos_sin_defecto.get(rid, 999.0)
                if tiempo_desde_ult <= 10.0:
                    # Dentro de la ventana → racha continúa
                    racha = self._racha_defectos.get(rid, 0)
                else:
                    # Fuera de ventana → racha nueva
                    racha = 0
                multiplicador = 2.0 ** racha
                dopamina += 6.0 * multiplicador
                self._racha_defectos[rid] = racha + 1
                # Resetear temporizador de tiempo sin defecto
                self._pasos_sin_defecto[rid] = 0.0
                robot._defecto_hallado_tick = False
            else:
                # Solo resetear racha si pasaron más de 10s sin defecto
                if self._pasos_sin_defecto.get(rid, 0.0) > 10.0:
                    self._racha_defectos[rid] = 0

            # Penalización proporcional al tiempo sin encontrar defecto nuevo
            t_sin = self._pasos_sin_defecto[rid]
            if t_sin > 2.0:  # Gracia de 2s antes de penalizar
                dopamina -= 0.05 * min(t_sin - 2.0, 30.0)

            # Castigo por estancamiento / estar quieto
            if self._tiempo_quieto[rid] > 1.0:
                dopamina -= 0.1 * min(self._tiempo_quieto[rid], 5.0)

            # La dopamina se acumula por trayectoria y se inyecta al elegir el
            # siguiente rumbo (credit assignment), no en cada tick sobre una
            # activacion arbitraria.
            self._acumular_dopamina(rid, dopamina)

            # 2. Si el robot necesita nuevo objetivo, el cerebro decide
            if robot.inspeccionando:
                continue

            if robot.goal is None or robot.goal_alcanzado:
                # Creditar la trayectoria anterior justo antes de decidir
                self._inyectar_pendiente(rid, cerebro)
                sensorial = self._construir_percepcion(robot, mundo)
                target_angle, speed_factor = cerebro.procesar(
                    sensorial, robot.angle, robot.omega, dt
                )

                # Proyectar el rumbo hacia una celda inspeccionable. La accion
                # de inspeccion ahora se cierra con pausa sobre el centro de la
                # celda; por eso la estrategia de mosca debe emitir centros de
                # celdas libres no inspeccionadas, no waypoints arbitrarios.
                dist_avance = gs * 3.5 * speed_factor
                objetivo = self._seleccionar_objetivo_inspeccionable(
                    robot, target_angle, dist_avance, m, objetivos_ocupados
                )

                if objetivo is not None:
                    tr, tc = objetivo
                    tx = tc * gs + gs / 2
                    ty = tr * gs + gs / 2
                    robot.asignar_objetivo(tx, ty, m)
                    objetivos_ocupados.add((tr, tc))
                    n_asignados += 1
                else:
                    # Fallback suave: si el cono elegido no tiene trabajo
                    # disponible, tomar cualquier celda pendiente cercana.
                    if self._auto_assign_bio(robot, m, objetivos_ocupados):
                        n_asignados += 1

        return n_asignados

    def _celdas_objetivo_activas(self, mundo) -> set[tuple[int, int]]:
        """Celdas que otros robots ya tienen como meta o estan probando."""
        gs = self.cfg.grid_size
        ocupadas: set[tuple[int, int]] = set()
        for rb in mundo.robots:
            if rb.celda_probando is not None:
                ocupadas.add(tuple(rb.celda_probando))
            elif rb.goal_mode and rb.goal is not None and not rb.goal_alcanzado:
                c = int(round((rb.goal[0] - gs / 2) / gs))
                r = int(round((rb.goal[1] - gs / 2) / gs))
                if 0 <= r < mundo.rows and 0 <= c < mundo.cols:
                    ocupadas.add((r, c))
        return ocupadas

    def _celda_pendiente(self, mundo, r: int, c: int,
                         ocupadas: set[tuple[int, int]]) -> bool:
        return (0 <= r < mundo.rows and 0 <= c < mundo.cols
                and mundo.grid[r, c] == 0
                and not mundo.inspec[r, c]
                and (r, c) not in ocupadas)

    def _seleccionar_objetivo_inspeccionable(self, robot, angulo: float,
                                             distancia: float, mundo,
                                             ocupadas: set[tuple[int, int]]):
        """Convierte el rumbo bioinspirado en una celda pendiente concreta.

        Se priorizan celdas no inspeccionadas que caen cerca del rayo elegido
        por el Central Complex. Esto conserva la decision direccional aprendida,
        pero la adapta al contrato actual de inspeccion por centros de celda.
        """
        gs = self.cfg.grid_size
        px = robot.x + math.cos(angulo) * distancia
        py = robot.y + math.sin(angulo) * distancia
        px = max(gs * 1.5, min(mundo.width_px - gs * 1.5, px))
        py = max(gs * 1.5, min(mundo.height_px - gs * 1.5, py))
        r0 = int(py // gs)
        c0 = int(px // gs)

        if self._celda_pendiente(mundo, r0, c0, ocupadas):
            return r0, c0

        max_radio = max(2, self.patch_size // 2)
        ux = math.cos(angulo)
        uy = math.sin(angulo)
        mejor = None
        mejor_score = float("inf")
        for radio in range(1, max_radio + 1):
            for dr in range(-radio, radio + 1):
                for dc in range(-radio, radio + 1):
                    if max(abs(dr), abs(dc)) != radio:
                        continue
                    r = r0 + dr
                    c = c0 + dc
                    if not self._celda_pendiente(mundo, r, c, ocupadas):
                        continue
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    vx = cx - robot.x
                    vy = cy - robot.y
                    dist = math.hypot(vx, vy)
                    if dist <= 1e-6:
                        alineacion = 0.0
                    else:
                        alineacion = max(0.0, 1.0 - ((vx / dist) * ux + (vy / dist) * uy))
                    score = radio + alineacion * 2.0 + dist / max(gs, 1)
                    if score < mejor_score:
                        mejor_score = score
                        mejor = (r, c)
            if mejor is not None:
                return mejor
        return None

    def _auto_assign_bio(self, robot, mundo, ocupadas: set[tuple[int, int]]):
        gs = self.cfg.grid_size
        r0 = int(robot.y // gs)
        c0 = int(robot.x // gs)
        max_radio = max(mundo.rows, mundo.cols)
        for radio in range(max_radio + 1):
            for dr in range(-radio, radio + 1):
                for dc in range(-radio, radio + 1):
                    if max(abs(dr), abs(dc)) != radio:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if self._celda_pendiente(mundo, r, c, ocupadas):
                        robot.asignar_objetivo(c * gs + gs / 2, r * gs + gs / 2, mundo)
                        ocupadas.add((r, c))
                        return True
        return False
