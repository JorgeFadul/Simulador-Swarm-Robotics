import math
import numpy as np
from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED


class Robot:
    STOPPED = 0
    ROTATING = 1
    MOVING = 2

    def __init__(self, x: float, y: float, cfg: Config, rng: np.random.Generator):
        self.x = x
        self.y = y
        self.cfg = cfg
        self.rng = rng

        self.radius = cfg.robot_radius_px
        self.max_speed = cfg.robot_max_speed_px_s
        self.detection_radius = cfg.detection_radius_px

        self.angle = rng.uniform(0, 2 * math.pi)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.omega = 0.0

        self.timer = 0.0
        self.duration = 0.0

        self.target_angle = self.angle
        self.target_speed = 0.0

        self.state = self.STOPPED
        self.turn_speed = cfg.robot_turn_speed_rad_s
        self.turn_threshold = math.radians(5)

        self.scouteadas_robot = 0
        self.visitadas_robot = 0
        self.paredes_robot = 0
        self.explored_robot = np.zeros((cfg.rows, cfg.cols), dtype=np.int8)

        self._elegir_nuevo_recorrido()

    def _elegir_nuevo_recorrido(self):
        self.target_angle = self.rng.uniform(0, 2 * math.pi)
        self.target_speed = self.rng.uniform(0.5, 1.0) * self.max_speed
        self.duration = self.rng.uniform(
            self.cfg.random_walk_interval_min, self.cfg.random_walk_interval_max
        )
        self.timer = 0.0
        self._iniciar_giro(self.target_angle)

    def _iniciar_giro(self, target_angle):
        self.target_angle = target_angle
        diff = self._diferencia_angular(self.target_angle, self.angle)
        if abs(diff) <= self.turn_threshold:
            self.angle = self.target_angle
            self.state = self.MOVING
            self._poner_en_marcha()
        else:
            self.state = self.ROTATING
            direction = 1.0 if diff > 0 else -1.0
            self.omega = direction * self.turn_speed
            self.speed = 0.0
            self.vx = 0.0
            self.vy = 0.0

    def _poner_en_marcha(self):
        self.speed = self.target_speed
        self.omega = 0.0
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

    def _frenar(self):
        self.omega = 0.0
        self.speed = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def _diferencia_angular(self, target, current):
        diff = target - current
        return math.atan2(math.sin(diff), math.cos(diff))

    @property
    def intent_tiempo_restante(self):
        """Segundos que restan del movimiento planificado."""
        return max(0.0, self.duration - self.timer)

    @property
    def intent_distancia(self):
        """Distancia prevista si el robot siguiera la intencion actual."""
        return self.target_speed * self.intent_tiempo_restante

    def _explorar_desde_propio(self, mundo, radius):
        gs = self.cfg.grid_size
        rows, cols = self.cfg.rows, self.cfg.cols
        min_c = max(0, int((self.x - radius) // gs))
        max_c = min(cols - 1, int((self.x + radius) // gs))
        min_r = max(0, int((self.y - radius) // gs))
        max_r = min(rows - 1, int((self.y + radius) // gs))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.explored_robot[r, c]:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if (math.hypot(cx - self.x, cy - self.y) < radius and
                        mundo.tiene_linea_de_vista(self.x, self.y, cx, cy)):
                    if mundo.grid[r, c] == 0:
                        self.explored_robot[r, c] = SCOUTED
                    else:
                        self.explored_robot[r, c] = DISCOVERED

    def _explorar_propio_colision(self, mundo):
        """Sincroniza explored_robot con el mapa global tras una colision."""
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                if mundo.explored[r, c] and not self.explored_robot[r, c]:
                    self.explored_robot[r, c] = mundo.explored[r, c]

    def update(self, dt: float, mundo: Mundo):
        rep_dx, rep_dy, has_obs = mundo.detectar_en_rango(
            self.x, self.y, self.detection_radius
        )

        colision_inminente = mundo.colision_frontal(
            self.x, self.y,
            math.cos(self.angle) * self.detection_radius,
            math.sin(self.angle) * self.detection_radius,
            self.radius, mundo.robots,
            solo_descubiertas=True
        )

        hay_peligro = colision_inminente or has_obs

        if self.state == self.MOVING:
            self.timer += dt
            if hay_peligro:
                self._frenar()
                if has_obs:
                    direccion_libre = math.atan2(rep_dy, rep_dx)
                    self._iniciar_giro(direccion_libre)
                else:
                    self._elegir_nuevo_recorrido()
            elif self.timer >= self.duration:
                self._elegir_nuevo_recorrido()
            else:
                self._poner_en_marcha()

        elif self.state == self.ROTATING:
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))

            diff = self._diferencia_angular(self.target_angle, self.angle)
            if abs(diff) <= self.turn_threshold:
                self.angle = self.target_angle
                self.state = self.MOVING
                self._poner_en_marcha()

        # Movimiento
        if self.state == self.MOVING and self.speed > 0:
            if self.speed < self.target_speed:
                self.speed = min(self.target_speed, self.speed + 2.0 * self.max_speed * dt)
            self.vx = math.cos(self.angle) * self.speed
            self.vy = math.sin(self.angle) * self.speed

            nx = self.x + self.vx * dt
            ny = self.y + self.vy * dt

            if mundo.colision_frontal(self.x, self.y, self.vx * dt, self.vy * dt, self.radius, mundo.robots):
                self._frenar()
                p = mundo.descubrir_pared_frontal(self.x, self.y, self.angle, self.radius)
                self.paredes_robot += p
                self._explorar_propio_colision(mundo)
                self._elegir_nuevo_recorrido()
            else:
                self.x, self.y = nx, ny
                if mundo.marcar_visitado(self.x, self.y):
                    self.visitadas_robot += 1

        exp_radius = self.detection_radius * self.cfg.exploration_radius_multiplier
        s, p = mundo.explorar_desde(self.x, self.y, exp_radius)
        self.scouteadas_robot += s
        self.paredes_robot += p
        self._explorar_desde_propio(mundo, exp_radius)
        gs = self.cfg.grid_size
        rc = int(self.x // gs)
        rr = int(self.y // gs)
        rows, cols = self.cfg.rows, self.cfg.cols
        if 0 <= rr < rows and 0 <= rc < cols and mundo.grid[rr, rc] == 0:
            self.explored_robot[rr, rc] = VISITED