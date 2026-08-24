import math
import numpy as np
from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED
from ..algorithms.pathfinding import encontrar_ruta


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

        self.goal = None
        self.goal_mode = False
        self.goal_alcanzado = False
        self._goal_timer = 0.0
        self.timed_out = False
        self.goal_reached_radius = cfg.goal_reached_radius_px
        self.goal_timeout = cfg.goal_timeout_s
        self.ruta = []
        self._ult_pos = (x, y)
        self._sin_avance = 0.0

        if self.cfg.strategy == "Random Walk":
            self._elegir_nuevo_recorrido()
        else:
            self._esperar_objetivo()

    def _esperar_objetivo(self):
        """Detiene el robot a la espera de que el planificador le asigne una meta.

        Filosofia propia de las estrategias dirigidas (Frontier/Sample/Hybrid):
        al quedarse sin objetivo el robot no hace random walk, se queda quieto
        hasta que el planificador le asigne la siguiente meta.
        """
        self.state = self.STOPPED
        self.target_speed = 0.0
        self.omega = 0.0
        self.speed = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.timer = 0.0
        self.duration = 0.0

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

    def asignar_objetivo(self, x, y, mundo):
        """Asigna un objetivo (px) al robot y planifica una ruta hacia el."""
        self.goal = (float(x), float(y))
        self.goal_mode = True
        self.goal_alcanzado = False
        self._goal_timer = 0.0
        self.timed_out = False
        self._ult_pos = (self.x, self.y)
        self._sin_avance = 0.0
        self.ruta = encontrar_ruta(mundo, self.x, self.y, self.goal[0], self.goal[1])
        if not self.ruta:
            self.goal_alcanzado = True
            self.timed_out = True
        else:
            largo = sum(
                math.hypot(self.ruta[i][0] - self.ruta[i - 1][0],
                           self.ruta[i][1] - self.ruta[i - 1][1])
                for i in range(1, len(self.ruta))
            )
            largo += math.hypot(self.goal[0] - self.ruta[-1][0],
                                self.goal[1] - self.ruta[-1][1])
            self.goal_timeout = max(self.cfg.goal_timeout_s,
                                    largo / max(self.max_speed, 1e-6) * 3.0)

    def cancelar_objetivo(self):
        self.goal = None
        self.goal_mode = False
        self.goal_alcanzado = False
        self._goal_timer = 0.0
        self.timed_out = False
        self.ruta = []

    def reiniciar_objetivo(self):
        self.cancelar_objetivo()

    def reiniciar_exploracion(self):
        self.explored_robot.fill(0)
        self.scouteadas_robot = 0
        self.visitadas_robot = 0
        self.paredes_robot = 0

    @property
    def necesita_objetivo(self):
        """True si el robot esta libre para recibir un objetivo nuevo."""
        return self.goal is None or self.goal_alcanzado

    @property
    def intent_tiempo_restante(self):
        """Segundos que restan del movimiento planificado (o hasta el objetivo)."""
        if self.goal_mode and self.goal is not None and not self.goal_alcanzado:
            if self.ruta:
                d = math.hypot(self.ruta[0][0] - self.x, self.ruta[0][1] - self.y)
            else:
                d = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
            return d / max(self.target_speed, 1e-6)
        return max(0.0, self.duration - self.timer)

    @property
    def intent_distancia(self):
        """Distancia prevista si el robot siguiera la intencion actual."""
        if self.goal_mode and self.goal is not None and not self.goal_alcanzado:
            if self.ruta:
                return math.hypot(self.ruta[0][0] - self.x, self.ruta[0][1] - self.y)
            return math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
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

    def _actualizar_random_walk(self, dt, hay_peligro, has_obs, rep_dx, rep_dy):
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

    def _dirigir_hacia_objetivo(self, dt, mundo, rep_dx, rep_dy, has_obs, colision_inminente):
        """Movimiento dirigido continuo (tipo diferencial).

        El robot avanza mientras gira hacia el siguiente waypoint de la ruta:
        no hay duracion fija del tramo, ni abandono por timeout, ni la secuencia
        de girar-total-y-luego-avanzar del random walk. La evitacion reactiva
        mezcla la direccion al objetivo con el vector de repulsion para esquivar
        paredes y otros robots sin detenerse.
        """
        gx, gy = self.goal

        if not self.ruta:
            self.ruta = encontrar_ruta(mundo, self.x, self.y, gx, gy)
            if not self.ruta:
                self.goal_alcanzado = True
                self.timed_out = True
                self._esperar_objetivo()
                return

        tx, ty = self.ruta[0]
        dist = math.hypot(tx - self.x, ty - self.y)
        if dist <= self.goal_reached_radius:
            self.ruta.pop(0)
            if not self.ruta:
                self.goal_alcanzado = True
                self._esperar_objetivo()
                return
            tx, ty = self.ruta[0]

        destino = math.atan2(ty - self.y, tx - self.x)

        if colision_inminente and not has_obs:
            destino = self.angle + math.radians(45)
        elif has_obs:
            # La repulsion solo desvia el rumbo si el obstaculo esta adelante
            # (producto punto negativo con la direccion al destino). Si esta
            # a un lado o detras, se mantiene el rumbo al objetivo.
            gdx, gdy = math.cos(destino), math.sin(destino)
            dot = gdx * rep_dx + gdy * rep_dy
            if colision_inminente:
                w = 0.0 if dot < 0.0 else self.cfg.dirigido_peso_evasion_imminente
            elif dot < -0.15:
                w = self.cfg.dirigido_peso_evasion_fuerte
            elif dot < 0.3:
                w = self.cfg.dirigido_peso_evasion_suave
            else:
                w = 1.0
            destino = math.atan2(w * gdy + (1.0 - w) * rep_dy,
                                 w * gdx + (1.0 - w) * rep_dx)

        diff = self._diferencia_angular(destino, self.angle)
        umbral = math.radians(self.cfg.dirigido_umbral_giro_deg)
        if abs(diff) <= umbral:
            self.omega = 0.0
            self.angle = destino
        else:
            self.omega = math.copysign(self.turn_speed, diff)
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))

        # Velocidad lineal escalada por el alineamiento con el destino: gira
        # (pivotando si hace falta) mientras avanza, sin parar por completo.
        # El factor minimo de avance (dirigido_avance_min) marca cuanto se
        # avanza mientras se esta girando para alinearse.
        alineado = max(0.0, math.cos(self._diferencia_angular(destino, self.angle)))
        self.speed = self.max_speed * (self.cfg.dirigido_avance_min
                                       + (1.0 - self.cfg.dirigido_avance_min) * alineado)
        self.target_speed = self.speed
        self.target_angle = destino
        self.state = self.MOVING
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

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

        if self.goal_mode and self.goal is not None and not self.goal_alcanzado:
            self._dirigir_hacia_objetivo(dt, mundo, rep_dx, rep_dy, has_obs, colision_inminente)
        elif self.cfg.strategy == "Random Walk":
            self._actualizar_random_walk(dt, hay_peligro, has_obs, rep_dx, rep_dy)
        else:
            self._esperar_objetivo()

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
                if self.goal_mode and self.goal is not None and not self.goal_alcanzado:
                    self.ruta = encontrar_ruta(mundo, self.x, self.y, self.goal[0], self.goal[1])
                    if not self.ruta:
                        self.goal_alcanzado = True
                        self.timed_out = True
                        self._esperar_objetivo()
                else:
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