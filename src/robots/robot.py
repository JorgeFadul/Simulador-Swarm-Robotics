import math
import numpy as np
from ..core.config import Config
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED
from ..core.energia import ConsumoEnergia
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

        # Inspeccion de baldosas por este robot (contadores y mapa con resultado)
        self.inspeccionadas_robot = 0
        self.defectuosas_robot = 0
        self.inspec_robot = np.zeros((cfg.rows, cfg.cols), dtype=np.int8)
        self.resultado_robot = np.full((cfg.rows, cfg.cols), -1, dtype=np.int8)

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

        # Cola de tareas (Fase 3): lista de dicts {x, y, costo} ordenada por el
        # planificador por prioridad (menor costo primero). El robot la consume
        # automaticamente: al terminar la tarea actual (llego, inspecciono o
        # abandono) toma la siguiente sin esperar al planificador, salvo que
        # quede vacia (entonces espera un objetivo nuevo).
        self.cola_tareas = []
        self.con_cola = False

        # Pausa de inspeccion (percusion): el robot se detiene sobre la celda
        # mientras "prueba" antes de registrar el resultado (etapa 2E).
        self.inspeccionando = False
        self.inspeccionando_timer = 0.0
        self.celda_probando = None   # (r, c) de la celda en prueba, o None

        self.energia = ConsumoEnergia(cfg)
        self.inspeccion_individual = cfg.inspeccion_individual

        strat = cfg._estrategia_activa()
        if strat in ("Random Walk", "Random Walk Hibrido"):
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

    def asignar_cola_tareas(self, tareas, mundo):
        """Fija la cola de tareas del robot (Fase 3) y arranca con la primera.

        `tareas` es una lista de dicts {x, y, costo} que el planificador ya
        ordeno por prioridad (menor costo primero). El robot conserva la cola
        completa (puede ser repriorizada por el planificador entre ciclos) y
        toma la primera como objetivo actual; las siguientes se consumen de
        forma automatica en `update` sin esperar al planificador.
        """
        self.cola_tareas = [dict(t) for t in tareas]
        self.con_cola = bool(self.cola_tareas)
        self._consumir_de_cola(mundo)

    def repriorizar_cola(self, tareas, mundo):
        """Reemplaza la cola pendiente del robot por una repriorizada.

        Solo conserva por delante las tareas que aun no ha empezado; la tarea
        que ya esta en curso no se interrumpe (el robot termina lo que inicio y
        luego sigue con la nueva orden). Esto permite repriorizacion dinamica
        sin cancelar movimientos a medio camino.
        """
        actual = self.goal
        self.cola_tareas = [dict(t) for t in tareas]
        self.con_cola = bool(self.cola_tareas)
        if (self.goal_mode and self.goal is not None
                and not self.goal_alcanzado and actual == self.goal):
            # la tarea en curso se mantiene; la cola nueva queda por detras.
            self.cola_tareas = [t for t in self.cola_tareas
                                if not (t.get("x") == actual[0] and t.get("y") == actual[1])]

    def _quitar_tarea(self, x, y):
        self.cola_tareas = [t for t in self.cola_tareas
                            if not (t.get("x") == x and t.get("y") == y)]
        self.con_cola = bool(self.cola_tareas)

    def _consumir_de_cola(self, mundo):
        """Toma la siguiente tarea pendiente como objetivo (si la hay).

        Devuelve True si asigno un objetivo nuevo desde la cola, False si la
        cola termino y el robot queda libre esperando al planificador.
        """
        while self.cola_tareas:
            t = self.cola_tareas.pop(0)
            self.con_cola = bool(self.cola_tareas)
            if self.cfg.grid_size <= 0:
                continue
            gs = self.cfg.grid_size
            r = int(t["y"] // gs)
            c = int(t["x"] // gs)
            # Descarta tareas cuyo objetivo ya no aplica (celda fuera del mapa o
            # ya inspeccionada por otro robot): son trabajos obsoletos.
            if (0 <= r < mundo.rows and 0 <= c < mundo.cols
                    and mundo.inspec[r, c]):
                continue
            self.asignar_objetivo(t["x"], t["y"], mundo)
            return True
        return False

    def cancelar_objetivo(self):
        self.goal = None
        self.goal_mode = False
        self.goal_alcanzado = False
        self._goal_timer = 0.0
        self.timed_out = False
        self.ruta = []
        self.inspeccionando = False
        self.inspeccionando_timer = 0.0
        self.celda_probando = None
        self.cola_tareas = []
        self.con_cola = False

    def reiniciar_objetivo(self):
        self.cancelar_objetivo()

    def reiniciar_exploracion(self):
        self.explored_robot.fill(0)
        self.scouteadas_robot = 0
        self.visitadas_robot = 0
        self.paredes_robot = 0
        self.inspec_robot.fill(0)
        self.resultado_robot.fill(-1)
        self.inspeccionadas_robot = 0
        self.defectuosas_robot = 0

    def reiniciar_energia(self):
        self.energia.reiniciar()

    def consumir_energia_inspeccion(self):
        """Gasto fijo por ciclo de inspeccion (percusion + FFT)."""
        return self.energia.consumir_energia_inspeccion()

    def inspeccionar_actual(self, mundo):
        """Inspecciona la celda libre sobre la que esta el robot y registra el
        resultado binario (sana/defectuosa) en el mundo y en este robot.

        Es el punto de extension de la accion de inspeccion: la decision de
        CUANDO inspeccionar pertenece a cada estrategia. Devuelve True si se
        realizo una inspeccion nueva, False si la celda no era inspeccionable o
        ya estaba inspeccionada.
        """
        res = mundo.inspeccionar_celda(self.x, self.y)
        if res is None:
            return False
        gs = self.cfg.grid_size
        c = int(self.x // gs)
        r = int(self.y // gs)
        self.inspec_robot[r, c] = 1
        self.resultado_robot[r, c] = res
        self.inspeccionadas_robot += 1
        if res:
            self.defectuosas_robot += 1
        self.consumir_energia_inspeccion()
        return True

    def _iniciar_inspeccion(self, mundo):
        """Inicia la prueba de la celda actual: el robot se detiene durante
        `inspeccion_pausa_s` segundos sobre la celda antes de registrar el
        resultado (percusion). Durante la pausa queda marcado `celda_probando`
        para que el renderer lo resalte en amarillo, y no se le asigna otra meta
        hasta que termine la prueba.

        La celda solo se prueba SI no ha sido inspeccionada segun el modo:
        - colectivo (default): si `mundo.inspec[r,c] == 0`
        - individual: si `self.inspec_robot[r,c] == 0`

        Retorna True si se inicio la inspeccion, False si la celda ya estaba
        inspeccionada (el robot debe elegir otro rumbo).
        """
        gs = self.cfg.grid_size
        r = int(self.y // gs)
        c = int(self.x // gs)
        if r < 0 or r >= mundo.rows or c < 0 or c >= mundo.cols:
            return False
        if mundo.grid[r, c] == 1:
            return False
        # Verificar si ya fue inspeccionada (colectivo o individual)
        if self.inspeccion_individual:
            ya_inspeccionada = self.inspec_robot[r, c] != 0
        else:
            ya_inspeccionada = mundo.inspec[r, c]
        if ya_inspeccionada:
            return False
        self.inspeccionando = True
        self.inspeccionando_timer = self.cfg.inspeccion_pausa_s
        self.celda_probando = (r, c)
        self._frenar()
        self.state = self.STOPPED
        self._goal_timer = 0.0
        return True

    def _avanzar_inspeccion(self, dt, mundo):
        """Decrementa el temporizador de la pausa de inspeccion; al agotarse
        ejecuta la percusion, marca la celda y deja al robot libre.
        Para Random Walk elige un nuevo recorrido; para estrategias dirigidas
        espera a que el planificador asigne la siguiente meta.
        """
        self.inspeccionando_timer -= dt
        if self.inspeccionando_timer <= 0:
            self.inspeccionar_actual(mundo)
            self.inspeccionando = False
            self.celda_probando = None
            self.goal_alcanzado = True
            self.timed_out = False
            strat = self.cfg._estrategia_activa()
            if strat in ("Random Walk", "Random Walk Hibrido"):
                self._elegir_nuevo_recorrido()
            else:
                self._esperar_objetivo()

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

    def _sincronizar_area_visitada(self, mundo):
        """Refleja en explored_robot las celdas que el area de visita acaba de
        marcar (dentro del radio del robot) para el mapa individual."""
        gs = self.cfg.grid_size
        radio = self.radius
        min_c = max(0, int((self.x - radio) // gs))
        max_c = min(self.cfg.cols - 1, int((self.x + radio) // gs))
        min_r = max(0, int((self.y - radio) // gs))
        max_r = min(self.cfg.rows - 1, int((self.y + radio) // gs))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.explored_robot[r, c] == VISITED:
                    continue
                if mundo.explored[r, c] == VISITED and mundo.grid[r, c] == 0:
                    self.explored_robot[r, c] = VISITED

    def _actualizar_random_walk(self, dt, mundo, hay_peligro, has_obs, rep_dx, rep_dy):
        """Movimiento aleatorio con inspeccion al final del tramo.

        En modos de inspeccion (RW Inspeccion o RW Hibrido), al terminar
        cada tramo el robot inspecciona la celda actual si aun no ha sido
        inspeccionada, antes de elegir un nuevo rumbo.
        """
        if self.state == self.MOVING:
            self.timer += dt
            if hay_peligro:
                self._frenar()
                if has_obs:
                    direccion_libre = math.atan2(rep_dy, rep_dx)
                    self._iniciar_giro(direccion_libre)
                else:
                    self._decide_inspeccion_o_rumbo(mundo)
            elif self.timer >= self.duration:
                self._decide_inspeccion_o_rumbo(mundo)
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

    def _decide_inspeccion_o_rumbo(self, mundo):
        """Al final de un tramo de random walk: inspecciona si aplica, luego
        elige un nuevo rumbo aleatorio."""
        strat = self.cfg._estrategia_activa()
        hacer_inspeccion = (
            strat == "Random Walk Hibrido" or
            (strat == "Random Walk" and self.cfg.modo_fase == "inspeccion")
        )
        if hacer_inspeccion:
            if self._iniciar_inspeccion(mundo):
                return
        self._elegir_nuevo_recorrido()

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
                # En la etapa de Inspeccion (2E), al llegar al objetivo el robot
                # se detiene un segundo "probando" la celda (percusion) antes de
                # registrar el resultado. Fuera de la etapa de Inspeccion, la
                # meta dirigida se da por alcanzada al llegar.
                if mundo.etapa == "inspeccion":
                    self._iniciar_inspeccion(mundo)
                else:
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

        # Consumo automatico de la cola (Fase 3): si el robot esta libre (sin
        # objetivo en curso ni inspeccion en pausa) y su cola de tareas aun
        # tiene trabajo, toma la siguiente tarea sin esperar al planificador.
        # Esto encadena los objetivos como la etapa de Market Auction los
        # coloco, permitiendo repriorizacion dinamica entre ciclos.
        if (not self.inspeccionando and self.necesita_objetivo and self.cola_tareas):
            self._consumir_de_cola(mundo)

        strat = self.cfg._estrategia_activa()
        if self.inspeccionando:
            # Pausa de percusion: el robot se queda detenido sobre la celda que
            # esta probando. No avanza ni se le asigna otra meta hasta terminar.
            self._avanzar_inspeccion(dt, mundo)
        elif self.goal_mode and self.goal is not None and not self.goal_alcanzado:
            # Timeout de objetivo: si el robot se traba (maniobras contra una
            # pared o una celda inalcanzable), abandona la meta pasados
            # `goal_timeout` segundos para que el planificador le asigne otra.
            # Evita que un robot quede bloqueado indefinidamente en la etapa de
            # Inspeccion (o en la exploracion dirigida).
            self._goal_timer += dt
            if self._goal_timer >= self.goal_timeout:
                self.goal_alcanzado = True
                self.timed_out = True
                self._esperar_objetivo()
            else:
                self._dirigir_hacia_objetivo(dt, mundo, rep_dx, rep_dy, has_obs, colision_inminente)
        elif strat in ("Random Walk", "Random Walk Hibrido"):
            self._actualizar_random_walk(dt, mundo, hay_peligro, has_obs, rep_dx, rep_dy)
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
                # Rasterizacion de visita por area (radio del robot): cubre
                # tambien las casillas libres adyacentes a paredes, evitando
                # huecos de casillas sin visitar en los bordes de las paredes.
                nuevas = mundo.visitar_area(self.x, self.y, self.radius)
                if nuevas:
                    self.visitadas_robot += nuevas
                    self._sincronizar_area_visitada(mundo)
                    # Inspeccion al visitar: en 1E-H se inspecciona durante la
                    # exploracion; en 2E solo en la etapa de Inspeccion (para no
                    # gastar energia de percusion antes de tiempo).
                    if self.cfg.inspeccionar_al_visitar and (
                            mundo.etapa == "inspeccion"):
                        self.inspeccionar_actual(mundo)

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

        # Consumo energetico del movimiento (modelo tractivo + giro + inercia)
        ppcm = self.cfg.pixels_per_cm
        v_m = self.speed / (ppcm * 100.0)                      # px/s -> m/s
        a_m = (v_m - self.energia.velocidad_anterior) / dt if dt > 0 else 0.0
        self.energia.consumir_movimiento(dt, v_m, a_m, self.omega)