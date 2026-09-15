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

        # Estado de conduccion Boustrophedon (zigzag autonomo celda a celda).
        # None = inactivo (se inicializa en el primer update con el mundo real).
        # Dict con: modo ("avanzar"/"salto"/"perdido"), orientacion ("h"/"v"),
        # dir (dr, dc), rumbo (angulo cardinal), pasada (+1/-1 lateral),
        # objetivo_celda ((r, c) o None).
        self.zigzag = None
        self.celdas_visitadas = set()   # (r, c) celdas por las que paso el robot

        # Pausa de inspeccion (percusion): el robot se detiene sobre la celda
        # mientras "prueba" antes de registrar el resultado (etapa 2E).
        self.inspeccionando = False
        self.inspeccionando_timer = 0.0
        self.celda_probando = None   # (r, c) de la celda en prueba, o None
        self._inspeccion_visita = False  # pausa iniciada al visitar (sin abandono de meta)

        self.energia = ConsumoEnergia(cfg)
        self.inspeccion_individual = cfg.inspeccion_individual

        strat = cfg._estrategia_activa()
        if strat in ("Random Walk", "Random Walk Hibrido"):
            self._elegir_nuevo_recorrido()
        elif strat == "Boustrophedon":
            self.zigzag = {"modo": "perdido"}
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
        self._aprox_insp_timer = 0.0
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
        self._aprox_insp_timer = 0.0
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
        """Inspecciona la celda libre objetivo y registra el resultado binario
        (sana/defectuosa) en el mundo y en este robot.

        Usa la celda registrada al iniciar la pausa (`celda_probando`): el
        robot se posiciono en el centro exacto de esa celda, por lo que la
        percusion se aplica sobre el centro y nunca cae en una casilla vecina.
        Si no hay celda registrada (inspeccion al visitar durante la
        exploracion), usa la celda que el robot esta pisando.

        Es el punto de extension de la accion de inspeccion: la decision de
        CUANDO inspeccionar pertenece a cada estrategia. Devuelve True si se
        realizo una inspeccion nueva, False si la celda no era inspeccionable o
        ya estaba inspeccionada.
        """
        if self.celda_probando is not None:
            r, c = self.celda_probando
        else:
            gs = self.cfg.grid_size
            c = int(self.x // gs)
            r = int(self.y // gs)
        gs = self.cfg.grid_size
        cx = c * gs + gs / 2
        cy = r * gs + gs / 2
        res = mundo.inspeccionar_celda(cx, cy)
        if res is None:
            return False
        self.inspec_robot[r, c] = 1
        self.resultado_robot[r, c] = res
        self.inspeccionadas_robot += 1
        self._nuevas_inspec_tick = True
        if res:
            self.defectuosas_robot += 1
            self._defecto_hallado_tick = True
        self.consumir_energia_inspeccion()
        return True


    def _radio_centrado_inspeccion(self):
        """Radio (px) alrededor del centro de la celda dentro del cual el robot
        ya se considera posicionado para la percusion.

        Al quedar dentro se ajusta al centro exacto sin que el salto se note;
        si esta mas lejos, en vez de teletransportarse al centro el robot
        sigue aproximandose hasta acercarse.
        """
        gs = max(self.cfg.grid_size, 1)
        return max(2.0, gs * 0.2)

    def _budget_aproximacion_insp(self):
        """Tiempo (s) maximo que el robot intenta acercarse al centro de la
        celda antes de la percusion.

        Con la llegada permisiva (a `goal_reached_radius` del centro) el robot
        suele detenerse a ~11-15 px del medio. Si la celda esta despejada, un
        breve tramo final lo deja a pocos px y el centrado es un retoque que no
        se nota. Pero en zonas congestionadas (otros robots aparcados) puede
        estar fisicamente impedido de acercarse mas: este presupuesto evita que
        se estanque y fuerza la percusion desde donde quedo, sin saltos.
        """
        return max(0.6, self.cfg.inspeccion_pausa_s * 1.0)

    def _iniciar_inspeccion(self, mundo, forzar=False):
        """Inicia la prueba de la celda objetivo: el robot se detiene durante
        `inspeccion_pausa_s` segundos antes de registrar el resultado
        (percusion). Durante la pausa queda marcado `celda_probando` para que
        el renderer lo resalte en amarillo, y no se le asigna otra meta hasta
        que termine la prueba.

        La celda a probar es la de la meta asignada cuando el robot llego a un
        objetivo dirigido; si no hay meta (Random Walk) se usa la celda sobre
        la que quedo el robot. Calcular la celda desde la meta evita registrar
        la casilla anterior o posterior si el robot aterriza cerca del borde.

        En objetivos dirigidos la percusion solo arranca si el robot ya quedo
        cerca del centro (a menos de `_radio_centrado_inspeccion`); si aun esta
        apartado devuelve False y el conductor sigue con el tramo final de
        aproximacion hacia el medio. Ese tramo esta acotado por
        `_budget_aproximacion_insp`: al agotarse el presupuesto (celda
        congestionada, otros robots aparcados) se llama con `forzar=True` y la
        percusion se hace desde donde quedo el robot, sin saltos. Al quedar
        dentro del radio se ajusta al centro exacto: un retoque pequeno, no un
        salto visible.

        La celda solo se prueba SI no ha sido inspeccionada segun el modo:
        - colectivo (default): si `mundo.inspec[r,c] == 0`
        - individual: si `self.inspec_robot[r,c] == 0`

        Retorna True si se inicio la inspeccion, False si la celda ya estaba
        inspeccionada o, sin `forzar`, el robot aun no esta sobre su centro
        (debe seguir acercandose o elegir otro rumbo).
        """
        gs = self.cfg.grid_size
        if self.goal_mode and self.goal is not None:
            c = int(round((self.goal[0] - gs / 2) / gs))
            r = int(round((self.goal[1] - gs / 2) / gs))
        else:
            c = int(round((self.x - gs / 2) / gs))
            r = int(round((self.y - gs / 2) / gs))
        return self._iniciar_inspeccion_celda(mundo, r, c, forzar=forzar)

    def _iniciar_inspeccion_celda(self, mundo, r, c, forzar=False):
        """Arranca la percusion con pausa sobre la celda (r, c) dada.

        Nucleo compartido por todas las estrategias: la pausa de
        `inspeccion_pausa_s` es la UNICA via de registrar una baldosa nueva
        (en lugar de registrarla al vuelo). Durante la pausa el robot queda
        detenido (`inspeccionando=True`, celda resaltada en amarillo) y recien
        al terminar `_avanzar_inspeccion` ejecuta `inspeccionar_actual`.
        """
        gs = self.cfg.grid_size
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
        cx = c * gs + gs / 2
        cy = r * gs + gs / 2
        # Con meta dirigida, la percusion espera a que el robot este sobre el
        # centro: si aun esta apartado no arranca y el conductor sigue el tramo
        # final de aproximacion. `forzar=True` salta esta condicion cuando el
        # presupuesto de aproximacion se agoto (celda congestionada).
        if self.goal_mode and self.goal is not None and not forzar:
            if math.hypot(cx - self.x, cy - self.y) > self._radio_centrado_inspeccion():
                return False
        # Ajuste fino al centro exacto solo cuando el robot ya esta cerca;
        # si quedo mas apartado prueba la celda desde donde se detuvo, sin
        # desplazamientos bruscos.
        if math.hypot(cx - self.x, cy - self.y) <= self._radio_centrado_inspeccion():
            self.x = cx
            self.y = cy
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
            if self._inspeccion_visita and self.goal_mode and self.goal is not None:
                self._inspeccion_visita = False
                self.ruta = []
                self._goal_timer = 0.0
                self._aprox_insp_timer = 0.0
                self.timed_out = False
                return
            self._inspeccion_visita = False
            self.goal_alcanzado = True
            self.timed_out = False
            strat = self.cfg._estrategia_activa(etapa=mundo.etapa)
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
        strat = self.cfg._estrategia_activa(etapa=mundo.etapa)
        hacer_inspeccion = (
            strat == "Random Walk Hibrido" or
            (strat == "Random Walk" and mundo.etapa == "inspeccion")
        )
        if hacer_inspeccion:
            if self._iniciar_inspeccion(mundo):
                return
        self._elegir_nuevo_recorrido()

    def _celda_actual(self, mundo):
        gs = mundo.cfg.grid_size
        return int(self.y // gs), int(self.x // gs)

    def _angulo_para(self, dr, dc):
        return math.atan2(dr, dc)

    def _celda_practicable(self, mundo, r, c, ignorar_visitadas=False):
        """True si el robot puede intentar ir a la celda (r, c).

        Restriccion de ceguera: solo una pared DISCOVERED bloquea. Una pared
        aun no descubierta se trata como transitable: el robot intenta
        avanzar, la choca fisicamente, la descubre y en el siguiente turno
        decide el salto (asi "encuentra la pared" en vez de adivinarla).
        `ignorar_visitadas=True` se usa solo para medir el largo fisico del
        pasillo: el avance frontal del zigzag prefiere suelo fresco y solo
        pisa estela propia si no hay alternativa."""
        if not (0 <= r < mundo.rows and 0 <= c < mundo.cols):
            return False
        if not ignorar_visitadas and (r, c) in self.celdas_visitadas:
            return False
        if mundo.grid[r, c] == 1:
            return mundo.explored[r, c] != DISCOVERED
        gs = mundo.cfg.grid_size
        for o in mundo.robots:
            if o is self:
                continue
            if int(o.y // gs) == r and int(o.x // gs) == c:
                return False
        return True

    def _contar_borde_libre(self, mundo, r, c, dr, dc):
        """Cuantas celdas practicables consecutivas hay desde (r, c) en
        (dr, dc). Mide el largo fisico del pasillo (ignora la estela
        visitada) para orientar el zigzag en pasadas largas."""
        largo = 0
        nr, nc = r, c
        while self._celda_practicable(mundo, nr, nc, ignorar_visitadas=True):
            largo += 1
            nr += dr
            nc += dc
        return largo

    def _celda_alcanzada(self, celda, mundo):
        if celda is None:
            return False
        r, c = celda
        gs = mundo.cfg.grid_size
        if int(self.y // gs) == r and int(self.x // gs) == c:
            return True
        cx = c * gs + gs / 2
        cy = r * gs + gs / 2
        return math.hypot(cx - self.x, cy - self.y) <= self.goal_reached_radius

    def _celda_visitada(self, mundo, r, c):
        """True si la celda ya fue pisada por este robot en la sesion actual."""
        return (r, c) in self.celdas_visitadas

    def _lateral_disponible(self, mundo, r, c, zz):
        """True si al menos un sentido lateral (pasada) tiene celda no visitada
        y practicable."""
        for paso in (zz["pasada"], -zz["pasada"]):
            if zz["orientacion"] == "h":
                nr, nc = r + paso, c
            else:
                nr, nc = r, c + paso
            if self._celda_practicable(mundo, nr, nc):
                return True
        return False

    def _iniciar_zigzag(self, mundo):
        """Elige la orientacion del barrido y arranca el zigzag.

        Solo considera celdas no visitadas al medir pasillos. La "pasada"
        lateral (sentido del salto perpendicular) apunta hacia donde hay mas
        area contigua sin visitar. Si no hay por donde moverse, queda
        "perdido" y se rescata hacia una casilla vista cercana."""
        r, c = self._celda_actual(mundo)
        self.celdas_visitadas.add((r, c))
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        largos = [self._contar_borde_libre(mundo, r + dr, c + dc, dr, dc)
                  for dr, dc in dirs]
        mejor = max(largos)
        if mejor <= 0:
            self.zigzag = {"modo": "perdido", "orientacion": "h",
                           "dir": (0, 1), "rumbo": 0.0, "pasada": 1,
                           "objetivo_celda": None}
            return
        candidatos = [dirs[i] for i, l in enumerate(largos) if l == mejor]
        dir_avance = candidatos[int(self.rng.integers(len(candidatos)))]
        if dir_avance[0] == 0:
            p1, p2 = (1, 0), (-1, 0)
            orientacion = "h"
        else:
            p1, p2 = (0, 1), (0, -1)
            orientacion = "v"
        l1 = self._contar_borde_libre(mundo, r + p1[0], c + p1[1], p1[0], p1[1])
        l2 = self._contar_borde_libre(mundo, r + p2[0], c + p2[1], p2[0], p2[1])
        pasada = 1 if l1 >= l2 else -1
        self.zigzag = {
            "modo": "avanzar", "orientacion": orientacion,
            "dir": dir_avance, "rumbo": self._angulo_para(*dir_avance),
            "pasada": pasada, "objetivo_celda": None,
        }
        self._iniciar_giro(self.zigzag["rumbo"])

    def _candidata_salto(self, mundo, r, c, zz):
        """Celda lateral (perpendicular a la marcha) a la que saltar.

        Prueba primero el sentido de la pasada actual y luego el opuesto.
        Devuelve (r, c) o None si ambos lados estan bloqueados (encerrado)."""
        for paso in (zz["pasada"], -zz["pasada"]):
            if zz["orientacion"] == "h":
                nr, nc = r + paso, c
            else:
                nr, nc = r, c + paso
            if self._celda_practicable(mundo, nr, nc):
                zz["pasada"] = paso
                return (nr, nc)
        return None

    def _casillas_reclamadas(self, mundo):
        """Celulas que otros robots ya "tienen en su plan" (rescate en curso o
        celda de salto): (r, c) del goal de rescate o del objetivo_celda de
        salto de cada robot Boustrophedon. Estas zonas son de baja prioridad
        para este robot y solo se usan como ultimo recurso."""
        reclamadas = set()
        gs = mundo.cfg.grid_size
        for o in mundo.robots:
            if o is self:
                continue
            zz = getattr(o, "zigzag", None)
            if zz is not None and zz.get("modo") == "salto" and zz.get("objetivo_celda"):
                reclamadas.add(zz["objetivo_celda"])
            if o.goal_mode and not o.goal_alcanzado and o.goal is not None:
                reclamadas.add((int(o.goal[1] // gs), int(o.goal[0] // gs)))
        return reclamadas

    def _casilla_vista_mas_cercana(self, mundo):
        """Busca la casilla vista (SCOUTED, aun no visitada y no en
        celdas_visitadas) mas cercana que sea alcanzable con ruta.

        Prefiere casillas que ningun otro robot tenga ya como objetivo de su
        plan (evita duplicar el area barrida). Si no queda ninguna libre,
        cae a las ya reclamadas por otros.
        Devuelve (x, y, ruta) o None."""
        idx = np.argwhere(mundo.explored == SCOUTED)
        if len(idx) == 0:
            return None
        gs = mundo.cfg.grid_size
        r_act, c_act = self._celda_actual(mundo)
        reclamadas = self._casillas_reclamadas(mundo)

        cx = idx[:, 1] * gs + gs / 2
        cy = idx[:, 0] * gs + gs / 2
        d = np.hypot(cx - self.x, cy - self.y)
        orden = np.argsort(d)[:60]

        def candidata(r0, c0):
            return (r0, c0) not in self.celdas_visitadas and not (r0 == r_act and c0 == c_act)

        def primera_por(usar_reclamadas):
            for i in orden:
                r0, c0 = int(idx[i][0]), int(idx[i][1])
                if not candidata(r0, c0):
                    continue
                es_reclamada = (r0, c0) in reclamadas
                if usar_reclamadas != es_reclamada:
                    continue
                x = c0 * gs + gs / 2
                y = r0 * gs + gs / 2
                ruta = encontrar_ruta(mundo, self.x, self.y, x, y)
                if ruta:
                    return (x, y, ruta)
            return None

        res = primera_por(False)
        if res is None:
            res = primera_por(True)
        return res

    def _fijar_objetivo_con_ruta(self, x, y, ruta, mundo):
        """Fija un objetivo propio (rescate del zigzag) sin pasar por el
        asignador del planificador. Similar a asignar_objetivo pero con ruta
        ya calculada."""
        self.goal = (float(x), float(y))
        self.goal_mode = True
        self.goal_alcanzado = False
        self._goal_timer = 0.0
        self.timed_out = False
        self.ruta = ruta
        self._ult_pos = (self.x, self.y)
        self._sin_avance = 0.0
        largo = sum(
            math.hypot(self.ruta[i][0] - self.ruta[i - 1][0],
                       self.ruta[i][1] - self.ruta[i - 1][1])
            for i in range(1, len(self.ruta))
        )
        largo += math.hypot(x - self.ruta[-1][0], y - self.ruta[-1][1])
        self.goal_timeout = max(self.cfg.goal_timeout_s,
                                largo / max(self.max_speed, 1e-6) * 3.0)

    def _intentar_rescate(self, mundo):
        """El robot se encerro: busca la casilla vista mas cercana no
        visitada y se reubica en ella (goal_mode). Al llegar, limpia
        celdas_visitadas y reinicia el zigzag."""
        res = self._casilla_vista_mas_cercana(mundo)
        if res is None:
            self._frenar()
            return False
        x, y, ruta = res
        self._fijar_objetivo_con_ruta(x, y, ruta, mundo)
        self.zigzag["modo"] = "perdido"
        self.zigzag["objetivo_celda"] = None
        return True

    def _mezclar_rumbo_repulsion(self, rumbo, rep_x, rep_y, colision_inminente,
                                 has_obs):
        """Mezcla el rumbo hacia el objetivo con el vector de repulsion.

        La repulsion SOLO desvia el rumbo si el obstaculo esta realmente
        adelante (producto punto negativo rumbo·repulsion). Con dot >= 0 el
        obstaculo esta a un lado o detras (paso junto a una pared, pasillo),
        y el rumbo NO se toca: el despeje lateral lo resuelven la fisica
        (`colision_frontal` frena) y el pivote de esquina, no el heading.

        Dot fuertemente negativo (frente bloqueado) usa evasion fuerte; dot
        levemente negativo (esquina que se avecina) usa evasion suave. Con
        colision inminente se mantiene la rama de emergencia: repulsion pura
        si el obstaculo esta adelante, sino el peso de emergencia."""
        if not has_obs:
            return rumbo
        gdx, gdy = math.cos(rumbo), math.sin(rumbo)
        dot = gdx * rep_x + gdy * rep_y
        if colision_inminente:
            w = 0.0 if dot < 0.0 else self.cfg.dirigido_peso_evasion_imminente
        elif dot < self.cfg.dirigido_gate_fuerte_dot:
            w = self.cfg.dirigido_peso_evasion_fuerte
        elif dot < self.cfg.dirigido_gate_suave_dot:
            w = self.cfg.dirigido_peso_evasion_suave
        else:
            w = 1.0
        return math.atan2(w * gdy + (1.0 - w) * rep_y,
                          w * gdx + (1.0 - w) * rep_x)

    def _avanzar_recto(self, rumbo, dt, mundo, rep_dx, rep_dy, has_obs,
                       colision_inminente):
        """Conduce en linea recta hacia el rumbo cardinal con evitacion
        reactiva. La pared frontal desconocida la detiene la fisica del
        update (choque -> descubrir_pared_frontal)."""
        destino = self._mezclar_rumbo_repulsion(
            rumbo, rep_dx, rep_dy, colision_inminente, has_obs)
        if not has_obs and colision_inminente:
            destino = self.angle + math.radians(45)

        diff = self._diferencia_angular(destino, self.angle)
        umbral = math.radians(self.cfg.dirigido_umbral_giro_deg)
        if abs(diff) <= umbral:
            self.omega = 0.0
            self.angle = destino
        else:
            self.omega = math.copysign(self.turn_speed, diff)
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))

        self.speed = self.max_speed
        self.target_speed = self.max_speed
        self.target_angle = destino
        self.state = self.MOVING
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

    def _actualizar_boustrophedon(self, dt, mundo, rep_dx, rep_dy, has_obs,
                                  colision_inminente):
        """Motor de conduccion del zigzag Boustrophedon (celda a celda).

        - "avanzar": apunta a la direccion de barrido; si la celda frontal no
          es practicable (pared conocida u otro robot) salta lateralmente.
        - "salto":   gira 90 grados hacia la celda lateral y al alcanzarla
          invierte la marcha (zigzag).
        - "perdido": frontal y ambos laterales bloqueados: se reubica en la
          casilla vista mas cercana y reinicia el patron.
        """
        if self.goal_alcanzado:
            self.goal_alcanzado = False
            self.goal = None
            self.goal_mode = False
            self.ruta = []
            self.zigzag = None
            self.celdas_visitadas = set()
        if self.zigzag is None:
            self._iniciar_zigzag(mundo)
        if self.zigzag["modo"] == "perdido":
            self._intentar_rescate(mundo)
            return

        zz = self.zigzag

        if self.state == self.ROTATING:
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
            diff = self._diferencia_angular(self.target_angle, self.angle)
            if abs(diff) <= self.turn_threshold:
                self.angle = self.target_angle
                self.state = self.MOVING
                self._poner_en_marcha()
            return

        r, c = self._celda_actual(mundo)
        self.celdas_visitadas.add((r, c))

        if zz["modo"] == "salto":
            if self._celda_alcanzada(zz["objetivo_celda"], mundo):
                zz["dir"] = (-zz["dir"][0], -zz["dir"][1])
                zz["rumbo"] = self._angulo_para(*zz["dir"])
                zz["modo"] = "avanzar"
                zz["objetivo_celda"] = None
                self._iniciar_giro(zz["rumbo"])
            else:
                self._avanzar_recto(zz["rumbo"], dt, mundo, rep_dx, rep_dy,
                                    has_obs, colision_inminente)
            return

        dr, dc = zz["dir"]
        nr, nc = r + dr, c + dc
        # Suelo fresco al frente: seguir la pasada larga.
        if self._celda_practicable(mundo, nr, nc):
            self._avanzar_recto(zz["rumbo"], dt, mundo, rep_dx, rep_dy,
                                has_obs, colision_inminente)
            return
        # Frontal no fresco (visitado o bloqueo fisico): buscar salto
        # lateral a suelo fresco (cambio de linea del zigzag).
        lateral = self._candidata_salto(mundo, r, c, zz)
        if lateral is not None:
            zz["modo"] = "salto"
            zz["objetivo_celda"] = lateral
            zz["rumbo"] = self._angulo_para(lateral[0] - r, lateral[1] - c)
            self._iniciar_giro(zz["rumbo"])
            return
        # Rodeado (solo estela visitada o bloqueo total): reubicarse en
        # otra zona de interes y retomar direccion horizontal/vertical alli.
        zz["modo"] = "perdido"
        self._intentar_rescate(mundo)

    def _radio_evasion_activa(self):
        """Radio (px) de la repulsion que se mezcla con el rumbo dirigido.

        Mucho mas corto que el sensor (`detection_radius`, ~50 px): las paredes
        y robots lejanos no tiran del heading (eso hacia que el robot perdiera
        la direccion al objetivo en pasillos anchos y orbitara deslizandose
        en paralelo a la pared). Con 2 radios de cobertura hay tiempo de
        reaccion para el obstaculo mas proximo.
        """
        return max(2.0 * self.radius, 2.0 * self.cfg.goal_reached_radius_px)

    def _lookahead_colision(self):
        """Alcance (px) del chequeo de colision frontal.

        Normalmente `detection_radius` (50 px), pero se acorta en la
        aproximacion final: cerca del objetivo no debe mirarse mas alla de la
        meta, porque un obstaculo que queda DETRAS de la celda objetivo (p. ej.
        la pared que bordea una celda de fin de pasillo) dispara una colision
        inminente falsa a 50 px y el robot pivota en sitio sin poder cerrar la
        percusion. El frenado fisico por paso (colision_frontal del movimiento)
        sigue deteniendo cualquier choque real durante el tramo corto final."""
        if self.goal_mode and self.goal is not None and not self.goal_alcanzado:
            dg = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
            return min(self.detection_radius, dg)
        return self.detection_radius

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

        if ((mundo.etapa == "inspeccion" or self.cfg.modo_fase == "hibrido"
                or self.cfg.fly_auto_inspeccion)
                and math.hypot(gx - self.x, gy - self.y) <= self.goal_reached_radius):
            # Llegada permisiva (dentro del radio de llegada de la celda meta):
            # tramo final dirigido al CENTRO de la celda con presupuesto de
            # aproximacion. El presupuesto se acumula AUNQUE otros robots
            # empujen al robot y re-planifiquen ruta: antes ese timer solo
            # corria con la ruta vacia y los choques la repoblaban en bucle,
            # de modo que el robot orbitaba alrededor de la celda (12-43 px)
            # sin poder cerrar la percusion.
            self._aprox_insp_timer += dt
            if self._aprox_insp_timer >= self._budget_aproximacion_insp():
                if not self._iniciar_inspeccion(mundo, forzar=True):
                    self.goal_alcanzado = True
                    self.timed_out = False
                    self._esperar_objetivo()
                return
            tx, ty = gx, gy
        elif not self.ruta:
            # En la etapa de Inspeccion (2E), al llegar al objetivo el robot
            # hace una ultima aproximacion hacia el centro de la celda y se
            # detiene un segundo "probando" (percusion) antes de registrar
            # el resultado. La celda probada es la de la meta (no la del
            # punto de llegada). Si no logra acercarse dentro del
            # presupuesto `_budget_aproximacion_insp` (celda congestionada
            # por otros robots), fuerza la percusion desde donde quedo sin
            # saltos.
            if (mundo.etapa == "inspeccion" or self.cfg.modo_fase == "hibrido"
                    or self.cfg.fly_auto_inspeccion):
                if math.hypot(gx - self.x, gy - self.y) <= self._radio_centrado_inspeccion():
                    self._iniciar_inspeccion(mundo)
                    return
                self._aprox_insp_timer += dt
                if self._aprox_insp_timer >= self._budget_aproximacion_insp():
                    self._iniciar_inspeccion(mundo, forzar=True)
                    return
                tx, ty = gx, gy
            else:
                self.goal_alcanzado = True
                self._esperar_objetivo()
                return
        else:
            tx, ty = self.ruta[0]

        destino = math.atan2(ty - self.y, tx - self.x)

        # Repulsion de corto alcance para el rumbo. El sensor largo (50 px)
        # tira del heading con paredes lejanas: en pasillos anchos anula la
        # direccion al objetivo y el robot se desliza en paralelo a la pared
        # (orbitando sin avanzar). Aqui solo se mezcla la repulsion con
        # obstaculos REALMENTE cercanos; la colision frontal la maneja la rama
        # de evasion o el pivote, no el heading.
        rep_cx, rep_cy, has_obs_c = mundo.detectar_en_rango(
            self.x, self.y, self._radio_evasion_activa()
        )

        if colision_inminente and not has_obs_c:
            # Evasion consciente del objetivo (frente bloqueado y sin vector de
            # repulsion util): se toma un rumbo de evasion FIJO en el mundo, 45
            # grados hacia el flanco que acorta el angulo hacia la meta (derecha
            # si ya esta alineado). Mientras el frente siga bloqueado se
            # mantiene ese rumbo, y si ya se llego y aun sigue bloqueado se
            # barre otro paso en el mismo sentido. Asi el robot desliza a lo
            # largo del obstaculo hacia el objetivo sin perseguir una esquina
            # (evita el giro infinito del +45 relativo re-evaluado cada frame).
            paso_ev = math.radians(45.0)
            umbral_ev = math.radians(self.cfg.dirigido_umbral_giro_deg)
            diff_meta = self._diferencia_angular(destino, self.angle)
            lado = math.copysign(1.0, diff_meta) if abs(diff_meta) > 1e-6 else 1.0
            rumbo = getattr(self, "_evadir_rumbo", None)
            if rumbo is None:
                rumbo = self.angle + lado * paso_ev
            elif abs(self._diferencia_angular(rumbo, self.angle)) <= umbral_ev:
                rumbo = self.angle + lado * paso_ev
            self._evadir_rumbo = rumbo
            destino = rumbo
        elif has_obs_c:
            # La repulsion solo desvia el rumbo si el obstaculo esta adelante
            # (producto punto negativo con la direccion al destino). Si esta
            # a un lado o detras, se mantiene el rumbo al objetivo: asi el
            # robot puede correr junto a paredes sin que lo empujen.
            destino = self._mezclar_rumbo_repulsion(
                destino, rep_cx, rep_cy, colision_inminente, True)
        else:
            self._evadir_rumbo = None

        diff = self._diferencia_angular(destino, self.angle)
        umbral = math.radians(self.cfg.dirigido_umbral_giro_deg)
        pivote_umbral = math.radians(self.cfg.dirigido_pivote_umbral_deg)
        kp = self.cfg.dirigido_giro_kp
        radio_cerrado = max(self.radius, 1e-3)

        # Giro de esquina en el sitio: si el siguiente tramo de la ruta exige
        # girar fuerte y el robot ya esta practicamente sobre el waypoint (o ya
        # lo estaba pivoteando), pivota en su centro (v=0) hasta alinear. Asi el
        # arco de giro en marcha no se abre sobre la esquina: sin esto el robot
        # cortaba las esquinas, chocaba con la pared y entraba en la reactivacion
        # infinita (colision -> jitter -> re-ruta -> misma esquina) que se ve en
        # pantalla como orbitar alrededor del punto.
        esperando = getattr(self, "_esperando_esquina", False)
        if esperando and abs(diff) <= pivote_umbral:
            esperando = False
        giro_esquina = (not colision_inminente and self.ruta
                        and abs(diff) > pivote_umbral
                        and (esperando or dist <= self.goal_reached_radius))
        self._esperando_esquina = esperando
        if giro_esquina:
            self._esperando_esquina = True
            self.omega = math.copysign(min(self.turn_speed, kp * abs(diff)), diff)
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
            self.speed = 0.0
            self.target_speed = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.state = self.ROTATING
            self.target_angle = self.angle
            return

        if colision_inminente and abs(diff) > pivote_umbral:
            # El arco cerrado no es accesible (obstruccion frontal y se necesita
            # un giro grande): se pivota en el centro, giro en 360° sobre una
            # rueda quieta, sin avance (v=0), como un robot diferencial real.
            self.omega = math.copysign(self.turn_speed, diff)
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
            self.speed = 0.0
            self.target_speed = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.state = self.ROTATING
            self.target_angle = self.angle
            return

        if abs(diff) <= umbral:
            self.omega = 0.0
            self.angle = destino
            self.speed = self.max_speed
            self.target_speed = self.max_speed
        else:
            # Arco cerrado (una rueda quieta): el giro es proporcional (no
            # bang-bang) y la velocidad de avance se limita a `omega*R` para que
            # el radio de giro nunca sea mas abierto que el radio real del robot
            # v/omega >= R. Si la ganancia pide girar mas rapido de lo que el
            # arco permite a esa velocidad, la velocidad baja y el giro se va
            # cerrando (transicion suave hacia el pivotaje).
            omega_deseada = min(self.turn_speed, kp * abs(diff))
            self.omega = math.copysign(omega_deseada, diff)
            self.angle += self.omega * dt
            self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
            # Velocidad lineal limitada por el radio de giro físico del robot
            self.speed = min(self.max_speed, omega_deseada * radio_cerrado)
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
            math.cos(self.angle) * self._lookahead_colision(),
            math.sin(self.angle) * self._lookahead_colision(),
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

        strat = self.cfg._estrategia_activa(etapa=mundo.etapa)
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
        elif strat == "Boustrophedon":
            self._actualizar_boustrophedon(dt, mundo, rep_dx, rep_dy, has_obs, colision_inminente)
        elif strat in ("Random Walk", "Random Walk Hibrido"):
            self._actualizar_random_walk(dt, mundo, hay_peligro, has_obs, rep_dx, rep_dy)
        else:
            self._esperar_objetivo()

        # Movimiento
        if self.state == self.MOVING and self.speed > 0:
            if self.speed < self.target_speed:
                self.speed = min(self.target_speed, self.speed + 2.0 * self.max_speed * dt)
            # Proteccion: la velocidad nunca debe superar la maxima
            if self.speed > self.max_speed:
                self.speed = self.max_speed
            self.vx = math.cos(self.angle) * self.speed
            self.vy = math.sin(self.angle) * self.speed

            nx = self.x + self.vx * dt
            ny = self.y + self.vy * dt

            if mundo.colision_frontal(self.x, self.y, self.vx * dt, self.vy * dt, self.radius, mundo.robots):
                self._frenar()
                p = mundo.descubrir_pared_frontal(self.x, self.y, self.angle, self.radius)
                self.paredes_robot += p
                self._explorar_propio_colision(mundo)

                # --- Rutina de recuperación leve (Push-out & Anti-Stuck) ---
                # Si el robot está atascado contra un obstáculo, aplique una leve separación cinemática
                # (1 a 2 px) en la dirección del vector de repulsión si existe espacio libre,
                # o gire 45° a 90° lejos de la pared para desbloquear su trayectoria.
                if has_obs and (rep_dx != 0 or rep_dy != 0):
                    sep_x = self.x + rep_dx * 2.0
                    sep_y = self.y + rep_dy * 2.0
                    if mundo.circulo_en_libre(sep_x, sep_y, self.radius):
                        self.x, self.y = sep_x, sep_y

                self._sin_avance += dt
                if self._sin_avance > 0.5:
                    # Ajuste leve de rumbo para desbloquear orientaciones pegadas a la pared
                    self.angle += self.rng.choice([math.radians(45), math.radians(-45), math.radians(90)])
                    self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))
                    self.target_angle = self.angle
                    self._sin_avance = 0.0

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
                self._sin_avance = 0.0
                # Rasterizacion de visita por area (radio del robot): cubre
                # tambien las casillas libres adyacentes a paredes, evitando
                # huecos de casillas sin visitar en los bordes de las paredes.
                nuevas = mundo.visitar_area(self.x, self.y, self.radius)
                if nuevas:
                    self.visitadas_robot += nuevas
                    self._sincronizar_area_visitada(mundo)
                    # En CUALQUIER estrategia la inspeccion de una baldosa usa
                    # la percusion con pausa (`inspeccion_pausa_s`): el robot se
                    # detiene 1s antes de registrar el resultado. Nunca se
                    # registra al vuelo. Los robots con meta activa inspeccionan
                    # al llegar a su destino (misma pausa); los libres
                    # (p. ej. Random Walk) prueban la baldosa que acaban de
                    # pisar sin detener su itinerario previo.
                    if self.cfg.inspeccionar_al_visitar and (
                            mundo.etapa == "inspeccion" or self.cfg.modo_fase == "hibrido"
                            or self.cfg.fly_auto_inspeccion):
                        if not (self.goal_mode and self.goal is not None
                                and not self.goal_alcanzado):
                            gs = self.cfg.grid_size
                            c = int(self.x // gs)
                            r = int(self.y // gs)
                            self._iniciar_inspeccion_celda(mundo, r, c, forzar=True)


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