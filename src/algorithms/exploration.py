"""Generacion de objetivos y asignacion de tareas para exploracion cooperativa.

Implementa las tres familias de target generation descritas en Wang et al.
("Multi-Robot System for Cooperative Exploration in Unknown Environments:
A Survey"): frontier-based (Yamauchi), sample-based / next-best-view y hybrid.

La planificacion es centralizada y usa el mapa global combinado de todos los
robots (mundo.explored): ningun robot conoce el mapa real total, solo lo que
el enjambre ha ido descubriendo.

Fronteras como ENTIDADES (Frontier Based / Hybrid):
- Las celdas de frontera (adyacentes a lo desconocido en 8-direcciones) se
  agrupan en regiones 8-conectadas y las regiones grandes se dividen en
  chunks de a lo mas `frontier_max_size` celdas (por defecto grid_size // 2,
  o sea 10 celdas con grid 20, para un mapa 50x50).
- Cada chunk es una entidad de frontera: un unico objetivo cuyo punto es la
  celda de la entidad mas cercana a su centroide (navegable). Una frontera de
  1 celda se comporta como celda individual.

Task allocation (Hybrid Based / Frontier Based) -> `_asignar_dinamico`:
- La funcion de costo de una frontera es cost = distancia / (celdas + 1):
  mas celdas -> menor costo; mas distancia -> mayor costo. La lista de
  prioridad ordena TODOS los pares (robot, frontera) por costo creciente y
  cada frontera se reserva en exclusiva para UN solo robot (los demas no van).
- La asignacion se recalcula cada plan_interval_s con la posicion ACTUAL de
  cada robot: un robot en transito se REASIGNA a una frontera nueva que le
  salga significativamente mas barata por el camino (ve las fronteras
  cercanas primero). Si un robot se queda sin frontera y no quedan libres,
  cae a la de menor costo aunque ya este reservada (cooperacion forzada).
"""

import math

import numpy as np

from ..core.mundo import DISCOVERED, Mundo

_RNG = np.random.default_rng()


def _celdas_frontera(mundo):
    """Mascara: celda libre y conocida adyacente (8-direcciones) a una celda
    desconocida."""
    libre = mundo.grid == 0
    conocido = mundo.explored != 0
    desconocido = mundo.explored == 0
    vecina = (
        np.roll(desconocido, 1, axis=0)
        | np.roll(desconocido, -1, axis=0)
        | np.roll(desconocido, 1, axis=1)
        | np.roll(desconocido, -1, axis=1)
        | np.roll(np.roll(desconocido, 1, axis=0), 1, axis=1)
        | np.roll(np.roll(desconocido, 1, axis=0), -1, axis=1)
        | np.roll(np.roll(desconocido, -1, axis=0), 1, axis=1)
        | np.roll(np.roll(desconocido, -1, axis=0), -1, axis=1)
    )
    return libre & conocido & vecina


def _vecinos(r, c, rows, cols, diag=True):
    """Vecinos dentro de limites. diag=True usa 8-direcciones, False 4."""
    base = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    if diag:
        base += [(r + 1, c + 1), (r + 1, c - 1), (r - 1, c + 1), (r - 1, c - 1)]
    return [(nr, nc) for nr, nc in base if 0 <= nr < rows and 0 <= nc < cols]


def _dividir_region(region, max_size, rows, cols):
    """Divide una region de celdas conectadas en chunks conectados de a lo
    mas `max_size` celdas cada uno. Cada chunk queda internamente 8-conectado
    (flood-fill BFS desde una celda no reclamada)."""
    if len(region) <= max_size:
        return [region]
    celdas_set = set(region)
    reclamadas = set()   # celdas ya contadas en algun chunk
    chunks = []
    for semilla in region:
        if semilla in reclamadas:
            continue
        chunk = []
        pila = [semilla]
        en_pila = {semilla}
        while pila and len(chunk) < max_size:
            actual = pila.pop()
            en_pila.discard(actual)
            if actual in reclamadas:
                continue
            # Se reclama al CONTAR en el chunk (no al apilar), para no
            # perder celdas que quedan en pila cuando el chunk se llena.
            reclamadas.add(actual)
            chunk.append(actual)
            for nr, nc in _vecinos(actual[0], actual[1], rows, cols, diag=True):
                v = (nr, nc)
                if (v in celdas_set and v not in reclamadas
                        and v not in en_pila):
                    en_pila.add(v)
                    pila.append(v)
        chunks.append(chunk)
    return chunks


def _agrupar_regiones(mask, min_size, max_size=None, rows=None, cols=None):
    """Agrupa celdas de frontera conectadas (8-vecindad) en regiones.

    Si se indica `max_size`, cada region resultante se divide ademas en chunks
    conectados de a lo mas `max_size` celdas (fronteras grandes no se tratan
    como un solo objetivo: cada chunk es una entidad asignable)."""
    h, w = mask.shape
    if rows is None:
        rows = h
    if cols is None:
        cols = w
    visitado = np.zeros_like(mask, dtype=bool)
    regiones = []
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not visitado[r, c]:
                region = []
                stack = [(r, c)]
                visitado[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for nr, nc in _vecinos(cr, cc, rows, cols, diag=True):
                        if mask[nr, nc] and not visitado[nr, nc]:
                            visitado[nr, nc] = True
                            stack.append((nr, nc))
                if len(region) >= min_size:
                    if max_size and len(region) > max_size:
                        regiones.extend(_dividir_region(region, max_size, rows, cols))
                    else:
                        regiones.append(region)
    return regiones


def _centroide_celda(celdas, gs):
    """Centro de la celda de frontera MAS CERCANA al centroide medio de la
    entidad. Garantiza que el objetivo sea una celda libre/conocida navegable
    (el punto medio continuo podria caer en una pared o fuera de rango)."""
    rs = sum(r for r, _ in celdas) / len(celdas)
    cs = sum(c for _, c in celdas) / len(celdas)
    # Celda de la entidad con menor distancia al punto medio.
    mejor = min(celdas, key=lambda rc: (rc[0] - rs) ** 2 + (rc[1] - cs) ** 2)
    return mejor[1] * gs + gs / 2, mejor[0] * gs + gs / 2


def _radio_exploracion(mundo):
    return mundo.cfg.detection_radius_px * mundo.cfg.exploration_radius_multiplier


def _ganancia(mundo, x, y, radio):
    """Celdas desconocidas que quedarian vistas si el robot se parara en (x,y)."""
    gs = mundo.cfg.grid_size
    min_c = max(0, int((x - radio) // gs))
    max_c = min(mundo.cols - 1, int((x + radio) // gs))
    min_r = max(0, int((y - radio) // gs))
    max_r = min(mundo.rows - 1, int((y + radio) // gs))
    gain = 0
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if mundo.explored[r, c] != 0:
                continue
            cx = c * gs + gs / 2
            cy = r * gs + gs / 2
            if math.hypot(cx - x, cy - y) < radio and mundo.tiene_linea_de_vista(x, y, cx, cy):
                gain += 1
    return gain


def _costo_mas_cercano(mundo, x, y):
    costo = None
    for r in mundo.robots:
        d = math.hypot(r.x - x, r.y - y)
        if costo is None or d < costo:
            costo = d
    return costo if costo is not None else 0.0


def _costo_frontera(robot, celdas, gs):
    """Costo de asignar una frontera (entidad de `celdas`) a un robot.

    Formula: cost = distancia_al_centroide / (num_celdas + 1).
    Mas celdas -> menor costo; mas distancia -> mayor costo. El robot debe
    ir a la frontera de MENOR costo (fronteras grandes y cercanas primero)."""
    n = len(celdas)
    gx = sum(c for _, c in celdas) / n * gs + gs / 2
    gy = sum(r for r, _ in celdas) / n * gs + gs / 2
    dist = math.hypot(robot.x - gx, robot.y - gy)
    return dist / (n + 1)


def _muestrear_en_zona(mundo, cx0, cy0, radio, rng):
    """Devuelve una posicion explorada y libre dentro de radio de (cx0, cy0)."""
    gs = mundo.cfg.grid_size
    for _ in range(30):
        ang = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0, radio * 0.8)
        x = cx0 + math.cos(ang) * d
        y = cy0 + math.sin(ang) * d
        c = int(x // gs)
        r = int(y // gs)
        if not (0 <= r < mundo.rows and 0 <= c < mundo.cols):
            continue
        if mundo.grid[r, c] != 0 or mundo.explored[r, c] == 0:
            continue
        if not mundo.circulo_en_libre(x, y, mundo.cfg.robot_radius_px):
            continue
        return x, y
    return None


def _paso_barrido(mundo):
    """Separacion entre lineas de barrido del Boustrophedon: una pasada debe
    avanzar al menos un radio de deteccion para no dejar huecos entre fajas."""
    return max(mundo.cfg.detection_radius_px, mundo.cfg.grid_size)


def _pared_conocida(mundo, r, c):
    """True si la celda es una pared que el enjambre YA ha descubierto (LOS o
    choque). Las paredes aun no vistas no se conocen (restriccion de ceguera del
    Boustrophedon: el patron solo conoce las 4 esquinas del rectangulo exterior)."""
    return mundo.grid[r, c] == 1 and mundo.explored[r, c] == DISCOVERED


def _tramos_barrido(mundo, fila, direccion):
    """Tramos conexos de celdas libres a lo largo de la fila horizontal `fila`,
    recorrida en `direccion` (1 derecha, -1 izquierda). Cada pared CONOCIDA
    parte el tramo: es la adaptacion dinamica que bordea paredes internas que
    un barrido ciego (solo 4 esquinas) no preveia. Devuelve lista de listas de
    (r, c) en orden de recorrido."""
    cols = (range(mundo.cols) if direccion > 0
            else range(mundo.cols - 1, -1, -1))
    tramos = []
    actual = []
    for c in cols:
        if mundo.grid[fila, c] == 0 and not _pared_conocida(mundo, fila, c):
            actual.append((fila, c))
        else:
            if actual:
                tramos.append(actual)
                actual = []
    if actual:
        tramos.append(actual)
    return tramos


def _waypoints_tramo(mundo, tramo, direccion, paso):
    """Subdivide un tramo de barrido en waypoints espaciados `paso` px a lo largo
    de su recorrido. Con un waypoint cada `paso` el zigzag es denso y el robot
    recorre la linea por segmentos en vez de saltar directo al extremo. El tramo
    ya viene ordenado segun la direccion de la pasada. Cada waypoint es una
    region asignable (celda individual)."""
    gs = mundo.cfg.grid_size
    puntos = []
    ultimo_x = None
    for r, c in tramo:
        x = c * gs + gs / 2
        if ultimo_x is None or (x - ultimo_x) >= paso:
            puntos.append({"x": x, "y": r * gs + gs / 2, "score": 1.0,
                           "region": [(r, c)]})
            ultimo_x = x
    return puntos


class Boustrophedon:
    """Barrido en zigzag (boustrophedon) del mapa.

    Bajo la restriccion de ceguera el patron se genera sobre el RECTANGULO
    exterior conocido (las 4 esquinas del mapa), sin pre-conocer las paredes
    internas. El barrido alterna la direccion de cada fila horizontal (izq -> der,
    der -> izq). Cuando el enjambre descubre una pared interna por linea de vista
    o choque (`_pared_conocida`), la fila se parte en tramos y la pasada lo
    bordea: es el recalculo dinamico que el enunciado pide.

    1E-H: se usa como exploracion e inspecciona al visitar.
    2E:   se usa como exploracion; la inspeccion la gestiona InspectionPlanner.
    """

    nombre = "Boustrophedon"

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        gs = mundo.cfg.grid_size
        paso = _paso_barrido(mundo)
        direccion = 1
        objetivos = []
        y = paso / 2
        while y < mundo.cfg.map_height:
            fila = int(y // gs)
            tramos = _tramos_barrido(mundo, fila, direccion)
            for t in tramos:
                objetivos.extend(_waypoints_tramo(mundo, t, direccion, paso))
            y += paso
            direccion = -direccion
        # Se conserva el ORDEN ESPACIAL del zigzag (fila a fila, alternando
        # direccion) y se recorta a un tope para no generar miles de objetivos;
        # la asignacion por cercania los reparte a los robots en ese orden.
        tope = max(self.cfg.sample_targets, len(mundo.robots) * 8)
        return objetivos[:tope]


class VoronoiExploracion:
    """Voronoi + Boustrophedon + Hybrid Frontier (1E-H).

    El mapa DESCUBIERTO y libre se particiona con un Diagrama de Voronoi discreto:
    cada celda libre conocida se asigna al sitio (posicion de un robot) mas
    cercano. Cada robot recibe su region en exclusiva y, dentro de ella, el
    objetivo es la mejor celda de frontera por ganancia menos costo (Hybrid
    Frontier dentro de la region). Cuando una region queda cubierta o bloqueada
    por una pared nueva, el sensor de frontera deja de reportar celdas en ella y
    el planificador recalcula los centroides y emite una nueva particion
    (recalculo dinamico de la Fase 2).
    """

    nombre = "Voronoi"

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.mask = None
        # Anotaciones de la particion actual para el renderer (ver_voronoi).
        self._viz = {"mapa": None, "sitios": [], "objetivos": []}

    def _particion(self, mundo):
        """Devuelve una lista de regiones (una por robot), cada una con sus celdas."""
        gs = mundo.cfg.grid_size
        robots = mundo.robots
        self._viz = {"mapa": None, "sitios": [], "objetivos": []}
        if not robots:
            return []
        sitios = [(r.x, r.y) for r in robots]
        regiones = [{"sitio": s, "celdas": [], "robot": k}
                    for k, s in enumerate(sitios)]
        libre = mundo.grid == 0
        conocido = mundo.explored != 0
        celdas = np.argwhere(libre & conocido)
        region_map = np.full((mundo.rows, mundo.cols), -1, dtype=np.int16)
        for r_i, c_i in celdas:
            gx = c_i * gs + gs / 2
            gy = r_i * gs + gs / 2
            mejor = min(range(len(sitios)),
                        key=lambda k: (sitios[k][0] - gx) ** 2 + (sitios[k][1] - gy) ** 2)
            region_map[int(r_i), int(c_i)] = mejor
            regiones[mejor]["celdas"].append((int(r_i), int(c_i)))
        self._viz["mapa"] = region_map
        self._viz["sitios"] = sitios
        return regiones

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        gs = mundo.cfg.grid_size
        radio = _radio_exploracion(mundo)
        regiones = self._particion(mundo)
        objetivos = []
        for reg in regiones:
            if not reg["celdas"]:
                continue
            mejor = None
            for r_i, c_i in reg["celdas"]:
                if not self.mask[r_i, c_i]:
                    continue
                x = c_i * gs + gs / 2
                y = r_i * gs + gs / 2
                gain = _ganancia(mundo, x, y, radio)
                costo = math.hypot(reg["sitio"][0] - x, reg["sitio"][1] - y)
                score = gain - self.cfg.sample_cost_weight * costo
                if mejor is None or score > mejor["score"]:
                    mejor = {"x": x, "y": y, "score": score,
                             "region": reg["celdas"], "robot": reg["robot"]}
            if mejor is not None:
                objetivos.append(mejor)
        objetivos.sort(key=lambda t: -t["score"])
        self._viz["objetivos"] = [
            {"x": t["x"], "y": t["y"], "robot": t.get("robot")} for t in objetivos
        ]
        return objetivos


class FrontierBased:
    """Fronteras = celdas libres conocidas adyacentes a lo desconocido.

    Objetivos = centroides de las regiones de frontera. Rapido y con poca
    informacion nueva por objetivo (Yamauchi [11], [22]).
    """

    nombre = "Frontier Based"

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        rows, cols = mundo.rows, mundo.cols
        regiones = _agrupar_regiones(self.mask, self.cfg.frontier_min_size,
                                     self.cfg.frontier_max_size, rows, cols)
        targets = []
        for region in regiones:
            x, y = _centroide_celda(region, self.cfg.grid_size)
            targets.append({"x": x, "y": y, "score": float(len(region)),
                            "region": region, "celdas": region,
                            "n_celdas": len(region)})
        targets.sort(key=lambda t: -t["score"])
        return targets


class SampleBased:
    """Next-best-view: muestrea candidatos cerca de la frontera y los puntua
    por ganancia de informacion menos costo de traslado (Gonzalez-Banos [26]).
    """

    nombre = "Sample Based"

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        idx = np.argwhere(self.mask)
        if len(idx) == 0:
            return []
        radio = _radio_exploracion(mundo)
        gs = mundo.cfg.grid_size
        candidatos = []
        n = self.cfg.sample_candidates
        for _ in range(n):
            r0, c0 = idx[self.rng.integers(len(idx))]
            cx0 = c0 * gs + gs / 2
            cy0 = r0 * gs + gs / 2
            punto = _muestrear_en_zona(mundo, cx0, cy0, radio, self.rng)
            if punto is None:
                continue
            x, y = punto
            gain = _ganancia(mundo, x, y, radio)
            costo = _costo_mas_cercano(mundo, x, y)
            score = gain - self.cfg.sample_cost_weight * costo
            candidatos.append((x, y, score))
        candidatos.sort(key=lambda t: -t[2])
        return candidatos[:self.cfg.sample_targets]


class Hybrid:
    """Combina frontier + sample, con cada frontera tratada como entidad.

    Agrupa las celdas de frontera (8-direcciones) en regiones y divide las
    grandes en chunks de a lo mas `grid_size // 2` celdas (10 con grid 20).
    Cada chunk es una ENTIDAD de frontera: un unico objetivo cuyo punto es el
    centroide de sus celdas, con score de costo = dist/(celdas+1). Asi un
    robot va al centroide de la entidad, no a una celda individual (salvo que
    la frontera sea de una sola celda).
    """

    nombre = "Hybrid Based"

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        rows, cols = mundo.rows, mundo.cols
        regiones = _agrupar_regiones(self.mask, self.cfg.frontier_min_size,
                                     self.cfg.frontier_max_size, rows, cols)
        gs = mundo.cfg.grid_size
        targets = []
        for region in regiones:
            celdas = region
            # Celda mas cercana al centroide medio de la entidad (siempre
            # navegable). Con 1 celda es la celda misma -> entidad individual.
            x, y = _centroide_celda(celdas, gs)
            # Costo de la entidad respecto al robot mas cercano (el asignado
            # se decide luego en la asignacion dinamica). Menor costo = mejor.
            costo = None
            for r in mundo.robots:
                cc = _costo_frontera(r, celdas, gs)
                if costo is None or cc < costo:
                    costo = cc
            if costo is None:
                continue
            # score invertido para ordenar descendente por bondad (menor costo
            # = mayor score). El costo real se recalcula por robot en la
            # asignacion dinamica; aqui solo se ordena la lista.
            score = 1.0 / (costo + 1e-6)
            targets.append({"x": x, "y": y, "score": score, "region": celdas,
                            "celdas": celdas, "n_celdas": len(celdas)})
        targets.sort(key=lambda t: -t["score"])
        return targets


class ExplorationPlanner:
    """Orquesta target generation + task allocation.

    Cada plan_interval_s regenera las regiones de frontera a partir del mapa
    global combinado. En Frontier Based y Hybrid se reserva cada region en
    exclusiva para el robot asignado (asignacion dinamica por costo
    dist/(celdas+1), mayor detalle en la docstring de _asignar_dinamico) hasta
    que completa su objetivo; Sample Based usa su asignacion por puntaje.
    """

    def __init__(self, cfg, method, rng=None):
        self.cfg = cfg
        self._rng = rng or np.random.default_rng()
        if method == "Frontier Based":
            self._generador = FrontierBased(cfg, self._rng)
        elif method == "Sample Based":
            self._generador = SampleBased(cfg, self._rng)
        elif method == "Boustrophedon":
            self._generador = Boustrophedon(cfg, self._rng)
        elif method == "Voronoi":
            self._generador = VoronoiExploracion(cfg, self._rng)
        else:
            self._generador = Hybrid(cfg, self._rng)
        self._objetivos = []
        self._regiones = []
        self._reservas = {}
        self._ultima_generacion = -1e9
        self._no_go = {}
        self._alguna_vez_hubo_fronteras = False

    @property
    def objetivos_pendientes(self):
        return len(self._objetivos)

    @property
    def frontera_mask(self):
        return self._generador.mask

    def _celda_objetivo(self, x, y, gs):
        return int(y // gs), int(x // gs)

    def actualizar(self, mundo, tiempo_simulado):
        gs = mundo.cfg.grid_size
        ahora = tiempo_simulado
        self._no_go = {k: v for k, v in self._no_go.items() if v > ahora}

        for r in mundo.robots:
            if r.timed_out and r.goal is not None:
                self._no_go[self._celda_objetivo(r.goal[0], r.goal[1], gs)] = ahora + self.cfg.goal_timeout_s
                r.timed_out = False
                r.goal = None
                r.ruta = []
            elif r.goal_alcanzado and r.goal is not None:
                self._no_go[self._celda_objetivo(r.goal[0], r.goal[1], gs)] = ahora + self.cfg.goal_timeout_s
                r.goal = None
                r.ruta = []

        es_sample = isinstance(self._generador, SampleBased)
        es_voronoi = isinstance(self._generador, VoronoiExploracion)

        if ahora - self._ultima_generacion >= self.cfg.plan_interval_s:
            generados = self._generador.generar(mundo)
            if generados:
                self._alguna_vez_hubo_fronteras = True
            self._ultima_generacion = ahora
            if es_sample:
                self._objetivos = [
                    t for t in generados
                    if self._celda_objetivo(t[0], t[1], gs) not in self._no_go
                ]
            else:
                self._regiones = [
                    t for t in generados
                    if self._celda_objetivo(t["x"], t["y"], gs) not in self._no_go
                ]
                self._objetivos = [(t["x"], t["y"], t["score"]) for t in self._regiones]
        elif es_sample:
            self._objetivos = [
                t for t in self._objetivos
                if self._celda_objetivo(t[0], t[1], gs) not in self._no_go
            ]
        else:
            self._regiones = [
                t for t in self._regiones
                if self._celda_objetivo(t["x"], t["y"], gs) not in self._no_go
            ]
            self._objetivos = [(t["x"], t["y"], t["score"]) for t in self._regiones]

        if es_sample:
            self._asignar_legacy(mundo)
        elif es_voronoi:
            self._asignar_voronoi(mundo, gs)
        else:
            self._asignar_dinamico(mundo, gs)

    def _asignar_legacy(self, mundo):
        """Asignacion original de Sample Based: objetivos por puntaje de
        ganancia, robot libre mas cercano con preferencia de linea de vista."""
        if not self._objetivos:
            return
        libres = [r for r in mundo.robots if r.necesita_objetivo]
        if not libres:
            return
        for objetivo in sorted(self._objetivos, key=lambda t: -t[2]):
            con_los = [r for r in libres if mundo.tiene_linea_de_vista(r.x, r.y, objetivo[0], objetivo[1])]
            pool = con_los if con_los else libres
            mejor_robot = min(pool, key=lambda r: math.hypot(r.x - objetivo[0], r.y - objetivo[1]))
            mejor_robot.asignar_objetivo(objetivo[0], objetivo[1], mundo)
            libres.remove(mejor_robot)
            if not libres:
                break

    def _frontera_de_robot(self, r, regiones, gs):
        """Devuelve la frontera (target) cuyo centroide corresponde al goal
        actual del robot ocupado, o None si no casa con ninguna. Permite saber
        que frontera esta persiguiendo cada robot para comparar costos."""
        if r.goal is None:
            return None
        mejor = None
        mejor_d = None
        for t in regiones:
            d = math.hypot(r.goal[0] - t["x"], r.goal[1] - t["y"])
            if mejor_d is None or d < mejor_d:
                mejor_d = d
                mejor = t
        return mejor

    def _asignar_dinamico(self, mundo, gs):
        """Task allocation dinamico por lista de prioridad de costo.

        Cada plan_interval_s se recalculan los costos de TODAS las fronteras
        contra TODOS los robots usando su posicion ACTUAL (el robot va
        descubriendo mientras camina). La lista de prioridad ordena los pares
        por costo creciente (menor costo = fronteras grandes y cercanas arriba).

        Reglas:
        - Cada frontera (entidad) se reserva en exclusiva para UN solo robot.
        - Un robot ocupado conserva su frontera mientras no aparezca otra
          disponible con costo mucho menor (umbral de reasignacion): asi primero
          atiende las fronteras cercanas/grandes que van apareciendo en vez de
          clavarse en una lejana.
        - Si un robot se queda sin obje/asignable (libre), recibe la frontera
          de menor costo disponible aunque otras ya fueron tomadas.
        """
        if not self._regiones:
            self._reservas = {}
            return

        robots = list(mundo.robots)
        if not robots:
            return

        fracciona = getattr(self.cfg, "reasignar_umbral_costo", 0.7)

        # Frontera que persigue cada robot en este ciclo (si su goal casa).
        objetivos_previos = {k: self._frontera_de_robot(r, self._regiones, gs)
                             for k, r in enumerate(robots)}

        # Lista de prioridad por costo: (costo, robot_idx, target) ascendente.
        pares = []
        for k, r in enumerate(robots):
            for t in self._regiones:
                pares.append((_costo_frontera(r, t["celdas"], gs), k, t))
        pares.sort(key=lambda p: (p[0], p[1]))

        reservadas = set()   # fronteras ya tomadas (una sola vez)
        asignados = {}       # robot_idx -> frontera asignada este ciclo
        tomado_esta = set()  # robots que ya consiguieron frontera este ciclo

        for costo, k, t in pares:
            if id(t) in reservadas:
                continue  # exclusividad: ya la persigue otro robot
            if k in tomado_esta:
                continue  # este robot ya consiguio su frontera de menor costo
            r = robots[k]
            previo = objetivos_previos.get(k)
            # Identidad de la frontera por celda del centroide (los objetos
            # target se regeneran en cada plan_interval_s, asi que id() no
            # sirve para comparar igualdad entre ciclos).
            celda_t = (int(t["y"] // gs), int(t["x"] // gs))
            misma_frontera = False
            if previo is not None:
                celda_previo = (int(previo["y"] // gs), int(previo["x"] // gs))
                misma_frontera = celda_previo == celda_t
            # Robot ocupado persiguiendo OTRA frontera distinta a la actual:
            # solo cambia si la nueva es significativamente mas barata.
            if (previo is not None and not misma_frontera
                    and not r.necesita_objetivo):
                c_actual = _costo_frontera(r, previo["celdas"], gs)
                if costo >= c_actual * fracciona:
                    continue  # no mejora lo suficiente, conserva su objetivo
            # El par vino en orden ascendente de costo: es la frontera de menor
            # costo disponible para este robot.
            asignados[k] = t
            tomado_esta.add(k)
            reservadas.add(id(t))

        # Barrido final: robots libres/sin frontera que quedaron fuera reciben la
        # frontera disponible de menor costo (nunca quedan parados). Si no
        # quedan fronteras LIBRES, reciben la de menor costo aunque ya este
        # reservada: como prioridad respetan la exclusividad, pero si faltan
        # fronteras los robots no se quedan parados (cooperan en la misma).
        disponibles = [t for t in self._regiones if id(t) not in reservadas]
        pendientes = [k for k, r in enumerate(robots)
                      if (k not in tomado_esta
                          and (r.necesita_objetivo or k not in asignados))]
        for k in pendientes:
            r = robots[k]
            if disponibles:
                mejor_t = min(disponibles,
                              key=lambda t: _costo_frontera(r, t["celdas"], gs))
                disponibles.remove(mejor_t)
            else:
                mejor_t = min(self._regiones,
                              key=lambda t: _costo_frontera(r, t["celdas"], gs))
            asignados[k] = mejor_t
            tomado_esta.add(k)
            reservadas.add(id(mejor_t))

        # Aplicar asignaciones. Si el robot ya tiene un goal practicamente igual al
        # centroide (misma celda), no se re-planifica: conserva su ruta en curso.
        for k, t in asignados.items():
            r = robots[k]
            celda_goal = (int(r.goal[1] // gs), int(r.goal[0] // gs)) if r.goal else None
            celda_target = (int(t["y"] // gs), int(t["x"] // gs))
            if (r.goal is not None and not r.necesita_objetivo
                    and celda_goal == celda_target):
                continue  # ya va hacia esa frontera
            r.asignar_objetivo(t["x"], t["y"], mundo)
            self._reservas[k] = (int(t["y"] // gs), int(t["x"] // gs))

    def _asignar_voronoi(self, mundo, gs):
        """Asignacion Voronoi: cada robot recibe en exclusiva la region cuyo
        sitio (centroide de su particion) le corresponde. Si su region no tiene
        objetivo disponible (cubierta o bloqueada por una pared nueva), el robot
        cae al objetivo mas cercano no reservado de otra region, lo que en el
        siguiente plan_interval_s dispara el recalculo de la particion."""
        if not self._regiones:
            return
        libres = [r for r in mundo.robots if r.necesita_objetivo]
        if not libres:
            return
        for r in libres:
            idx = mundo.robots.index(r)
            propias = [t for t in self._regiones if t.get("robot") == idx]
            disponibles = [t for t in propias
                           if self._celda_objetivo(t["x"], t["y"], gs) not in self._no_go]
            if not disponibles:
                disponibles = [t for t in self._regiones
                               if self._celda_objetivo(t["x"], t["y"], gs) not in self._no_go]
            if not disponibles:
                continue
            t = min(disponibles, key=lambda tt: math.hypot(r.x - tt["x"], r.y - tt["y"]))
            r.asignar_objetivo(t["x"], t["y"], mundo)
            self._reservas[idx] = (int(t["y"] // gs), int(t["x"] // gs))
