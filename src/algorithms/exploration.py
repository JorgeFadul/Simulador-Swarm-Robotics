"""Generacion de objetivos y asignacion de tareas para exploracion cooperativa.

Implementa las tres familias de target generation descritas en Wang et al.
("Multi-Robot System for Cooperative Exploration in Unknown Environments:
A Survey"): frontier-based (Yamauchi), sample-based / next-best-view y hybrid.

La planificacion es centralizada y usa el mapa global combinado de todos los
robots (mundo.explored): ningun robot conoce el mapa real total, solo lo que
el enjambre ha ido descubriendo. En Frontier Based y Hybrid la asignacion
empareja por cercania (emparejamiento greedy minimo global): cada region de
frontera se asigna al robot libre mas cercano y queda reservada en exclusiva
hasta que el robot completa su objetivo; entonces se reasigna dinamicamente
la siguiente region mas cercana. Sample Based conserva su asignacion por
puntaje de ganancia de informacion.
"""

import math

import numpy as np

from ..core.mundo import Mundo

_RNG = np.random.default_rng()


def _celdas_frontera(mundo):
    """Mascara: celda libre y conocida adyacente a una celda desconocida."""
    libre = mundo.grid == 0
    conocido = mundo.explored != 0
    desconocido = mundo.explored == 0
    vecina = (
        np.roll(desconocido, 1, axis=0)
        | np.roll(desconocido, -1, axis=0)
        | np.roll(desconocido, 1, axis=1)
        | np.roll(desconocido, -1, axis=1)
    )
    return libre & conocido & vecina


def _agrupar_regiones(mask, min_size):
    """Agrupa celdas de frontera conectadas (4-vecindad) en regiones."""
    rows, cols = mask.shape
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
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if (0 <= nr < rows and 0 <= nc < cols and
                                mask[nr, nc] and not visitado[nr, nc]):
                            visitado[nr, nc] = True
                            stack.append((nr, nc))
                if len(region) >= min_size:
                    regiones.append(region)
    return regiones


def _centroide(region, gs):
    rs = sum(p[0] for p in region) / len(region)
    cs = sum(p[1] for p in region) / len(region)
    return cs * gs + gs / 2, rs * gs + gs / 2


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


def _muestrear_en_zona(mundo, cx0, cy0, radio):
    """Devuelve una posicion explorada y libre dentro de radio de (cx0, cy0)."""
    gs = mundo.cfg.grid_size
    for _ in range(30):
        ang = _RNG.uniform(0, 2 * math.pi)
        d = _RNG.uniform(0, radio * 0.8)
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


class FrontierBased:
    """Fronteras = celdas libres conocidas adyacentes a lo desconocido.

    Objetivos = centroides de las regiones de frontera. Rapido y con poca
    informacion nueva por objetivo (Yamauchi [11], [22]).
    """

    nombre = "Frontier Based"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        regiones = _agrupar_regiones(self.mask, self.cfg.frontier_min_size)
        targets = []
        for region in regiones:
            x, y = _centroide(region, self.cfg.grid_size)
            targets.append({"x": x, "y": y, "score": float(len(region)),
                            "region": region})
        targets.sort(key=lambda t: -t["score"])
        return targets


class SampleBased:
    """Next-best-view: muestrea candidatos cerca de la frontera y los puntua
    por ganancia de informacion menos costo de traslado (Gonzalez-Banos [26]).
    """

    nombre = "Sample Based"

    def __init__(self, cfg):
        self.cfg = cfg
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
            r0, c0 = idx[_RNG.integers(len(idx))]
            cx0 = c0 * gs + gs / 2
            cy0 = r0 * gs + gs / 2
            punto = _muestrear_en_zona(mundo, cx0, cy0, radio)
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
    """Combina ambos: agrupa la frontera en regiones y muestrea varios puntos
    de vista por region, quedandose con el mejor de cada una (Charrow [86]).
    """

    nombre = "Hybrid"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mask = None

    def generar(self, mundo):
        self.mask = _celdas_frontera(mundo)
        regiones = _agrupar_regiones(self.mask, self.cfg.frontier_min_size)
        radio = _radio_exploracion(mundo)
        gs = mundo.cfg.grid_size
        targets = []
        for region in regiones:
            mejor = None
            for _ in range(self.cfg.sample_per_cluster):
                r0, c0 = region[_RNG.integers(len(region))]
                cx0 = c0 * gs + gs / 2
                cy0 = r0 * gs + gs / 2
                punto = _muestrear_en_zona(mundo, cx0, cy0, radio)
                if punto is None:
                    continue
                x, y = punto
                gain = _ganancia(mundo, x, y, radio)
                costo = _costo_mas_cercano(mundo, x, y)
                score = gain - self.cfg.sample_cost_weight * costo
                if mejor is None or score > mejor["score"]:
                    mejor = {"x": x, "y": y, "score": score, "region": region}
            if mejor is not None:
                targets.append(mejor)
        targets.sort(key=lambda t: -t["score"])
        return targets[:self.cfg.sample_targets]


class ExplorationPlanner:
    """Orquesta target generation + task allocation.

    Cada plan_interval_s regenera las regiones de frontera a partir del mapa
    global combinado. En Frontier Based y Hybrid se reserva cada region en
    exclusiva para el robot asignado (emparejamiento por cercania) hasta que
    completa su objetivo; Sample Based usa su asignacion por puntaje.
    """

    def __init__(self, cfg, method):
        self.cfg = cfg
        if method == "Frontier Based":
            self._generador = FrontierBased(cfg)
        elif method == "Sample Based":
            self._generador = SampleBased(cfg)
        else:
            self._generador = Hybrid(cfg)
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
        else:
            self._asignar_regiones(mundo, gs)

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

    def _reservar_regiones(self, mundo, regiones, gs):
        """Marca como reservadas las regiones cuyo centroide sigue cerca del
        objetivo de cada robot ocupado. Devuelve el conjunto de regiones
        reservadas y actualiza self._reservas (robot -> celda centroide)."""
        radio = _radio_exploracion(mundo)
        reservadas = set()
        nuevas = {}
        for idx, r in enumerate(mundo.robots):
            if r.necesita_objetivo or r.goal is None:
                continue
            gx, gy = r.goal
            mejor = None
            for t in regiones:
                if id(t) in reservadas:
                    continue
                d = math.hypot(gx - t["x"], gy - t["y"])
                if d <= radio and (mejor is None or d < mejor[0]):
                    mejor = (d, t)
            if mejor is not None:
                t = mejor[1]
                reservadas.add(id(t))
                nuevas[idx] = (int(t["y"] // gs), int(t["x"] // gs))
        self._reservas = nuevas
        return reservadas

    def _asignar_regiones(self, mundo, gs):
        """Emparejamiento greedy minimo global: en cada paso se asigna el par
        (robot libre, region no reservada) de menor distancia. La region queda
        reservada en exclusiva para el robot hasta que completa su objetivo."""
        if not self._regiones:
            return
        reservadas = self._reservar_regiones(mundo, self._regiones, gs)
        libres = [r for r in mundo.robots if r.necesita_objetivo]
        if not libres:
            return
        disponibles = [t for t in self._regiones if id(t) not in reservadas]
        while libres and disponibles:
            mejor = None
            for r in libres:
                for t in disponibles:
                    d = math.hypot(r.x - t["x"], r.y - t["y"])
                    if mejor is None or d < mejor[0]:
                        mejor = (d, r, t)
            if mejor is None:
                break
            _, r, t = mejor
            r.asignar_objetivo(t["x"], t["y"], mundo)
            self._reservas[mundo.robots.index(r)] = (int(t["y"] // gs), int(t["x"] // gs))
            libres.remove(r)
            disponibles.remove(t)
