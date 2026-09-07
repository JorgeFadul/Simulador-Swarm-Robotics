"""Planificador de la etapa de Inspeccion (Fase 3).

En las estrategias 2E, tras la transicion global Exploracion -> Inspeccion en
el enjambre, los robots pasan a inspeccionar las baldosas sin revisar. Esta
es la implementacion definitiva (Fase 3) que reemplaza al marcador de posicion
de Fase 1. Combina tres mecanismos:

1. **Colas de tareas** en `Robot`: cada robot lleva una cola de objetivos
   ordenada por prioridad (menor costo de mercado primero) y la consume de
   forma automatica, permitiendo repriorizacion dinamica entre ciclos.

2. **Voronoi estatico jerarquico (triangulacion en abanico)**: al entrar en la
   etapa de inspeccion se congela la descomposicion completa. El primer
   Diagrama de Voronoi usa como sitios las posiciones de los robots en ese
   mismo instante (raices, nivel 0), con distancias geodesicas (camino libre)
   para respetar las paredes internas. Cada region se subdivide recursivamente
   por triangulacion en abanico: con su centroide como punto central y los
   vertices de su contorno, se forman tantas piezas como aristas; el centroide
   de cada pieza se recalcula y se repite hasta que la pieza queda en una sola
   celda (ahi se detiene la recursion). El arbol completo (todos sus nodos)
   se construye una sola vez y no se reconstruye: cada nodo aporta una tarea
   (su centroide) y las tareas de las celdas internas se deduplican quedandose
   con el nodo mas superficial. Como cada hoja es una celda, el haz de tareas
   cubre exactamente todas las celdas libres sin repetir ningun punto.

3. **Market Auction Method (Subasta)**: la asignacion de las tareas pendientes
   a los robots se hace por subasta de primer precio greedy. El costo de una
   tarea considera el costo de A*/Dijkstra (distancia esquivando obstaculos a
   la velocidad del robot) mas el tiempo estimado en resolver las tareas
   previas de su cola, y es **multiplicativo por generacion**: `peso = g^n`
   (nivel 0-based) premiando lo poco profundo (padres antes que hijos), y por
   un **ajuste dinamico** que refleja el resultado de las inspecciones:

   * una pieza **defectuosa** multiplica por `inspeccion_factor_defectivo`
     todas las tareas de SU sub-arbol (apilando si hay varios defectuosos
     ancestros): esa descendencia se vuelve mas urgente (costo menor);
   * una pieza **sana** multiplica por `inspeccion_factor_sano` unicamente la
     propia tarea (el punto ya inspeccionado, sin efecto operativo).

   Si `inspeccion_padre_antes_hijo` esta activo (estricto), un hijo solo entra
   a la subasta cuando su padre fue inspeccionado; las raices (nivel 0)
   siempre son elegibles. La trama de colores del renderer usa este mismo
   predicado para revelar los hijos de forma progresiva.

La instancia consume la cola automaticamente (`Robot._consumir_de_cola`) y solo
asigna trabajo a los robots libres, de modo que la repriorizacion dinamica no
cancela movimientos a medio camino.

La clase `InspectionPlanner` (greedy por cercania de Fase 1) se conserva como
referencia y respaldo, pero la etapa de Inspeccion usa `InspeccionVoronoi`.
"""

import bisect
import heapq
import math
from collections import deque

import numpy as np
from scipy import ndimage

from .pathfinding import encontrar_ruta


# Estado de inspeccion / resultado (mismo dominio que Mundo):
#   inspec:   0 = no inspeccionada, 1 = inspeccionada
#   resultado: -1 = desconocido, 0 = sana, 1 = defectuosa
RESULTADO_DESCONOCIDO = -1
SANA = 0
DEFECTUOSA = 1


def _paredes_descubiertas(mundo):
    """Mascara booleana de las paredes reales que el enjambre ha descubierto
    (mundo.explored == DISCOVERED = 3). Es la fuente de distancia del DT."""
    return (mundo.grid == 1) & (mundo.explored == 3)


def _distancia_transform(mundo):
    """Distance Transform euclideo de cada celda libre a la pared descubierta
    mas cercana (bordes del mapa incluidos via las paredes perimetrales).

    Se construye una imagen `dominio` con 0 en las paredes descubiertas y 1 en
    el resto; `scipy.ndimage.distance_transform_edt` devuelve en cada celda la
    distancia euclidea a su pared mas proxima. Las celdas libres profundas (el
    esqueleto) tienen DT alto; son las semillas del primer Voronoi.
    """
    dominio = np.ones((mundo.rows, mundo.cols), dtype=np.float64)
    dominio[_paredes_descubiertas(mundo)] = 0.0
    dt = ndimage.distance_transform_edt(dominio)
    return dt


def _particionar_geodesico(mundo, celdas, sitios):
    """Asigna cada celda libre de `celdas` al sitio mas cercano por distancia de
    camino (BFS multi-fuente sobre el espacio libre, 4-vecinos). Respeta las
    paredes internas: las regiones no cortan obstaculos y quedan conectadas por
    espacio libre. Si una componente quedara inalcanzable desde todos los
    sitios (recinto sin abertura), sus celdas se asignan por distancia
    euclidea como respaldo visual.

    Devuelve dict sitio -> lista de (r, c).
    """
    sitios = list(dict.fromkeys(sitios))
    if not sitios:
        return {}
    rows, cols = mundo.rows, mundo.cols
    libre = (mundo.grid == 0)
    prop = np.full((rows, cols), -1, dtype=np.int32)
    dq = deque()
    for i, (r, c) in enumerate(sitios):
        if (0 <= r < rows and 0 <= c < cols
                and libre[r, c] and prop[r, c] < 0):
            prop[r, c] = i
            dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        o = prop[r, c]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < rows and 0 <= nc < cols
                    and libre[nr, nc] and prop[nr, nc] < 0):
                prop[nr, nc] = o
                dq.append((nr, nc))
    regiones = {s: [] for s in sitios}
    sin_propietario = []
    for (r, c) in celdas:
        if prop[r, c] >= 0:
            regiones[sitios[prop[r, c]]].append((r, c))
        else:
            sin_propietario.append((r, c))
    for (r, c) in sin_propietario:
        mejor = sitios[0]
        mejor_d = float("inf")
        for s in sitios:
            d = math.hypot(r - s[0], c - s[1])
            if d < mejor_d:
                mejor_d = d
                mejor = s
        regiones[mejor].append((r, c))
    return regiones


class _Nodo:
    """Nodo del arbol estatico de Voronoi: una pieza de la descomposicion.

    El arbol completo se congela al arrancar la inspeccion. El nivel es 0-based
    (las raices son las regiones del primer Voronoi). Cada nodo aporta una tarea
    (su centroide); el peso `g^nivel` hace que la subasta tienda a resolver
    primero los padres.
    """
    __slots__ = ("rid", "celdas", "centroide", "nivel", "peso", "padre_rid",
                 "hijos")

    def __init__(self, rid, celdas, centroide, nivel, peso, padre_rid):
        self.rid = rid
        self.celdas = celdas
        self.centroide = centroide
        self.nivel = nivel
        self.peso = peso
        self.padre_rid = padre_rid
        self.hijos = []


class InspeccionVoronoi:
    """Voronoi estatico jerarquico + Market Auction (Fase 3).

    La descomposicion se congela **una sola vez** al arrancar la inspeccion:
      1. el primer Voronoi usa como sitios las posiciones de los robots en ese
         instante (raices, nivel 0) y distancias geodesicas (respeta paredes
         internas),
      2. cada region se subdivide recursivamente por triangulacion en abanico
         (centroide + vertices del contorno) hasta que la pieza queda en una
         sola celda, momento en que se detiene la recursion,
      3. ejecuta la subasta mercantil en cada ciclo sobre las tareas pendientes
         (padres antes que hijos en modo estricto) para los robots libres, y
         llena su cola por prioridad de costo.
    El arbol no se reconstruye despues de la creacion.
    """

    # Numero maximo de tareas cotizadas con A* por robot en cada ronda de la
    # subasta (el estimador barato preselecciona sin pathfinding).
    _K_CANDIDATOS = 48

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self._reservadas = set()     # (r, c) asignadas en el ultimo ciclo
        self._inicializado = False
        self._recursion_pts = []
        self._nodos = {}             # rid -> _Nodo
        self._tareas = []            # tareas de subasta (una por celda libre)
        self._tarea_por_celda = {}   # (r, c) -> tarea (dedup mas superficial)
        self._desc_tareas = {}       # rid -> [tareas del sub-arbol, sin la propia]
        self._rid_con_tarea = set()
        self._inspec_visto = None    # checkpoint contra `mundo.inspec`
        self._next_rid = 0
        self._viz_dat = None

    @property
    def viz(self):
        return self._viz_dat

    @property
    def objetivos_pendientes(self):
        return len(self._reservadas)

    def tarea_revelada(self, mundo, t):
        """Revelacion progresiva (independiente de la subasta): la tarea se
        muestra/dibuja solo si su padre ya fue inspeccionado. En modo no
        estricto todo es visible de inmediato."""
        if not self.cfg.inspeccion_padre_antes_hijo:
            return True
        if t["padre"] is None:
            return True
        return bool(mundo.inspec[t["padre"]])

    def _sitios_nivel1(self, mundo):
        """Sitios del primer Voronoi (raices, nivel 0): la posicion (celda) de
        cada robot en el momento en que arranca la estrategia de inspeccion, sin
        duplicados."""
        gs = self.cfg.grid_size
        sitios = []
        for rb in mundo.robots:
            rr = int(rb.y // gs)
            cc = int(rb.x // gs)
            if (0 <= rr < mundo.rows and 0 <= cc < mundo.cols
                    and mundo.grid[rr, cc] == 0 and (rr, cc) not in sitios):
                sitios.append((rr, cc))
        return sitios

    def _vertices_poligono(self, mundo, celdas):
        """Vertices (esquinas) del contorno del poligomo definido por `celdas`,
        en pixeles: esquinas convexas y concavas (bloques 2x2 con 1 o 3 celdas).
        Son las aristas del abanico de triangulacion."""
        gs = self.cfg.grid_size
        rows, cols = mundo.rows, mundo.cols
        occ = np.zeros((rows + 1, cols + 1), dtype=np.int16)
        for (r, c) in celdas:
            occ[r, c] = 1
        cnt = np.zeros((rows + 1, cols + 1), dtype=np.int16)
        cnt[1:, 1:] += occ[:-1, :-1]
        cnt[1:, :-1] += occ[:-1, 1:]
        cnt[:-1, 1:] += occ[1:, :-1]
        cnt[:-1, :-1] += occ[1:, 1:]
        esquinas = np.argwhere((cnt == 1) | (cnt == 3))
        return [(int(j) * gs, int(i) * gs) for (i, j) in esquinas]

    def _centroide_pieza(self, mundo, dt, celdas):
        """Centroide de una pieza: la celda mas cercana al centro geometrico
        (media de los centros de las celdas). En empate, la mas profunda."""
        gs = self.cfg.grid_size
        cx = sum(c * gs + gs / 2 for _, c in celdas) / len(celdas)
        cy = sum(r * gs + gs / 2 for r, _ in celdas) / len(celdas)
        mejor = celdas[0]
        mejor_d = float("inf")
        mejor_dt = -1.0
        for (r, c) in celdas:
            px = c * gs + gs / 2
            py = r * gs + gs / 2
            d = (px - cx) ** 2 + (py - cy) ** 2
            ddt = float(dt[r, c])
            if d < mejor_d - 1e-9 or (abs(d - mejor_d) <= 1e-9 and ddt > mejor_dt):
                mejor_d = d
                mejor_dt = ddt
                mejor = (r, c)
        return mejor

    def _particionar_pieza(self, mundo, dt, celdas, centroide):
        """Divide una pieza por triangulacion en abanico alrededor del centroide:
        los sectores (triangulos) se forman con los vertices del contorno
        ordenados por angulo. Si el abanico degenera (toda la pieza en un solo
        sector) se parte por el eje mayor para garantizar la terminacion de la
        recursion. Devuelve la lista de piezas (no vacias)."""
        gs = self.cfg.grid_size
        r0, c0 = centroide
        px = c0 * gs + gs / 2
        py = r0 * gs + gs / 2
        vertices = self._vertices_poligono(mundo, celdas)
        if len(vertices) >= 3:
            angs = sorted(math.atan2(y - py, x - px) for (x, y) in vertices)
            piezas = [[] for _ in angs]
            for (r, c) in celdas:
                a = math.atan2(r * gs + gs / 2 - py, c * gs + gs / 2 - px)
                piezas[bisect.bisect_right(angs, a) % len(angs)].append((r, c))
            # Garantia de progreso: si el abanico degenerado deja una pieza del
            # mismo tamano (toda la pieza en un solo sector), partir por el eje
            # mayor para asegurar que la recursion termine.
            if max((len(p) for p in piezas), default=0) >= len(celdas):
                fila = sorted(celdas)
                return [fila[: len(fila) // 2], fila[len(fila) // 2:]]
            return [p for p in piezas if p]
        # contorno degradado: partir por el eje mayor
        fila = sorted(celdas)
        return [fila[: len(fila) // 2], fila[len(fila) // 2:]]

    def _construir_arbol_pieza(self, mundo, dt, celdas, centroide, nivel,
                               padre_rid, g):
        """Recursion estatica: registra el nodo de `celdas` (con su peso
        `g^nivel` y su centroide) y construye recursivamente sus hijos por
        triangulacion en abanico hasta piezas de una celda. Devuelve el nodo."""
        rid = self._next_rid
        self._next_rid += 1
        nodo = _Nodo(rid, celdas, centroide, nivel, max(0.0, g ** nivel),
                     padre_rid)
        self._nodos[rid] = nodo
        if len(celdas) > 1:
            piezas = self._particionar_pieza(mundo, dt, celdas, centroide)
            self._recursion_pts.append((centroide[0], centroide[1], nivel))
            for pieza in piezas:
                if not pieza:
                    continue
                cent = self._centroide_pieza(mundo, dt, pieza)
                hijo = self._construir_arbol_pieza(mundo, dt, pieza, cent,
                                                   nivel + 1, rid, g)
                nodo.hijos.append(hijo.rid)
        return nodo

    def _construir_estatico(self, mundo):
        """Congela la descomposicion completa al arrancar la inspeccion:
        1. sitios = posiciones actuales de los robots (celdas) -> raices nivel 0,
        2. primer Voronoi geodesico sobre el espacio libre (respeta paredes),
        3. recursion de triangulacion en abanico hasta piezas de una celda,
        4. tareas = un centroide por nodo, deduplicado por celda conservando el
           nodo mas superficial, y tablas de propagacion de ajustes."""
        celdas = list(zip(*np.where(mundo.grid == 0)))
        if not celdas:
            self._tareas = []
            self._tarea_por_celda = {}
            self._desc_tareas = {}
            self._rid_con_tarea = set()
            self._viz_dat = None
            return
        dt = _distancia_transform(mundo)
        sitios = self._sitios_nivel1(mundo)
        if not sitios:
            sitios = [self._centroide_pieza(mundo, dt, celdas)]
        regiones0 = _particionar_geodesico(mundo, celdas, sitios)

        g = self.cfg.inspeccion_peso_generacion
        self._recursion_pts = []
        self._nodos = {}
        self._next_rid = 0
        for s, sub in regiones0.items():
            if not sub:
                continue
            self._construir_arbol_pieza(mundo, dt, sub, s, 0, None, g)

        self._tareas = []
        self._tarea_por_celda = {}
        for rid in sorted(self._nodos):
            nodo = self._nodos[rid]
            r, c = nodo.centroide
            if (r, c) in self._tarea_por_celda:
                continue
            t = {
                "rid": rid,
                "r": r, "c": c,
                "x": c * self.cfg.grid_size + self.cfg.grid_size / 2,
                "y": r * self.cfg.grid_size + self.cfg.grid_size / 2,
                "nivel": nodo.nivel,
                "peso": nodo.peso,
                "ajuste": 1.0,
                "costo": 0.0,
                "padre": (self._nodos[nodo.padre_rid].centroide
                          if nodo.padre_rid is not None else None),
            }
            self._tarea_por_celda[(r, c)] = t
            self._tareas.append(t)
        self._rid_con_tarea = {t["rid"] for t in self._tareas}

        # Tabla de propagacion de ajustes: descendientes (tareas) por nodo,
        # acumulada en post-orden.
        self._desc_tareas = {}
        for rid in sorted(self._nodos, reverse=True):
            nodo = self._nodos[rid]
            acc = []
            for h in nodo.hijos:
                if h in self._rid_con_tarea:
                    acc.append(self._tarea_por_celda[self._nodos[h].centroide])
                acc.extend(self._desc_tareas[h])
            self._desc_tareas[rid] = acc

        # --- Anotaciones para el renderer (ver_voronoi) ---
        nivel0 = np.full((mundo.rows, mundo.cols), -1, dtype=np.int16)
        for idx, (s, sub) in enumerate(regiones0.items()):
            for rr, cc in sub:
                nivel0[rr, cc] = idx
        niveles = np.full((mundo.rows, mundo.cols), -1, dtype=np.int16)
        for nodo in self._nodos.values():
            for rr, cc in nodo.celdas:
                niveles[rr, cc] = nodo.nivel
        self._viz_dat = {
            "dt": dt,
            "semillas": sitios,
            "nivel0": nivel0,
            "niveles": niveles,
            "tareas": list(self._tareas),
            "recursion": list(self._recursion_pts),
        }

    def _aplicar_resultados(self, mundo):
        """Aplica el resultado de las inspecciones nuevas (checkpoint contra
        `mundo.inspec` y `mundo.resultado`) sobre los ajustes de mercado:

        * celda DEFECTUOSA: multiplica a todas las tareas de SU sub-arbol por
          `inspeccion_factor_defectivo` (apilando entre ancestros defectuosos):
          la descendencia se vuelve mas urgente (costo menor, rojo en el mapa);
        * celda SANA: multiplica a la propia tarea por `inspeccion_factor_sano`.
        """
        if self._inspec_visto is None:
            self._inspec_visto = np.zeros((mundo.rows, mundo.cols), dtype=bool)
        inspec = (mundo.inspec > 0)
        nuevos = np.argwhere(inspec & ~self._inspec_visto)
        if nuevos.size:
            fd = self.cfg.inspeccion_factor_defectivo
            fs = self.cfg.inspeccion_factor_sano
            for (r, c) in nuevos:
                (r, c) = (int(r), int(c))
                t = self._tarea_por_celda.get((r, c))
                if t is None:
                    continue
                if mundo.resultado[r, c] == DEFECTUOSA:
                    for dt in self._desc_tareas.get(t["rid"], ()):
                        dt["ajuste"] *= fd
                elif mundo.resultado[r, c] == SANA:
                    t["ajuste"] *= fs
        self._inspec_visto |= inspec

    def _reservadas_de_robots(self, mundo, gs):
        """Celdas objetivo en curso: goal activo de cada robot + toda su cola."""
        en_curso = set()
        for rb in mundo.robots:
            if rb.goal is not None and (rb.goal_mode and not rb.goal_alcanzado):
                gx, gy = rb.goal
                en_curso.add((int(gy // gs), int(gx // gs)))
            for t in rb.cola_tareas:
                en_curso.add((int(t["y"] // gs), int(t["x"] // gs)))
        return en_curso

    def _tiempo_viaje(self, mundo, rb, x, y):
        """Tiempo de viaje A*/Dijkstra desde el robot hasta (x, y): largo de la
        ruta (esquivando obstaculos) dividido por la velocidad maxima."""
        ruta = encontrar_ruta(mundo, rb.x, rb.y, x, y)
        if not ruta:
            return float("inf")
        return len(ruta) * self.cfg.grid_size / max(self.cfg.robot_max_speed_px_s, 1e-6)

    def _costo_mercado(self, mundo, rb, tarea, tiempo_previo_cola):
        """Costo de mercado exacto de la tarea para el robot `rb`:

            C = (tau(viaje) + TP(colas previas) + tau(inspeccion)) * peso * ajuste

        donde:
          * tau(viaje):             tiempo A*/Dijkstra hasta el centroide.
          * TP:                     tiempo estimado de las tareas previas de la
                                    cola del robot.
          * peso:                   ponderacion por generacion `g^nivel`
                                    (nivel 0-based), premiando lo poco profundo
                                    (padres antes que hijos).
          * ajuste:                 factor dinamico por resultados de inspeccion
                                    (defectuoso abarata su descendencia; sano
                                    encarece el propio punto).
        """
        tau_viaje = self._tiempo_viaje(mundo, rb, tarea["x"], tarea["y"])
        tau_ins = self.cfg.inspeccion_pausa_s
        tau_total = tau_viaje + tiempo_previo_cola + tau_ins
        return max(0.0, tau_total * tarea["peso"] * tarea["ajuste"])

    def _preseleccionar(self, rb, pendientes, asignadas, tp):
        """Devuelve las `_K_CANDIDATOS` tareas mas prometedoras para `rb` usando
        un costo estimado barato (Euclides * peso * ajuste). El verdadero costo
        de mercado se cotiza luego solo sobre este conjunto."""
        gs = self.cfg.grid_size
        par = self.cfg.inspeccion_pausa_s

        def est(t):
            d = math.hypot(rb.x - (t["c"] * gs + gs / 2),
                           rb.y - (t["r"] * gs + gs / 2))
            return (d + tp + par) * t["peso"] * t["ajuste"]

        return heapq.nsmallest(self._K_CANDIDATOS, pendientes, key=est)

    def actualizar(self, mundo):
        gs = self.cfg.grid_size
        if not mundo.robots:
            return

        # 1) Estatico: la descomposicion se congela una sola vez, al arrancar
        #    la inspeccion, con las posiciones de los robots en ese instante.
        if not self._inicializado:
            self._construir_estatico(mundo)
            self._inicializado = True

        # 2) Resultados de inspeccion nuevos -> ajustes de mercado.
        self._aplicar_resultados(mundo)

        # 3) Tareas pendientes: tareas del arbol estatico aun sin inspeccionar,
        #    sin duplicar las ya asignadas (en curso) y, en modo estricto, solo
        #    las reveladas (cuyo padre ya fue inspeccionado).
        en_curso = self._reservadas_de_robots(mundo, gs)
        pendientes = [t for t in self._tareas
                      if (t["r"], t["c"]) not in en_curso
                      and self.tarea_revelada(mundo, t)
                      and not mundo.inspec[t["r"], t["c"]]]
        if not pendientes:
            self._reservadas = set()
            return

        # 4) Subasta mercantil (primer precio greedy): mientras haya robots
        #    libres y tareas sin adjudicar, se asigna la pareja (robot, tarea)
        #    de menor costo de mercado. Se llama solo a robots con la cola vacia
        #    y sin objetivo actual (los ocupados conservan su plan).
        libres = [i for i, rb in enumerate(mundo.robots)
                  if rb.necesita_objetivo and not rb.cola_tareas]
        if not libres:
            self._reservadas = set()
            return

        asignaciones = {}   # robot idx -> lista de tareas
        asignadas = set()
        while libres:
            mejor_par = None
            mejor_costo = float("inf")
            mejor_rid = -1
            for idx in libres:
                rb = mundo.robots[idx]
                tp = sum(t.get("costo", 0.0) for t in asignaciones.get(idx, []))
                for t in self._preseleccionar(rb, pendientes, asignadas, tp):
                    c = self._costo_mercado(mundo, rb, t, tp)
                    if (c < mejor_costo - 1e-9
                            or (abs(c - mejor_costo) <= 1e-9
                                and t["rid"] < mejor_rid)):
                        mejor_costo = c
                        mejor_rid = t["rid"]
                        mejor_par = (idx, t)
            if mejor_par is None:
                break
            idx, t = mejor_par
            # tiempo operacional de esta tarea (sin su propio peso: es el costo
            # real de trabajo para acumular en la cola).
            t["costo"] = self._tiempo_viaje(mundo, mundo.robots[idx], t["x"], t["y"]) \
                + self.cfg.inspeccion_pausa_s
            asignaciones.setdefault(idx, []).append(t)
            asignadas.add((t["r"], t["c"]))
            # el robot ya tiene una tarea ganadora; no recibe mas en este ciclo
            # (la repriorizacion completa llegara en el proximo actualizar).
            libres.remove(idx)

        # 5) Entrega: se reprioriza la cola de cada robot ganador.
        for idx, lista in asignaciones.items():
            rb = mundo.robots[idx]
            lista.sort(key=lambda t: (t["peso"] * t["ajuste"] * (t["costo"] + 1e-6),
                                      t["r"], t["c"]))
            rb.repriorizar_cola(lista, mundo)

        self._reservadas = {(t["r"], t["c"])
                            for idx, lista in asignaciones.items() for t in lista}


class InspectionPlanner:
    """Asigna a cada robot libre la celda libre mas cercana sin inspeccionar.

    Implementacion de *Fase 1* (greedy por cercania), conservada como referencia
    y respaldo. La etapa de Inspeccion usa `InspeccionVoronoi` (Fase 3).
    """

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self._reservas = {}
        self._fallidas = set()

    @property
    def objetivos_pendientes(self):
        return len(self._reservas)

    def _celda_objetivo(self, x, y, gs):
        return int(y // gs), int(x // gs)

    def _celdas_por_inspeccionar(self, mundo):
        gs = self.cfg.grid_size
        filas = []
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                if mundo.grid[r, c] == 0 and not mundo.inspec[r, c]:
                    filas.append((r, c))
        return filas

    def actualizar(self, mundo):
        gs = self.cfg.grid_size

        a_liberar = [k for k in list(self._reservas)
                     if k < len(mundo.robots) and mundo.robots[k].necesita_objetivo]
        for k in a_liberar:
            if mundo.robots[k].timed_out:
                self._fallidas.add(self._reservas[k])
            del self._reservas[k]

        if not mundo.robots:
            return

        pendientes = self._celdas_por_inspeccionar(mundo)
        if not pendientes:
            return

        reservadas = set(v for k, v in self._reservas.items())

        libres = [i for i, r in enumerate(mundo.robots) if r.necesita_objetivo]
        for idx in libres:
            if idx in self._reservas:
                continue
            r = mundo.robots[idx]
            mejor = None
            for (cr, cc) in pendientes:
                if (cr, cc) in reservadas or (cr, cc) in self._fallidas:
                    continue
                cx = cc * gs + gs / 2
                cy = cr * gs + gs / 2
                d = math.hypot(r.x - cx, r.y - cy)
                if mejor is None or d < mejor[0]:
                    mejor = (d, cr, cc)
            if mejor is None:
                continue
            _, cr, cc = mejor
            cx = cc * gs + gs / 2
            cy = cr * gs + gs / 2
            r.asignar_objetivo(cx, cy, mundo)
            self._reservas[idx] = (cr, cc)
            reservadas.add((cr, cc))
            pendientes = [p for p in pendientes if p != (cr, cc)]