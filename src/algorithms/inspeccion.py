"""Planificador de la etapa de Inspeccion (Fase 3).

En las estrategias 2E, tras la transicion global Exploracion -> Inspeccion en
el enjambre, los robots pasan a inspeccionar las baldosas sin revisar. Esta
es la implementacion definitiva (Fase 3) que reemplaza al marcador de posicion
de Fase 1. Combina tres mecanismos:

1. **Colas de tareas** en `Robot`: cada robot lleva una cola de objetivos
   ordenada por prioridad (menor costo de mercado primero) y la consume de
   forma automatica, permitiendo repriorizacion dinamica entre ciclos.

2. **Voronoi recursivo usando bordes de distancia**: al entrar en la etapa de
   inspeccion y en cada ciclo de planificacion se calcula un *Distance
   Transform* desde las paredes descubiertas. Los maximos locales del DT
   (esqueleto / crestas) y las posiciones de los robots sirven de semillas para
   el primer Diagrama de Voronoi. Cuando un robot inspecciona el centroide de
   una region, el resultado altera el peso del area y se aplica la recursion:
   el centroide se convierte en semilla junto con los extremos (vertices) de la
   region, creando sub-Voronois. La parada por region es independiente: un
   sub-Voronoi de una sola casilla no se subdivide.

3. **Market Auction Method (Subasta)**: la asignacion de las regiones (tareas)
   a los robots se hace por subasta de primer precio greedy. El costo de una
   tarea considera el costo de A*/Dijkstra (distancia esquivando obstaculos a
   la velocidad del robot) mas el tiempo estimado en resolver las tareas
   previas de su cola, y es **multiplicativo basado en el padre**: el costo del
   sub-Voronoi hereda el peso de sus Voronois padres, proporcional a la
   distancia y a la paternidad. Si el padre detecto un defecto, la zona se
   vuelve mas atractiva (menor costo); si no, menos atractiva. El multiplicador
   nunca es 0.

La instancia consume la cola automaticamente (`Robot._consumir_de_cola`) y solo
asigna trabajo a los robots libres, de modo que la repriorizacion dinamica no
cancela movimientos a medio camino.

La clase `InspectionPlanner` (greedy por cercania de Fase 1) se conserva como
referencia y respaldo, pero la etapa de Inspeccion usa `InspeccionVoronoi`.
"""

import math

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


def _celdas_pendientes(mundo):
    """Lista de (r, c) de celdas libres aun sin inspeccionar."""
    r, c = np.where((mundo.grid == 0) & (mundo.inspec == 0))
    return list(zip(r.tolist(), c.tolist()))


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


def _centroide_local(mundo, dt, celdas):
    """Celda de una region mas profunda (mayor DT), es decir, la mas alejada de
    las paredes descubiertas: representa el 'centroide' del area."""
    mejor = celdas[0]
    mejor_d = -1.0
    for (r, c) in celdas:
        d = float(dt[r, c])
        if d > mejor_d:
            mejor_d = d
            mejor = (r, c)
    return mejor


def _subsitios(mundo, dt, celdas, centroide):
    """Semillas para subdividir una region ya inspeccionada en su centroide:
    el propio centroide (ahora semilla confirmada) mas los extremos/vertices de
    la region (maximos locales de DT dentro de ella). Si la region es pequena,
    basta con el centroide; se devuelve siempre al menos un sitio."""
    sitios = [centroide]
    orden = sorted(celdas, key=lambda cc: dt[cc[0], cc[1]], reverse=True)
    for cc in orden:
        if len(sitios) >= 3:
            break
        if cc == centroide:
            continue
        # exige separacion minima (al menos 1 casilla) para no duplicar sitios
        if all(math.hypot(cc[0] - s[0], cc[1] - s[1]) >= 1.5 for s in sitios):
            sitios.append(cc)
    return sitios


def _particionar(celdas, sitios):
    """Asigna cada celda de `celdas` al sitio de menor distancia euclidea;
    devuelve dict sitio -> lista de celdas."""
    regiones = {s: [] for s in sitios}
    for (r, c) in celdas:
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
    """Hoja activa del arbol recursivo de Voronoi: una region a la que aun le
    falta inspeccionar su centroide. Lleva el peso de mercado heredado de forma
    multiplicativa desde la raiz."""
    __slots__ = ("celdas", "centroide", "sitio", "peso", "nivel")

    def __init__(self, celdas, centroide, sitio, peso, nivel):
        self.celdas = celdas
        self.centroide = centroide
        self.sitio = sitio
        self.peso = peso
        self.nivel = nivel


class InspeccionVoronoi:
    """Voronoi recursivo por bordes de distancia + Market Auction (Fase 3).

    Cada ciclo de planificacion:
      1. calcula el Distance Transform desde las paredes descubiertas,
      2. construye recursivamente el arbol de regiones: una region cuyo
         centroide aun no se inspecciono es una 'tarea' (hoja); una cuya
         centroide ya se inspecciono se subdivide en sub-Voronois (recursion)
         y su peso de mercado se propaga multiplicativamente a los hijos,
         con parada por region cuando el sub-Voronoi tiene una sola casilla,
      3. ejecuta la subasta mercantil para asignar las tareas pendientes a los
         robots libres, y llena su cola por prioridad de costo.
    """

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self._reservadas = {}   # tarea (r,c) -> robot idx (asignada en curso)
        self._inicializado = False

    @property
    def objetivos_pendientes(self):
        return len(self._reservadas)

    def _semillas_iniciales(self, mundo, dt, pendientes):
        """Semillas del primer Voronoi: maximos locales del DT (esqueleto) mas
        las posiciones de los robots. Toma hasta `num_robots` sitios bien
        separados por muestreo de puntos mas lejanos (farthest-point)."""
        n = max(1, len(mundo.robots) or 1)
        orden = sorted(pendientes, key=lambda cc: dt[cc[0], cc[1]], reverse=True)
        sitios = []
        for cc in orden:
            if len(sitios) >= n:
                break
            if all(math.hypot(cc[0] - s[0], cc[1] - s[1]) >= 2.0 for s in sitios):
                sitios.append(cc)
        # anade la posicion de cada robot como sitio (prioridad local)
        gs = self.cfg.grid_size
        for rb in mundo.robots:
            rr = int(rb.y // gs)
            cc = int(rb.x // gs)
            if (0 <= rr < mundo.rows and 0 <= cc < mundo.cols
                    and mundo.grid[rr, cc] == 0 and (rr, cc) not in sitios):
                sitios.append((rr, cc))
        return sitios

    def _recursar(self, mundo, dt, celdas, sitio, peso, nivel):
        """Devuelve la lista de hojas (tareas pendientes) del sub-arbol que
        nace de la region `celdas` con semilla `sitio` y peso heredado `peso`."""
        celdas = [cc for cc in celdas if not mundo.inspec[cc[0], cc[1]]]
        if not celdas:
            return []
        if len(celdas) <= 1:
            # Parada por region: sub-Voronoi de una sola casilla.
            return [_Nodo(celdas, celdas[0], sitio, peso, nivel)]
        r, c = _centroide_local(mundo, dt, celdas)
        resul = int(mundo.resultado[r, c])
        if resul < 0:
            # centroide sin inspeccionar: es una tarea pendiente (hoja)
            return [_Nodo(celdas, (r, c), sitio, peso, nivel)]
        # El centroide ya se inspecciono -> recursion: se vuelve semilla junto
        # a los extremos de la figura, creando sub-Voronois, y el peso del area
        # se altera segun el resultado (factor multiplicativo, nunca 0).
        if resul == DEFECTUOSA:
            factor = 1.0 - self.cfg.auction_defectivo_factor
        else:
            factor = 1.0 + self.cfg.auction_sano_factor
        peso_hijo = max(0.0, peso * factor
                        * (self.cfg.auction_profundidad_p ** (nivel + 1)))
        sin_centroide = [cc for cc in celdas if cc != (r, c)]
        subsitios = _subsitios(mundo, dt, sin_centroide, (r, c))
        regiones = _particionar(sin_centroide, subsitios)
        hojas = []
        for s, sub in regiones.items():
            hojas.extend(self._recursar(mundo, dt, sub, s, peso_hijo, nivel + 1))
        return hojas

    def _tarea_dict(self, mundo, nodo):
        """Convierte una hoja en una tarea de subasta {x, y, costo, peso, nivel}."""
        gs = self.cfg.grid_size
        r, c = nodo.centroide
        return {
            "r": r, "c": c,
            "x": c * gs + gs / 2,
            "y": r * gs + gs / 2,
            "peso": nodo.peso,
            "nivel": nodo.nivel,
            "costo": 0.0,   # tiempo operacional rellenado en la subasta
        }

    def _tiempo_viaje(self, mundo, rb, x, y):
        """Tiempo de viaje A*/Dijkstra desde el robot hasta (x, y): largo de la
        ruta (esquivando obstaculos) dividido por la velocidad maxima."""
        ruta = encontrar_ruta(mundo, rb.x, rb.y, x, y)
        if not ruta:
            return float("inf")
        return len(ruta) * self.cfg.grid_size / max(self.cfg.robot_max_speed_px_s, 1e-6)

    def _costo_mercado(self, mundo, rb, tarea, tiempo_previo_cola):
        """Costo de mercado exacto de la tarea para el robot `rb`:

            C = (tau(viaje) + TP(colas previas) + tau(inspeccion)) * rho * p^n

        donde:
          * tau(viaje):             tiempo A*/Dijkstra hasta el centroide.
          * TP:                     tiempo estimado de las tareas previas de la
                                    cola del robot.
          * rho:                    peso de mercado multiplicativo heredado
                                    desde la raiz del arbol recursivo.
          * p^n:                    factor de paternidad (profundidad del nodo),
                                    proporcional a la distancia en el arbol.
        """
        tau_viaje = self._tiempo_viaje(mundo, rb, tarea["x"], tarea["y"])
        tau_ins = self.cfg.inspeccion_pausa_s
        tau_total = tau_viaje + tiempo_previo_cola + tau_ins
        rho = tarea["peso"]
        p_n = self.cfg.auction_profundidad_p ** tarea["nivel"]
        return max(0.0, tau_total * rho * p_n)

    def actualizar(self, mundo):
        gs = self.cfg.grid_size
        if not mundo.robots:
            return

        # 1) Distribucion continua de pesos: reconstruye el arbol recursivo de
        #    Voronoi a partir del estado actual del mapa (celdas inspeccionadas
        #    disparan la recursion). Las hojas pendientes son las tareas.
        dt = _distancia_transform(mundo)
        pendientes_totales = _celdas_pendientes(mundo)
        if not pendientes_totales:
            self._reservadas = {}
            return

        semillas = self._semillas_iniciales(mundo, dt, pendientes_totales)
        regiones0 = _particionar(pendientes_totales, semillas)
        tareas = []
        for s, sub in regiones0.items():
            peso = self.cfg.auction_profundidad_p ** 0  # raiz: peso inicial = 1.0
            for nodo in self._recursar(mundo, dt, sub, s, 1.0, 1):
                tareas.append(self._tarea_dict(mundo, nodo))

        # 2) Tareas ya en curso (asignadas y aun pendientes de completar) no se
        #    vuelven a subastar; se marcan para no duplicarse.
        en_curso = set()
        for idx, rb in enumerate(mundo.robots):
            if rb.goal is not None and (rb.goal_mode and not rb.goal_alcanzado):
                rr = int(rb.y // gs)
                cc = int(rb.x // gs)
                # objetivo actual de un robot (su celda meta)
                gx, gy = rb.goal
                gr = int(gy // gs)
                gc = int(gx // gs)
                en_curso.add((gr, gc))
            for t in rb.cola_tareas:
                r = int(t["y"] // gs)
                c = int(t["x"] // gs)
                en_curso.add((r, c))

        libres_tareas = [t for t in tareas if (t["r"], t["c"]) not in en_curso
                         and not mundo.inspec[t["r"], t["c"]]]

        # 3) Subasta mercantil (primer precio greedy): mientras haya robots
        #    libres y tareas sin adjudicar, se asigna la pareja (robot, tarea)
        #    de menor costo de mercado. Se llama solo a robots con la cola vacia
        #    y sin objetivo actual (los ocupados conservan su plan).
        libres = [i for i, rb in enumerate(mundo.robots)
                  if rb.necesita_objetivo and not rb.cola_tareas]
        if not libres or not libres_tareas:
            self._reservadas = {}
            return

        asignaciones = {}   # robot idx -> lista de tareas
        asignadas = set()
        while libres:
            mejor_par = None
            mejor_costo = float("inf")
            for idx in libres:
                rb = mundo.robots[idx]
                tp = sum(t.get("costo", 0.0) for t in asignaciones.get(idx, []))
                for t in libres_tareas:
                    if (t["r"], t["c"]) in asignadas:
                        continue
                    c = self._costo_mercado(mundo, rb, t, tp)
                    if c < mejor_costo:
                        mejor_costo = c
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

        # 4) Entrega: se reprioriza la cola de cada robot ganador.
        for idx, lista in asignaciones.items():
            rb = mundo.robots[idx]
            lista.sort(key=lambda t: (t["peso"] * (t["costo"] + 1e-6), t["r"], t["c"]))
            rb.repriorizar_cola(lista, mundo)

        self._reservadas = {((t["r"], t["c"]), idx)
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
