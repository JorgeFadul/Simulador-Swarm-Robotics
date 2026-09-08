"""Planificador de la etapa de Inspeccion (Fase 3).

En las estrategias 2E, tras la transicion global Exploracion -> Inspeccion en
el enjambre, los robots pasan a inspeccionar las baldosas sin revisar. Esta
es la implementacion definitiva (Fase 3). Combina tres mecanismos:

1. **Colas de tareas** en `Robot`: cada robot lleva una cola de objetivos
   ordenada por prioridad (menor costo de mercado primero) y la consume de
   forma automatica, permitiendo repriorizacion dinamica entre ciclos.

2. **Voronoi y Triangulacion Recursiva Homogenea por Figuras**:
   Al entrar en la etapa de inspeccion se congela la descomposicion geometrica:
   * **Voronoi (Gen 0):** Toma en cuenta las esquinas exteriores del mapa y las
     posiciones de los robots como centroides (Gen 0). Se obtienen exactamente
     tantas figuras Gen 0 como robots, recortadas limpiamente con las
     mediatrices de Voronoi (Sutherland-Hodgman).
   * **Triangulacion (Gen 1):** Toma cada par de vertices adyacentes del
     poligono de Voronoi y el centroide de Gen 0 (que se transforma en vertice),
     formando los triangulos en abanico (tantos triangulos como vertices del
     poligono Gen 0).
   * **Triangulaciones siguientes (Gen 2 a Gen n):** Metodo recursivo uniforme.
     Cada triangulo se divide en 3 subtriangulos tomando su centroide como
     vertice central. Si el centroide de un subtriangulo cae en la misma celda
     que la generacion anterior, en pared o fuera del mapa, esa iteracion es
     descartada y esa rama de recursion termina.
   * Todas las figuras contienen: id, generacion, padre, hijos, vertices,
     centroide (xy y celda rc), estado de inspeccion ("null", "sana", "defectuosa"),
     coste individual y coste acumulado.

3. **Subasta por generaciones con barrera local por region raiz**:
   La asignacion de tareas pendientes a los robots se hace por subasta de primer
   precio greedy. La barrera local sincroniza por generacion de forma
   independiente en cada Voronoi raiz: en la region V solo se subastan tareas de
   su generacion activa minima pendiente. Si una celda resulta defectuosa al
   inspeccionarse, abarata el coste acumulado de todos sus descendientes en el
   arbol; si resulta sana, encarece su descendencia.
"""

import bisect
import heapq
import math
import sys
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
    """Mascara booleana de las paredes reales que el enjambre ha descubierto."""
    return (mundo.grid == 1) & (mundo.explored == 3)


def _distancia_transform(mundo):
    """Distance Transform euclideo de cada celda libre a la pared descubierta mas cercana."""
    dominio = np.ones((mundo.rows, mundo.cols), dtype=np.float64)
    dominio[_paredes_descubiertas(mundo)] = 0.0
    dt = ndimage.distance_transform_edt(dominio)
    return dt


def _line_intersection(p1, p2, p3, p4):
    """Interseccion de la linea orientada p1-p2 con la linea p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _clip_polygon(poly, p1, p2):
    """Recorta un poligono convexo contra el semiplano a la izquierda de p1 -> p2
    (Sutherland-Hodgman)."""
    def is_inside(p):
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= -1e-7

    out = []
    if not poly:
        return out
    s = poly[-1]
    for e in poly:
        if is_inside(e):
            if not is_inside(s):
                out.append(_line_intersection(s, e, p1, p2))
            out.append(e)
        elif is_inside(s):
            out.append(_line_intersection(s, e, p1, p2))
        s = e
    return out


def _poligono_voronoi_2d(box, site, other_sites):
    """Calcula el poligono de Voronoi convexo de un sitio dentro de la caja delimitadora,
    recortando contra las mediatrices con todos los demas sitios."""
    poly = list(box)
    for other in other_sites:
        mx = (site[0] + other[0]) / 2.0
        my = (site[1] + other[1]) / 2.0
        dx = -(site[1] - other[1])
        dy = site[0] - other[0]
        p1 = (mx, my)
        p2 = (mx + dx, my + dy)
        poly = _clip_polygon(poly, p1, p2)
    return poly


def _particionar_geodesico(mundo, celdas, sitios):
    """Asigna cada celda libre de `celdas` al sitio mas cercano por distancia de
    camino (BFS multi-fuente sobre el espacio libre, 4-vecinos). Respeta paredes
    internas para pintar el fondo nivel0."""
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


class Figura:
    """Figura geometrica del arbol recursivo de Voronoi / Triangulacion.

    Atributos:
      id: identificador unico entero
      generacion: nivel de generacion (0 para Voronoi inicial, 1..n para triangulos)
      padre: referencia a la Figura padre (None para Gen 0)
      hijos: lista de Figuras hijas (vacio para hojas terminales)
      vertices: lista de tuplas (x, y) en pixeles
      centroide_xy: (x, y) continuo centro de la celda en pixeles
      celda: tupla (r, c) de cuadricula discreta
      inspeccion: "null" (no inspeccionada), "sana", "defectuosa"
      coste_individual: coste base de inspeccion (tiempo de pausa / viaje)
      coste_acumulado: coste acumulado ponderado por herencia de resultados
      raiz_id: indice de la region Voronoi Gen 0 de la cual desciende
    """
    __slots__ = ("id", "generacion", "padre", "hijos", "vertices",
                 "centroide_xy", "celda", "inspeccion", "coste_individual",
                 "coste_acumulado", "raiz_id")

    def __init__(self, fig_id, generacion, vertices, centroide_xy, celda_rc,
                 padre=None, raiz_id=0):
        self.id = fig_id
        self.generacion = generacion
        self.padre = padre
        self.hijos = []
        self.vertices = vertices
        self.centroide_xy = centroide_xy
        self.celda = celda_rc
        self.inspeccion = "null"
        self.coste_individual = 0.0
        self.coste_acumulado = 0.0
        self.raiz_id = raiz_id


def _subdividir_figura(figura, mundo, gs, max_gen=15, id_counter=None):
    """Subdivision recursiva uniforme para cualquier figura triangular.
    Divide el triangulo en 3 subtriangulos usando el centroide continuo como vertice
    central. Se detiene y descarta la iteracion si el nuevo centroide cae
    en la misma celda que el padre, en pared o fuera del mapa."""
    if figura.generacion >= max_gen:
        return
    tri = figura.vertices
    if len(tri) != 3:
        return
    v1, v2, v3 = tri
    m_xy = figura.centroide_xy
    subs = [
        [m_xy, v1, v2],
        [m_xy, v2, v3],
        [m_xy, v3, v1]
    ]
    for sub in subs:
        area2 = abs(sub[0][0] * (sub[1][1] - sub[2][1]) +
                    sub[1][0] * (sub[2][1] - sub[0][1]) +
                    sub[2][0] * (sub[0][1] - sub[1][0]))
        if area2 < 1.0:
            continue
        sub_mx = (sub[0][0] + sub[1][0] + sub[2][0]) / 3.0
        sub_my = (sub[0][1] + sub[1][1] + sub[2][1]) / 3.0
        r = int(sub_my // gs)
        c = int(sub_mx // gs)
        if r < 0 or r >= mundo.rows or c < 0 or c >= mundo.cols:
            continue
        if mundo.grid[r, c] == 1:
            continue
        # Criterio de parada exacto: celda repetida respecto a la generacion anterior
        if (r, c) == figura.celda:
            continue

        sub_cell_xy = (c * gs + gs / 2.0, r * gs + gs / 2.0)
        id_counter[0] += 1
        hijo = Figura(
            fig_id=id_counter[0],
            generacion=figura.generacion + 1,
            vertices=sub,
            centroide_xy=sub_cell_xy,
            celda_rc=(r, c),
            padre=figura,
            raiz_id=figura.raiz_id
        )
        figura.hijos.append(hijo)
        _subdividir_figura(hijo, mundo, gs, max_gen=max_gen, id_counter=id_counter)


class InspeccionVoronoi:
    """Voronoi estatico jerarquico + Subasta por generaciones (Fase 3)."""

    _K_CANDIDATOS = 48

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self._reservadas = set()     # (r, c) asignadas en el ultimo ciclo
        self._inicializado = False
        self._raices = []            # Figuras de Gen 0
        self._todas_figuras = []     # Todas las figuras del arbol
        self._figuras_por_id = {}    # id -> Figura
        self._figuras_por_celda = {} # (r, c) -> [Figuras]
        self._tarea_por_celda = {}   # (r, c) -> tarea representativa
        self._tareas = []            # Lista de tareas para subasta
        self._inspec_visto = None    # checkpoint contra mundo.inspec
        self._viz_dat = None
        self._fallos = {}            # (r, c) -> fallos consecutivos
        self._cooldown = {}          # (r, c) -> segundos restantes de aparcado

    @property
    def viz(self):
        return self._viz_dat

    @property
    def objetivos_pendientes(self):
        return len(self._reservadas)

    def _figura_a_dict(self, f):
        """Serializa la figura y sus hijos a dict para el visualizador."""
        return {
            "id": f.id,
            "generacion": f.generacion,
            "centroide": f.celda,
            "centroide_xy": f.centroide_xy,
            "vertices": f.vertices,
            "inspeccion": f.inspeccion,
            "coste_acumulado": f.coste_acumulado,
            "hijos": [self._figura_a_dict(h) for h in f.hijos]
        }

    def _construir_estatico(self, mundo):
        """Congela la descomposicion completa de figuras al arrancar la inspeccion:
        1. Voronoi Gen 0: a partir de las esquinas del mapa y posiciones de robots,
           generando N figuras convexas con sus vertices exactos.
        2. Triangulacion Gen 1: triangulos en abanico (C0, v_j, v_{j+1}).
        3. Triangulacion Gen 2..n: recursion ternaria homogenea (M, v_1, v_2),
           con parada y descarte cuando el centroide coincide con la celda del padre.
        4. Inicializacion de costes individuales y acumulados."""
        gs = self.cfg.grid_size
        box = [(0.0, 0.0), (float(mundo.width_px), 0.0),
               (float(mundo.width_px), float(mundo.height_px)),
               (0.0, float(mundo.height_px))]
        sitios = [(rb.x, rb.y) for rb in mundo.robots]
        if not sitios:
            return

        sitios_celda = []
        for sx, sy in sitios:
            sr = int(sy // gs)
            sc = int(sx // gs)
            if 0 <= sr < mundo.rows and 0 <= sc < mundo.cols and mundo.grid[sr, sc] == 0:
                sitios_celda.append((sr, sc))
            else:
                sitios_celda.append((max(0, min(mundo.rows - 1, sr)), max(0, min(mundo.cols - 1, sc))))

        celdas_libres = list(zip(*np.where(mundo.grid == 0)))
        regiones0 = _particionar_geodesico(mundo, celdas_libres, sitios_celda)
        nivel0 = np.full((mundo.rows, mundo.cols), -1, dtype=np.int16)
        for rindex, (s, sub) in enumerate(regiones0.items()):
            for rr, cc in sub:
                nivel0[rr, cc] = rindex

        self._raices = []
        id_counter = [0]
        triangulos_viz = []

        for i, (sx, sy) in enumerate(sitios):
            others = [sitios[j] for j in range(len(sitios)) if j != i]
            poly = _poligono_voronoi_2d(box, (sx, sy), others)
            sr, sc = sitios_celda[i]
            c0_xy = (sx, sy)
            id_counter[0] += 1
            fig0 = Figura(
                fig_id=id_counter[0],
                generacion=0,
                vertices=poly,
                centroide_xy=c0_xy,
                celda_rc=(sr, sc),
                padre=None,
                raiz_id=i
            )
            self._raices.append(fig0)

            # Gen 1: Triangulacion en abanico
            k = len(poly)
            for j in range(k):
                v1 = poly[j]
                v2 = poly[(j + 1) % k]
                tri = [c0_xy, v1, v2]
                area2 = abs(tri[0][0] * (tri[1][1] - tri[2][1]) +
                            tri[1][0] * (tri[2][1] - tri[0][1]) +
                            tri[2][0] * (tri[0][1] - tri[1][0]))
                if area2 < 1.0:
                    continue
                sub_mx = (c0_xy[0] + v1[0] + v2[0]) / 3.0
                sub_my = (c0_xy[1] + v1[1] + v2[1]) / 3.0
                r1 = int(sub_my // gs)
                c1 = int(sub_mx // gs)
                if r1 < 0 or r1 >= mundo.rows or c1 < 0 or c1 >= mundo.cols:
                    continue
                if mundo.grid[r1, c1] == 1:
                    continue
                if (r1, c1) == (sr, sc):
                    continue

                cell_xy = (c1 * gs + gs / 2.0, r1 * gs + gs / 2.0)
                id_counter[0] += 1
                fig1 = Figura(
                    fig_id=id_counter[0],
                    generacion=1,
                    vertices=tri,
                    centroide_xy=cell_xy,
                    celda_rc=(r1, c1),
                    padre=fig0,
                    raiz_id=i
                )
                fig0.hijos.append(fig1)
                _subdividir_figura(fig1, mundo, gs, max_gen=15, id_counter=id_counter)

        # Recorrer todo el arbol para registrar todas las figuras
        def registrar_figuras(f):
            self._todas_figuras.append(f)
            self._figuras_por_id[f.id] = f
            self._figuras_por_celda.setdefault(f.celda, []).append(f)
            if f.generacion >= 1:
                triangulos_viz.append((f.vertices, f.generacion, f.raiz_id))
            for h in f.hijos:
                registrar_figuras(h)

        self._todas_figuras = []
        self._figuras_por_id = {}
        self._figuras_por_celda = {}
        for r0 in self._raices:
            registrar_figuras(r0)

        # Inicializar costes individuales y acumulados
        pausa = self.cfg.inspeccion_pausa_s
        peso_gen = self.cfg.inspeccion_peso_generacion
        for f in self._todas_figuras:
            f.coste_individual = pausa
            if f.padre is None:
                f.coste_acumulado = f.coste_individual
            else:
                f.coste_acumulado = (f.padre.coste_acumulado + f.coste_individual) * (peso_gen ** f.generacion)

        # Deduplicar tareas para subasta: cada celda discreta representada por su figura mas superficial
        self._tarea_por_celda = {}
        for (r, c), fig_list in self._figuras_por_celda.items():
            mejor_fig = min(fig_list, key=lambda f: f.generacion)
            padre_celda = mejor_fig.padre.celda if mejor_fig.padre else None
            t = {
                "rid": mejor_fig.id,
                "figura": mejor_fig,
                "r": r, "c": c,
                "x": mejor_fig.centroide_xy[0],
                "y": mejor_fig.centroide_xy[1],
                "nivel": mejor_fig.generacion,
                "peso": (peso_gen ** mejor_fig.generacion),
                "ajuste": 1.0,
                "costo": 0.0,
                "coste_acumulado": mejor_fig.coste_acumulado,
                "raiz_id": mejor_fig.raiz_id,
                "padre": padre_celda,
            }
            self._tarea_por_celda[(r, c)] = t

        self._tareas = list(self._tarea_por_celda.values())

        # Anotaciones para visualizacion
        self._viz_dat = {
            "nivel0": nivel0,
            "tareas": self._tareas,
            "triangulos": triangulos_viz,
            "figuras_gen0": [(r0.vertices, r0.centroide_xy, r0.raiz_id) for r0 in self._raices],
            "arbol": [self._figura_a_dict(r0) for r0 in self._raices],
        }

    def _aplicar_resultados(self, mundo):
        """Aplica los resultados de nuevas inspecciones:
        - Marca figura.inspeccion como 'sana' o 'defectuosa'.
        - Propaga el multiplicador a todos los descendientes recursivamente:
          * DEFECTUOSA: abarata los descendientes (coste_acumulado *= factor_defectivo < 1).
          * SANA: encarece los descendientes (coste_acumulado *= factor_sano > 1)."""
        if self._inspec_visto is None:
            self._inspec_visto = np.zeros((mundo.rows, mundo.cols), dtype=bool)
        inspec = (mundo.inspec > 0)
        nuevos = np.argwhere(inspec & ~self._inspec_visto)
        if nuevos.size:
            fd = self.cfg.inspeccion_factor_defectivo
            fs = self.cfg.inspeccion_factor_sano
            for (r, c) in nuevos:
                (r, c) = (int(r), int(c))
                figs = self._figuras_por_celda.get((r, c), [])
                if not figs:
                    continue
                res_str = "defectuosa" if mundo.resultado[r, c] == DEFECTUOSA else "sana"
                factor = fd if mundo.resultado[r, c] == DEFECTUOSA else fs

                for f in figs:
                    f.inspeccion = res_str
                    self._propagar_factor_descendientes(f, factor)

            for t in self._tareas:
                t["coste_acumulado"] = t["figura"].coste_acumulado
                if t["figura"].inspeccion != "null":
                    t["inspeccion"] = t["figura"].inspeccion

        self._inspec_visto |= inspec

    def _propagar_factor_descendientes(self, figura, factor):
        """Propaga recursivamente el factor multiplicativo a toda la descendencia."""
        for h in figura.hijos:
            h.coste_acumulado *= factor
            self._propagar_factor_descendientes(h, factor)

    def generaciones_activas(self, mundo):
        """G_activa(V) para cada region raiz V: el minimo nivel de las tareas de
        V aun sin inspeccionar. Barrera local independiente por Voronoi raiz."""
        g = {}
        for t in self._tareas:
            if not mundo.inspec[t["r"], t["c"]]:
                ri = t["raiz_id"]
                if ri not in g or t["nivel"] < g[ri]:
                    g[ri] = t["nivel"]
        return g

    def tarea_elegible(self, t, generaciones, mundo):
        """True si la tarea pertenece a la generacion activa de su region raiz
        (barrera local) Y su celda padre ya fue inspeccionada (o es Gen 0)."""
        if generaciones.get(t["raiz_id"]) != t["nivel"]:
            return False
        if t["padre"] is not None:
            pr, pc = t["padre"]
            if not mundo.inspec[pr, pc]:
                return False
        return True

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

    def _tarea_fallida(self, t):
        """True si la tarea esta aparcada por timeouts repetidos."""
        celda = (t["r"], t["c"])
        return (self._cooldown.get(celda, 0.0) > 0.0
                and self._fallos.get(celda, 0) >= self.cfg.inspeccion_fallos_para_perdonar)

    def _registrar_fallos(self, mundo, gs):
        """Registra timeouts recientes para aparcar tareas inalcanzables."""
        for rb in mundo.robots:
            if rb.goal is None or not rb.goal_alcanzado or not rb.timed_out:
                continue
            if rb.cola_tareas:
                continue
            celda = (int(rb.goal[1] // gs), int(rb.goal[0] // gs))
            t = self._tarea_por_celda.get(celda)
            if t is None or mundo.inspec[t["r"], t["c"]]:
                continue
            self._fallos[celda] = self._fallos.get(celda, 0) + 1
            if self._fallos[celda] >= self.cfg.inspeccion_fallos_para_perdonar:
                self._cooldown[celda] = self.cfg.inspeccion_cooldown_fallo_s

    def _decrementar_cooldowns(self, dt):
        """Descuenta el cooldown de tareas aparcadas."""
        for celda, restante in list(self._cooldown.items()):
            restante -= dt
            if restante > 0.0:
                self._cooldown[celda] = restante
                continue
            del self._cooldown[celda]
            if self._fallos.get(celda, 0) >= self.cfg.inspeccion_fallos_para_perdonar:
                self._fallos[celda] = 0
            else:
                self._fallos.pop(celda, None)

    def _tiempo_viaje(self, mundo, rb, x, y):
        """Tiempo de viaje A* desde el robot hasta (x, y)."""
        ruta = encontrar_ruta(mundo, rb.x, rb.y, x, y)
        if not ruta:
            d = math.hypot(rb.x - x, rb.y - y)
            return d / max(self.cfg.robot_max_speed_px_s, 1e-6)
        return len(ruta) * self.cfg.grid_size / max(self.cfg.robot_max_speed_px_s, 1e-6)

    def _costo_mercado(self, mundo, rb, tarea, tiempo_previo_cola):
        """Costo de mercado exacto de la tarea para el robot `rb`."""
        tau_viaje = self._tiempo_viaje(mundo, rb, tarea["x"], tarea["y"])
        tau_ins = self.cfg.inspeccion_pausa_s
        tau_total = tau_viaje + tiempo_previo_cola + tau_ins
        return max(0.0, tau_total * tarea["coste_acumulado"])

    def _preseleccionar(self, rb, pendientes, asignadas, tp):
        """Preselecciona las candidatas mas cercanas para agilizar cotizacion A*."""
        gs = self.cfg.grid_size
        par = self.cfg.inspeccion_pausa_s

        def est(t):
            d = math.hypot(rb.x - (t["c"] * gs + gs / 2),
                           rb.y - (t["r"] * gs + gs / 2))
            return (d + tp + par) * t["coste_acumulado"]

        return heapq.nsmallest(self._K_CANDIDATOS, pendientes, key=est)

    def actualizar(self, mundo, dt=0.05):
        gs = self.cfg.grid_size
        if not mundo.robots:
            return

        # 1) Estatico: descomposicion congelada una sola vez
        if not self._inicializado:
            self._construir_estatico(mundo)
            self._inicializado = True

        # 2) Actualizar resultados de inspeccion y cooldowns
        self._aplicar_resultados(mundo)
        self._registrar_fallos(mundo, gs)
        self._decrementar_cooldowns(dt)

        # 3) Filtrar tareas pendientes de la generacion activa de su Voronoi raiz
        en_curso = self._reservadas_de_robots(mundo, gs)
        generaciones = self.generaciones_activas(mundo)
        pendientes = [t for t in self._tareas
                      if (t["r"], t["c"]) not in en_curso
                      and not mundo.inspec[t["r"], t["c"]]
                      and self.tarea_elegible(t, generaciones, mundo)]
        if not pendientes:
            self._reservadas = set()
            return

        subastables = [t for t in pendientes if not self._tarea_fallida(t)]
        if not subastables:
            subastables = pendientes

        # 4) Subasta mercantil para robots libres
        libres = [i for i, rb in enumerate(mundo.robots)
                  if rb.necesita_objetivo and not rb.cola_tareas]
        if not libres:
            self._reservadas = set()
            return

        asignaciones = {}
        asignadas = set()
        while libres:
            mejor_par = None
            mejor_costo = float("inf")
            mejor_rid = -1
            for idx in libres:
                rb = mundo.robots[idx]
                tp = sum(t.get("costo", 0.0) for t in asignaciones.get(idx, []))
                for t in self._preseleccionar(rb, subastables, asignadas, tp):
                    c = self._costo_mercado(mundo, rb, t, tp)
                    if (c < mejor_costo - 1e-9
                            or (abs(c - mejor_costo) <= 1e-9 and t["rid"] < mejor_rid)):
                        mejor_costo = c
                        mejor_rid = t["rid"]
                        mejor_par = (idx, t)
            if mejor_par is None:
                break
            idx, t = mejor_par
            t["costo"] = self._tiempo_viaje(mundo, mundo.robots[idx], t["x"], t["y"]) + self.cfg.inspeccion_pausa_s
            asignaciones.setdefault(idx, []).append(t)
            asignadas.add((t["r"], t["c"]))
            libres.remove(idx)

        # 5) Asignar tareas a la cola de cada robot ganador
        for idx, lista in asignaciones.items():
            rb = mundo.robots[idx]
            lista.sort(key=lambda t: (t["coste_acumulado"] * (t["costo"] + 1e-6), t["r"], t["c"]))
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
