"""Planificador de la etapa de Inspeccion (Fase 3).

En las estrategias 2E, tras la transicion global Exploracion -> Inspeccion en
el enjambre, los robots pasan a inspeccionar las baldosas sin revisar. Esta
es la implementacion definitiva (Fase 3). Combina tres mecanismos:

1. **Colas de tareas** en `Robot`: cada robot lleva una cola de objetivos
   ordenada por prioridad (menor costo de mercado primero) y la consume de
   forma automatica, permitiendo repriorizacion dinamica entre ciclos.

2. **Voronoi Geodesico y Triangulacion Recursiva Homogenea por Figuras**:
   Al entrar en la etapa de inspeccion se congela la descomposicion geometrica:
   * **Voronoi (Gen 0):** Geodesico CENTROIDAL sobre el espacio libre. El
     numero de regiones es el numero de robots, pero las semillas iniciales se
     distribuyen por el espacio libre (farthest-point sampling geodesico,
     deterministico, sin usar la posicion de los robots) y se itera (Lloyd)
     reemplazando cada sitio por el centroide geometrico de su region, de modo
     que al converger los centros del Voronoi coinciden con los centroides de
     las figuras Gen 0 creadas. Cada celda libre se asigna al sitio de menor
     distancia de camino (BFS multi-fuente sobre el espacio libre, paredes
     omitidas). El poligono de cada region se extrae de
     sus celdas y sus aristas coinciden con las aristas internas de las paredes
     del perimetro (no se usan las esquinas del mapa como caja) y con las
     fronteras entre regiones. Los dientes de sierra de las fronteras entre
     regiones se colapsan a tramos oblicuos entre vertices de la cuadricula.
   * **Triangulacion (Gen 1):** Abanico desde el centroide geometrico de cada
     region: cada arista del contorno exterior genera (M, vi, vi+1), de modo
     que el centroide de la figura Gen 0 actua como vertice central del primer
     nivel (homogeneo con la recursion de Gen 2+). Las zonas que el centroide
     no "ve" (region no estrellada u hoyos) se rellenan con ear-clipping
     exacto de las celdas libres restantes.
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
   precio greedy. El INTERES de cada tarea es

   `interes = interes_inspeccion / ((generacion + 1) * distancia_A**alpha * 2^R)`

   * `interes_inspeccion` (coste_acumulado) parte de 1.0 para toda figura y
     cambia SOLO por el resultado de sus ancestros inspeccionados en la misma
     rama: si la gen1 resulta DEFECTUOSA, todas las generaciones siguientes de
     esa rama quedan multiplicadas por `inspeccion_factor_defectivo` (> 1); si
     la gen2 tambien resulta defectuosa, se vuelve a multiplicar (acumulacion
     generacion a generacion). Si resulta SANA, se multiplica por
     `inspeccion_factor_sano` (< 1).
   * `(generacion + 1)`: cuota decremental por nivel del arbol (evita el 0 en
     el denominador): cada generacion siguiente parte de un interes de base
     menor.
   * `distancia_A**alpha`: la distancia (tiempo de viaje A*) pesa DEBILMENTE
     (exponente `alpha = inspeccion_peso_distancia`, default 0.5), de forma que
     las zonas con mas fallas dominan a las simplemente cercanas.
   * `2^R`: termino DINAMICO anti-hacinamiento EXPONENCIAL, recalculado en cada
     evaluacion del interes: `R` es el numero de robots dentro de la region
     Gen0 (Voronoi de primera division) de la tarea en ese instante. Reduce
     fuerte el interes de mandar mas robots a una region ya concurrida (1 robot
     divide entre 2, 2 entre 4, 3 entre 8...): los robots se interfieren. Con
     `R=0` el factor es 1 (sin penalizacion); nunca anula el interes.

   El interes es siempre estrictamente positivo (nunca 0 ni negativo). La
   eligibilidad de cada tarea la controla `inspeccion_padre_antes_hijo`:
   * `True` (default): orden estricto padre->hijo por rama: cada figura hijo se
     subasta solo cuando su padre directo ya fue inspeccionado.
   * `False`: sin gate de nivel ni padre, toda tarea pendiente es subastable
     (`generaciones_activas` queda como diagnostico y su resultado no filtra).
   Cada ciclo adjudica UNA tarea por robot libre: el robot toma la primera
   prioridad de su lista y el resto queda libre para otros robots; al completar
   la tarea, la siguiente subasta repite.
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


def _clip_polygon_against_walls(poly, mundo, gs):
    """Retorna el poligono tal cual.

    El poligono de la region ya fue extraido de las celdas libres y sus aristas
    coinciden con las aristas internas de las paredes; no se necesita recorte
    adicional."""
    return list(poly)
def _clip_polygon_halfplane(poly, p1, p2, site):
    """Clip polygon against bisector half‑plane containing the site.

    poly: list of (x, y) vertices (convex).
    p1, p2: two points defining the bisector line.
    site: (x, y) of the current site; we keep the side where site lies.
    """
    # Compute cross product for the site to determine which side to keep
    site_cross = (p2[0] - p1[0]) * (site[1] - p1[1]) - (p2[1] - p1[1]) * (site[0] - p1[0])
    new_poly = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cur_cross = (p2[0] - p1[0]) * (cur[1] - p1[1]) - (p2[1] - p1[1]) * (cur[0] - p1[0])
        nxt_cross = (p2[0] - p1[0]) * (nxt[1] - p1[1]) - (p2[1] - p1[1]) * (nxt[0] - p1[0])
        cur_in = cur_cross * site_cross >= 0
        nxt_in = nxt_cross * site_cross >= 0
        if cur_in:
            new_poly.append(cur)
        if cur_in != nxt_in:
            inter = _line_intersection(cur, nxt, p1, p2)
            new_poly.append(inter)
    return new_poly



def _normalizar_poligono(poly, tol=1e-6):
    """Elimina vertices duplicados consecutivos y colineales de un poligono
    convexo ordenado (conserva la convexidad y el orden de giro). Los
    duplicados aparecian en `_clip_polygon_halfplane` cuando una mediatriz
    pasaba exactamente por un vertice existente, lo que creaba aristas de
    longitud ~0 y triangulos degenerados en el abanico Gen 1."""
    if not poly:
        return []
    pts = []
    for p in poly:
        if pts and abs(p[0] - pts[-1][0]) < tol and abs(p[1] - pts[-1][1]) < tol:
            continue
        pts.append(p)
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < tol and abs(pts[0][1] - pts[-1][1]) < tol:
        pts.pop()
    n = len(pts)
    if n < 3:
        return []
    out = []
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-9:
            out.append(b)
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
        poly = _clip_polygon_halfplane(poly, p1, p2, site)
    return _normalizar_poligono(poly)


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


def _celda_libre_mas_cercana(mundo, r, c):
    """Celda libre (grid == 0) mas cercana a (r, c) medida en pasos de BFS.

    Los sitios que caen sobre una pared interna (o fuera del mapa) se
    "snappean" al punto no-pared mas cercano, de modo que toda celda libre
    quede dentro de alguna region y no existan huecos de cobertura."""
    rows, cols = mundo.rows, mundo.cols
    if 0 <= r < rows and 0 <= c < cols and mundo.grid[r, c] == 0:
        return (r, c)
    dq = deque([(r, c)])
    visto = {(r, c)}
    while dq:
        cr, cc = dq.popleft()
        if 0 <= cr < rows and 0 <= cc < cols and mundo.grid[cr, cc] == 0:
            return (cr, cc)
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if (nr, nc) not in visto:
                visto.add((nr, nc))
                dq.append((nr, nc))
    return None


def _area_bucle(loop):
    """Area signada (shoelace) de un poligono cerrado en pixels."""
    s = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _quitar_colineales(loop):
    """Elimina vertices duplicados consecutivos y colineales de un bucle."""
    pts = []
    for p in loop:
        if pts and p == pts[-1]:
            continue
        pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-9:
            out.append(b)
    return out


def _loops_de_region(celdas, rows, cols, gs):
    """Contornos de una region de celdas libres (4-conexa) como poligonos en
    pixels, recorridos SIEMPRE con la region a la izquierda.

    Las aristas del contorno son las aristas internas de las paredes (bordes
    del perimetro incluidos: no se usan las esquinas del mapa como caja) y las
    fronteras entre regiones vecinas. Devuelve una lista de loops cerrados."""
    celdas = set(celdas)
    aristas = {}
    for (r, c) in celdas:
        x0 = c * gs
        y0 = r * gs
        x1 = x0 + gs
        y1 = y0 + gs
        # Borde contra una pared o el exterior: la arista interna dela pared.
        if r - 1 < 0 or (r - 1, c) not in celdas:
            aristas[(x0, y0)] = (x1, y0)
        if c + 1 >= cols or (r, c + 1) not in celdas:
            aristas[(x1, y0)] = (x1, y1)
        if r + 1 >= rows or (r + 1, c) not in celdas:
            aristas[(x1, y1)] = (x0, y1)
        if c - 1 < 0 or (r, c - 1) not in celdas:
            aristas[(x0, y1)] = (x0, y0)
    loops = []
    restantes = set(aristas.items())
    while restantes:
        inicio, siguiente = min(restantes)
        restantes.discard((inicio, siguiente))
        loop = [inicio, siguiente]
        actual = siguiente
        while actual != inicio:
            nxt = aristas.get(actual)
            if nxt is None or (actual, nxt) not in restantes:
                break
            restantes.discard((actual, nxt))
            loop.append(nxt)
            actual = nxt
        loops.append(_quitar_colineales(loop))
    return [l for l in loops if len(l) >= 3]


def _centroid_poligono(verts):
    """Centroide geometrico de un poligono cerrado (shoelace)."""
    n = len(verts)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return verts[0]
    if n == 2:
        return ((verts[0][0] + verts[1][0]) / 2.0,
                (verts[0][1] + verts[1][1]) / 2.0)
    A6 = 0.0
    cx_n = 0.0
    cy_n = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A6 += cross
        cx_n += (x0 + x1) * cross
        cy_n += (y0 + y1) * cross
    if abs(A6) < 1e-12:
        return (sum(v[0] for v in verts) / n,
                sum(v[1] for v in verts) / n)
    return (cx_n / (3.0 * A6), cy_n / (3.0 * A6))


def _colapsar_escaleras(loop, mundo, gs, quiebre=None):
    """Reemplaza las escaleras (zigzag monotono) de la frontera de una region
    por tramos oblicuos con un criterio CANONICO e independiente de la
    orientacion del recorrido.

    Dos regiones adyacentes comparten la misma escalera de celdas en su
    frontera comun. Si cada una colapsa por su cuenta, pueden elegir ventanas
    distintas de la misma escalera y sus diagonales quedan desplazadas entre
    si, superponiendo/separando las areas. Para evitarlo, cada escalera
    (bloque maximal de aristas unitarias monotono en fila y columna) se
    colapsa SIEMPRE desde su vertice canonico de inicio (min y, min x):
    - Si el tramo recto inicio->fin no corta pared, todo el bloque se
      reemplaza por una sola linea oblicua (2 vertices).
    - Si el tramo pleno cortaria una pared, se busca, anclada al mismo inicio
      canonico, la ventana balanceada (igual numero de pasos horizontales y
      verticales) mas larga cuya diagonal sea segura, y el resto se conserva.
    Un tramo que cortaria una pared no se colapsa (no se permiten diagonales
    sobre paredes).

    `quiebre` (opcional) es el conjunto de vertices "union" de la malla de
    regiones (vertices donde se tocan 3 o mas regiones). En esos puntos la
    escalera cambia de par de regiones vecinas, por lo que el bloque se corta:
    asi dos regiones que comparten SOLO un tramo de escalera lo colapsan de
    forma identica, sin arrastrar el tramo que cada una comparte con una
    tercera region (que generaria diagonales distintas y solapamiento)."""
    rows, cols = mundo.rows, mundo.cols
    grid = mundo.grid

    def _card_dir(pa, pb):
        c0 = (int(pa[0] // gs), int(pa[1] // gs))
        c1 = (int(pb[0] // gs), int(pb[1] // gs))
        dr = c1[0] - c0[0]
        dc = c1[1] - c0[1]
        return (0 if dr == 0 else (1 if dr > 0 else -1),
                0 if dc == 0 else (1 if dc > 0 else -1))

    def _diag_segura(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return False
        pasos = max(2, int(round(max(abs(dx), abs(dy)) / gs)))
        for k in range(1, pasos):
            t = k / float(pasos)
            px = a[0] + dx * t
            py = a[1] + dy * t
            r = int(py // gs)
            c = int(px // gs)
            if not (0 <= r < rows and 0 <= c < cols and grid[r, c] == 0):
                return False
            for s in (-1.0, 1.0):
                qx = px + s * gs * 0.25 * (-dy) / L
                qy = py + s * gs * 0.25 * dx / L
                r = int(qy // gs)
                c = int(qx // gs)
                if not (0 <= r < rows and 0 <= c < cols and grid[r, c] == 0):
                    return False
        return True

    def _max_block(esc, i):
        """Longitud (en aristas) del bloque maximal de aristas unitarias
        monotoas que empieza en la arista i. Se corta en los vertices de
        quiebre (union de 3 o mas regiones)."""
        n = len(esc)
        sr = 0
        sc = 0
        L = 0
        j = i
        while j < i + n:
            k = j % n
            if k != i and quiebre is not None and tuple(esc[k]) in quiebre:
                break
            a = esc[k]
            b = esc[(k + 1) % n]
            plen = math.hypot(b[0] - a[0], b[1] - a[1])
            if abs(plen - gs) > 1e-6:
                break
            rv, ch = _card_dir(a, b)
            if (rv != 0 and ch != 0) or (rv == 0 and ch == 0):
                break
            if rv != 0 and sr != 0 and rv != sr:
                break
            if ch != 0 and sc != 0 and ch != sc:
                break
            if rv != 0:
                sr = rv
            if ch != 0:
                sc = ch
            L += 1
            j += 1
        return L, sr, sc

    def _colapsar_una_pasada(esc):
        n = len(esc)
        if n < 4:
            return esc, False
        usados = [False] * n
        quitar = []
        for i in range(n):
            if usados[i]:
                continue
            L, sr, sc = _max_block(esc, i)
            if L < 2 or sr == 0 or sc == 0:
                continue
            for t in range(L):
                usados[(i + t) % n] = True
            verts = [esc[(i + t) % n] for t in range(L + 1)]
            vin = min(verts, key=lambda p: (p[1], p[0]))
            if tuple(verts[0]) == tuple(vin):
                arc = verts
            else:
                arc = list(reversed(verts))
            vend = list(arc[-1])
            if not _diag_segura(vin, vend):
                # fallback: ventana balanceada anclada al inicio canonico
                h = 0
                v = 0
                mejor_len = 0
                mejor_end = None
                for t in range(1, len(arc)):
                    a = arc[t - 1]
                    b = arc[t]
                    dr = int(b[1] // gs) - int(a[1] // gs)
                    dc = int(b[0] // gs) - int(a[0] // gs)
                    if dr != 0:
                        v += abs(dr)
                    if dc != 0:
                        h += abs(dc)
                    if v == h and v >= 1 and (v + h) > mejor_len \
                            and _diag_segura(vin, b):
                        mejor_len = v + h
                        mejor_end = tuple(b)
                if mejor_end is None:
                    continue
                intermedios = []
                for p in arc:
                    intermedios.append(tuple(p))
                    if tuple(p) == mejor_end:
                        break
                quitar.extend(intermedios[1:-1])
            else:
                quitar.extend(tuple(p) for p in arc[1:-1])
        if not quitar:
            return esc, False
        qset = set(quitar)
        nuevo = [p for p in esc if tuple(p) not in qset]
        return _quitar_colineales(nuevo), len(nuevo) < n

    esc = list(loop)
    for _paso in range(8):
        esc, cambio = _colapsar_una_pasada(esc)
        if not cambio:
            break
    return esc


def _vertices_quiebre(loops_por_region):
    """Vertices de la malla donde se tocan 3 o mas regiones (grado > 2 en la
    union no orientada de todos los bordes). En esos puntos las escaleras
    cambian de par de regiones vecinas."""
    vecinos = {}
    for loops in loops_por_region.values():
        for l in loops:
            n = len(l)
            for i in range(n):
                a = tuple(l[i])
                b = tuple(l[(i + 1) % n])
                if a == b:
                    continue
                vecinos.setdefault(a, set()).add(b)
                vecinos.setdefault(b, set()).add(a)
    return {v for v, ns in vecinos.items() if len(ns) > 2}


def _poligono_de_region(celdas, mundo, gs, quiebre=None):
    """Poligono pixel de una region de celdas: (exterior, huecos).

    Las escaleras de celdas de su frontera se colapsan a tramos oblicuos
    (sin cortar paredes). El exterior se orienta CCW (area signada positiva)
    y los huecos CW, de modo que la region quede siempre al lado izquierdo
    del recorrido."""
    loops = _loops_de_region(celdas, mundo.rows, mundo.cols, gs)
    loops = [_quitar_colineales(_colapsar_escaleras(l, mundo, gs, quiebre))
             for l in loops]
    loops = [l for l in loops if len(l) >= 3]
    if not loops:
        return [], []
    areas = [_area_bucle(l) for l in loops]
    k = max(range(len(loops)), key=lambda i: abs(areas[i]))
    exterior = loops[k]
    if areas[k] < 0:
        exterior = list(reversed(exterior))
    huecos = []
    for i, (l, a) in enumerate(zip(loops, areas)):
        if i == k:
            continue
        huecos.append(list(reversed(l)) if a > 0 else l)
    return exterior, huecos


def _dist_a_sitios(mundo, celdas_libres, sitios):
    """Distancia de camino (BFS multi-fuente 4-vecinos) de cada celda libre al
    sitio mas cercano. Devuelve un arreglo (rows, cols) con -1 en las celdas
    no alcanzables (paredes o componentes sin sitio)."""
    rows, cols = mundo.rows, mundo.cols
    dist = np.full((rows, cols), -1, dtype=np.int32)
    dq = deque()
    for (r, c) in sitios:
        r = int(r)
        c = int(c)
        if (0 <= r < rows and 0 <= c < cols
                and mundo.grid[r, c] == 0 and dist[r, c] < 0):
            dist[r, c] = 0
            dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        d = dist[r, c]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < rows and 0 <= nc < cols
                    and mundo.grid[nr, nc] == 0 and dist[nr, nc] < 0):
                dist[nr, nc] = d + 1
                dq.append((nr, nc))
    return dist


def _semillas_distribuidas(mundo, celdas_libres, k):
    """Genera `k` sitios iniciales repartidos por el espacio libre con
    farthest-point sampling geodesico (determinista): el sitio i + 1 es la
    celda libre mas lejana (distancia de camino) a los sitios ya elegidos.
    No depende de las posiciones de los robots, solo de la geometria."""
    celdas = [tuple(map(int, c)) for c in celdas_libres]
    if not celdas:
        return []
    k = min(k, len(celdas))
    sitios = [celdas[0]]
    while len(sitios) < k:
        dist = _dist_a_sitios(mundo, celdas, sitios)
        mejor = None
        mejor_d = -1
        for c in celdas:
            d = dist[c[0], c[1]]
            if d > mejor_d:
                mejor_d = d
                mejor = c
        if mejor is None or mejor_d <= 0:
            break
        sitios.append(mejor)
    return sitios


def _quiebres_de_regiones(regiones, rows, cols, gs):
    """Vertices "union" de la malla de regiones (grado > 2 en la union no
    orientada de todos los bordes). En esos puntos las escaleras de celdas
    cambian de par de regiones vecinas, por lo que los bloques de colapso se
    cortan para que dos regiones que comparten SOLO un tramo de escalera lo
    colapsen de forma identica."""
    return _vertices_quiebre(
        {s: _loops_de_region(r, rows, cols, gs) for s, r in regiones.items()}
    )


def _area2_tri(a, b, c):
    """Doble area signada del triangulo (a, b, c)."""
    return ((b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0]))


def _punto_en_triangulo(p, a, b, c):
    """True si p esta dentro o sobre la frontera de (a, b, c)."""
    d1 = _area2_tri(a, b, p)
    d2 = _area2_tri(b, c, p)
    d3 = _area2_tri(c, a, p)
    pos = any(d > 1e-9 for d in (d1, d2, d3))
    neg = any(d < -1e-9 for d in (d1, d2, d3))
    return not (pos and neg)


def _segmento_dentro_de_region(a, b, celdas, gs):
    """True si el segmento a-b queda (a muestreo denso) dentro de la region."""
    for k in range(1, 16):
        t = k / 16.0
        px = a[0] + (b[0] - a[0]) * t
        py = a[1] + (b[1] - a[1]) * t
        if (int(py // gs), int(px // gs)) not in celdas:
            return False
    return True


def _puentear_huecos(exterior, huecos, celdas, gs):
    """Une cada hueco al contorno exterior con un puente interior, devolviendo
    una unica cadena (poligono debilmente simple) para el ear-clipping."""
    pts = list(exterior)
    for h in huecos:
        mejor = None
        mejor_d2 = None
        for i, v in enumerate(pts):
            for j, u in enumerate(h):
                d2 = (v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2
                if mejor_d2 is not None and d2 >= mejor_d2:
                    continue
                if not _segmento_dentro_de_region(v, u, celdas, gs):
                    continue
                mejor = (i, j)
                mejor_d2 = d2
        if mejor is None:
            continue
        i, j = mejor
        rev = list(reversed(h[j:] + h[:j + 1]))
        pts = pts[:i + 1] + rev + pts[i:]
    return pts


def _ear_clip(pts):
    """Triangulacion por ear-clipping de un poligono simple CCW.

    Devuelve triangulos (CCW) que rellenan exactamente el poligono; valido
    tambien para poligonos concavos como las regiones geodesicas."""
    if len(pts) < 3:
        return []
    if _area_bucle(pts) < 0:
        pts = list(reversed(pts))
    idx = list(range(len(pts)))
    tris = []
    max_pases = len(pts) * len(pts) + 8
    n_iter = 0
    while len(idx) > 3 and n_iter < max_pases:
        n_iter += 1
        n = len(idx)
        cortado = False
        for k in range(n):
            i0 = idx[(k - 1) % n]
            i1 = idx[k]
            i2 = idx[(k + 1) % n]
            p0, p1, p2 = pts[i0], pts[i1], pts[i2]
            if _area2_tri(p0, p1, p2) <= 1e-9:
                continue
            bloqueado = False
            for j in idx:
                if j == i0 or j == i1 or j == i2:
                    continue
                q = pts[j]
                if q == p0 or q == p1 or q == p2:
                    continue
                if _punto_en_triangulo(q, p0, p1, p2):
                    bloqueado = True
                    break
            if bloqueado:
                continue
            tris.append([list(p0), list(p1), list(p2)])
            del idx[k]
            cortado = True
            break
        if not cortado:
            break
    if len(idx) == 3:
        tris.append([list(pts[idx[0]]), list(pts[idx[1]]), list(pts[idx[2]])])
    return tris


def _triangulo_bbox(gs, tri):
    """(rmin, rmax, cmin, cmax) del bbox del triangulo recortado al mapa."""
    xmin = min(p[0] for p in tri)
    xmax = max(p[0] for p in tri)
    ymin = min(p[1] for p in tri)
    ymax = max(p[1] for p in tri)
    return (max(0, int(ymin // gs)), max(0, int(xmin // gs)),
            int(ymax // gs), int(xmax // gs))


def _triangulo_no_cruza_paredes(mundo, tri, gs):
    """True si el triangulo (a, b, c) no contiene el centro de ninguna celda
    de pared (grid != 0). Detecta abanicos que cruzan paredes internas u
    hoyos (los huecos de una region son paredes grid == 1)."""
    rmin, cmin, rmax, cmax = _triangulo_bbox(gs, tri)
    a, b, c = tri
    for r in range(rmin, rmax + 1):
        for cc in range(cmin, cmax + 1):
            px = cc * gs + gs / 2.0
            py = r * gs + gs / 2.0
            if _punto_en_triangulo((px, py), a, b, c) and mundo.grid[r, cc] != 0:
                return False
    return True


def _celdas_en_triangulo(mundo, tri, gs):
    """Celdas libres cuyo centro cae dentro del triangulo (a, b, c)."""
    rmin, cmin, rmax, cmax = _triangulo_bbox(gs, tri)
    a, b, c = tri
    out = []
    for r in range(rmin, rmax + 1):
        for cc in range(cmin, cmax + 1):
            if mundo.grid[r, cc] != 0:
                continue
            px = cc * gs + gs / 2.0
            py = r * gs + gs / 2.0
            if _punto_en_triangulo((px, py), a, b, c):
                out.append((int(r), int(cc)))
    return out


def _componentes_celdas(celdas, mundo):
    """Componentes conexas (4-vecinos) de un conjunto de celdas libres."""
    celdas_set = set(celdas)
    vistos = set()
    comps = []
    for semilla in celdas:
        if semilla in vistos:
            continue
        dq = deque([semilla])
        vistos.add(semilla)
        comp = []
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mundo.rows and 0 <= nc < mundo.cols
                        and (nr, nc) in celdas_set and (nr, nc) not in vistos):
                    vistos.add((nr, nc))
                    dq.append((nr, nc))
        comps.append(comp)
    return comps


def _triangulacion_region(exterior, huecos, celdas, gs, mundo):
    """Triangulos (Gen 1) que cubren exactamente el poligono de una region
    geodesica, tomando el CENTROIDE como vertice central.

    Se genera un abanico (M, vi, vi+1) por cada arista del contorno exterior,
    con M = centroide geometrico de la region: asi el centroide de la figura
    Gen 0 actua como vertice central del primer nivel, homogeneo con la
    subdivision recursiva de las generaciones siguientes (cada generacion
    transforma centroides en vertices). Los triangulos del abanico que
    cruzan paredes, hoyos o zonas ajenas (region no estrellada respecto de M)
    se descartan, y las celdas libres que quedan sin cubrir se agrupan en
    componentes conexas que se rellenan con ear-clipping exacto."""
    if len(exterior) < 3:
        return []
    celdas_set = set(celdas)
    M = _centroid_poligono(exterior)
    n = len(exterior)
    tris = []
    cubiertas = set()
    for i in range(n):
        a = exterior[i]
        b = exterior[(i + 1) % n]
        tri = [list(M), list(a), list(b)]
        if not _triangulo_no_cruza_paredes(mundo, tri, gs):
            continue
        cubre = _celdas_en_triangulo(mundo, tri, gs)
        if not cubre or any(c not in celdas_set for c in cubre):
            continue
        tris.append(tri)
        cubiertas.update(cubre)
    restantes = [c for c in celdas if c not in cubiertas]
    if restantes:
        for comp in _componentes_celdas(restantes, mundo):
            exterior2, huecos2 = _poligono_de_region(comp, mundo, gs)
            if len(exterior2) < 3:
                continue
            pts = _puentear_huecos(exterior2, huecos2, comp, gs)
            tris.extend(_ear_clip(pts))
    if not tris:
        # Cobertura residual defensiva (centroide sin visibilidad util).
        return _ear_clip(exterior) if not huecos \
            else _ear_clip(_puentear_huecos(exterior, huecos, celdas, gs))
    return tris


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
      coste_acumulado: INTERES DE INSPECCION acumulado por rama. Empieza en 1.0
        para toda figura y cambia SOLO por el resultado de los ancestros
        inspeccionados: sana lo multiplica por `inspeccion_factor_sano` (< 1) y
        defectuosa por `inspeccion_factor_defectivo` (> 1), acumulandose en
        cada generacion (gen1 defectuosa y gen2 defectuosa => x4 en la rama).
      raiz_id: indice de la region Voronoi Gen 0 de la cual desciende
    """
    __slots__ = ("id", "generacion", "padre", "hijos", "vertices",
                 "centroide_xy", "celda", "inspeccion", "coste_individual",
                 "coste_acumulado", "raiz_id", "huecos")

    def __init__(self, fig_id, generacion, vertices, centroide_xy, celda_rc,
                 padre=None, raiz_id=0):
        self.id = fig_id
        self.generacion = generacion
        self.padre = padre
        self.hijos = []
        self.huecos = []
        self.vertices = vertices
        self.centroide_xy = centroide_xy
        self.celda = celda_rc
        self.inspeccion = "null"
        self.coste_individual = 0.0
        self.coste_acumulado = 0.0
        self.raiz_id = raiz_id


def _subdividir_figura(figura, mundo, gs, max_gen=15, id_counter=None):
    """Subdivision recursiva uniforme para cualquier figura triangular.

    Divide el triangulo en 3 subtriangulos usando su CENTROIDE GEOMETRICO
    como vertice central: (M, v1, v2), (M, v2, v3) y (M, v3, v1). El centroide
    de cada sub-figura es a su vez el centroide geometrico del subtriangulo,
    de modo que cada generacion transforma el centroide de los triangulos
    recien creados en vertices de la generacion siguiente (3 figuras nuevas
    por triangulo en cada generacion). Se detiene y descarta la iteracion si
    el centroide del sub-triangulo cae fuera del mapa, en pared o en la misma
    celda que el padre, o cuando el triangulo ya es mas pequeno que una celda
    del mapa (resolucion de la cuadricula)."""
    if figura.generacion >= max_gen:
        return
    tri = figura.vertices
    if len(tri) != 3:
        return
    v1, v2, v3 = tri
    # Centroide geometrico del triangulo actual: el nuevo vertice central.
    m_xy = ((v1[0] + v2[0] + v3[0]) / 3.0,
            (v1[1] + v2[1] + v3[1]) / 3.0)
    area2p = abs(v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) +
                 v3[0] * (v1[1] - v2[1]))
    if area2p < 1.0:
        return  # triangulo degenerado: no se subdivide
    if area2p < 2.0 * gs * gs:
        return  # area menor que una celda: resolucion del mapa
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
        if (r, c) == figura.celda:
            continue

        id_counter[0] += 1
        hijo = Figura(
            fig_id=id_counter[0],
            generacion=figura.generacion + 1,
            vertices=sub,
            centroide_xy=(sub_mx, sub_my),
            celda_rc=(r, c),
            padre=figura,
            raiz_id=figura.raiz_id
        )
        figura.hijos.append(hijo)
        _subdividir_figura(hijo, mundo, gs, max_gen=max_gen, id_counter=id_counter)


def _punto_en_poligono(x, y, poly):
    """Test de inclusion punto-en-poligono (ray casting) sobre `poly`, una lista
    de (px, py) en pixeles (admite vertices flotantes)."""
    dentro = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            dentro = not dentro
        j = i
    return dentro


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
        self._nivel0 = None         # capa task allocation (solo diagnostico)
        self._region_de_celda = None  # celda -> gen0 raiz_id (para contar robots)
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
            "huecos": [list(h) for h in f.huecos],
            "inspeccion": f.inspeccion,
            "coste_acumulado": f.coste_acumulado,
            "hijos": [self._figura_a_dict(h) for h in f.hijos]
        }

    def _construir_estatico(self, mundo):
        """Congela la descomposicion completa de figuras al arrancar la inspeccion.
        El Voronoi Gen 0 ahora es GEODESICO CENTROIDAL:
        1. El numero de regiones es el numero de robots, pero las posiciones de
           los robots NO se usan para centrar el Voronoi: las semillas iniciales
           se distribuyen por el espacio libre (farthest-point sampling
           geodesico, determinista).
        2. Se itera (Lloyd): cada sitio se reemplaza por el centroide geometrico
           de la region que genera (extraido de sus celdas, con las escaleras de
           frontera colapsadas) hasta que los sitios dejan de moverse. Al
           converger, los centros del Voronoi coinciden con los centroides de
           las figuras Gen 0 creadas. Cada celda libre se asigna al sitio con
           menor distancia de camino (BFS multi-fuente 4-vecinos sobre el
           espacio libre); las paredes quedan omitidas de todo el analisis.
        3. Triangulacion Gen 1: abanico desde el CENTROIDE geometrico de cada
           poligono de region: cada arista produce (M, vi, vi+1), de modo que
           la figura Gen 0 tiene su centroide como vertice central. Las zonas
           no estrelladas u hoyos se rellenan con ear-clipping exacto de las
           celdas libres restantes. Cada triangulo tiene centroide_xy =
           centroide GEOMETRICO del triangulo.
        4. Triangulacion Gen 2..n: recursion ternaria homogenea (M, v_1, v_2)
           donde M es el centroide geometrico del triangulo padre (cada
           triangulo genera 3 nuevos por generacion), con parada cuando el
           centroide coincide con la celda del padre o el triangulo es menor
           que una celda del mapa.
        5. Inicializacion de costes individuales y acumulados."""
        gs = self.cfg.grid_size
        rows, cols = mundo.rows, mundo.cols
        sitios = [(rb.x, rb.y) for rb in mundo.robots]
        if not sitios:
            return

        # 1. Numero de regiones = robots agrupados por celda libre (solo el
        #    CONJUNTO: las posiciones de los robots NO se usan para centrar
        #    el Voronoi).
        sitios_celda = []
        for sx, sy in sitios:
            sr = int(sy // gs)
            sc = int(sx // gs)
            if 0 <= sr < rows and 0 <= sc < cols and mundo.grid[sr, sc] == 0:
                sitios_celda.append((sr, sc))
            else:
                libre = _celda_libre_mas_cercana(mundo, sr, sc)
                sitios_celda.append(libre if libre is not None
                                    else (max(0, min(rows - 1, sr)),
                                          max(0, min(cols - 1, sc))))
        k = len(dict.fromkeys(sitios_celda))

        # 2. Voronoi GEODESICO CENTROIDAL: las semillas se distribuyen por el
        #    espacio libre (farthest-point sampling geodesico, deterministico,
        #    sin robots) y se itera (Lloyd) reemplazando cada sitio por el
        #    centroide de la region que genera, de modo que al converger los
        #    centros del Voronoi COINCIDEN con los centroides de las figuras
        #    Gen 0 creadas.
        celdas_libres = list(zip(*np.where(mundo.grid == 0)))
        sitios = _semillas_distribuidas(mundo, celdas_libres, k)
        if len(sitios) < 1:
            return
        for _ in range(60):
            regiones0 = _particionar_geodesico(mundo, celdas_libres, sitios)
            quiebres = _quiebres_de_regiones(regiones0, rows, cols, gs)
            nuevos = []
            for s in sitios:
                region = regiones0.get(s)
                if not region:
                    nuevos.append(s)
                    continue
                exterior, _ = _poligono_de_region(region, mundo, gs, quiebres)
                if len(exterior) < 3:
                    nuevos.append(s)
                    continue
                cx, cy = _centroid_poligono(exterior)
                nr = int(cy // gs)
                nc = int(cx // gs)
                nr = max(0, min(rows - 1, nr))
                nc = max(0, min(cols - 1, nc))
                if mundo.grid[nr, nc] != 0:
                    libre = _celda_libre_mas_cercana(mundo, nr, nc)
                    nuevos.append(libre if libre is not None else s)
                else:
                    nuevos.append((nr, nc))
            nuevos = list(dict.fromkeys(nuevos))
            if nuevos == sitios:
                break
            sitios = nuevos
        regiones0 = _particionar_geodesico(mundo, celdas_libres, sitios)
        quiebres = _quiebres_de_regiones(regiones0, rows, cols, gs)
        sitios_unicos = sitios

        # 3. Particion geodesica discreta final: distancia de camino real
        #    (paredes omitidas). `nivel0` queda como capa de task allocation.
        nivel0 = np.full((rows, cols), -1, dtype=np.int16)
        for rindex, (s, sub) in enumerate(regiones0.items()):
            for rr, cc in sub:
                nivel0[rr, cc] = rindex

        self._raices = []
        id_counter = [0]
        triangulos_viz = []
        sitio_a_root = {}

        for i, (sr, sc) in enumerate(sitios_unicos):
            region = regiones0.get((sr, sc))
            if not region:
                continue
            exterior, huecos = _poligono_de_region(region, mundo, gs, quiebres)
            if len(exterior) < 3:
                continue
            c0_xy = _centroid_poligono(exterior)
            c0_r = int(c0_xy[1] // gs)
            c0_c = int(c0_xy[0] // gs)
            c0_r = max(0, min(rows - 1, c0_r))
            c0_c = max(0, min(cols - 1, c0_c))
            id_counter[0] += 1
            fig0 = Figura(
                fig_id=id_counter[0],
                generacion=0,
                vertices=exterior,
                centroide_xy=c0_xy,
                celda_rc=(c0_r, c0_c),
                padre=None,
                raiz_id=i
            )
            fig0.huecos = huecos
            self._raices.append(fig0)
            sitio_a_root[(sr, sc)] = fig0

            # Gen 1: abanico desde el centroide de la region (M, vi, vi+1),
            # con ear-clipping residual en zonas no estrelladas. Se descartan
            # los triangulos con centroide en pared, fuera del mapa o en la
            # misma celda de la raiz.
            for tri in _triangulacion_region(exterior, huecos, region, gs, mundo):
                mx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
                my = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
                r1 = int(my // gs)
                c1 = int(mx // gs)
                if not (0 <= r1 < rows and 0 <= c1 < cols):
                    continue
                if mundo.grid[r1, c1] == 1:
                    continue
                if (r1, c1) == (sr, sc):
                    continue
                id_counter[0] += 1
                fig1 = Figura(
                    fig_id=id_counter[0],
                    generacion=1,
                    vertices=tri,
                    centroide_xy=(mx, my),
                    celda_rc=(r1, c1),
                    padre=fig0,
                    raiz_id=i
                )
                fig0.hijos.append(fig1)
                _subdividir_figura(fig1, mundo, gs, max_gen=15,
                                   id_counter=id_counter)

        # Mapa celda -> gen0 raiz_id (los rindex de `nivel0` pueden desalinearse
        # con `raiz_id` si alguna region degenerada se descarto; por eso se
        # etiqueta con el raiz_id real de su raiz).
        self._nivel0 = nivel0
        self._region_de_celda = np.full((rows, cols), -1, dtype=np.int16)
        for (s, sub) in regiones0.items():
            fig0 = sitio_a_root.get(s)
            if fig0 is None:
                continue
            for rr, cc in sub:
                self._region_de_celda[rr, cc] = fig0.raiz_id

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

        # 5b. ULTIMA GENERACION: celdas libres cuyo centro no coincide con el
        # punto de interes (centroide) de ninguna figura de la triangulacion
        # recursiva. Cada una se cuelga de la figura contenedora mas profunda
        # de su region como una hoja de generacion `padre.generacion + 1`, sin
        # triangulacion: solo se convierten ellas mismas en puntos de interes
        # con las mismas funciones de interes que las demas celdas (hereda el
        # coste acumulado por la rama y queda subyugada al mismo gate de
        # elegibilidad de su padre).
        def _contenedora_mas_profunda(fig, x, y):
            """Desciende por el arbol hasta la figura mas profunda cuyo poligono
            contiene el punto (x, y); devuelve None si la raiz no lo contiene."""
            if not _punto_en_poligono(x, y, fig.vertices):
                return None
            for h in fig.hijos:
                sub = _contenedora_mas_profunda(h, x, y)
                if sub is not None:
                    return sub
            return fig

        cubiertas = set(self._figuras_por_celda.keys())
        for (rr, cc) in celdas_libres:
            if (rr, cc) in cubiertas:
                continue
            propietario_sitio = None
            for (s, sub) in regiones0.items():
                if (rr, cc) in sub:
                    propietario_sitio = s
                    break
            if propietario_sitio is None:
                continue
            raiz = sitio_a_root.get(propietario_sitio)
            if raiz is None:
                continue
            cx = cc * gs + gs / 2.0
            cy = rr * gs + gs / 2.0
            padre = _contenedora_mas_profunda(raiz, cx, cy)
            if padre is None:
                continue
            id_counter[0] += 1
            fig_final = Figura(
                fig_id=id_counter[0],
                generacion=padre.generacion + 1,
                vertices=((cx - gs / 2, cy - gs / 2),
                          (cx + gs / 2, cy - gs / 2),
                          (cx + gs / 2, cy + gs / 2),
                          (cx - gs / 2, cy + gs / 2)),
                centroide_xy=(cx, cy),
                celda_rc=(rr, cc),
                padre=padre,
                raiz_id=padre.raiz_id
            )
            fig_final.huecos = []
            padre.hijos.append(fig_final)
            self._todas_figuras.append(fig_final)
            self._figuras_por_id[fig_final.id] = fig_final
            self._figuras_por_celda.setdefault((rr, cc), []).append(fig_final)
            triangulos_viz.append((fig_final.vertices, fig_final.generacion,
                                   fig_final.raiz_id))

        # Inicializar interes de inspeccion (interes > 0 siempre).
        # `coste_acumulado` empieza en 1.0 para TODAS las figuras y solo cambia
        # por el resultado de la inspeccion de sus ancestros en la misma rama:
        # `_aplicar_resultados` lo multiplica por `inspeccion_factor_defectivo`
        # (defectuosa, sube) o `inspeccion_factor_sano` (sana, baja) y propaga
        # el factor a TODA la descendencia, acumulandose generacion a generacion.
        # El descuento por generacion NO se aplica aqui: se hace en el
        # denominador de `_interes_total` como `(generacion + 1)`.
        pausa = self.cfg.inspeccion_pausa_s
        for f in self._todas_figuras:
            f.coste_individual = pausa
            f.coste_acumulado = 1.0

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
                "peso": 1.0,
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
            "figuras_gen0": [{
                "indice": i,
                "raiz_id": r0.raiz_id,
                "centroide": r0.celda,
                "centroide_xy": r0.centroide_xy,
                "vertices": list(r0.vertices),
                "huecos": [list(h) for h in r0.huecos],
            } for i, r0 in enumerate(self._raices)],
            "arbol": [self._figura_a_dict(r0) for r0 in self._raices],
        }

    def _aplicar_resultados(self, mundo):
        """Aplica los resultados de nuevas inspecciones:
        - Marca figura.inspeccion como 'sana' o 'defectuosa'.
        - Propaga el multiplicador a todos los descendientes recursivamente:
          * DEFECTUOSA: aumenta el interes de los descendientes (coste_acumulado *= factor_defectivo, p.ej. 2.0x).
          * SANA: reduce el interes de los descendientes (coste_acumulado *= factor_sano, p.ej. 0.5x).
          * El interes nunca es 0 ni negativo."""
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
        """Propaga recursivamente el factor multiplicativo a toda la descendencia, manteniendo el interes > 0."""
        for h in figura.hijos:
            h.coste_acumulado = max(1e-6, h.coste_acumulado * factor)
            self._propagar_factor_descendientes(h, factor)

    def generaciones_activas(self, mundo):
        """G_activa(V) para cada region raiz V: el minimo nivel de las tareas de
        V aun sin inspeccionar."""
        g = {}
        for t in self._tareas:
            if not mundo.inspec[t["r"], t["c"]]:
                ri = t["raiz_id"]
                if ri not in g or t["nivel"] < g[ri]:
                    g[ri] = t["nivel"]
        return g

    def tarea_elegible(self, t, generaciones, mundo):
        """True si la tarea es elegible para inspeccion.

        Con `inspeccion_padre_antes_hijo = True` (default): orden estricto
        padre->hijo por rama: una figura hijo se puede inspeccionar solo cuando
        su figura padre directa ya fue inspeccionada (o si es Gen 0).
        Con `= False`: sin gate de nivel ni padre: toda tarea pendiente (sin
        inspeccionar y sin estar en curso) es elegible y las ramas avanzan sin
        esperar a su padre."""
        if not self.cfg.inspeccion_padre_antes_hijo:
            return True
        if t["figura"].padre is not None:
            pr, pc = t["figura"].padre.celda
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
        """Tiempo de viaje A* (ruta suavizada y consciente del footprint) desde
        el robot hasta (x, y): longitud de la polilinea / max_speed."""
        ruta = encontrar_ruta(mundo, rb.x, rb.y, x, y)
        if not ruta:
            d = math.hypot(rb.x - x, rb.y - y)
            return d / max(self.cfg.robot_max_speed_px_s, 1e-6)
        dist = 0.0
        aux = ruta[0]
        for i in range(1, len(ruta)):
            dist += math.hypot(ruta[i][0] - aux[0], ruta[i][1] - aux[1])
            aux = ruta[i]
        return dist / max(self.cfg.robot_max_speed_px_s, 1e-6)

    def _robots_en_region(self, mundo, raiz_id):
        """Cuantos robots estan DENTRO de la region Gen0 `raiz_id` en este
        instante (segun su celda actual). Se usa como termino anti-hacinamiento
        del interes: a mas robots compartiendo region, menos interes tiene
        mandar aun mas."""

        reg = self._region_de_celda
        if reg is None or not mundo.robots:
            return 0
        gs = self.cfg.grid_size
        n = 0
        for rb in mundo.robots:
            r = int(rb.y // gs)
            c = int(rb.x // gs)
            if 0 <= r < reg.shape[0] and 0 <= c < reg.shape[1] and reg[r, c] == raiz_id:
                n += 1
        return n

    def _interes_total(self, tarea, tau_viaje, mundo=None):
        """Interes de una tarea:

        `interes = interes_inspeccion / ((nivel + 1) * tau_viaje^alpha * 2^R)`

        * `interes_inspeccion` (`coste_acumulado`): parte de 1 y cambia por el
          resultado de los ancestros inspeccionados en su rama (sana lo baja,
          defectuosa lo sube, acumulando el factor).
        * `(nivel + 1)`: cuota decremental por generacion (evita 0 en el
          denominador): cada nivel del arbol reduce el interes de base.
        * `tau_viaje^alpha` con `alpha = inspeccion_peso_distancia`: la
          distancia (tiempo de viaje A*) pesa debilmente (exponente = alpha).
        * `2^R`: termino dinamico anti-hacinamiento exponencial, calculado en
          cada evaluacion: `R` = numero de robots dentro de la region Gen0 de
          la tarea en ese momento. 1 robot divida entre 2, 2 robots entre 4,
          3 entre 8, etc.; con 0 robots es 1 (sin penalizacion).
        El valor es siempre estrictamente positivo (nunca 0 ni negativo)."""
        alpha = max(0.0, float(self.cfg.inspeccion_peso_distancia))
        tau = max(1e-6, tau_viaje)
        interes_insp = max(1e-6, tarea["coste_acumulado"])
        r_region = self._robots_en_region(mundo, tarea["raiz_id"]) if mundo is not None else 0
        return interes_insp / ((tarea["nivel"] + 1) * (tau ** alpha) * (2 ** r_region))

    def _costo_mercado(self, mundo, rb, tarea):
        """Interes exacto de la tarea para el robot `rb`.

        Sin cola previa: cada ciclo de subasta asigna UNA tarea por robot libre
        y el resto permanece libre para otros robots."""
        tau_viaje = self._tiempo_viaje(mundo, rb, tarea["x"], tarea["y"])
        return self._interes_total(tarea, tau_viaje, mundo)

    def _preseleccionar(self, rb, pendientes, asignadas, mundo):
        """Preselecciona las candidatas con mayor interes para agilizar A*,
        excluyendo las celdas ya adjudicadas en este ciclo."""
        gs = self.cfg.grid_size
        vel = max(float(self.cfg.robot_max_speed_px_s), 1e-6)

        def est(t):
            d = math.hypot(rb.x - (t["c"] * gs + gs / 2),
                           rb.y - (t["r"] * gs + gs / 2))
            return self._interes_total(t, d / vel, mundo)

        candidas = [t for t in pendientes
                    if (t["r"], t["c"]) not in asignadas]
        return heapq.nlargest(self._K_CANDIDATOS, candidas, key=est)

    def _pendientes_actuales(self, mundo):
        """Puntos de interes que siguen pendientes de inspeccion: sin inspeccionar,
        sin estar como objetivo en curso de un robot y elegibles segun el flag
        `inspeccion_padre_antes_hijo`."""
        gs = self.cfg.grid_size
        if not self._inicializado:
            self._construir_estatico(mundo)
            self._inicializado = True
        en_curso = self._reservadas_de_robots(mundo, gs)
        generaciones = self.generaciones_activas(mundo)
        return [t for t in self._tareas
                if (t["r"], t["c"]) not in en_curso
                and not mundo.inspec[t["r"], t["c"]]
                and self.tarea_elegible(t, generaciones, mundo)]

    def puntos_restantes(self, mundo):
        """Puntos de interes que quedan por tratar: pendientes de inspeccion o ya
        en curso de un robot (goal activo o en cola). Devuelve 0 cuando se
        agotaron todos los puntos de interes de la inspeccion."""
        gs = self.cfg.grid_size
        if not self._inicializado:
            self._construir_estatico(mundo)
            self._inicializado = True
        en_curso = self._reservadas_de_robots(mundo, gs)
        return len(self._pendientes_actuales(mundo)) + len(en_curso)

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

        # 3) Filtrar puntos de interes pendientes (no inspeccionados, no en
        #    curso y elegibles segun `inspeccion_padre_antes_hijo`)
        pendientes = self._pendientes_actuales(mundo)
        if not pendientes:
            self._reservadas = set()
            return

        subastables = [t for t in pendientes if not self._tarea_fallida(t)]
        if not subastables:
            subastables = pendientes

        # 4) Subasta mercantil para robots libres (gana la tarea con MAYOR
        #    interes). Una tarea por robot por ciclo: el robot toma la primera
        #    prioridad de su lista sin importar el resto; las demas opciones
        #    quedan libres para otros robots y se re-subastan al terminar.
        libres = [i for i, rb in enumerate(mundo.robots)
                  if rb.necesita_objetivo and not rb.cola_tareas]
        if not libres:
            self._reservadas = set()
            return

        asignaciones = {}
        asignadas = set()
        while libres:
            mejor_par = None
            mejor_interes = -1.0
            mejor_rid = -1
            for idx in libres:
                rb = mundo.robots[idx]
                for t in self._preseleccionar(rb, subastables, asignadas, mundo):
                    interes = self._costo_mercado(mundo, rb, t)
                    if (interes > mejor_interes + 1e-9
                            or (abs(interes - mejor_interes) <= 1e-9 and t["rid"] < mejor_rid)):
                        mejor_interes = interes
                        mejor_rid = t["rid"]
                        mejor_par = (idx, t)
            if mejor_par is None:
                break
            idx, t = mejor_par
            t["costo"] = self._tiempo_viaje(mundo, mundo.robots[idx], t["x"], t["y"]) + self.cfg.inspeccion_pausa_s
            asignaciones.setdefault(idx, []).append(t)
            asignadas.add((t["r"], t["c"]))
            libres.remove(idx)

        # 5) Asignar tareas a la cola de cada robot ganador (ordenadas de MAYOR
        #    a menor interes; una unica tarea por robot en este ciclo)
        pausa = self.cfg.inspeccion_pausa_s
        for idx, lista in asignaciones.items():
            lista.sort(key=lambda t: (-self._interes_total(t, max(1e-6, t["costo"] - pausa), mundo),
                                      t["r"], t["c"]))
            rb = mundo.robots[idx]
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
