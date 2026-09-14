"""Planificacion de movimiento (motion planning) sobre el grid.

A* ponderado: las celdas libres ya exploradas tienen costo bajo y las celdas
libres desconocidas costo alto (se pueden cruzar de forma optimista, como en
exploracion por fronteras, pero se prefiere el camino ya conocido). Las
paredes reales no son transitables.

La ruta es CONSCIENTE DEL FOOTPRINT del robot: cada arista entre celdas solo
se acepta si el disco del robot centrado en el punto medio de la arista no
toca ninguna pared, de modo que el planificador nunca envia al robot por
pasillos donde fisicamente no cabe (el antiguo A* que solo miraba centros de
celda llegaba a rutear por huecos de 1 celda e hizo que los robots orbitaran
atascados en la entrada). Ademas los waypoints se suavizan: los tramos rectos
que quedan libres sustituyen a la poligonal de centro en centro, de modo que
el robot navega directo y sin oscilaciones de esquina en esquina.
"""

import heapq
import math


def _disco_libre(mundo, x, y, radio):
    """True si el disco del robot en (x, y) no solapa ninguna pared real."""
    return mundo.circulo_en_libre(x, y, radio)


def _construir_tabla_aristas(mundo):
    """Precomputa (una sola vez por mundo) que aristas entre celdas adyacentes
    son transitables por el disco del robot (el punto medio de la arista debe
    quedar libre). Asi el A* consulta cada arista en O(1) sin recalcular
    circulos de despeje miles de veces (era el costo dominante del footprint).

    direcciones: d=0 (+fila), 1 (-fila), 2 (+col), 3 (-col)."""
    import numpy as np
    rows, cols = mundo.rows, mundo.cols
    gs = mundo.cfg.grid_size
    radio = float(mundo.cfg.robot_radius_px)
    tabla = np.zeros((rows, cols, 4), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            for d, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if mundo.grid[r, c] != 0 or mundo.grid[nr, nc] != 0:
                    continue
                mx = ((c + nc) * gs + gs) / 2
                my = ((r + nr) * gs + gs) / 2
                tabla[r, c, d] = _disco_libre(mundo, mx, my, radio)
    return tabla


def _arista_libre(mundo, r0, c0, r1, c1):
    """True si el robot (disco de radio `radio`) puede pasar recto entre los
    centros de dos celdas adyacentes: el disco en el punto medio de la arista
    debe quedar libre. Esto bloquea pasillos de 1 celda de ancho y los giros
    que cortan esquinas (pared diagonal a menos de un radio)."""
    tabla = getattr(mundo, "_pf_tabla_arista", None)
    if tabla is None:
        tabla = _construir_tabla_aristas(mundo)
        mundo._pf_tabla_arista = tabla
        # la cache vieja por diccionario queda obsoleta
        for attr in ("_pf_arista",):
            if hasattr(mundo, attr):
                delattr(mundo, attr)
    if r1 > r0:
        d = 0
    elif r1 < r0:
        d = 1
    elif c1 > c0:
        d = 2
    else:
        d = 3
    return bool(tabla[r0, c0, d])


def _segmento_libre(mundo, ax, ay, bx, by, radio):
    """True si la poligonal recta (ax,ay)-(bx,by) es transitable por el robot:
    el disco de radio `radio` centrado en muestras a lo largo del segmento no
    toca paredes (aproximacion del barrido del disco, consistente con el
    modelo de colision en tiempo de ejecucion).

    El resultado se memoiza por par de celdas extremo: el suavizado greedy
    repite las mismas pruebas entre rutas (grid estatico), asi no se rescannea
    el mismo segmento decenas de miles de veces."""
    gs = mundo.cfg.grid_size
    a = (int(ay // gs), int(ax // gs))
    b = (int(by // gs), int(bx // gs))
    clave = a + b if a <= b else b + a
    cache = getattr(mundo, "_pf_segmento", None)
    if cache is None:
        cache = {}
        mundo._pf_segmento = cache
    if clave in cache:
        return cache[clave]
    libre = _segmento_libre_sin_cache(mundo, ax, ay, bx, by, radio)
    if len(cache) > 80000:
        cache.clear()
    cache[clave] = libre
    return libre


def _segmento_libre_sin_cache(mundo, ax, ay, bx, by, radio):
    d = math.hypot(bx - ax, by - ay)
    if d <= 1e-9:
        return _disco_libre(mundo, ax, ay, radio)
    n = max(1, int(d / max(radio * 0.5, 1e-6)))
    for i in range(n + 1):
        t = i / n
        if not _disco_libre(mundo, ax + (bx - ax) * t, ay + (by - ay) * t, radio):
            return False
    return True


def _suavizar(mundo, waypoints, radio):
    """Elimina waypoints intermedios cuando el tramo recto entre dos waypoints
    lejanos queda libre: produce rutas directas (menos giros y esquinas)."""
    if len(waypoints) <= 2:
        return waypoints
    suave = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        while j > i + 1 and not _segmento_libre(mundo,
                                                waypoints[i][0], waypoints[i][1],
                                                waypoints[j][0], waypoints[j][1],
                                                radio):
            j -= 1
        suave.append(waypoints[j])
        i = j
    return suave


def _a_estrella(mundo, r0, c0, r1, c1):
    rows, cols = mundo.rows, mundo.cols
    if mundo.grid[r0, c0] != 0 or mundo.grid[r1, c1] != 0:
        return None
    if (r0, c0) == (r1, c1):
        return [(r0, c0)]
    explore = mundo.explored
    open_set = [(0.0, 0.0, r0, c0)]
    g_cost = {(r0, c0): 0.0}
    came = {}

    def heur(r, c):
        return abs(r - r1) + abs(c - c1)

    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r, c) == (r1, c1):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]
                path.append((r, c))
            path.reverse()
            return path
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if mundo.grid[nr, nc] != 0:
                continue
            if not _arista_libre(mundo, r, c, nr, nc):
                continue
            costo = 1.0 if explore[nr, nc] != 0 else 6.0
            ng = g + costo
            if ng < g_cost.get((nr, nc), 1e18):
                g_cost[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (ng + heur(nr, nc), ng, nr, nc))
    return None


def encontrar_ruta(mundo, x1, y1, x2, y2):
    """Devuelve lista de waypoints (px) directos o lista vacia si no hay ruta.

    Las aristas del A* exigen despeje fisico del disco del robot (consciente
    del footprint) y la salida se suaviza en tramos rectos.

    Las rutas se cachean por (celda origen, celda meta) en el propio mundo: el
    grid de paredes es estatico durante la simulacion, por lo que una ruta
    calculada una vez es valida para siempre (las celdas libres desconocidas que
    el A* pueda preferir o no por `explored` siempre SON transitables). Esto
    evita recomputar el A* + suavizado decenas de miles de veces (subasta por
    candidato y re-planificaciones por choque), que era lo que hundia los FPS.
    Se devuelve una COPIA: los llamadores mutan la lista (pop de waypoints)."""
    gs = mundo.cfg.grid_size
    r0, c0 = int(y1 // gs), int(x1 // gs)
    r1, c1 = int(y2 // gs), int(x2 // gs)
    if not (0 <= r0 < mundo.rows and 0 <= c0 < mundo.cols and
            0 <= r1 < mundo.rows and 0 <= c1 < mundo.cols):
        return []
    clave = (r0, c0, r1, c1)
    rcache = getattr(mundo, "_pf_rutas", None)
    if rcache is None:
        rcache = {}
        mundo._pf_rutas = rcache
    if clave in rcache:
        v = rcache[clave]
        return [] if v is None else list(v)
    path = _a_estrella(mundo, r0, c0, r1, c1)
    if path is None:
        rcache[clave] = None
        return []
    radio = float(mundo.cfg.robot_radius_px)
    px = [(c * gs + gs / 2, r * gs + gs / 2) for r, c in path]
    res = _suavizar(mundo, px, radio)
    if len(rcache) > 40000:
        rcache.clear()
    rcache[clave] = res
    return list(res)
