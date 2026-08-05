"""Planificacion de movimiento (motion planning) sobre el grid.

A* ponderado: las celdas libres ya exploradas tienen costo bajo y las celdas
libres desconocidas costo alto (se pueden cruzar de forma optimista, como en
exploracion por fronteras, pero se prefiere el camino ya conocido). Las
paredes reales no son transitables.
"""

import heapq


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
            costo = 1.0 if explore[nr, nc] != 0 else 6.0
            ng = g + costo
            if ng < g_cost.get((nr, nc), 1e18):
                g_cost[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (ng + heur(nr, nc), ng, nr, nc))
    return None


def encontrar_ruta(mundo, x1, y1, x2, y2):
    """Devuelve lista de waypoints (px) o lista vacia si no hay ruta."""
    gs = mundo.cfg.grid_size
    r0, c0 = int(y1 // gs), int(x1 // gs)
    r1, c1 = int(y2 // gs), int(x2 // gs)
    if not (0 <= r0 < mundo.rows and 0 <= c0 < mundo.cols and
            0 <= r1 < mundo.rows and 0 <= c1 < mundo.cols):
        return []
    path = _a_estrella(mundo, r0, c0, r1, c1)
    if path is None:
        return []
    return [(c * gs + gs / 2, r * gs + gs / 2) for r, c in path]
