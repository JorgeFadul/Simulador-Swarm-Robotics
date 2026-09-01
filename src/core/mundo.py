import math
import numpy as np
from .config import Config


def _bresenham(r0, c0, r1, c1):
    """Genera celdas entre (r0,c0) y (r1,c1) exclusivo, en orden."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c1 > c0 else -1
    sr = 1 if r1 > r0 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
        if c == c1 and r == r1:
            return
        yield r, c


class Pared:
    def __init__(self, x, y, width, height, angle=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = math.radians(angle)
        self.cache_rect()

    def cache_rect(self):
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        hw, hh = self.width / 2, self.height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        self.corners = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            self.corners.append((rx, ry))


SCOUTED = 1
VISITED = 2
DISCOVERED = 3

# Estado de inspeccion: 0 = no inspeccionada, 1 = inspeccionada
NO_INSPECCIONADA = 0
INSPECCIONADA = 1

# Resultado de la inspeccion por celda: -1 = desconocido, 0 = sana, 1 = defectuosa
RESULTADO_DESCONOCIDO = -1
SANA = 0
DEFECTUOSA = 1


class Mundo:
    def __init__(self, cfg: Config, rng=None):
        self.cfg = cfg
        self.rows = cfg.rows
        self.cols = cfg.cols
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.explored = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.robots = []
        self.paredes = []
        self.width_px = cfg.map_width
        self.height_px = cfg.map_height
        self.exploration_threshold = cfg.exploration_threshold

        self._crear_paredes_perimetrales()
        self._cargar_paredes(cfg.walls)

        self.total_libres = int(np.sum(self.grid == 0))
        self.total_paredes = int(np.sum(self.grid == 1))
        self._scouteadas = 0
        self._visitadas = 0
        self._paredes_vistas = 0

        # Inspeccion de baldosas: estado y resultado por celda (curato ground truth).
        self.inspec = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.resultado = np.full((self.rows, self.cols), RESULTADO_DESCONOCIDO,
                                 dtype=np.int8)
        self._inicializar_defectos(rng)
        self._inspeccionadas = 0
        self._defectuosas = 0
        self.mapa_inicial = self.grabar_mapa_inicial()
        # Etapa global del enjambre: "exploracion" o "inspeccion" (Fase 1).
        # La autoridad es Simulacion; se refleja aqui para que los robots y
        # planificadores lean el estado de dos etapas.
        self.etapa = "exploracion"

    def cambiar_a_inspeccion(self):
        """Transicion global a la etapa de Inspeccion (estrategias 2E)."""
        self.etapa = "inspeccion"

    def _inicializar_defectos(self, rng):
        """Genera el ground truth de baldosas defectuosas sobre celdas libres.

        Dos modos:
          * Si `cfg.defecto_blobs` trae una lista de blobs {cx, cy, radio}
            (mapas de prueba generados algoritmicamente), se crean manchas
            irregulares de defectos con esas semillas (no se solapan con
            paredes ni entre si) y se ignoran `prob_defecto`.
          * Si no hay blobs, se usa la distribucion uniforme con
            `prob_defecto` (Fase 0).

        El rng llega de la simulacion (sembrado por --semilla). En el modo
        uniforme la semilla es local para reproducir la misma distribucion en
        corridas repetidas. En el modo blobs cada blob usa su propia semilla
        determinista de la definicion del mapa (reproducible sin importar la
        semilla de la simulacion). Los bordes y paredes nunca son
        inspeccionables.
        """
        self.defectos = np.zeros((self.rows, self.cols), dtype=np.int8)
        blobs = getattr(self.cfg, "defecto_blobs", None) or []
        if blobs:
            self._inicializar_defectos_blobs(blobs)
        else:
            prob = self.cfg.prob_defecto
            if rng is None:
                rng = np.random.default_rng(0)
            # Semilla local derivada del rng base para no consumir su secuencia
            semilla_local = int(rng.integers(0, 2 ** 31))
            local = np.random.default_rng(semilla_local)
            defectos = local.random((self.rows, self.cols))
            mask = (self.grid == 0) & (defectos < prob)
            self.defectos[mask] = 1
        self._defectuosas_gt = int(self.defectos.sum())

    def _inicializar_defectos_blobs(self, blobs):
        """Crea manchas irregulares de defectos a partir de blobs semilla.

        Cada blob se define con {cx, cy, radio} en px sobre el mapa. A partir
        de la celda central se "crece" una mancha con un paseo aleatorio sesgado
        radialmente, generando bordes irregulares (ver el algoritmo documentado
        en CHANGELOG_TESIS). Las celdas de pared y los bordes del mapa nunca
        se marcan, y cada blob usa una semilla determinista para ser
        reproducible.
        """
        gs = self.cfg.grid_size
        ultima = None
        for i, b in enumerate(blobs):
            cx_px, cy_px = float(b["cx"]), float(b["cy"])
            radio = max(int(b.get("radio", 0)), 1)
            semilla = int(b.get("semilla", 1000 + i))
            rngb = np.random.default_rng(semilla)
            rc = int(cy_px // gs)
            cc = int(cx_px // gs)
            if not (0 <= rc < self.rows and 0 <= cc < self.cols):
                continue
            # Centro llamado a ser defectuoso solo si es celda libre
            if self.grid[rc, cc] == 0:
                self.defectos[rc, cc] = 1
            # Crecimiento radial con ruido (caminata aleatoria amortiguada)
            for _ in range(radio * gs):
                # Cantidad de pasos proporcional al area esperada del blob
                nr = rc + int(rngb.integers(-1, 2))
                nc = cc + int(rngb.integers(-1, 2))
                # Sesgo radial: probar candidatos cercanos al centro del blob
                dr = nr - rc
                dc = nc - cc
                if (dr * dr + dc * dc) > radio * radio:
                    continue
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr, nc] != 0:
                    continue
                # Evita solape con blobs previos: no marcar si ya es defecto de
                # otro blob (aunque se permita adyacencia).
                self.defectos[nr, nc] = 1
            ultima = (rc, cc)

    def inspeccionar_celda(self, x, y, robot_id=None):
        """Marca una celda como inspeccionada y registra su resultado binario.

        Devuelve el resultado (SANA/DEFECTUOSA) si la inspeccion se realizo, o
        None si la celda no es inspeccionable (pared, borde, fuera de rango) o ya
        estaba inspeccionada. Es el punto de extension donde cada estrategia
        decide cuando inspeccionar.

        La deteccion es 100 % efectiva en esta fase (sigma de ruido = 0). El
        metodo `_resultado_con_ruido` es la costura preparada para inyectar
        ruido gaussiano en la probabilidad de deteccion en fases posteriores.
        """
        gs = self.cfg.grid_size
        c = int(x // gs)
        r = int(y // gs)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return None
        if self.grid[r, c] != 0:
            return None
        if self.inspec[r, c]:
            return None
        self.inspec[r, c] = INSPECCIONADA
        self._inspeccionadas += 1
        res = self._resultado_con_ruido(r, c)
        if res:
            self.resultado[r, c] = DEFECTUOSA
            self._defectuosas += 1
        else:
            self.resultado[r, c] = SANA
        return int(self.resultado[r, c])

    def _resultado_con_ruido(self, r, c):
        """Resultado de la percusion sobre la celda (r, c).

        Con `ruido_inspeccion_sigma == 0` devuelve el ground truth exacto
        (deteccion perfecta). Para sigma>0 se corrompe la deteccion con ruido
        gaussiano sobre una probabilidad de deteccion (a implementar en fases
        de robustez); en esta fase la estructura queda preparada pero sin
        ruido.
        """
        sigma = getattr(self.cfg, "ruido_inspeccion_sigma", 0.0)
        gt = int(self.defectos[r, c])
        if sigma <= 0.0:
            return gt
        # Costura para ruido futuro: pzelda defectuosa = gt; se corrompe la
        # decision con gaussiana. (No activo en Fase 1.)
        rng = np.random.default_rng()
        ruido = rng.normal(0.0, sigma)
        p_detect = gt + ruido
        return 1 if p_detect >= 0.5 else 0

    def estado_celdas(self):
        """Estado por celda del mapa con coordenadas (col, fila) y, para celdas
        libres, la clase de exploracion y el estado/resultado de inspeccion.

        Devuelve una lista de dicts: {x, y, fila, col, tipo, vista, visitada,
        inspeccionada, resultado}. "tipo" es 0=libre, 1=pared. Solo se listan
        celdas con algun estado relevante (libres vistas/visitadas/inspeccionadas
        o paredes descubiertas).
        """
        gs = self.cfg.grid_size
        filas = []
        for r in range(self.rows):
            for c in range(self.cols):
                libre = self.grid[r, c] == 0
                exp = int(self.explored[r, c])
                es_vista = libre and exp in (SCOUTED, VISITED)
                es_visitada = libre and exp == VISITED
                it = bool(self.inspec[r, c])
                res = int(self.resultado[r, c])
                if libre:
                    if not (es_vista or es_visitada or it):
                        continue
                else:
                    if exp != DISCOVERED:
                        continue
                filas.append({
                    "x": round(c * gs + gs / 2.0, 1),
                    "y": round(r * gs + gs / 2.0, 1),
                    "fila": r,
                    "col": c,
                    "tipo": 0 if libre else 1,
                    "vista": bool(es_vista),
                    "visitada": bool(es_visitada),
                    "inspeccionada": bool(it),
                    "resultado": res if it else RESULTADO_DESCONOCIDO,
                })
        return filas

    @property
    def total_libres_inspeccionables(self):
        return self.total_libres

    def grilla_ocupacion(self):
        """Devuelve la grilla inicial (0=libre, 1=pared) como lista de filas,
        para poder reconstruir el mapa exacto del escenario."""
        return self.grid.tolist()

    def snapshot_escenario(self):
        """Datos del area original de inspeccion (para reproducir el escenario):
        dimensiones, grilla de ocupacion, paredes y ground truth de defectos."""
        return {
            "map_width": self.width_px,
            "map_height": self.height_px,
            "grid_size": self.cfg.grid_size,
            "filas": self.rows,
            "columnas": self.cols,
            "total_celdas": self.rows * self.cols,
            "celdas_libres": self.total_libres,
            "celdas_pared": self.total_paredes,
            "grilla_ocupacion": self.grilla_ocupacion(),
            "paredes": [{"x": p.x, "y": p.y, "width": p.width, "height": p.height,
                         "angle": math.degrees(p.angle)} for p in self.paredes],
            "defectos_gt": self.defectos.tolist(),
            "prob_defecto": self.cfg.prob_defecto,
        }

    def grabar_mapa_inicial(self):
        """Almacena una copia del estado inicial del escenario (ocupacion y
        ground truth de defectos) tras crear paredes, para poder reproducirlo."""
        self._mapa_inicial = {
            "map_width": self.width_px,
            "map_height": self.height_px,
            "grid_size": self.cfg.grid_size,
            "grilla_ocupacion": self.grilla_ocupacion(),
            "paredes": [{"x": p.x, "y": p.y, "width": p.width, "height": p.height,
                         "angle": math.degrees(p.angle)} for p in self.paredes],
            "defectos_gt": self.defectos.tolist(),
        }
        return self._mapa_inicial

    def marcar_visitado(self, x, y):
        gs = self.cfg.grid_size
        c = int(x // gs)
        r = int(y // gs)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        actual = self.explored[r, c]
        if actual == SCOUTED:
            self.explored[r, c] = VISITED
            self._scouteadas -= 1
            self._visitadas += 1
            return True
        elif actual == 0 and self.grid[r, c] == 0:
            self.explored[r, c] = VISITED
            self._visitadas += 1
            return True
        return False

    def visitar_area(self, x, y, radio):
        """Marca como VISITED todas las celdas libres dentro de `radio` px del
        robot, y devuelve cuantas eran nuevas.

        Mejora de la rasterizacion de visita: el robot tiene extension fisica
        (radio), de modo que al pasar al lado de una pared su cuerpo cubre
        parcialmente la casilla libre adyacente. Marcar solo la celda del
        centro dejaba "huecos" de casillas sin visitar pegadas a los bordes de
        las paredes (el centro nunca caia sobre ellas). Al rasterizar el area
        cubierta por el robot, esas casillas limítrofes quedan marcadas y no
        aparecen casillas vacias junto a las paredes.
        """
        gs = self.cfg.grid_size
        min_c = max(0, int((x - radio) // gs))
        max_c = min(self.cols - 1, int((x + radio) // gs))
        min_r = max(0, int((y - radio) // gs))
        max_r = min(self.rows - 1, int((y + radio) // gs))
        nuevas = 0
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.explored[r, c] != 0 or self.grid[r, c] != 0:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if math.hypot(cx - x, cy - y) <= radio:
                    self.explored[r, c] = VISITED
                    self._visitadas += 1
                    nuevas += 1
        return nuevas

    def explorar_desde(self, x, y, radius):
        gs = self.cfg.grid_size
        min_c = max(0, int((x - radius) // gs))
        max_c = min(self.cols - 1, int((x + radius) // gs))
        min_r = max(0, int((y - radius) // gs))
        max_r = min(self.rows - 1, int((y + radius) // gs))
        scouted_ahora = 0
        paredes_ahora = 0
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.explored[r, c]:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if (math.hypot(cx - x, cy - y) < radius and
                        self.tiene_linea_de_vista(x, y, cx, cy)):
                    if self.grid[r, c] == 0:
                        self.explored[r, c] = SCOUTED
                        self._scouteadas += 1
                        scouted_ahora += 1
                    else:
                        self.explored[r, c] = DISCOVERED
                        self._paredes_vistas += 1
                        paredes_ahora += 1
        return scouted_ahora, paredes_ahora

    def reiniciar_exploracion(self):
        self.explored.fill(0)
        self._scouteadas = 0
        self._visitadas = 0
        self._paredes_vistas = 0
        self.inspec.fill(0)
        self.resultado.fill(RESULTADO_DESCONOCIDO)
        self._inspeccionadas = 0
        self._defectuosas = 0
        self.total_libres = int(np.sum(self.grid == 0))
        self.total_paredes = int(np.sum(self.grid == 1))

    def exploracion_completa(self):
        if self.total_libres == 0 or self.total_paredes == 0:
            return False
        libres_vistas = self._scouteadas + self._visitadas
        return (libres_vistas / self.total_libres >= self.exploration_threshold and
                self._paredes_vistas / self.total_paredes >= self.exploration_threshold)

    @property
    def porcentaje_descubierto(self):
        """Fraccion del mapa (libres + paredes) ya descubierta por el enjambre."""
        total = self.rows * self.cols
        if total == 0:
            return 0.0
        return (self._scouteadas + self._visitadas + self._paredes_vistas) / total

    @property
    def porcentaje_inspeccionado(self):
        """Fraccion de celdas libres inspeccionables ya inspeccionadas."""
        if self.total_libres == 0:
            return 0.0
        return self._inspeccionadas / self.total_libres

    def inspeccion_completa(self):
        """True cuando todas las celdas libres inspeccionables fueron inspeccionadas."""
        return self._inspeccionadas >= self.total_libres and self.total_libres > 0

    def _crear_paredes_perimetrales(self):
        gs = self.cfg.grid_size
        w, h = self.width_px, self.height_px
        t = gs
        perimetro = [
            (w / 2, t / 2, w, t, 0.0),
            (w / 2, h - t / 2, w, t, 0.0),
            (t / 2, h / 2, t, h, 0.0),
            (w - t / 2, h / 2, t, h, 0.0),
        ]
        for args in perimetro:
            p = Pared(*args)
            self.paredes.append(p)
            self._rasterizar_pared(p)

    def _cargar_paredes(self, wall_list: list[dict]):
        for w in wall_list:
            p = Pared(w["x"], w["y"], w["width"], w["height"], w.get("angle", 0))
            self.paredes.append(p)
            self._rasterizar_pared(p)

    def _rasterizar_pared(self, p: Pared):
        gs = self.cfg.grid_size
        xs = [int(c[0] // gs) for c in p.corners]
        ys = [int(c[1] // gs) for c in p.corners]
        min_c = max(0, min(xs))
        max_c = min(self.cols - 1, max(xs))
        min_r = max(0, min(ys))
        max_r = min(self.rows - 1, max(ys))

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if self._punto_en_pared(cx, cy, p):
                    self.grid[r, c] = 1

    def _punto_en_pared(self, px, py, p: Pared):
        dx = px - p.x
        dy = py - p.y
        cos_a = math.cos(-p.angle)
        sin_a = math.sin(-p.angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        return abs(lx) <= p.width / 2 and abs(ly) <= p.height / 2

    def tiene_linea_de_vista(self, x1, y1, x2, y2):
        """True si no hay ninguna pared entre (x1,y1) y (x2,y2)."""
        gs = self.cfg.grid_size
        r1, c1 = int(y1 // gs), int(x1 // gs)
        r2, c2 = int(y2 // gs), int(x2 // gs)
        if r1 == r2 and c1 == c2:
            return True
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols and
                0 <= r2 < self.rows and 0 <= c2 < self.cols):
            return False
        for r, c in _bresenham(r1, c1, r2, c2):
            if self.grid[r, c] == 1:
                return False
        return True

    def es_libre(self, x, y):
        c = int(x // self.cfg.grid_size)
        r = int(y // self.cfg.grid_size)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return self.grid[r, c] == 0

    def circulo_en_libre(self, x, y, radius):
        if not self.es_libre(x, y):
            return False
        gs = self.cfg.grid_size
        min_c = max(0, int((x - radius) // gs))
        max_c = min(self.cols - 1, int((x + radius) // gs))
        min_r = max(0, int((y - radius) // gs))
        max_r = min(self.rows - 1, int((y + radius) // gs))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.grid[r, c] == 0:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if math.hypot(cx - x, cy - y) < radius:
                    return False
        return True

    def _circulo_sin_descubiertas(self, x, y, radius):
        """True si el circulo en (x,y) no solapa ninguna pared DISCOVERED."""
        gs = self.cfg.grid_size
        min_c = max(0, int((x - radius) // gs))
        max_c = min(self.cols - 1, int((x + radius) // gs))
        min_r = max(0, int((y - radius) // gs))
        max_r = min(self.rows - 1, int((y + radius) // gs))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.grid[r, c] == 0 or self.explored[r, c] != DISCOVERED:
                    continue
                cx = c * gs + gs / 2
                cy = r * gs + gs / 2
                if math.hypot(cx - x, cy - y) < radius:
                    return False
        return True

    def descubrir_pared_frontal(self, x, y, angle, radius):
        """Marca como DISCOVERED las paredes en la direccion frontal tras un choque."""
        gs = self.cfg.grid_size
        look_ahead = radius * 3
        fx = x + math.cos(angle) * look_ahead
        fy = y + math.sin(angle) * look_ahead
        min_c = max(0, int((min(x, fx)) // gs) - 1)
        max_c = min(self.cols - 1, int((max(x, fx)) // gs) + 1)
        min_r = max(0, int((min(y, fy)) // gs) - 1)
        max_r = min(self.rows - 1, int((max(y, fy)) // gs) + 1)
        count = 0
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.grid[r, c] == 1 and self.explored[r, c] != DISCOVERED:
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    if self.tiene_linea_de_vista(x, y, cx, cy):
                        self.explored[r, c] = DISCOVERED
                        self._paredes_vistas += 1
                        count += 1
        return count

    def detectar_en_rango(self, x, y, radius, incluir_paredes=True):
        """Devuelve vector de alejamiento promedio de obstaculos dentro del radio.
        Retorna (dx, dy) normalizado, o (0, 0) si no hay obstaculos.
        Si incluir_paredes=False, solo considera a otros robots."""
        gs = self.cfg.grid_size
        min_c = max(0, int((x - radius) // gs))
        max_c = min(self.cols - 1, int((x + radius) // gs))
        min_r = max(0, int((y - radius) // gs))
        max_r = min(self.rows - 1, int((y + radius) // gs))

        rep_dx = 0.0
        rep_dy = 0.0
        count = 0

        # Paredes (solo con linea de vista)
        if incluir_paredes:
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if self.grid[r, c] == 0:
                        continue
                    cx = c * gs + gs / 2
                    cy = r * gs + gs / 2
                    dx = x - cx
                    dy = y - cy
                    dist = math.hypot(dx, dy)
                    if dist < radius and dist > 0.1 and self.tiene_linea_de_vista(x, y, cx, cy):
                        rep_dx += dx / dist
                        rep_dy += dy / dist
                        count += 1

        # Otros robots
        for robot in self.robots:
            dx = x - robot.x
            dy = y - robot.y
            dist = math.hypot(dx, dy)
            min_dist = radius + robot.radius
            if dist < min_dist and dist > 0.1:
                rep_dx += dx / dist
                rep_dy += dy / dist
                count += 1

        if count == 0:
            return 0.0, 0.0, False

        mag = math.hypot(rep_dx, rep_dy)
        if mag > 0:
            rep_dx /= mag
            rep_dy /= mag

        return rep_dx, rep_dy, True

    def colision_frontal(self, x, y, vx, vy, radius, robots, solo_descubiertas=False):
        """Chequea si el movimiento (vx, vy) hace chocar el frente del robot.
        El frente es el hemisferio en la direccion del vector velocidad.
        Si solo_descubiertas=True, solo considera paredes con explored==DISCOVERED."""
        if vx == 0 and vy == 0:
            return False

        speed = math.hypot(vx, vy)
        nx = x + vx
        ny = y + vy

        # Paredes: chequear punto destino
        if solo_descubiertas:
            if not self._circulo_sin_descubiertas(nx, ny, radius):
                return True
        else:
            if not self.circulo_en_libre(nx, ny, radius):
                return True

        # Otros robots: solo si vienen de frente (angulo < 90 grados)
        for r in robots:
            dx = r.x - x
            dy = r.y - y
            dist = math.hypot(dx, dy)
            if dist < radius + r.radius and dist > 0.1:
                # Vector hacia el otro robot
                to_other_x = dx / dist
                to_other_y = dy / dist
                # Vector velocidad normalizado
                vel_x = vx / speed
                vel_y = vy / speed
                # Producto escalar > 0 significa que el otro robot esta al frente
                if vel_x * to_other_x + vel_y * to_other_y > 0:
                    return True

        return False

    def posicion_libre_aleatoria(self, rng, radius):
        margin = radius + 5
        for _ in range(200):
            x = rng.uniform(margin, self.width_px - margin)
            y = rng.uniform(margin, self.height_px - margin)
            if not self.circulo_en_libre(x, y, radius):
                continue
            ok = True
            for r in self.robots:
                if math.hypot(x - r.x, y - r.y) < radius + r.radius + 5:
                    ok = False
                    break
            if ok:
                return x, y
        return None, None

    def es_borde(self, r, c):
        return r <= 0 or r >= self.rows - 1 or c <= 0 or c >= self.cols - 1

    def colocar_pared_celda(self, r, c):
        if self.es_borde(r, c):
            return False
        self.grid[r, c] = 1
        self.total_libres = int(np.sum(self.grid == 0))
        self.total_paredes = int(np.sum(self.grid == 1))
        return True

    def borrar_celda(self, r, c):
        if self.es_borde(r, c):
            return False
        self.grid[r, c] = 0
        self.explored[r, c] = 0
        self.total_libres = int(np.sum(self.grid == 0))
        self.total_paredes = int(np.sum(self.grid == 1))
        return True

    def limpiar(self):
        self.grid.fill(0)
        self.paredes.clear()
        self.robots.clear()
        self._crear_paredes_perimetrales()
        self.reiniciar_exploracion()

    def update(self, dt: float):
        for robot in self.robots:
            robot.update(dt, self)