import colorsys
import math

import numpy as np
import pygame
from ..core.mundo import (Mundo, SCOUTED, VISITED, DISCOVERED,
                          SANA, DEFECTUOSA, INSPECCIONADA, RESULTADO_DESCONOCIDO)
from ..algorithms.exploration import VoronoiExploracion


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self._thumb_bg = None

    def draw(self, mundo: Mundo, estado="running", plan=None,
             plan_inspeccion=None,
             ver_fronteras=True, ver_objetivos=True, ver_intencion=True,
             ver_defectos=False, ver_voronoi=False, ver_triangulacion=False):
        self.screen.fill((20, 20, 30))
        self._draw_exploracion(mundo, estado)
        if ver_defectos:
            self._draw_defectos(mundo)
        if ver_fronteras and plan is not None and getattr(plan, "frontera_mask", None) is not None:
            self._draw_fronteras(mundo, plan.frontera_mask)
        if ver_voronoi:
            self._draw_capa_voronoi(mundo, plan, plan_inspeccion,
                                    ver_triangulacion)
        self._draw_grid(mundo)
        self._draw_walls(mundo)
        # Los resultados de inspeccion (probando/sana/defectuosa) se dibujan
        # despues de la capa rosada de ground truth para que queden por encima
        # del toggle de defectos.
        self._draw_inspeccion(mundo)
        if ver_objetivos:
            self._draw_goals(mundo)
        if ver_intencion:
            self._draw_intents(mundo)
        self._draw_robots(mundo)
        if mundo.robots:
            self._draw_robot_thumbnails(mundo)

    def _draw_inspeccion(self, mundo: Mundo):
        """Dibuja el resultado visible de la inspeccion de baldosas, que es el
        foco de la etapa de Inspeccion (2E):

        - amarillo: la celda que un robot esta probando AHORA (pausa de percusion).
        - azul:     celda inspeccionada cuyo resultado fue SANA (sin defecto).
        - rojo:     celda inspeccionada cuyo resultado fue DEFECTUOSA.

        A diferencia de la capa oculta de `_draw_defectos` (ground truth, solo
        con el toggle del operador), estas celdas se muestran siempre, porque el
        resultado de la percusion es conocido por los robots y no es "oculto".
        """
        gs = mundo.cfg.grid_size
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                color = None
                if mundo.inspec[r, c] == INSPECCIONADA:
                    color = (70, 130, 230) if mundo.resultado[r, c] == SANA else (230, 60, 40)
                rect = (c * gs, r * gs, gs, gs)
                if color is not None:
                    pygame.draw.rect(self.screen, color, rect)
        # Celda en prueba (amarillo) por encima del resto
        for robot in mundo.robots:
            if robot.inspeccionando and robot.celda_probando is not None:
                r, c = robot.celda_probando
                rect = (c * gs, r * gs, gs, gs)
                pygame.draw.rect(self.screen, (255, 210, 0), rect)

    def _draw_defectos(self, mundo: Mundo):
        """Dibuja la capa oculta de areas afectadas (ground truth) visible solo
        para el operador (toggle de la pestaña Controles).

        Las celdas defectuosas (ground truth) se pintan en rosado; se dibujan
        sobre las celdas ya exploradas para que se distingan del fondo.
        """
        gs = mundo.cfg.grid_size
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                if not mundo.defectos[r, c]:
                    continue
                color = (255, 120, 190)
                rect = (c * gs, r * gs, gs, gs)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_fronteras(self, mundo: Mundo, mask):
        gs = mundo.cfg.grid_size
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    pygame.draw.rect(self.screen, (0, 90, 110), (c * gs, r * gs, gs, gs))

    def _draw_goals(self, mundo: Mundo):
        color = (255, 80, 255)
        for robot in mundo.robots:
            if robot.goal_mode and robot.goal is not None and not robot.goal_alcanzado:
                gx, gy = int(robot.goal[0]), int(robot.goal[1])
                pygame.draw.circle(self.screen, color, (gx, gy), 5, 1)
                pygame.draw.line(self.screen, color,
                                 (int(robot.x), int(robot.y)), (gx, gy), 1)

    def _draw_exploracion(self, mundo: Mundo, estado="running"):
        gs = mundo.cfg.grid_size
        # En la etapa de Inspeccion se atenua el rastro de la exploracion para
        # que el foco visual quede en las pruebas de las baldosas (la cobertura
        # de scouteo ya cumplio su funcion).
        dim = 0.30 if mundo.etapa == "inspeccion" else 1.0
        rows, cols = mundo.rows, mundo.cols
        w, h = cols * gs, rows * gs
        # La capa de celdas se dibuja por lotes en una Surface cacheadada que se
        # reconstruye solo cuando cambia la exploracion/etapa (muy barato en
        # comparacion con un draw.rect por celda cada frame).
        firma = (mundo.etapa, estado, hash(mundo.explored.tobytes()))
        if not hasattr(self, "_explo_surf") or self._explo_firma != firma:
            self._explo_firma = firma
            arr = np.zeros((rows, cols, 3), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    val = mundo.explored[r, c]
                    if mundo.grid[r, c] == 0:
                        if val == 1:
                            arr[r, c] = (0, int(55 * dim), 0)
                        elif val == 2:
                            arr[r, c] = (0, int(110 * dim), 0)
                        else:
                            arr[r, c] = (20, 20, 30)
                    else:
                        if val == 3:
                            arr[r, c] = (110, 0, 0)
                        elif estado == "edicion":
                            arr[r, c] = (60, 40, 40)
                        else:
                            arr[r, c] = (30, 15, 15)
            full = np.kron(arr, np.ones((gs, gs, 1), dtype=np.uint8))
            self._explo_surf = pygame.surfarray.make_surface(full.transpose(1, 0, 2))
        if w != self._explo_surf.get_width():
            self._explo_surf = pygame.transform.scale(self._explo_surf, (w, h))
        self.screen.blit(self._explo_surf, (0, 0))

    def _draw_grid(self, mundo: Mundo):
        # Lineas de grilla como trazos (una linea por fila/columna) en lugar de
        # un rectangulo por celda: mucho mas barato y visualmente identico.
        gs = mundo.cfg.grid_size
        w, h = mundo.cols * gs, mundo.rows * gs
        color = (50, 50, 60)
        for c in range(1, mundo.cols):
            x = c * gs
            pygame.draw.line(self.screen, color, (x, 0), (x, h), 1)
        for r in range(1, mundo.rows):
            y = r * gs
            pygame.draw.line(self.screen, color, (0, y), (w, y), 1)

    def _draw_walls(self, mundo: Mundo):
        for p in mundo.paredes:
            if len(p.corners) == 4:
                pygame.draw.polygon(self.screen, (160, 70, 70), p.corners, 2)

    def _intent_hit_distancia(self, mundo: Mundo, robot, dist):
        """Distancia hasta la primera pared DISCOVERED en la direccion de la intencion."""
        gs = mundo.cfg.grid_size
        dx = math.cos(robot.target_angle)
        dy = math.sin(robot.target_angle)
        step = gs * 0.5
        n = int(dist / step)
        for i in range(1, n + 1):
            px = robot.x + dx * step * i
            py = robot.y + dy * step * i
            c = int(px // gs)
            r = int(py // gs)
            if r < 0 or r >= mundo.rows or c < 0 or c >= mundo.cols:
                return step * (i - 1)
            if mundo.grid[r, c] == 1 and mundo.explored[r, c] == DISCOVERED:
                return step * (i - 1)
        return dist

    def _draw_intents(self, mundo: Mundo):
        """Dibuja la ruta prevista de la intencion de movimiento de cada robot."""
        color = (0, 230, 230)
        for robot in mundo.robots:
            dist = robot.intent_distancia
            if dist <= 0:
                continue
            dist = min(dist, 260.0)
            dist = self._intent_hit_distancia(mundo, robot, dist)
            if dist < 6:
                continue

            x0, y0 = robot.x, robot.y
            dx = math.cos(robot.target_angle)
            dy = math.sin(robot.target_angle)
            tip_x = x0 + dx * dist
            tip_y = y0 + dy * dist

            dash_len, gap_len = 9, 6
            pos = 0.0
            while pos < dist:
                e = min(pos + dash_len, dist)
                sx = int(x0 + dx * pos)
                sy = int(y0 + dy * pos)
                ex = int(x0 + dx * e)
                ey = int(y0 + dy * e)
                pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 2)
                pos += dash_len + gap_len

            arrow = 9
            for off in (math.radians(150), -math.radians(150)):
                ax = int(tip_x + math.cos(robot.target_angle + off) * arrow)
                ay = int(tip_y + math.sin(robot.target_angle + off) * arrow)
                pygame.draw.line(self.screen, color,
                                 (int(tip_x), int(tip_y)), (ax, ay), 2)

    def _draw_robots(self, mundo: Mundo):
        for robot in mundo.robots:
            cx, cy = int(robot.x), int(robot.y)
            r = int(robot.radius)

            # Detection radius (semi-transparente)
            pygame.draw.circle(self.screen, (80, 160, 220, 30), (cx, cy), int(robot.cfg.detection_radius_px), 1)

            # Robot body
            pygame.draw.circle(self.screen, (70, 160, 220), (cx, cy), r, 0)
            pygame.draw.circle(self.screen, (50, 120, 180), (cx, cy), r, 2)

            # Heading
            tip_x = int(robot.x + robot.radius * 1.5 * math.cos(robot.angle))
            tip_y = int(robot.y + robot.radius * 1.5 * math.sin(robot.angle))
            pygame.draw.line(self.screen, (180, 220, 255), (cx, cy), (tip_x, tip_y), 2)

            # Velocity vector
            if robot.speed > 0:
                vel_x = int(robot.x + robot.vx * 0.1)
                vel_y = int(robot.y + robot.vy * 0.1)
                pygame.draw.line(self.screen, (255, 200, 50), (cx, cy), (vel_x, vel_y), 1)

    def _draw_robot_thumbnails(self, mundo: Mundo):
        robots = mundo.robots
        if not robots:
            return

        rows, cols = mundo.rows, mundo.cols
        thumb_size = 50
        per_row = 4
        n = len(robots)
        grid_rows = (n + per_row - 1) // per_row
        gap, pad = 2, 10
        panel_w = per_row * thumb_size + (per_row - 1) * gap + pad * 2
        panel_h = grid_rows * thumb_size + (grid_rows - 1) * gap + pad * 2 + 14
        panel_x = self.screen.get_width() - panel_w - 8
        panel_y = self.screen.get_height() - panel_h - 8

        s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 170))
        self.screen.blit(s, (panel_x, panel_y))
        font = pygame.font.SysFont("Arial", 9)
        font.render("Indiv.", True, (180, 180, 180))

        if not hasattr(self, "_thumb_cache"):
            self._thumb_cache = {}

        for i, robot in enumerate(robots):
            col_idx = i % per_row
            row_idx = i // per_row
            tx = panel_x + pad + col_idx * (thumb_size + gap)
            ty = panel_y + pad + 12 + row_idx * (thumb_size + gap)

            # Caché del thumbnail: solo se regenera la superficie cuando cambia
            # el mapa explorado propio del robot (evita crear Surface por frame).
            firma = hash(robot.explored_robot.tobytes())
            key = (id(robot), firma)
            cached = self._thumb_cache.get(key)
            if cached is None:
                labels = {SCOUTED: (0, 55, 0), VISITED: (0, 110, 0),
                          DISCOVERED: (110, 0, 0)}
                bg_free = (25, 25, 35)
                bg_wall = (40, 20, 20)
                arr = np.zeros((rows, cols, 3), dtype=np.uint8)
                for r in range(rows):
                    for c in range(cols):
                        val = robot.explored_robot[r, c]
                        if mundo.grid[r, c] == 0:
                            arr[r, c] = labels.get(val, bg_free)
                        else:
                            arr[r, c] = labels.get(val, bg_wall)
                surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
                cached = pygame.transform.scale(surf, (thumb_size, thumb_size))
                # Limita el tamaño de la caché para no retener superficies viejas
                if len(self._thumb_cache) > 200:
                    self._thumb_cache.clear()
                self._thumb_cache[key] = cached
            self.screen.blit(cached, (tx, ty))

            lbl = font.render(str(i), True, (200, 200, 200))
            self.screen.blit(lbl, (tx + 2, ty + 2))

    # ------------------------------------------------------------------
    # Capa de visualizacion del Voronoi (toggle `V`)
    # ------------------------------------------------------------------

    def _paleta_regiones(self, n):
        """n colores vivos bien diferenciados (secuencia de hue dorada)."""
        paleta = []
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
            paleta.append((int(r * 255), int(g * 255), int(b * 255)))
        return paleta

    def _draw_capa_voronoi(self, mundo, plan, plan_inspeccion,
                           ver_triangulacion=False):
        """Orquesta la capa segun que voronoi esta activo.

        - Etapa de Inspeccion: `InspeccionVoronoi` (Fase 3, jerarquico). Muestra
          los bordes del primer Voronoi (nivel 0) y las tareas reveladas de
          forma progresiva; con `ver_triangulacion` (checkbox) pinta ademas el
          mapa de celdas por generacion de recursion.
        - Etapa de Exploracion: `VoronoiExploracion` (Fase 2, particion por robot).
        """
        if mundo.etapa == "inspeccion" and plan_inspeccion is not None:
            viz = getattr(plan_inspeccion, "viz", None)
            if viz is None:
                return
            self._dibujar_bordes(mundo, viz["nivel0"])
            if ver_triangulacion:
                self._dibujar_mapa_niveles(mundo, viz["niveles"])
            self._dibujar_tareas_inspeccion(mundo, plan_inspeccion,
                                            viz["tareas"])
            return
        gen = getattr(plan, "_generador", None)
        if isinstance(gen, VoronoiExploracion) and gen._viz.get("mapa") is not None:
            self._dibujar_mapa_particion(mundo, gen._viz["mapa"], alpha=80)
            self._dibujar_bordes(mundo, gen._viz["mapa"])
            self._dibujar_explo_objetivos(mundo, gen._viz.get("objetivos", []))

    def _color_interes(self, t):
        """Heatmap INVERTIDO del interes de mercado (`peso * ajuste`): t=0
        (interes/costo bajo) rojo/urgente, t=1 (interes/costo alto) azul/menos
        prioritario. El numero de cada tarea se pinta con este color."""
        t = max(0.0, min(1.0, t))
        r, g, b = colorsys.hsv_to_rgb(0.65 * t, 0.60, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _dibujar_mapa_niveles(self, mundo, niveles, alpha=100):
        """Colorea cada celda segun su generacion en el arbol estatico de
        triangulacion (0-based): raices (nivel 0) en azul, profundo en rojo.
        Solo se usa bajo el checkbox 'Ver triangulacion'."""
        gs = mundo.cfg.grid_size
        vals = niveles[niveles >= 0]
        if not vals.size:
            return
        vmax = int(vals.max()) or 1
        rows, cols = mundo.rows, mundo.cols
        overlay = pygame.Surface((cols * gs, rows * gs), pygame.SRCALPHA)
        for r in range(rows):
            for c in range(cols):
                nv = int(niveles[r, c])
                if nv < 0:
                    continue
                col = self._hue_escala(nv, vmax)
                pygame.draw.rect(overlay, (col[0], col[1], col[2], alpha),
                                 (c * gs, r * gs, gs, gs))
        self.screen.blit(overlay, (0, 0))

    def _hue_escala(self, nivel, vmax):
        """Hue escalado de profundidad 0-based: azul (nivel 0) -> rojo (vmax)."""
        t = 0.0 if vmax <= 0 else nivel / vmax
        hue = 0.65 - 0.65 * t
        r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _dibujar_mapa_particion(self, mundo, mapa, alpha=110):
        """Pinta cada celda segun su sitio (region). `mapa` es un array de
        indices de region con -1 en las celdas no participadas."""
        gs = mundo.cfg.grid_size
        unicos = [v for v in np.unique(mapa) if v >= 0]
        if not unicos:
            return
        paleta = self._paleta_regiones(len(unicos))
        color_idx = {v: paleta[i] for i, v in enumerate(unicos)}
        rows, cols = mundo.rows, mundo.cols
        overlay = pygame.Surface((cols * gs, rows * gs), pygame.SRCALPHA)
        for r in range(rows):
            for c in range(cols):
                idx = mapa[r, c]
                if idx < 0:
                    continue
                col = color_idx[idx]
                pygame.draw.rect(overlay, (col[0], col[1], col[2], alpha),
                                 (c * gs, r * gs, gs, gs))
        self.screen.blit(overlay, (0, 0))

    def _dibujar_bordes(self, mundo, mapa, color=(240, 230, 255)):
        """Bordes del diagrama de Voronoi: lineas entre celdas de distinta region."""
        gs = mundo.cfg.grid_size
        rows, cols = mundo.rows, mundo.cols
        for r in range(rows):
            for c in range(cols):
                if mapa[r, c] < 0:
                    continue
                if r + 1 < rows and mapa[r + 1, c] != mapa[r, c]:
                    pygame.draw.line(self.screen, color,
                                     (c * gs, (r + 1) * gs), (c * gs + gs, (r + 1) * gs), 1)
                if c + 1 < cols and mapa[r, c + 1] != mapa[r, c]:
                    pygame.draw.line(self.screen, color,
                                     ((c + 1) * gs, r * gs), ((c + 1) * gs, r * gs + gs), 1)

    def _dibujar_tareas_inspeccion(self, mundo, planificador, tareas):
        """Tareas reveladas de forma progresiva (padre antes que hijo): solo se
        dibuja la tarea cuyo padre ya fue inspeccionado (misma condicion que la
        subasta estricta). El numero es la generacion de recursion (0 = region
        raiz del primer Voronoi) y su color es el heatmap INVERTIDO del interes
        (`peso * ajuste`): rojo = interes bajo = urgente; azul = interes alto =
        menos prioritario. La celda inspeccionada ya esta pintada por el
        resultado (SANA/DEFECTUOSA), asi que no se repite aqui."""
        gs = mundo.cfg.grid_size
        visibles = [t for t in tareas
                    if planificador.tarea_revelada(mundo, t)
                    and not mundo.inspec[t["r"], t["c"]]]
        valores = [t["peso"] * t["ajuste"] for t in visibles]
        vmin = min(valores) if valores else 1.0
        vmax = max(valores) if valores else 1.0
        rango = max(vmax - vmin, 1e-6)
        font = pygame.font.SysFont("Arial", 10)
        for t in visibles:
            px, py = int(t["x"]), int(t["y"])
            r, c = int(t["r"]), int(t["c"])
            niv = int(t["nivel"])
            if niv == 0:
                tam = gs - max(2, gs // 4)
                borde = (235, 220, 255)
                relleno = (58, 56, 80)
                fuente = pygame.font.SysFont("Arial", max(10, gs // 2))
            else:
                tam = max(4, gs // 2)
                borde = (120, 118, 150)
                relleno = (46, 46, 66)
                fuente = font
            interes = t["peso"] * t["ajuste"]
            tt = 0.5 if rango <= 0 else (interes - vmin) / rango
            col = self._color_interes(tt)
            lado = (c * gs + (gs - tam) // 2, r * gs + (gs - tam) // 2, tam, tam)
            pygame.draw.rect(self.screen, relleno, lado)
            pygame.draw.rect(self.screen, borde, lado, 1)
            surf = fuente.render(str(niv), True, col)
            self.screen.blit(surf, (px - surf.get_width() // 2,
                                    py - surf.get_height() // 2))

    def _dibujar_explo_objetivos(self, mundo, objetivos):
        """Mejor celda de frontera de cada region (rombo)."""
        for t in objetivos:
            px, py = int(t["x"]), int(t["y"])
            pygame.draw.line(self.screen, (255, 255, 255), (px - 6, py), (px, py - 6), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (px, py - 6), (px + 6, py), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (px + 6, py), (px, py + 6), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (px, py + 6), (px - 6, py), 1)