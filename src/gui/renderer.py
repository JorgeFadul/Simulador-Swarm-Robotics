import math
import numpy as np
import pygame
from ..core.mundo import Mundo, SCOUTED, VISITED, DISCOVERED


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self._thumb_bg = None

    def draw(self, mundo: Mundo, estado="running"):
        self.screen.fill((20, 20, 30))
        self._draw_exploracion(mundo, estado)
        self._draw_grid(mundo)
        self._draw_walls(mundo)
        self._draw_intents(mundo)
        self._draw_robots(mundo)
        if mundo.robots:
            self._draw_robot_thumbnails(mundo)

    def _draw_exploracion(self, mundo: Mundo, estado="running"):
        gs = mundo.cfg.grid_size
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                val = mundo.explored[r, c]
                rect = (c * gs, r * gs, gs, gs)
                if mundo.grid[r, c] == 0:
                    if val == 1:
                        color = (0, 55, 0)
                    elif val == 2:
                        color = (0, 110, 0)
                    else:
                        continue
                else:
                    if val == 3:
                        color = (110, 0, 0)
                    elif estado == "edicion":
                        color = (60, 40, 40)
                    else:
                        color = (30, 15, 15)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_grid(self, mundo: Mundo):
        gs = mundo.cfg.grid_size
        for r in range(mundo.rows):
            for c in range(mundo.cols):
                color = (60, 50, 50) if mundo.grid[r, c] else (35, 35, 45)
                rect = (c * gs, r * gs, gs, gs)
                pygame.draw.rect(self.screen, color, rect, 1)

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

        for i, robot in enumerate(robots):
            col_idx = i % per_row
            row_idx = i // per_row
            tx = panel_x + pad + col_idx * (thumb_size + gap)
            ty = panel_y + pad + 12 + row_idx * (thumb_size + gap)

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
            scaled = pygame.transform.scale(surf, (thumb_size, thumb_size))
            self.screen.blit(scaled, (tx, ty))

            lbl = font.render(str(i), True, (200, 200, 200))
            self.screen.blit(lbl, (tx + 2, ty + 2))