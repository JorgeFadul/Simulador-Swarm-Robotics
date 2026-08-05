import pygame
import numpy as np

from .config import Config
from .mundo import Mundo
from ..robots.robot import Robot
from ..gui.renderer import Renderer
from ..algorithms.exploration import ExplorationPlanner


class Simulacion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng()

        pygame.init()
        self.screen = pygame.display.set_mode((cfg.map_width, cfg.map_height))
        pygame.display.set_caption("Simulador Swarm 2D - Editor")
        self.clock = pygame.time.Clock()

        self.mundo = Mundo(cfg)
        self.renderer = Renderer(self.screen)

        self.estado = "edicion"
        self.herramienta = "robot"
        self._mouse_down = False
        self._ult_r = -1
        self._ult_c = -1
        self.tiempo_simulado = 0.0
        self.pasos_totales = 0
        self.fps_actual = 0.0
        self.tiempo_final = None
        self.planner = None
        self.ver_fronteras = True
        self.ver_objetivos = True
        self.ver_intencion = True

    def _reconstruir_mundo(self):
        self.mundo = Mundo(self.cfg)
        self.tiempo_simulado = 0.0
        self.pasos_totales = 0
        self.tiempo_final = None
        self.planner = None

    def _crear_planner(self):
        strat = self.cfg.strategy
        if strat in ("Frontier Based", "Sample Based", "Hybrid"):
            return ExplorationPlanner(self.cfg, strat)
        return None

    def iniciar(self):
        """Arranca la simulacion desde el editor: resetea exploracion, crea el
        planner segun la estrategia y pone los robots en modo dirigido."""
        if self.mundo is None or not self.mundo.robots:
            return
        self.mundo.reiniciar_exploracion()
        for r in self.mundo.robots:
            r.reiniciar_exploracion()
            r.reiniciar_objetivo()
        self.planner = self._crear_planner()
        self.tiempo_simulado = 0.0
        self.pasos_totales = 0
        self.tiempo_final = None
        self.estado = "running"

    def poblar_robots(self, n):
        for _ in range(n):
            x, y = self.mundo.posicion_libre_aleatoria(self.rng, self.cfg.robot_radius_px)
            if x is None:
                break
            robot = Robot(x, y, self.cfg, self.rng)
            self.mundo.robots.append(robot)

    def _termino(self, mundo):
        if mundo.porcentaje_descubierto >= self.cfg.exploration_goal:
            return True
        if self.planner is None or not self.planner._alguna_vez_hubo_fronteras:
            return False
        fm = self.planner.frontera_mask
        if fm is None:
            return False
        return int(fm.sum()) == 0

    def step(self):
        if self.estado != "running" or not self.mundo:
            return
        if self._termino(self.mundo):
            self.estado = "completado"
            self.tiempo_final = self.tiempo_simulado
            return
        if self.planner:
            self.planner.actualizar(self.mundo, self.tiempo_simulado)
        self.mundo.update(0.05)
        self.tiempo_simulado += 0.05
        self.pasos_totales += 1

    def manejar_evento(self, event):
        if self.estado != "edicion":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.estado = "edicion"
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.herramienta = "robot"
            elif event.key == pygame.K_w:
                self.herramienta = "pared"
            elif event.key == pygame.K_e:
                self.herramienta = "borrar"

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._mouse_down = True
            self._colocar_en(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._borrar_en(event.pos)
        elif event.type == pygame.MOUSEMOTION and self._mouse_down:
            self._colocar_en(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._mouse_down = False
            self._ult_r = -1
            self._ult_c = -1

    def _colocar_en(self, pos):
        x, y = pos
        m = self.mundo
        gs = m.cfg.grid_size
        c = int(x // gs)
        r = int(y // gs)
        if r < 0 or r >= m.rows or c < 0 or c >= m.cols:
            return
        if m.es_borde(r, c):
            return

        if self.herramienta == "robot":
            if m.grid[r, c] != 0:
                return
            cx = c * gs + gs / 2
            cy = r * gs + gs / 2
            for existing in m.robots:
                if np.hypot(cx - existing.x, cy - existing.y) < existing.radius + 5:
                    return
            robot = Robot(cx, cy, self.cfg, self.rng)
            m.robots.append(robot)

        elif self.herramienta == "pared":
            if (r, c) == (self._ult_r, self._ult_c):
                return
            self._ult_r, self._ult_c = r, c
            m.colocar_pared_celda(r, c)
            m.reiniciar_exploracion()

        elif self.herramienta == "borrar":
            self._borrar_en(pos)

    def _borrar_en(self, pos):
        x, y = pos
        m = self.mundo
        gs = m.cfg.grid_size
        c = int(x // gs)
        r = int(y // gs)
        if r < 0 or r >= m.rows or c < 0 or c >= m.cols:
            return
        if m.es_borde(r, c):
            return

        for i, robot in enumerate(m.robots):
            if np.hypot(x - robot.x, y - robot.y) < robot.radius + 5:
                m.robots.pop(i)
                m.reiniciar_exploracion()
                return

        if m.grid[r, c] == 1:
            m.borrar_celda(r, c)
            m.reiniciar_exploracion()

    def render(self):
        if not self.mundo:
            return
        self.renderer.draw(self.mundo, self.estado, plan=self.planner,
                           ver_fronteras=self.ver_fronteras,
                           ver_objetivos=self.ver_objetivos,
                           ver_intencion=self.ver_intencion)
        self._draw_mode_hud()
        pygame.display.flip()

    def _draw_mode_hud(self):
        if self.estado == "edicion":
            atajos = [
                f"[R]obot  [P]ared  [B]orrar",
                f"Herramienta: {self.herramienta.upper()}",
                "Click izq: colocar   Click der: borrar",
            ]
            fuente = pygame.font.SysFont("Arial", 14)
            y = 10
            for texto in atajos:
                surf = fuente.render(texto, True, (200, 200, 200))
                self.screen.blit(surf, (10, y))
                y += 20

            titulo = fuente.render("EDITOR", True, (100, 200, 255))
            self.screen.blit(titulo, (self.cfg.map_width - 80, 10))
            return

        fuente = pygame.font.SysFont("Arial", 14)
        pct = self.mundo.porcentaje_descubierto * 100.0
        if self.estado == "completado":
            msg = f"COMPLETADO - mapa {pct:.1f}% en {self.tiempo_final:.1f} s"
            surf = fuente.render(msg, True, (255, 220, 60))
        else:
            msg = f"{self.cfg.strategy} - descubierto {pct:.1f}%"
            surf = fuente.render(msg, True, (150, 220, 255))
        self.screen.blit(surf, (10, 10))