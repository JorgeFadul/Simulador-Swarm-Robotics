import pygame
import numpy as np

from .config import Config, ESTRATEGIAS_EXPLORACION, ESTRATEGIAS_INSPECCION
from .mundo import Mundo
from ..robots.robot import Robot
from ..gui.renderer import Renderer
from ..algorithms.exploration import ExplorationPlanner, _celdas_frontera
from ..algorithms.inspeccion import InspeccionVoronoi, InspectionPlanner


class Simulacion:
    def __init__(self, cfg: Config, seed=None, headless=False):
        self.cfg = cfg
        self.headless = headless
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((cfg.map_width, cfg.map_height))
            pygame.display.set_caption("Simulador Swarm 2D - Editor")
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(self.screen)
        else:
            self.screen = None
            self.clock = None
            self.renderer = None

        self.mundo = Mundo(cfg, rng=self.rng)

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
        self.planner_inspeccion = None
        self.ver_fronteras = True
        self.ver_objetivos = True
        self.ver_intencion = True
        self.ver_defectos = False
        # Estado de fase
        self.etapa = "exploracion"
        self._total_descubierto_previo = 0
        self._ultimo_nuevo_descubrimiento = 0.0
        self._total_inspeccionadas_previo = 0
        self._ultimo_nuevo_inspeccionado = 0.0

    @property
    def transicionado(self):
        """True si ya se transiciono a la etapa de Inspeccion (modo hibrido)."""
        return self.etapa == "inspeccion"

    def cambiar_a_inspeccion(self):
        """Transicion a la etapa de Inspeccion (modo hibrido)."""
        if self.mundo is None:
            return
        self.etapa = "inspeccion"
        self.mundo.cambiar_a_inspeccion()
        self._ultimo_nuevo_inspeccionado = self.tiempo_simulado

    def _reconstruir_mundo(self):
        self.mundo = Mundo(self.cfg, rng=self.rng)
        self.tiempo_simulado = 0.0
        self.pasos_totales = 0
        self.tiempo_final = None
        self.planner = None
        self.planner_inspeccion = None
        self.etapa = "exploracion"
        self._total_descubierto_previo = 0
        self._ultimo_nuevo_descubrimiento = 0.0
        self._total_inspeccionadas_previo = 0
        self._ultimo_nuevo_inspeccionado = 0.0

    def _crear_planner(self):
        """Crea el planner segun el modo de fase y la estrategia activa."""
        modo = self.cfg.modo_fase

        if modo == "exploracion":
            strat = self.cfg.estrategia_exploracion
            if strat == "Random Walk":
                return None  # Random Walk no usa planner
            if strat in ("Frontier Based", "Sample Based", "Hybrid Based",
                         "Boustrophedon"):
                return ExplorationPlanner(self.cfg, strat, self.rng)
            # RL Exploracion: por ahora retorna None (placeholder)
            return None

        elif modo == "inspeccion":
            strat = self.cfg.estrategia_inspeccion
            if strat == "Voronoi":
                return InspeccionVoronoi(self.cfg, self.rng)
            if strat == "Boustrophedon Inspeccion":
                return InspectionPlanner(self.cfg, self.rng)
            # RL Inspeccion: por ahora retorna None (placeholder)
            return None

        elif modo == "hibrido":
            strat = self.cfg.estrategia_hibrido
            if strat == "Random Walk Hibrido":
                return None  # Random Walk hibrido: world.etapa maneja inspeccion
            # RL Hibrido: placeholder
            return None

        return None

    def _actualizar_titulo(self):
        if self.headless or self.screen is None:
            return
        if self.estado == "edicion":
            titulo = "Simulador Swarm 2D - Editor"
        elif self.estado == "running":
            titulo = "Simulador Swarm 2D - Ejecutando"
        elif self.estado == "paused":
            titulo = "Simulador Swarm 2D - Pausado"
        else:
            titulo = "Simulador Swarm 2D - Completado"
        pygame.display.set_caption(titulo)

    def iniciar(self):
        """Arranca la simulacion desde el editor."""
        if self.mundo is None or not self.mundo.robots:
            return
        self.mundo.reiniciar_exploracion()
        self.etapa = "exploracion"
        self.mundo.etapa = "exploracion"
        self._total_descubierto_previo = 0
        self._ultimo_nuevo_descubrimiento = 0.0
        self._total_inspeccionadas_previo = 0
        self._ultimo_nuevo_inspeccionado = 0.0
        self.planner_inspeccion = None

        modo = self.cfg.modo_fase
        for r in self.mundo.robots:
            r.reiniciar_exploracion()
            r.reiniciar_objetivo()
            r.reiniciar_energia()
            # Modo inspeccion: pre-explorar el entorno del robot
            if modo == "inspeccion":
                exp_radius = r.detection_radius * self.cfg.exploration_radius_multiplier
                s, p = self.mundo.explorar_desde(r.x, r.y, exp_radius)
                r.scouteadas_robot += s
                r.paredes_robot += p
                r._explorar_desde_propio(self.mundo, exp_radius)

        self.planner = self._crear_planner()

        # Modo inspeccion: crear planner de inspeccion directamente
        if modo == "inspeccion":
            self.planner_inspeccion = self.planner
            self.planner = None
            self.etapa = "inspeccion"
            self.mundo.etapa = "inspeccion"

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
        """Condicion de fin de la simulacion segun el modo de fase."""
        modo = self.cfg.modo_fase

        if modo == "exploracion":
            return self._termino_exploracion(mundo)
        elif modo == "inspeccion":
            return self._termino_inspeccion(mundo)
        elif modo == "hibrido":
            return self._termino_hibrido(mundo)
        return False

    def _termino_exploracion(self, mundo):
        """Modo exploracion: termina por meta de descubrimiento o estancamiento."""
        if mundo.porcentaje_descubierto >= self.cfg.exploration_goal:
            return True
        # Random Walk (sin planner): solo por meta de descubrimiento
        if self.planner is None:
            return False
        if not self.planner._alguna_vez_hubo_fronteras:
            return False
        # Estancamiento de exploracion
        if (self._ultimo_nuevo_descubrimiento > 0 and
                self.tiempo_simulado - self._ultimo_nuevo_descubrimiento >
                self.cfg.transicion_tiempo_sin_scouteo_s):
            return True
        # Guarda de arranque
        if self.tiempo_simulado < self.cfg.plan_interval_s * 2:
            return False
        # Fronteras agotadas
        return int(_celdas_frontera(mundo).sum()) == 0

    def _termino_inspeccion(self, mundo):
        """Modo inspeccion: termina por inspeccion completa o estancamiento."""
        if mundo.inspeccion_completa():
            return True
        if (self.tiempo_simulado - self._ultimo_nuevo_inspeccionado > 30):
            return True
        return False

    def _termino_hibrido(self, mundo):
        """Modo hibrido: termina cuando no quedan celdas por ver/visitar/inspeccionar.

        Para Random Walk Hibrido: basta con `inspeccion_completa()` (implica
        que todas las celdas fueron vistas, visitadas e inspeccionadas).
        Para otras estrategias hibridas: requiere exploracion completada
        Y inspeccion completada.
        """
        strat = self.cfg.estrategia_hibrido
        if strat == "Random Walk Hibrido":
            return mundo.inspeccion_completa()
        # Otras estrategias hibridas: exploracion + inspeccion
        exploracion_ok = mundo.porcentaje_descubierto >= self.cfg.exploration_goal
        if not exploracion_ok and self.planner is not None:
            if self.planner._alguna_vez_hubo_fronteras:
                if (self._ultimo_nuevo_descubrimiento > 0 and
                        self.tiempo_simulado - self._ultimo_nuevo_descubrimiento >
                        self.cfg.transicion_tiempo_sin_scouteo_s):
                    exploracion_ok = True
                if (self.tiempo_simulado >= self.cfg.plan_interval_s * 2 and
                        int(_celdas_frontera(mundo).sum()) == 0):
                    exploracion_ok = True
        inspeccion_ok = mundo.inspeccion_completa()
        if not inspeccion_ok and self.etapa == "inspeccion":
            if (self.tiempo_simulado - self._ultimo_nuevo_inspeccionado > 30):
                inspeccion_ok = True
        return exploracion_ok and inspeccion_ok

    def _verificar_transicion(self, mundo):
        """Verifica transicion de exploracion a inspeccion (solo modo hibrido)."""
        if self.cfg.modo_fase != "hibrido" or self.etapa == "inspeccion":
            return False
        ahora = self.tiempo_simulado
        total = mundo._scouteadas + mundo._visitadas + mundo._paredes_vistas
        if total > self._total_descubierto_previo:
            self._ultimo_nuevo_descubrimiento = ahora
            self._total_descubierto_previo = total

        if mundo.porcentaje_descubierto >= self.cfg.transicion_umbral_descubierto:
            self.cambiar_a_inspeccion()
            return True
        # Estancamiento: si hace mas de `transicion_tiempo_sin_scouteo_s`
        # simulados sin progreso de descubrimiento, la exploracion util termino.
        # Guarda de arranque: no evaluar antes de 2 ciclos de planificacion.
        if ahora >= self.cfg.plan_interval_s * 2:
            tiempo_ref = self._ultimo_nuevo_descubrimiento if self._ultimo_nuevo_descubrimiento > 0 else 0
            if ahora - tiempo_ref > self.cfg.transicion_tiempo_sin_scouteo_s:
                self.cambiar_a_inspeccion()
                return True
        return False

    def forzar_transicion_inspeccion(self):
        """Transicion manual a la etapa de Inspeccion (modo hibrido)."""
        if self.cfg.modo_fase != "hibrido" or self.etapa == "inspeccion":
            return
        self.cambiar_a_inspeccion()

    def step(self):
        if self.estado != "running" or not self.mundo:
            return
        m = self.mundo
        if self._termino(m):
            self.estado = "completado"
            self.tiempo_final = self.tiempo_simulado
            return

        modo = self.cfg.modo_fase

        # Modo hibrido: verificar transicion exploracion -> inspeccion
        if modo == "hibrido":
            if self._verificar_transicion(m):
                self.planner = None
                self.planner_inspeccion = InspeccionVoronoi(self.cfg, self.rng)

        # Actualizar planners segun modo
        if self.planner_inspeccion is not None and (
                (modo == "hibrido" and self.etapa == "inspeccion") or
                modo == "inspeccion"):
            self.planner_inspeccion.actualizar(m)
        elif self.planner:
            if modo in ("exploracion", "hibrido"):
                self.planner.actualizar(m, self.tiempo_simulado)

        m.update(0.05)
        self.tiempo_simulado += 0.05
        self.pasos_totales += 1

        # Registra avance de descubrimiento
        total_desc = m._scouteadas + m._visitadas + m._paredes_vistas
        if total_desc > self._total_descubierto_previo:
            self._total_descubierto_previo = total_desc
            self._ultimo_nuevo_descubrimiento = self.tiempo_simulado

        # Registra avance de inspeccion
        if m._inspeccionadas > self._total_inspeccionadas_previo:
            self._total_inspeccionadas_previo = m._inspeccionadas
            self._ultimo_nuevo_inspeccionado = self.tiempo_simulado

    def manejar_evento(self, event):
        if self.estado != "edicion":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.estado = "edicion"
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.herramienta = "robot"
            elif event.key == pygame.K_p:
                self.herramienta = "pared"
            elif event.key == pygame.K_b:
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
        if self.headless or not self.mundo or self.screen is None or self.renderer is None:
            return
        self.renderer.draw(self.mundo, self.estado, plan=self.planner,
                           ver_fronteras=self.ver_fronteras,
                           ver_objetivos=self.ver_objetivos,
                           ver_intencion=self.ver_intencion,
                           ver_defectos=self.ver_defectos)
        self._draw_mode_hud()
        self._actualizar_titulo()
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
        modo = self.cfg.modo_fase
        strat = self.cfg._estrategia_activa()

        if self.estado == "completado":
            msg = f"COMPLETADO - mapa {pct:.1f}% en {self.tiempo_final:.1f} s"
            surf = fuente.render(msg, True, (255, 220, 60))
        elif modo == "exploracion":
            msg = f"[EXPLORACION] {strat} - descubierto {pct:.1f}%"
            surf = fuente.render(msg, True, (150, 220, 255))
        elif modo == "inspeccion":
            pct_ins = self.mundo.porcentaje_inspeccionado * 100.0
            msg = f"[INSPECCION] {strat} - inspeccionado {pct_ins:.1f}%"
            surf = fuente.render(msg, True, (120, 255, 180))
        elif modo == "hibrido":
            if self.etapa == "inspeccion":
                pct_ins = self.mundo.porcentaje_inspeccionado * 100.0
                msg = f"[HIBRIDO-INSPECCION] {strat} - insp {pct_ins:.1f}%"
                color = (120, 255, 180)
            else:
                msg = f"[HIBRIDO-EXPLORACION] {strat} - descubierto {pct:.1f}%"
                color = (255, 200, 100)
            surf = fuente.render(msg, True, color)
        else:
            msg = f"{strat} - descubierto {pct:.1f}%"
            surf = fuente.render(msg, True, (150, 220, 255))

        self.screen.blit(surf, (10, 10))
