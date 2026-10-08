"""Tramo final congelado (sin flicker pre-inspeccion) y RW con latch.

1. Robot dirigido en tramo final con pared parpadeante: omega no alterna y
   llega al centrado de inspeccion.
2. Random Walk con peligro persistente: emite pocos giros y avanza.
3. Kick anti-atasco: no dispara en un giro normal con progreso angular, pero
   si cuando la rotacion esta netamente quieta.
"""
import math
import numpy as np

from src.core.config import Config
from src.core.mundo import Mundo
from src.robots.robot import Robot


def _mundo_chico(cfg, seed=11):
    cfg.map_width = 200
    cfg.map_height = 200
    cfg.rows = 10
    cfg.cols = 10
    cfg.grid_size = 20
    rng = np.random.default_rng(seed)
    mundo = Mundo(cfg, rng=rng)
    mundo.grid[:, :] = 0
    mundo.robots = []
    return mundo, rng


def test_tramo_final_sin_flicker_con_pared_parpadeante():
    cfg = Config()
    mundo, rng = _mundo_chico(cfg)
    mundo.etapa = "inspeccion"
    rb = Robot(110.0, 120.0, cfg, rng)
    rb.angle = 0.0
    mundo.robots.append(rb)
    # Meta al centro de la celda (5,5); robot a 10 px (en radio 15, fuera de 4).
    rb.goal = (110.0, 110.0)
    rb.goal_mode = True
    rb.goal_alcanzado = False
    rb._goal_timer = 0.0
    rb._aprox_insp_timer = 0.0
    rb.goal_timeout = 60.0
    rb.ruta = [(110.0, 110.0)]
    signos = set()
    for i in range(200):
        # Pared cercana que aparece/desaparece: antes hacia parpadear destino.
        mundo.grid[5, 6] = 1 if i % 2 == 0 else 0
        rb.update(0.05, mundo)
        if rb.inspeccionando:
            break
        if abs(rb.omega) > 1e-9:
            signos.add(1 if rb.omega > 0 else -1)
    assert rb.inspeccionando, "debio centrar e iniciar la percusion"
    assert len(signos) <= 1, f"omega alterno de sentido: {signos}"


def test_rw_peligro_persistente_avanza_con_pocos_giros():
    cfg = Config()
    cfg.estrategia_exploracion = "Random Walk"
    mundo, rng = _mundo_chico(cfg)
    mundo.etapa = "exploracion"
    # Muro al este del robot, dentro del rango de deteccion.
    mundo.grid[5, 7] = 1
    mundo.grid[4, 7] = 1
    mundo.grid[6, 7] = 1
    rb = Robot(100.0, 110.0, cfg, rng)
    rb.angle = 0.0  # mirando al muro
    mundo.robots.append(rb)
    llamadas = {"n": 0}
    orig = rb._iniciar_giro

    def contando(target):
        llamadas["n"] += 1
        return orig(target)

    rb._iniciar_giro = contando
    x0 = rb.x
    for _ in range(200):
        rb.update(0.05, mundo)
    assert abs(rb.x - x0) > 5.0, "el RW quedo clavado mirando al muro"
    assert llamadas["n"] <= 6, f"re-giros por frame: {llamadas['n']}"


def test_kick_no_dispara_en_giro_normal():
    cfg = Config()
    cfg.estrategia_exploracion = "Random Walk"
    mundo, rng = _mundo_chico(cfg)
    mundo.etapa = "exploracion"
    rb = Robot(100.0, 100.0, cfg, rng)
    mundo.robots.append(rb)
    rb.state = rb.ROTATING
    rb.target_angle = rb.angle + 1.0
    rb.omega = rb.turn_speed
    rb.target_speed = rb.max_speed
    for _ in range(20):
        rb.update(0.05, mundo)
    assert abs(rb._diferencia_angular(rb.target_angle, rb.angle)) < math.radians(10)
    assert rb._giro_stuck_timer == 0.0


def test_kick_dispara_en_rotacion_quieta():
    cfg = Config()
    cfg.estrategia_exploracion = "Random Walk"
    mundo, rng = _mundo_chico(cfg)
    mundo.etapa = "exploracion"
    rb = Robot(100.0, 100.0, cfg, rng)
    mundo.robots.append(rb)
    rb.state = rb.ROTATING
    rb.target_angle = rb.angle + 2.0
    rb.omega = 0.0  # rotacion netamente quieta (flicker sin progreso)
    rb.target_speed = 0.0
    a0 = rb.angle
    for _ in range(12):  # 0.6 s > timeout 0.4 s
        rb.update(0.05, mundo)
    assert abs(rb._diferencia_angular(rb.angle, a0)) > math.radians(45)
