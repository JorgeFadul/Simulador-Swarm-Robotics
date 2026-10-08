"""Giro trasero ~180 junto a pared: sin oscilacion de sentido."""
import math
import numpy as np

from src.core.config import Config
from src.core.mundo import Mundo
from src.robots.robot import Robot
from src.algorithms.pathfinding import _suavizar


def _robot_simple():
    cfg = Config()
    rng = np.random.default_rng(7)
    rb = Robot(100.0, 100.0, cfg, rng)
    rb.angle = 0.0
    return cfg, rb


def test_histeresis_conserva_sentido_en_180():
    _, rb = _robot_simple()
    d1 = rb._elegir_dir_giro(math.radians(179.0))
    # Ruido al otro lado del umbral: debe conservar el sentido previo.
    d2 = rb._elegir_dir_giro(math.radians(-179.0))
    d3 = rb._elegir_dir_giro(math.radians(178.5))
    assert d1 == d2 == d3
    assert rb.omega == 0.0  # el helper no toca omega directamente


def test_histeresis_se_libera_al_alinear():
    _, rb = _robot_simple()
    rb._elegir_dir_giro(math.radians(175.0))
    assert rb._giro_bloqueado
    rb._elegir_dir_giro(math.radians(10.0))
    assert not rb._giro_bloqueado


def test_suavizar_conserva_wp_con_giro_trasero_y_pared():
    cfg = Config()
    cfg.map_width = 200
    cfg.map_height = 200
    cfg.rows = 10
    cfg.cols = 10
    cfg.grid_size = 20
    rng = np.random.default_rng(3)
    mundo = Mundo(cfg, rng=rng)
    mundo.grid[:, :] = 0
    # Pared pegada al robot (origen) para activar la guardia.
    mundo.grid[5, 4] = 1
    radio = float(cfg.robot_radius_px)
    x0, y0 = 5 * 20 + 10, 5 * 20 + 10  # centro celda (4,5) junto a pared
    wps = [(x0, y0), (x0 + 20, y0), (x0 - 60, y0)]  # salto directo = giro ~180
    out = _suavizar(mundo, wps, radio, angulo_actual=0.0,
                    x_actual=x0, y_actual=y0)
    # Sin guardia el suavizado saltaria al ultimo punto; con guardia conserva
    # el intermedio para no girar de espaldas contra la pared.
    assert len(out) >= 3
