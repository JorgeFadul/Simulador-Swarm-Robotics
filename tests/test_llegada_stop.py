"""Llegada al punto: frena dentro de 1 radio y se detiene a la primera.

Sin rebase con vueltas: el giro total acumulado hasta la percusion esta
acotado y la velocidad cae por debajo de la maxima solo muy cerca.
"""
import math
import numpy as np

from src.core.config import Config
from src.core.mundo import Mundo
from src.robots.robot import Robot


def test_llegada_frena_y_no_orbita():
    cfg = Config()
    cfg.map_width = 400
    cfg.map_height = 400
    cfg.rows = 20
    cfg.cols = 20
    cfg.grid_size = 20
    rng = np.random.default_rng(21)
    mundo = Mundo(cfg, rng=rng)
    mundo.grid[:, :] = 0
    mundo.etapa = "inspeccion"
    mundo.robots = []
    # Aproximacion con desvio lateral (el caso que antes cazaba orbitando).
    rb = Robot(90.0, 140.0, cfg, rng)
    rb.angle = 0.0
    mundo.robots.append(rb)
    rb.goal = (170.0, 110.0)
    rb.goal_mode = True
    rb.goal_alcanzado = False
    rb._goal_timer = 0.0
    rb._aprox_insp_timer = 0.0
    rb.goal_timeout = 120.0
    rb.ruta = [(170.0, 110.0)]
    giro_total = 0.0
    ang_prev = rb.angle
    freno_visto = False
    for _ in range(600):
        rb.update(0.05, mundo)
        giro_total += abs(rb._diferencia_angular(rb.angle, ang_prev))
        ang_prev = rb.angle
        d = math.hypot(170.0 - rb.x, 110.0 - rb.y)
        if d < rb.radius and rb.speed < rb.max_speed * 0.9:
            freno_visto = True
        if rb.inspeccionando:
            break
    assert rb.inspeccionando, "no llego a la percusion a la primera"
    assert giro_total < 1.5 * math.pi, f"orbito el punto: {giro_total:.2f} rad"
    assert freno_visto, "nunca freno dentro del radio"
