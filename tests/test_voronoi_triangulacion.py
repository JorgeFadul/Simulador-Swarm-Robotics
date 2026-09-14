import math
import numpy as np
import pytest

from src.core.config import Config
from src.core.mundo import Mundo, DEFECTUOSA, SANA
from src.robots.robot import Robot
from src.algorithms.inspeccion import (
    Figura,
    InspeccionVoronoi,
    _poligono_voronoi_2d,
    _subdividir_figura,
    _centroid_poligono,
    _colapsar_escaleras
)


def _crear_mundo_con_robots(num_robots=3):
    cfg = Config()
    cfg.map_width = 400
    cfg.map_height = 400
    cfg.rows = 20
    cfg.cols = 20
    cfg.grid_size = 20
    cfg.num_robots = num_robots
    cfg.inspeccion_factor_defectivo = 0.85
    cfg.inspeccion_factor_sano = 1.25

    rng = np.random.default_rng(42)
    mundo = Mundo(cfg, rng=rng)
    # Add robots in free cells
    posiciones = [(50.0, 50.0), (350.0, 50.0), (200.0, 350.0)]
    for i in range(num_robots):
        px, py = posiciones[i % len(posiciones)]
        rb = Robot(px, py, cfg, rng)
        mundo.robots.append(rb)
    return cfg, mundo


def test_voronoi_gen0_vertices():
    box = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]
    sites = [(100.0, 100.0), (300.0, 100.0), (200.0, 300.0)]
    polys = []
    for i, s in enumerate(sites):
        others = [sites[j] for j in range(len(sites)) if j != i]
        p = _poligono_voronoi_2d(box, s, others)
        polys.append(p)
    assert len(polys) == 3
    # Every Voronoi cell should have between 3 and 6 clean vertices, not dozens
    for p in polys:
        assert 3 <= len(p) <= 6


def test_figura_clase_atributos():
    fig = Figura(
        fig_id=1,
        generacion=0,
        vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
        centroide_xy=(5.0, 5.0),
        celda_rc=(0, 0),
        padre=None,
        raiz_id=0
    )
    assert fig.id == 1
    assert fig.generacion == 0
    assert fig.padre is None
    assert fig.hijos == []
    assert fig.inspeccion == "null"
    assert fig.coste_individual == 0.0
    assert fig.coste_acumulado == 0.0


def test_triangulacion_recursion_parada():
    cfg, mundo = _crear_mundo_con_robots(3)
    plan = InspeccionVoronoi(cfg)
    
    # Run static construction
    plan._construir_estatico(mundo)
    
    # 1. Number of Gen 0 figures equals number of robots
    assert len(plan._raices) == 3
    
    # 2. Number of Gen 1 figures for each root equals number of vertices in its Voronoi polygon (minus any landing on same cell)
    for r0 in plan._raices:
        # Una figura por arista del contorno geodesico Gen 0 (abanico C0-vj-vj+1);
        # las zonas concavas que el centroide no ve se rellenan con ear-clipping
        # residual y pueden anadir unos pocos triangulos mas (nunca duplicando el
        # abanico), y nunca menos de 3 (poligono minimo).
        assert 3 <= len(r0.hijos) <= 2 * len(r0.vertices) + 2
        
        # 3. Every child of Gen 0 must be Gen 1
        for h in r0.hijos:
            assert h.generacion == 1
            assert h.padre == r0
            # Centroid must NOT be in a wall
            assert mundo.grid[h.celda[0], h.celda[1]] == 0
            # Centroid must not repeat parent cell
            assert h.celda != r0.celda

    # 4. Check all figures in the tree
    for fig in plan._todas_figuras:
        if fig.padre is not None:
            assert fig.generacion == fig.padre.generacion + 1
            assert fig.celda != fig.padre.celda
            assert mundo.grid[fig.celda[0], fig.celda[1]] == 0


def test_subasta_barrera_local_independiente():
    cfg, mundo = _crear_mundo_con_robots(3)
    plan = InspeccionVoronoi(cfg)
    plan._construir_estatico(mundo)
    
    # Initially, all roots should have active generation 0
    g_act = plan.generaciones_activas(mundo)
    assert all(g == 0 for g in g_act.values())
    
    # If all Gen 0 tasks of root 0 are inspected:
    r0_tasks = [t for t in plan._tareas if t["raiz_id"] == 0 and t["nivel"] == 0]
    for t in r0_tasks:
        mundo.inspec[t["r"], t["c"]] = 1
        mundo.resultado[t["r"], t["c"]] = SANA
    
    g_act_after = plan.generaciones_activas(mundo)
    # Root 0 should advance to generation 1
    assert g_act_after[0] == 1
    # Other roots should still remain at generation 0!
    assert g_act_after[1] == 0
    assert g_act_after[2] == 0


def test_propagacion_costes_defecto():
    cfg, mundo = _crear_mundo_con_robots(3)
    plan = InspeccionVoronoi(cfg)
    plan._construir_estatico(mundo)
    
    # Pick a root with children
    r0 = plan._raices[0]
    child = r0.hijos[0]
    costo_previo = child.coste_acumulado
    
    # Mark r0 as inspected and DEFECTUOSA
    mundo.inspec[r0.celda[0], r0.celda[1]] = 1
    mundo.resultado[r0.celda[0], r0.celda[1]] = DEFECTUOSA
    
    plan._aplicar_resultados(mundo)
    
    # Child's coste_acumulado should be reduced (cheaper / more urgent)
    assert child.coste_acumulado == pytest.approx(costo_previo * 0.85, rel=1e-5)
    assert r0.inspeccion == "defectuosa"


def _mundo_dummy(grid, gs, rows, cols):
    m = type("M", (), {})()
    m.grid = grid
    m.rows = rows
    m.cols = cols
    return m


def test_centroid_poligono():
    sq = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    cx, cy = _centroid_poligono(sq)
    assert (cx, cy) == pytest.approx((150.0, 150.0))

    tri = [(0.0, 0.0), (0.0, 300.0), (300.0, 0.0)]
    cx, cy = _centroid_poligono(tri)
    assert (cx, cy) == pytest.approx((100.0, 100.0))

    # Degenerado: todos los vertices colineales -> promedio (sin division por cero)
    line = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    cx, cy = _centroid_poligono(line)
    assert (cx, cy) == pytest.approx((10.0, 0.0))

    # El centroide de una figura debe quedar dentro de sus limites
    asim = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 800.0)]
    cx, cy = _centroid_poligono(asim)
    assert 0.0 <= cx <= 400.0


def test_colapsar_escaleras_diagonal_pura():
    gs = 10
    rows = cols = 20
    grid = np.zeros((rows, cols), dtype=int)
    m = _mundo_dummy(grid, gs, rows, cols)

    loop = [
        (0, 0), (10, 0), (10, 10), (20, 10), (20, 20), (30, 20), (30, 30),
        (30, 40), (100, 40), (100, 100), (0, 100), (0, 0),
    ]
    out = _colapsar_escaleras(loop, m, gs)
    # Toda la escalera SE (0,0)->(30,40) sin obstaculos se reemplaza por una
    # sola diagonal del vertice canonico (0,0) al final del bloque.
    assert (0, 0) in out and (30, 40) in out
    assert (10, 0) not in out and (10, 10) not in out and (20, 10) not in out
    assert (20, 20) not in out and (30, 20) not in out and (30, 30) not in out
    # Solo deben quedar los 2 vertices del tramo oblicuo mas el resto del bucle
    assert len(out) == 5


def test_colapsar_escaleras_respeta_paredes():
    gs = 10
    rows = cols = 20
    grid = np.zeros((rows, cols), dtype=int)
    grid[1, 1] = 1              # pared en el camino de la diagonal
    m = _mundo_dummy(grid, gs, rows, cols)

    loop = [
        (0, 0), (10, 0), (10, 10), (20, 10), (20, 20), (30, 20), (30, 30),
        (30, 40), (100, 40), (100, 100), (0, 100), (0, 0),
    ]
    out = _colapsar_escaleras(loop, m, gs)
    # La diagonal plena (0,0)->(30,40) corta la pared en (1,1): no se colapsa.
    # Solo se colapsa la ventana segura anclada al inicio canonico (0,0)->(10,10);
    # el resto de la escalera se conserva (y (30,30) queda fusionado con la
    # cola vertical colineal (30,20)->(30,40)).
    assert (0, 0) in out and (10, 10) in out     # ventana segura colapsada
    assert (30, 40) in out
    assert (10, 0) not in out
    assert (20, 10) in out and (20, 20) in out and (30, 20) in out
    assert (30, 30) not in out                   # fusionado colineal
    assert len(out) == 9


def test_colapsar_no_altera_poligono_recto():
    gs = 10
    rows = cols = 10
    grid = np.zeros((rows, cols), dtype=int)
    m = _mundo_dummy(grid, gs, rows, cols)

    loop = [(10, 10), (90, 10), (90, 90), (10, 90)]
    assert _colapsar_escaleras(loop, m, gs) == loop