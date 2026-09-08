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
    _subdividir_figura
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
        assert len(r0.hijos) <= len(r0.vertices)
        assert len(r0.hijos) >= 3
        # Must not be 40 or 80!
        assert len(r0.hijos) <= 8
        
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