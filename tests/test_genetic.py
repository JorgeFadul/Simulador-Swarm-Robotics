"""Pruebas unitarias del meta-optimizador GA NSGA-II (Parte A)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.genetic import (  # noqa: E402
    GEN_BOUNDS,
    Individual,
    NSGA2Optimizer,
)


def _en_bounds(g):
    for v, (lo, hi) in zip(g, GEN_BOUNDS):
        assert lo - 1e-9 <= v <= hi + 1e-9, f"{v} fuera de [{lo},{hi}]"


def test_sbx_respeta_bounds():
    opt = NSGA2Optimizer(seed=1)
    a = Individual([b[0] for b in GEN_BOUNDS])
    b = Individual([b[1] for b in GEN_BOUNDS])
    for _ in range(50):
        c1, c2 = opt.crossover_sbx(a, b)
        _en_bounds(c1.genome)
        _en_bounds(c2.genome)


def test_mutacion_respeta_bounds():
    opt = NSGA2Optimizer(seed=2)
    a = Individual([ (lo + hi) / 2 for lo, hi in GEN_BOUNDS])
    for _ in range(50):
        _en_bounds(opt.mutate_polynomial(a).genome)


def test_non_dominated_sort():
    # f1,f2 minimizacion: p0 y p1 no dominados, p2 dominado por ambos.
    p0 = Individual([ (lo + hi) / 2 for lo, hi in GEN_BOUNDS])
    p1 = Individual([ (lo + hi) / 2 for lo, hi in GEN_BOUNDS])
    p2 = Individual([ (lo + hi) / 2 for lo, hi in GEN_BOUNDS])
    p0.objectives = (1.0, 2.0, 0.0)
    p1.objectives = (2.0, 1.0, 0.0)
    p2.objectives = (2.0, 2.0, 1.0)
    fronts = NSGA2Optimizer.fast_non_dominated_sort([p0, p1, p2])
    assert len(fronts[0]) == 2
    assert p2.rank == 1


def test_crowding_extremos_inf():
    pop = []
    for o in [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 0.5, 1.5)]:
        ind = Individual([(lo + hi) / 2 for lo, hi in GEN_BOUNDS])
        ind.objectives = o
        pop.append(ind)
    NSGA2Optimizer.calculate_crowding_distance(pop)
    assert sum(1 for i in pop if i.crowding_distance == float("inf")) >= 2


def test_aplicar_genoma_mapea_8_genes():
    from src.core.config import Config, aplicar_genoma
    cfg = Config()
    g = np.array([0.05, 2.0, 5.0, 1.0, 3.0, 2.0, 0.4, 4.0])
    aplicar_genoma(cfg, g)
    assert cfg.sample_cost_weight == 0.05
    assert cfg.ga_w_info == 2.0
    assert cfg.ga_w_defect == 5.0
    assert cfg.inspeccion_peso_distancia == 1.0
    assert cfg.ga_beta_repulsion == 3.0
    assert cfg.inspeccion_factor_defectivo == 2.0
    assert cfg.inspeccion_factor_sano == 0.4
    assert cfg.ga_apf_repulsion == 4.0


def test_lhs_cubre_rangos():
    opt = NSGA2Optimizer(seed=3)
    pop = opt.initialize_population(20)
    assert len(pop) == 20
    arr = np.array([i.genome for i in pop])
    for j, (lo, hi) in enumerate(GEN_BOUNDS):
        assert arr[:, j].min() >= lo - 1e-9
        assert arr[:, j].max() <= hi + 1e-9
