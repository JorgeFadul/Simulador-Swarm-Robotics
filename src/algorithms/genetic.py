"""Meta-algoritmo genetico NSGA-II (Parte A - Offline).

Genoma continuo D=8 (rangos del plan de tesis, prevalecen sobre default.yaml):

  g0 w_cost        [0.001, 0.20] def 0.02  -> sample_cost_weight
  g1 w_info        [0.1, 10.0]   def 1.0   -> ga_w_info (nuevo)
  g2 w_defect      [1.0, 15.0]   def 3.0   -> ga_w_defect (nuevo)
  g3 alpha_dist    [0.1, 2.5]    def 0.5   -> inspeccion_peso_distancia
  g4 beta_rep      [1.2, 4.0]    def 2.0   -> ga_beta_repulsion (nuevo)
  g5 k_defectuoso  [1.1, 3.0]    def 1.5   -> inspeccion_factor_defectivo
  g6 k_sano        [0.2, 0.9]    def 0.5   -> inspeccion_factor_sano
  g7 apf_repulsion [0.5, 8.0]    def 2.5   -> ga_apf_repulsion (nuevo)

Estrategia fija (no codificada en el genoma): exploracion "Hybrid Based"
+ inspeccion "Voronoi" en modo dos_etapas, mapa sin paredes internas.
"""

import numpy as np

GEN_NAMES = (
    "w_cost",
    "w_info",
    "w_defect",
    "alpha_dist",
    "beta_rep",
    "k_defectuoso",
    "k_sano",
    "apf_repulsion",
)

GEN_BOUNDS = (
    (0.001, 0.20),
    (0.1, 10.0),
    (1.0, 15.0),
    (0.1, 2.5),
    (1.2, 4.0),
    (1.1, 3.0),
    (0.2, 0.9),
    (0.5, 8.0),
)

GEN_DEFAULTS = (0.02, 1.0, 3.0, 0.5, 2.0, 1.5, 0.5, 2.5)

D = len(GEN_NAMES)

P_INFACTIBLE = 1000.0


def genome_to_dict(genome):
    """Convierte un vector D=8 a dict de hiperparametros nombrados."""
    g = np.asarray(genome, dtype=float).ravel()
    return {name: float(g[i]) for i, name in enumerate(GEN_NAMES)}


def dict_to_genome(d):
    """Convierte un dict nombrado a vector D=8 (usa defaults si falta)."""
    return np.array(
        [float(d.get(name, default))
         for name, default in zip(GEN_NAMES, GEN_DEFAULTS)],
        dtype=float,
    )


def clip_to_bounds(genome):
    """Proyecta un genoma a las restricciones de caja [L_i, U_i]."""
    g = np.asarray(genome, dtype=float).copy()
    for i, (lo, hi) in enumerate(GEN_BOUNDS):
        g[i] = min(hi, max(lo, g[i]))
    return g


class Individual:
    """Un individuo: genoma + objetivos + atributos NSGA-II."""

    __slots__ = ("genome", "objectives", "rank", "crowding_distance",
                 "n_eval_ok", "tasa_validez")

    def __init__(self, genome):
        self.genome = clip_to_bounds(np.asarray(genome, dtype=float))
        self.objectives = None  # (f1_energia_mwh, f2_tiempo_s, f3_error)
        self.rank = 0
        self.crowding_distance = 0.0
        self.n_eval_ok = 0
        self.tasa_validez = 0.0

    def copy(self):
        ind = Individual(self.genome.copy())
        ind.objectives = self.objectives
        ind.rank = self.rank
        ind.crowding_distance = self.crowding_distance
        ind.n_eval_ok = self.n_eval_ok
        ind.tasa_validez = self.tasa_validez
        return ind


def dominates(a, b):
    """True si a domina a b (minimizacion en todos los objetivos)."""
    oa, ob = a.objectives, b.objectives
    return (all(x <= y for x, y in zip(oa, ob))
            and any(x < y for x, y in zip(oa, ob)))


class NSGA2Optimizer:
    """Operadores NSGA-II: inicializacion LHS, sort, crowding, SBX, mutacion."""

    def __init__(self, eta_c=20.0, eta_m=20.0, p_crossover=0.9, seed=None):
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.p_crossover = float(p_crossover)
        self.rng = np.random.default_rng(seed)

    # -- Poblacion inicial (LHS) -------------------------------------------
    def initialize_population(self, pop_size):
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=D, seed=self.rng.integers(2 ** 31))
            unit = sampler.random(n=pop_size)
        except Exception:
            # Fallback sin scipy: LHS manual con numpy.
            unit = np.zeros((pop_size, D))
            for j in range(D):
                perm = self.rng.permutation(pop_size)
                unit[:, j] = (perm + self.rng.random(pop_size)) / pop_size
        pop = []
        for i in range(pop_size):
            g = np.array([
                lo + unit[i, j] * (hi - lo)
                for j, (lo, hi) in enumerate(GEN_BOUNDS)
            ])
            pop.append(Individual(g))
        return pop

    # -- Non-dominated sorting ----------------------------------------------
    @staticmethod
    def fast_non_dominated_sort(population):
        fronts = [[]]
        dom_count = {}
        dominated = {}
        for p in population:
            dominated[id(p)] = []
            dom_count[id(p)] = 0
            for q in population:
                if p is q:
                    continue
                if dominates(p, q):
                    dominated[id(p)].append(q)
                elif dominates(q, p):
                    dom_count[id(p)] += 1
            if dom_count[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in dominated[id(p)]:
                    dom_count[id(q)] -= 1
                    if dom_count[id(q)] == 0:
                        q.rank = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return [f for f in fronts if f]

    # -- Crowding distance ----------------------------------------------------
    @staticmethod
    def calculate_crowding_distance(front):
        n = len(front)
        if n == 0:
            return
        for ind in front:
            ind.crowding_distance = 0.0
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return
        m = len(front[0].objectives)
        for k in range(m):
            front.sort(key=lambda ind: ind.objectives[k])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            fmin = front[0].objectives[k]
            fmax = front[-1].objectives[k]
            denom = (fmax - fmin) if fmax > fmin else 1.0
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[k] - front[i - 1].objectives[k]
                ) / denom

    # -- Seleccion por torneo binario -----------------------------------------
    def tournament_selection(self, population):
        i, j = self.rng.integers(len(population), size=2)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        return a if a.crowding_distance >= b.crowding_distance else b

    # -- SBX --------------------------------------------------------------------
    def crossover_sbx(self, parent1, parent2):
        g1 = parent1.genome.copy()
        g2 = parent2.genome.copy()
        c1 = g1.copy()
        c2 = g2.copy()
        eta = self.eta_c
        for i, (lo, hi) in enumerate(GEN_BOUNDS):
            if self.rng.random() > self.p_crossover:
                continue
            x1, x2 = float(g1[i]), float(g2[i])
            if abs(x1 - x2) < 1e-12:
                continue
            if x1 > x2:
                x1, x2 = x2, x1
            u = self.rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            v1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
            v2 = 0.5 * ((x1 + x2) + beta * (x2 - x1))
            c1[i] = min(hi, max(lo, v1))
            c2[i] = min(hi, max(lo, v2))
        return Individual(c1), Individual(c2)

    # -- Mutacion polinomial ------------------------------------------------------
    def mutate_polynomial(self, individual):
        g = individual.genome.copy()
        eta = self.eta_m
        p_m = 1.0 / D
        for i, (lo, hi) in enumerate(GEN_BOUNDS):
            if self.rng.random() > p_m:
                continue
            x = float(g[i])
            delta1 = (x - lo) / (hi - lo) if hi > lo else 0.0
            delta2 = (hi - x) / (hi - lo) if hi > lo else 0.0
            u = self.rng.random()
            if u < 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta + 1.0)
                dq = val ** (1.0 / (eta + 1.0)) - 1.0
            else:
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (eta + 1.0)
                dq = 1.0 - val ** (1.0 / (eta + 1.0))
            g[i] = min(hi, max(lo, x + dq * (hi - lo)))
        return Individual(g)

    # -- Reemplazo elitista ---------------------------------------------------------
    def environmental_selection(self, combined, pop_size):
        fronts = self.fast_non_dominated_sort(combined)
        nueva = []
        for front in fronts:
            self.calculate_crowding_distance(front)
            if len(nueva) + len(front) <= pop_size:
                nueva.extend(front)
            else:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                nueva.extend(front[:pop_size - len(nueva)])
                break
        return nueva
