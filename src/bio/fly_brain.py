"""Modelo computacional bioinspirado del cerebro de Drosophila melanogaster.

Modela dos subsistemas clave del conectoma del insecto:
1. Mushroom Body (MB): Red neuronal con proyeccion esparsa (Kenyon Cells, k-WTA),
   plasticidad sinaptica modulada por Dopamina (Three-Factor Learning Rule) y neuronas
   de salida (MBONs) para atraccion/evitacion.
2. Central Complex (CX): Red de atractor en anillo (Ring Attractor) para integracion de rumbo,
   rumbo preferente y coordinacion motriz.
"""

import math
import numpy as np


class MushroomBody:
    """Circuito Mushroom Body con plasticidad sinaptica por dopamina."""

    def __init__(self, n_inputs: int = 240, n_kc: int = 1000, n_mbon: int = 8,
                 sparsity_k: float = 0.05, lr: float = 0.02, decay: float = 0.0005,
                 rng: np.random.Generator = None):
        self.n_inputs = n_inputs
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.k_active = max(1, int(n_kc * sparsity_k))  # Solo 5% de KCs activas (biologia)
        self.lr = lr
        self.decay = decay
        self.rng = rng or np.random.default_rng(42)

        # Proyeccion PN -> KC: grado de entrada bajo (~5-8 inputs por cada KC como en la mosca)
        self.W_pn_kc = np.zeros((n_inputs, n_kc), dtype=np.float32)
        for j in range(n_kc):
            selected_inputs = self.rng.choice(n_inputs, size=min(7, n_inputs), replace=False)
            self.W_pn_kc[selected_inputs, j] = self.rng.uniform(0.5, 1.0, size=len(selected_inputs))

        # Sinapsis plasticas KC -> MBON: moduladas por dopamina
        self.W_kc_mbon = self.rng.uniform(0.1, 0.3, size=(n_kc, n_mbon)).astype(np.float32)

        self.last_kc_act = np.zeros(n_kc, dtype=np.float32)
        self.last_mbon_act = np.zeros(n_mbon, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propaga el vector sensorial hacia KCs (k-WTA esparsa) y hacia MBONs."""
        # 1. Activacion cruda en KCs
        kc_raw = np.dot(x, self.W_pn_kc)

        # 2. Inhibicion lateral tipo APL (k-Winner-Take-All)
        kc_act = np.zeros(self.n_kc, dtype=np.float32)
        if self.k_active > 0:
            top_k_idx = np.argpartition(kc_raw, -self.k_active)[-self.k_active:]
            # Solo los ganadores disparan con magnitud normalizada
            val_ganadores = np.maximum(0.0, kc_raw[top_k_idx])
            norm = np.linalg.norm(val_ganadores) + 1e-6
            kc_act[top_k_idx] = val_ganadores / norm
        self.last_kc_act = kc_act

        # 3. Respuesta en MBONs (valencia / atraccion motriz)
        mbon_act = np.dot(kc_act, self.W_kc_mbon)
        self.last_mbon_act = mbon_act
        return mbon_act

    def aplicar_dopamina(self, dopamina: float):
        """Regla de tres factores: actualiza las sinapsis KC->MBON segun la dopamina.

        Delta W_ij = lr * KC_i * MBON_j * Dopamina - decay * W_ij
        """
        # Outer product de activaciones
        coactivacion = np.outer(self.last_kc_act, self.last_mbon_act)
        delta_w = self.lr * coactivacion * float(dopamina)

        # Actualizar pesos manteniendo rango no negativo acotado
        self.W_kc_mbon += delta_w - (self.decay * self.W_kc_mbon)
        self.W_kc_mbon = np.clip(self.W_kc_mbon, 0.0, 2.5)


class CentralComplex:
    """Circuito del Complejo Central (CX) tipo Atractor en Anillo para orientacion."""

    def __init__(self, n_cuñas: int = 8):
        self.n_cuñas = n_cuñas
        self.angulos = np.linspace(0, 2 * math.pi, n_cuñas, endpoint=False)
        self.rumbo_actual = 0.0  # Heading interno

    def step(self, mbon_valencias: np.ndarray, heading_sensor: float,
             velocidad_angular: float, dt: float) -> tuple[float, float]:
        """Calcula el angulo deseado combinando el heading con la valencia de MBONs.

        Retorna (target_angle, target_speed_factor).
        """
        # Integracion de rumbo (path integration en anillo)
        self.rumbo_actual = (heading_sensor + velocidad_angular * dt) % (2 * math.pi)

        # Los MBONs representan 8 direcciones cardinales relativas al robot
        # mbon_valencias > 0 indican apetencia hacia ese rumbo
        if len(mbon_valencias) == self.n_cuñas:
            # Softmax con temperatura
            exp_v = np.exp(np.clip(mbon_valencias * 2.0, -10.0, 10.0))
            probs = exp_v / (np.sum(exp_v) + 1e-8)

            # Vector resultante en el plano circular
            vx = np.sum(probs * np.cos(self.angulos))
            vy = np.sum(probs * np.sin(self.angulos))

            angulo_deseado = (math.atan2(vy, vx) + self.rumbo_actual) % (2 * math.pi)
            magnitud = math.hypot(vx, vy)
            speed_factor = float(np.clip(magnitud * 1.5, 0.4, 1.0))
            return angulo_deseado, speed_factor

        return self.rumbo_actual, 0.8


class FlyBrain:
    """Cerebro integrado de Drosophila: Mushroom Body (MB) + Central Complex (CX)."""

    def __init__(self, n_inputs: int = 240, n_kc: int = 1000, n_direcciones: int = 8,
                 rng: np.random.Generator = None):
        self.mb = MushroomBody(n_inputs=n_inputs, n_kc=n_kc, n_mbon=n_direcciones, rng=rng)
        self.cx = CentralComplex(n_cuñas=n_direcciones)
        self.dopamina_actual = 0.0
        self.dopamina_acumulada = 0.0
        self.historico_dopamina = []

    def procesar(self, vector_sensorial: np.ndarray, heading_robot: float,
                 omega_robot: float, dt: float) -> tuple[float, float]:
        """Evalua la percepcion sensorial y emite comandos de rumbo y avance."""
        valencias = self.mb.forward(vector_sensorial)
        target_angle, speed_factor = self.cx.step(valencias, heading_robot, omega_robot, dt)
        return target_angle, speed_factor

    def inyectar_recompensa(self, reward: float):
        """Aplica un pulso de dopamina (+ o -) y actualiza la plasticidad del cerebro."""
        self.dopamina_actual = float(reward)
        self.dopamina_acumulada += self.dopamina_actual
        self.historico_dopamina.append(self.dopamina_actual)
        if len(self.historico_dopamina) > 200:
            self.historico_dopamina.pop(0)

        # Modulacion de sinapsis por dopamina
        self.mb.aplicar_dopamina(self.dopamina_actual)
