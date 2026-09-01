"""Modelo de energia del chasis del robot (plataforma tipo HeRo Swarm).

Calculo generico para llantas: la potencia mecanica entregada por los
motores es P_mech = tau * omega, con tau el torque en las ruedas y omega
la velocidad angular de la llanta. Se agregan tres terminos:

  1) Resistencia a la rodadura:  F_rod = mu_r * m * g   (constante)
  2) Fuerza de aceleracion:      F_acel = m * a          (solo si a > 0)
  3) Giro en el sitio: las ruedas contra-giran y disipan rodadura a la
     velocidad superficial de giro  v_turn = 0.5 * b * |omega_robot|,
     con b el ancho de trocha.

Ademas, cada vez que el robot pasa de v = 0 a v > 0 se suma un pico de
energia E_start por la inercia (energia cinetica de translacion + rotacion
de las ruedas) que hay que vencer para arrancar.

Fase 0: el robot NO muere por falta de bateria; solo se acumula el gasto
para telemetria. El corte por bateria se reserva para escenarios de
robustez posteriores (Capitulo 6 de la tesis).

Todas las cantidades estan en SI (J, W, kg, m, rad/s).
"""

import math

GRAVEDAD = 9.81  # m/s^2


class ConsumoEnergia:
    def __init__(self, cfg):
        self.cfg = cfg
        self.consumida_movimiento = 0.0
        self.consumida_inspeccion = 0.0
        self.picos_arranque = 0
        self.velocidad_anterior = 0.0
        self.ultimo_torque = 0.0
        self.ultima_potencia = 0.0
        self.ultimo_consumo = 0.0
        self._inercia_rueda = 0.5 * cfg.masa_rueda_kg * cfg.radio_rueda_m ** 2

    def reiniciar(self):
        self.consumida_movimiento = 0.0
        self.consumida_inspeccion = 0.0
        self.picos_arranque = 0
        self.velocidad_anterior = 0.0
        self.ultimo_torque = 0.0
        self.ultima_potencia = 0.0
        self.ultimo_consumo = 0.0

    @property
    def energia_consumida_total(self):
        return self.consumida_movimiento + self.consumida_inspeccion

    @property
    def bateria_restante(self):
        return max(0.0, self.cfg.bateria_capacidad_j - self.energia_consumida_total)

    @property
    def porcentaje_bateria(self):
        cap = self.cfg.bateria_capacidad_j
        if cap <= 0:
            return 0.0
        return 100.0 * self.bateria_restante / cap

    def pago_arranque(self, v):
        """Pico inercial al pasar de reposo a velocidad v (m/s).

        E_start = (0.5 * m * v^2 + n * 0.5 * I_rueda * (v / r_rueda)^2) / eta
        La energia cinetica (translacion + rotacion de ruedas) debe ser
        entregada por la fuente por encima de la marcha normal, afectada por
        la eficiencia de la transmision.
        """
        if v <= 0:
            return 0.0
        cfg = self.cfg
        n = cfg.num_ruedas_motor
        w_rueda = v / cfg.radio_rueda_m
        e = 0.5 * cfg.masa_kg * v * v + n * 0.5 * self._inercia_rueda * w_rueda * w_rueda
        return e / cfg.eficiencia_transmision

    def consumir_movimiento(self, dt, v, a, omega_robot):
        """Consume energia por movimiento en el intervalo dt.

        v  : velocidad lineal del robot (m/s, se usa |v|)
        a  : aceleracion lineal estimada (m/s^2)
        omega_robot : velocidad angular de giro del robot (rad/s)
        """
        cfg = self.cfg
        m = cfg.masa_kg
        rw = cfg.radio_rueda_m
        n = cfg.num_ruedas_motor

        v = abs(v)

        # Pico inercial de arranque (v pasa de 0 a >0)
        pico = 0.0
        if v > 0 and self.velocidad_anterior <= 0.0:
            pico = self.pago_arranque(v)
            self.picos_arranque += 1
        self.velocidad_anterior = v

        F_rod = cfg.coef_rodadura * m * GRAVEDAD
        F_acel = max(m * a, 0.0)
        F_t = F_rod + F_acel

        tau = (F_t / n) * rw if n > 0 else 0.0
        w_rueda = v / rw if rw > 0 else 0.0

        P_trac = F_t * v
        P_giro = F_rod * abs(omega_robot) * (cfg.ancho_trocha_m / 2.0)
        P_mech = P_trac + P_giro
        P_elec = P_mech / cfg.eficiencia_transmision

        dE = P_elec * dt + pico
        self.consumida_movimiento += dE
        self.ultimo_torque = tau
        self.ultima_potencia = P_elec
        self.ultimo_consumo = dE
        return dE

    def consumir_energia_inspeccion(self, veces=1):
        """Gasto fijo por ciclo de inspeccion (percusion + adquisicion + FFT).

        En fase 0 es una constante configurable (energia_inspeccion_j) que
        representa el costo de procesar la FFT y la percusion sobre cada
        baldosa inspeccionada.
        """
        dE = self.cfg.energia_inspeccion_j * veces
        self.consumida_inspeccion += dE
        self.ultimo_consumo = dE
        return dE