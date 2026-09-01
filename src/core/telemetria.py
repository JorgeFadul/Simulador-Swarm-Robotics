"""Telemetria para el modo headless (CLI).

Streaming en vivo: una linea JSON por paso hacia stdout (JSONL) con el
estado de cada robot (posiciones, bateria y tarea) para que un agente
externo (IA, pipeline de datos) pueda leerlo en tiempo real.

Al finalizar la corrida se exportan dos archivos:
  - resumen JSON con las metricas globales del enjambre;
  - serie CSV (una fila por paso) con cobertura y energia agregada,
    lista para graficar en la tesis.
"""

import csv
import json
import math

ESTADOS_ROBOT = {0: "STOPPED", 1: "ROTATING", 2: "MOVING"}


def _tarea(robot):
    if robot.goal_mode and robot.goal is not None and not robot.goal_alcanzado:
        return "objetivo:({0:.1f},{1:.1f})".format(*robot.goal)
    if robot.timed_out:
        return "timeout"
    return "ninguna"


def estado_robot(idx, robot, ppcm):
    return {
        "id": idx,
        "x": round(robot.x, 1),
        "y": round(robot.y, 1),
        "angulo_deg": round(math.degrees(robot.angle), 1),
        "velocidad_cm_s": round(robot.speed / ppcm, 2) if ppcm else round(robot.speed, 2),
        "estado": ESTADOS_ROBOT.get(robot.state, "DESCONOCIDO"),
        "tarea": _tarea(robot),
        "energia_consumida_j": round(robot.energia.energia_consumida_total, 3),
        "bateria_restante_j": round(robot.energia.bateria_restante, 3),
        "bateria_pct": round(robot.energia.porcentaje_bateria, 2),
        "celdas_vistas": robot.scouteadas_robot,
        "visitadas": robot.visitadas_robot,
        "inspeccionadas": robot.inspeccionadas_robot,
        "defectuosas": robot.defectuosas_robot,
    }


def _conteos_generales(m):
    libres = m.total_libres
    vistas = m._scouteadas + m._visitadas
    return {
        "vistas": vistas,
        "vistas_pct": round(100.0 * vistas / libres, 2) if libres else 0.0,
        "visitadas": m._visitadas,
        "visitadas_pct": round(100.0 * m._visitadas / libres, 2) if libres else 0.0,
        "inspeccionadas": m._inspeccionadas,
        "inspeccionadas_pct": round(100.0 * m._inspeccionadas / libres, 2) if libres else 0.0,
        "defectuosas": m._defectuosas,
        "defectuosas_pct": round(100.0 * m._defectuosas / m._inspeccionadas, 2)
        if m._inspeccionadas else 0.0,
    }


def estado_simulacion(sim):
    m = sim.mundo
    robots = m.robots
    total = m.rows * m.cols
    libres = m.total_libres
    paredes = m.total_paredes
    v = m._visitadas
    s = m._scouteadas
    d = m._paredes_vistas
    energia = sum(r.energia.energia_consumida_total for r in robots)
    ppcm = m.cfg.pixels_per_cm
    return {
        "t": round(sim.tiempo_simulado, 3),
        "paso": sim.pasos_totales,
        "estado": sim.estado,
        "estrategia": sim.cfg._estrategia_activa(),
        "modo_fase": sim.cfg.modo_fase,
        "cobertura": {
            "mapa_pct": round(100.0 * (v + s + d) / total, 2) if total else 0.0,
            "libres_pct": round(100.0 * (v + s) / libres, 2) if libres else 0.0,
            "paredes_pct": round(100.0 * d / paredes, 2) if paredes else 0.0,
        },
        "energia_total_j": round(energia, 3),
        "bateria_min_pct": round(min((r.energia.porcentaje_bateria for r in robots), default=0.0), 2),
        "celdas": _conteos_generales(m),
        "por_robot": [dict(_conteos_robot(r), id=i) for i, r in enumerate(robots)],
        "estado_celdas": m.estado_celdas(),
        "robots": [estado_robot(i, r, ppcm) for i, r in enumerate(robots)],
    }


def _conteos_robot(robot):
    return {
        "vistas": robot.scouteadas_robot,
        "visitadas": robot.visitadas_robot,
        "inspeccionadas": robot.inspeccionadas_robot,
        "defectuosas": robot.defectuosas_robot,
    }


def fila_serie(sim):
    m = sim.mundo
    total = m.rows * m.cols
    libres = m.total_libres
    v = m._visitadas
    s = m._scouteadas
    d = m._paredes_vistas
    energia = sum(r.energia.energia_consumida_total for r in m.robots)
    fila = {
        "paso": sim.pasos_totales,
        "t_s": round(sim.tiempo_simulado, 3),
        "cobertura_mapa_pct": round(100.0 * (v + s + d) / total, 3) if total else 0.0,
        "cobertura_libres_pct": round(100.0 * (v + s) / libres, 3) if libres else 0.0,
        "energia_total_j": round(energia, 3),
        "celdas_vistas": s + v,
        "celdas_visitadas": v,
        "celdas_inspeccionadas": m._inspeccionadas,
        "celdas_defectuosas": m._defectuosas,
    }
    return fila


def construir_resumen(sim, semilla=None, estrategia=None, resultado=None):
    m = sim.mundo
    robots = m.robots
    total = m.rows * m.cols
    libres = m.total_libres
    paredes = m.total_paredes
    v = m._visitadas
    s = m._scouteadas
    d = m._paredes_vistas
    ppcm = m.cfg.pixels_per_cm
    energia_total = sum(r.energia.energia_consumida_total for r in robots)
    return {
        "estrategia": estrategia or sim.cfg._estrategia_activa(),
        "semilla": semilla,
        "resultado": resultado or sim.estado,
        "tiempo_simulado_s": round(sim.tiempo_simulado, 3),
        "pasos": sim.pasos_totales,
        "tiempo_final_s": round(sim.tiempo_final, 3) if sim.tiempo_final is not None else None,
        "grid": {"filas": m.rows, "columnas": m.cols, "total": total,
                 "libres": libres, "paredes": paredes},
        "cobertura": {
            "libres_pct": round(100.0 * (v + s) / libres, 2) if libres else 0.0,
            "paredes_pct": round(100.0 * d / paredes, 2) if paredes else 0.0,
            "mapa_pct": round(100.0 * (v + s + d) / total, 2) if total else 0.0,
            "visitadas": v,
            "scouteadas": s,
            "paredes_descubiertas": d,
        },
        "celdas": _conteos_generales(m),
        "energia": {
            "total_j": round(energia_total, 3),
            "movimiento_j": round(sum(r.energia.consumida_movimiento for r in robots), 3),
            "inspeccion_j": round(sum(r.energia.consumida_inspeccion for r in robots), 3),
            "picos_arranque_total": sum(r.energia.picos_arranque for r in robots),
        },
        "por_robot": [dict(_conteos_robot(r), id=i) for i, r in enumerate(robots)],
        "robots": [estado_robot(i, r, ppcm) for i, r in enumerate(robots)],
        "escenario_original": m.mapa_inicial,
        "estado_final_celdas": m.estado_celdas(),
        "grilla_final": m.inspec.tolist(),
    }


def escribir_resumen_json(path, resumen):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


def escribir_serie_csv(path, filas):
    if not filas:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)