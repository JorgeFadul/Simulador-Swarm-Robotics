"""Generador de los 5 mapas de prueba de la Fase 1.

Cada mapa es una escena reproducible que combina:
  * paredes internas (no cerradas) definidas a mano por escenario;
  * areas afectadas (defectos) generadas en `Mundo._inicializar_defectos`
    por recursion probabilistica de blobs (semillas repartidas por el suelo
    con separacion minima, que crecen con probabilidades decrecientes).

El resultado se escribe como YAML (`mapas/mapa_test_<n>.yaml`) que
`cargar_mapa` sabe leer (map_width, map_height, grid_size, num_robots,
walls, defecto_num_blobs). La forma y tamano de los blobs se resuelven en
tiempo de corrida a partir de `defecto_num_blobs`; aqui solo se fija cuantas
semillas usa cada escenario.
"""

import os

import yaml

from ..core.config import Config

MAP_ANCHO = 400
MAP_ALTO = 400
GRID = 20
NUM_ROBOTS = 3

# Escenarios: paredes internas (no cerradas). Cada pared = {x, y, width, height}
# en px. `n_blobs` fija cuantas semillas de defecto usa cada escenario.
_ESCENARIOS = [
    {  # mapa_test_1: corredor central vertical
        "nombre": "Corredor Central",
        "descripcion": "Pocas paredes: dos bloques laterales crean un corredor central.",
        "walls": [
            {"x": 120, "y": 130, "width": 24, "height": 160, "angle": 0},
            {"x": 120, "y": 300, "width": 24, "height": 160, "angle": 0},
            {"x": 280, "y": 130, "width": 24, "height": 160, "angle": 0},
            {"x": 280, "y": 300, "width": 24, "height": 160, "angle": 0},
        ],
        "n_blobs": 3,
    },
    {  # mapa_test_2: columnas en retícula
        "nombre": "Columnas en Reticula",
        "descripcion": "Columnas internas distribuidas; manchas entre columnas.",
        "walls": [
            {"x": 80, "y": 80, "width": 24, "height": 24, "angle": 0},
            {"x": 200, "y": 80, "width": 24, "height": 24, "angle": 0},
            {"x": 320, "y": 80, "width": 24, "height": 24, "angle": 0},
            {"x": 80, "y": 200, "width": 24, "height": 24, "angle": 0},
            {"x": 320, "y": 200, "width": 24, "height": 24, "angle": 0},
            {"x": 80, "y": 320, "width": 24, "height": 24, "angle": 0},
            {"x": 200, "y": 320, "width": 24, "height": 24, "angle": 0},
            {"x": 320, "y": 320, "width": 24, "height": 24, "angle": 0},
        ],
        "n_blobs": 4,
    },
    {  # mapa_test_3: laberinto abierto (paredes no cerradas)
        "nombre": "Laberinto Abierto",
        "descripcion": "Pasillos sin cerrar que obligan a bordear las paredes.",
        "walls": [
            {"x": 100, "y": 100, "width": 200, "height": 22, "angle": 0},
            {"x": 100, "y": 290, "width": 200, "height": 22, "angle": 0},
            {"x": 210, "y": 195, "width": 22, "height": 200, "angle": 0},
            {"x": 70, "y": 195, "width": 22, "height": 110, "angle": 0},
        ],
        "n_blobs": 5,
    },
    {  # mapa_test_4: particiones diagonales (paredes con angulo)
        "nombre": "Particiones Diagonal",
        "descripcion": "Paredes inclinadas que crean pasillos diagonales.",
        "walls": [
            {"x": 120, "y": 120, "width": 22, "height": 140, "angle": 35},
            {"x": 280, "y": 120, "width": 22, "height": 140, "angle": -35},
            {"x": 120, "y": 290, "width": 22, "height": 140, "angle": -35},
            {"x": 280, "y": 290, "width": 22, "height": 140, "angle": 35},
        ],
        "n_blobs": 4,
    },
    {  # mapa_test_5: salones abiertos (sin cerrar)
        "nombre": "Salones Abiertos",
        "descripcion": "Paredes internas aisladas tipo estanterias/divisores.",
        "walls": [
            {"x": 160, "y": 100, "width": 26, "height": 90, "angle": 0},
            {"x": 245, "y": 100, "width": 26, "height": 90, "angle": 0},
            {"x": 160, "y": 310, "width": 26, "height": 90, "angle": 0},
            {"x": 245, "y": 310, "width": 26, "height": 90, "angle": 0},
            {"x": 80, "y": 200, "width": 90, "height": 26, "angle": 0},
            {"x": 320, "y": 200, "width": 90, "height": 26, "angle": 0},
        ],
        "n_blobs": 3,
    },
]


def construir_mapa_prueba(idx, cfg=None):
    """Devuelve el dict de configuracion YAML del mapa de prueba idx (1..5)."""
    esc = _ESCENARIOS[idx - 1]
    return {
        "map_width": MAP_ANCHO,
        "map_height": MAP_ALTO,
        "grid_size": GRID,
        "num_robots": NUM_ROBOTS,
        "nombre": esc["nombre"],
        "descripcion": esc["descripcion"],
        "walls": esc["walls"],
        "defecto_num_blobs": esc["n_blobs"],
    }


MAPAS_PRULEBA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "mapas",
)


def asegurar_mapas_prueba(cfg=None):
    """Genera los 5 mapas de prueba en `mapas/mapa_test_<n>.yaml` si aun no
    existen (idempotente). Devuelve la lista de rutas creadas/generadas."""
    cfg = cfg or Config()
    os.makedirs(MAPAS_PRULEBA_DIR, exist_ok=True)
    rutas = []
    for i in range(1, 6):
        ruta = os.path.join(MAPAS_PRULEBA_DIR, f"mapa_test_{i}.yaml")
        rutas.append(ruta)
        if os.path.exists(ruta):
            continue
        mapa = construir_mapa_prueba(i, cfg)
        with open(ruta, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapa, f, allow_unicode=True, sort_keys=False)
    return rutas


if __name__ == "__main__":
    rutas = asegurar_mapas_prueba()
    for ruta in rutas:
        print(f"generado: {ruta}")
