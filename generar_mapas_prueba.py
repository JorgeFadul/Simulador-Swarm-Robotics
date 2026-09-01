"""Genera (o regenera) los 5 mapas de prueba de la Fase 1 en `mapas/`.

Es un acceso directo a `main.py --generar-mapas`, util para (re)crear los
escenarios sin lanzar la simulacion. Cada mapa es un YAML reproducible con
paredes internas y areas afectadas (blobs) generadas algorithmicamente.

Uso:
    python generar_mapas_prueba.py
    python generar_mapas_prueba.py --semilla <n>   # semilla base opcional
"""

import argparse
import sys

sys.path.insert(0, "src")

from src.utils.mapas_prueba import asegurar_mapas_prueba  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semilla", type=int, default=None,
                        help="Semilla base para los blobs (opcional).")
    args = parser.parse_args()

    rutas = asegurar_mapas_prueba()
    for ruta in rutas:
        print(ruta)


if __name__ == "__main__":
    main()
