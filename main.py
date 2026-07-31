import sys
import tkinter as tk
import pygame

sys.path.insert(0, "src")

from src.core.config import Config
from src.core.simulacion import Simulacion
from src.gui.stats_window import StatsWindow


def main():
    cfg = Config()
    cfg.cargar_desde_archivo("sim_config.txt")
    sim = Simulacion(cfg)

    root = tk.Tk()
    root.withdraw()
    stats = StatsWindow(root, sim)

    reloj = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                sim.manejar_evento(event)

        if sim.estado == "running":
            sim.step()

        sim.render()

        sim.fps_actual = reloj.get_fps()
        stats.actualizar()
        root.update()

        reloj.tick(60)

    pygame.quit()
    root.destroy()


if __name__ == "__main__":
    main()