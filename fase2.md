Actúa como experto en algoritmos de planificación de cobertura (Coverage Path Planning). 

Tu tarea es implementar la "Fase 2" en nuestro Simulador Swarm 2D: integrar el método Boustrophedon y una estrategia Híbrida con Voronoi.

**Tareas a realizar:**
1. **Estrategia: Boustrophedon (1E-H y 2E):**
   - Implementa un planificador Boustrophedon (barrido en zigzag). 
   - Restricción de ceguera: El algoritmo SOLO conoce las 4 esquinas exteriores del mapa, no las paredes internas. 
   - Debe reaccionar usando un "Hybrid Frontier": cuando detecta un obstáculo no previsto (pared interna) mediante sus sensores (línea de vista), recalcula dinámicamente el zigzag para bordearlo y continuar.

2. **Estrategia: Voronoi + Boustrophedon + Hybrid Frontier (1E-H):**
   - Implementa un planificador central que haga una partición del mapa descubierto usando Diagramas de Voronoi.
   - Cada región de Voronoi se asigna a un robot.
   - Dentro de su región, el robot realiza el patrón Boustrophedon.
   - Cuando el sensor fronterizo de un robot detecta que su región está cubierta o bloqueada por una pared nueva, el planificador global recalcula los centroides y emite una nueva partición de Voronoi.

**Instrucción Final:**
En el archivo `CHANGELOG_TESIS.md`, explica detalladamente la matemática detrás de la partición de Voronoi dinámica en mapas parcialmente conocidos, y cómo el algoritmo Boustrophedon adapta sus waypoints al encontrar obstáculos.