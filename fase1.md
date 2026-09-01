Actúa como un Ingeniero de Software experto en robótica y Pygame. Estamos ampliando un Simulador Swarm 2D. 

Tu tarea es implementar la "Fase 1": la arquitectura de Dos Etapas (Exploración -> Inspección), la capa de "áreas afectadas" oculta y los mapas de prueba.

**Tareas a realizar:**
1. **Etiquetas de Estrategia:**
   - Modifica las estrategias existentes y la UI para que cada estrategia incluya un tag visual indicando si es de "1 Etapa Híbrida (1E-H)" o "2 Etapas (2E)".

2. **Transición Global a Inspección:**
   - Para estrategias 2E, añade la lógica de transición. El paso de Exploración a Inspección debe ser global para todo el enjambre.
   - Triggers de transición: Cuando el `porcentaje_descubierto` de celdas exploradas llegue al 99%, O cuando pase más de 60 segundos simulados sin que ninguna celda nueva sea "scouteada".
   - Añade un botón en la UI de Tkinter para forzar el paso a la Etapa de Inspección manualmente (solo habilitado si la estrategia es 2E).

3. **Capa Oculta de Áreas Afectadas (Defectos):**
   - Crea una estructura de datos acoplada a la grilla para representar el estado real de las baldosas (Sana / Desprendida). Esta capa es invisible para los robots.
   - Los robots solo conocerán el estado de una celda específica cuando ejecuten la prueba de percusión sobre ella (asume 100% de efectividad en la detección por ahora, pero deja preparada la estructura para inyectar ruido gaussiano futuro).
   - Añade un toggle en la UI (pestaña Controles) para que el operador humano pueda ver visualmente estas áreas afectadas (diferenciadas por color).

4. **Generación de 5 Mapas de Prueba:**
   - Escribe un script generador paramétrico o funciones predefinidas que creen 5 mapas de prueba al iniciar. 
   - Los mapas deben tener paredes internas (no cerradas) y áreas afectadas generadas algorítmicamente en forma de "blobs" o manchas irregulares. Las áreas afectadas no deben solaparse con las paredes ni entre sí.
   - Los robots no deben dejar casillas vacías en los bordes de las paredes al marcar las zonas como visitadas. Mejora la función de rasterización o visita para solucionar los huecos cerca de las paredes.

**Instrucción Final:**
Documenta la lógica de la transición de estados, el algoritmo generador de blobs (manchas irregulares) y cómo los robots interactúan con la capa oculta en el archivo `CHANGELOG_TESIS.md`.