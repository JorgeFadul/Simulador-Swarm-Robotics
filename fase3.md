Actúa como experto en sistemas multi-agente, Task Allocation y geometría computacional. Esta es la "Fase 3", la parte algorítmica más compleja del Simulador Swarm 2D.

**Tareas a realizar (Solo activado en la Etapa de Inspección):**
1. **Colas de Tareas para Robots:**
   - Rediseña la estructura de `Robot` y `ExplorationPlanner` para que los robots puedan tener una "cola de tareas" (múltiples objetivos ordenados) en lugar de un solo objetivo, permitiendo re-priorización dinámica.

2. **Voronoi Recursivo usando Bordes de Distancia:**
   - Al entrar en la etapa de inspección, haz un *Distance Transform* desde las paredes descubiertas. Usa los máximos locales/esqueletos como semillas (puntos de interés) para generar el primer Diagrama de Voronoi.
   - Cuando un robot inspecciona el centroide de un Voronoi, el resultado altera el peso del área. Se debe aplicar recursión: el centroide se convierte en semilla junto a los vértices de la figura, creando sub-Voronois. 
   - Condición de parada de la recursión (evaluada por región independiente): cuando el área del sub-Voronoi sea de solo 1 casilla.

3. **Market Auction Method (Subasta):**
   - La asignación de estas regiones a los robots se hará por subasta mercantil.
   - Fórmula de Costo/Prioridad: Debe considerar el costo de A*/Dijkstra (estimando distancia esquivando obstáculos y velocidad del robot) más el tiempo estimado en resolver las tareas previas de su cola.
   - **Multiplicador por Inspección:** La función de costo de mercado de un sub-Voronoi debe ser multiplicativa basada en el padre. Propón e implementa una fórmula donde el costo del sub-Voronoi herede el peso de sus "Voronois padres", siendo proporcional a la distancia y la "paternidad". Si el padre detectó un defecto, la zona se vuelve más atractiva (menor costo); si no, menos atractiva. ¡Nunca debe ser 0!

**Instrucción Final:**
Documenta en `CHANGELOG_TESIS.md` la función de costo de mercado exacta propuesta, el algoritmo de Distance Transform utilizado para las semillas iniciales, y cómo se gestiona la propagación de pesos en el árbol recursivo de Voronoi.