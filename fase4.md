Actúa como experto en Machine Learning (Reinforcement Learning) y entornos Gymnasium.

Tu tarea es implementar la "Fase 4": la integración de estrategias RL en nuestro Simulador Swarm 2D (que ya soporta ejecución headless).

**Tareas a realizar:**
1. **Wrapper de Entorno:**
   - Crea un entorno custom compatible con la API de `gymnasium` que envuelva el simulador `mundo.py` y `simulacion.py`.
   - El espacio de observación debe incluir: mapa conocido (tensor), posiciones propias y de aliados, niveles de energía y cola de tareas.
   - El espacio de acción debe traducir las acciones de alto nivel (asignar objetivo) o bajo nivel (moverse/girar) dependiendo del enfoque de control.

2. **Funciones de Recompensa (Reward Functions):**
   - Diseña e implementa dos arquitecturas de recompensas separadas:
     a) Minimizar el tiempo de descubrimiento e inspección total.
     b) Minimizar el gasto de energía del enjambre (usando el modelo de energía previamente implementado).

3. **Pipeline de Entrenamiento:**
   - Utiliza `Stable-Baselines3` (PPO o SAC) para construir el pipeline de entrenamiento.
   - El script de entrenamiento debe poder ejecutarse en paralelo (multiprocessing de entornos) aprovechando el modo headless.
   - Configura el entorno para soportar una estrategia "1 Etapa Híbrida (1E-H)" y una de "2 Etapas (2E)".
   - Guarda los modelos entrenados en una carpeta `models/`.
   - Crea un script de inferencia para cargar un modelo y visualizarlo en el simulador normal (con UI).

**Instrucción Final:**
Documenta detalladamente en `CHANGELOG_TESIS.md` el diseño del MDP (Markov Decision Process): justificación del espacio de estados, normalización, espacio de acciones, y el diseño matemático de la función de recompensa (Reward Shaping) con referencias bibliográficas estándar.