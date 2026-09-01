Actúa como un Ingeniero de Software experto en Python, Pygame y simulaciones robóticas. Estamos trabajando en un Simulador Swarm 2D para una tesis de maestría. 

Tu tarea es implementar la "Fase 0", que sienta las bases técnicas para simulaciones avanzadas, resolviendo problemas técnicos actuales y añadiendo el modelo de energía y el modo CLI (Headless).

**Tareas a realizar:**
1. **Correcciones de UX y Bugs:**
   - En `requirements.txt`, añade `numpy` y `Pillow`.
   - En `src/core/simulacion.py`, unifica los controles del editor: haz que las teclas de colocar pared y borrar sean 'P' y 'B' respectivamente (actualmente son W y E).
   - Corrige el título de la ventana de Pygame para que no diga siempre "... - Editor" cuando esté en estado `running`.
   - Mejora la disposición lógica de los menús en `stats_window.py` si encuentras solapamientos o mala distribución.

2. **Modelo de Energía (Batería):**
   - Crea una clase o módulo para llevar el registro del consumo energético de cada robot.
   - Implementa un cálculo de energía genérico para las llantas del chasis HeRo Swarm. La energía consumida por movimiento debe basarse en la velocidad y el torque.
   - Incluye el costo inercial de arranque: cada vez que el robot pasa de velocidad 0 a velocidad $v$, debe sumar un pico de energía $E_{start}$ por romper la inercia.
   - Crea un método `consumir_energia_inspeccion()` que sume un valor constante por ahora (representando el gasto de procesar la FFT y la percusión).
   - El robot no morirá por falta de batería en esta fase, solo llevará el conteo acumulativo. Refleja este gasto en la pestaña 'Individual' de la UI.

3. **Modo CLI (Headless) y Streaming de Datos:**
   - Modifica `main.py` para aceptar argumentos por línea de comandos (usando `argparse`). Argumentos esperados: `--headless`, `--estrategia`, `--semilla`, `--mapa`.
   - En modo `--headless`, Pygame y Tkinter NO deben inicializarse visualmente. El bucle debe correr en memoria lo más rápido posible.
   - Implementa un output en formato JSONL (una línea JSON por tick o intervalo) hacia `stdout` que contenga el estado de los robots (posiciones, batería, tareas) para que un agente de IA pueda leerlo en vivo.
   - Al finalizar la corrida headless, debe exportar un JSON o CSV con el resumen de métricas.

**Instrucción Final:**
Al terminar, no me des solo fragmentos de código sueltos. Explica qué archivos modificaste. Adicionalmente, crea un archivo llamado `CHANGELOG_TESIS.md` y documenta ahí las fórmulas físicas de energía que asumiste, justificando las variables (inercia, velocidad, consumo constante) para que pueda anexarlo a la tesis.