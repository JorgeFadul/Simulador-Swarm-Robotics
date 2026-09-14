# Plan de Implementación: Estrategia Bioinspirada "Cerebro de Mosca" (*Drosophila melanogaster*) con Plasticidad Dopaminérgica

## 1. Fundamento Biológico y Arquitectura
En la mosca de la fruta (*Drosophila melanogaster*), el aprendizaje por refuerzo y la navegación espacial no utilizan retropropagación (*backpropagation*). Operan a través de dos subsistemas anatómicos interconectados:

1. **Mushroom Body (MB - Cuerpos Fúngicos)**:
   - **Projection Neurons (PNs)**: Reciben el vector sensorial egocéntrico del robot:
     - Parche local $K \times K$ de celdas libres/paredes/defectos.
     - **Canal Social / Densidad de Compañeras**:
       - Número de otras moscas/robots en su radio de percepción social $N_{\text{cerca}}$ normalizado (para soportar los $\approx 10$ robots operando a la vez).
       - Distancia y orientación relativa de las compañeras más próximas (evitación de hacinamiento o reclutamiento colectivo).
     - Nivel de batería e inercia motriz.
   - **Kenyon Cells (KCs)**: Realizan una proyección de alta dimensión esparsa (*sparse coding*, $\approx 1000$ neuronas donde solo el $5\%$ de las más excitadas disparan). Esta representación descorrelaciona estados sensoriales similares (por ejemplo: "baldosa defectuosa solitaria" vs. "baldosa defectuosa con 2 compañeras cerca").
   - **Dopaminergic Neurons (DANs)**: Señal de valencia/recompensa. Liberan pulsos de dopamina cuando el robot experimenta eventos relevantes (+ baldosa nueva, ++ baldosa defectuosa, - colisión o estancamiento, y modulación social cooperativa).
   - **Mushroom Body Output Neurons (MBONs)**: Neuronas de salida que codifican preferencia de acción (vectores de aproximación vs. evitación).
   - **Regla de Tres Factores (Three-Factor Learning Rule)**:
     $$\Delta W_{ij}(t) = \eta \cdot a_{\text{KC}, i}(t) \cdot a_{\text{MBON}, j}(t) \cdot D(t) - \lambda W_{ij}(t)$$
     donde $D(t)$ es la concentración instantánea de dopamina y $\lambda$ es un decaimiento sináptico natural.


2. **Central Complex (CX - Complejo Central)**:
   - Modela una red de atractor en anillo (*Ring Attractor*) en el Protocerebral Bridge / Ellipsoid Body.
   - Integra el rumbo (*heading direction*) y la velocidad para mantener la orientación espacial y transformar los potenciales de acción de los MBONs en comandos continuos de dirección motriz o selección de waypoints discretos.

---

## 2. Ventajas del Enfoque para la Tesis
- **Aprendizaje Continuo en Línea (*Online / Continual Learning*)**: Cada robot del enjambre aprende *in situ* durante la simulación interactiva sin depender obligatoriamente de pre-entrenar 3 millones de pasos en GPU.
- **Invarianza Total a $N$ y Dimensiones**: Cada robot porta su propio "cerebro" individual que evalúa su entorno egocéntrico.
- **Alto Aporte Académico**: Compara un algoritmo de RL profundo tradicional (PPO) contra una arquitectura neuromórfica bioinspirada en insectos reales.

---

## 3. Componentes a Crear y Modificar

### A. Módulo Bioinspirado: `src/bio/fly_brain.py` [NEW]
- **`class MushroomBody`**:
  - Parámetros: `n_inputs` (tamaño vector sensorial, ej. 237), `n_kc` (1000 neuronas), `n_mbon` (número de acciones o direcciones cardinales, ej. 8 direcciones o $K \times K$ candidatos).
  - Matrices de conectividad fija $W_{\text{PN} \to \text{KC}}$ (aleatoria fija con grado de entrada bajo, como en biología).
  - Activación $k$-Winner-Take-All ($k$-WTA) en KCs (solo el top $5\%$ se activa a 1.0, el resto a 0.0).
  - Matriz de pesos plástica $W_{\text{KC} \to \text{MBON}}$.
  - Método `actualizar_plasticidad(dopamina, dt)`: Aplica la regla de aprendizaje por dopamina.
- **`class CentralComplex`**:
  - Modela el anillo atractor de 8 a 16 cuñas angulares para integración de rumbo (*path integration*) y estabilidad direccional.
- **`class FlyBrain`**:
  - Orquesta MB + CX para un robot individual. Mantiene el nivel instantáneo de dopamina $D(t)$ y su tasa de degradación enzimática.

### B. Controlador Bioinspirado: `src/bio/fly_controller.py` [NEW]
- **`class FlyBrainSwarmController`**:
  - Mantiene un diccionario `{robot_id: FlyBrain}` para que cada robot del enjambre tenga su propia mente plástica.
  - En cada ciclo:
    1. Construye la observación sensorial del robot (parche local, paredes, vecinos).
    2. Calcula la dopamina generada en el último paso:
       - $+1.0$ por baldosa libre recién explorada.
       - $+5.0$ por baldosa defectuosa encontrada.
       - $-0.5$ por colisión o proximidad de peligro.
       - $-0.02 \times t_{\text{quieto}}$ si el robot permanece estancado.
       - $-0.005 \times \Delta E$ por energía consumida.
    3. Alimenta la dopamina al cerebro del robot $\to$ actualización sináptica en caliente.
    4. Consulta la acción motriz al cerebro y la asigna a `robot.asignar_objetivo()`.

### C. Integración en el Simulador y Configuración
- **`src/core/config.py`** [MODIFY]:
  - Añadir `"Cerebro Mosca (Dopamina)"` a `ESTRATEGIAS_HIBRIDO` y `ESTRATEGIAS_EXPLORACION`.
  - Parámetros configurables: `dopamina_factor_defecto`, `dopamina_decay_rate`, `fly_kc_count`.
- **`src/core/simulacion.py`** [MODIFY]:
  - En `_crear_planner()` y `_crear_planner_inspeccion()`, instanciar `FlyBrainSwarmController` cuando la estrategia activa sea `"Cerebro Mosca (Dopamina)"`.
- **`src/gui/stats_window.py`** [MODIFY]:
  - Registrar la nueva estrategia en los dropdowns de la GUI.
  - Añadir telemetría de dopamina: mostrar en la pestaña "Individual" el nivel actual de dopamina del robot seleccionado.

---

## 4. Plan de Verificación

1. **Test Unitario de Plasticidad Sináptica**:
   - Comprobar que en `src/bio/fly_brain.py` estímulos asociados a dopamina positiva incrementan la fuerza de las conexiones sinápticas hacia esa acción, mientras que dopamina negativa las deprime (*Long-Term Depression - LTD*).
2. **Prueba de Inferencia y Online Learning**:
   - Iniciar la simulación en GUI y verificar en consola o telemetría que los robots comienzan con exploración exploratoria y, al encontrar la primera celda defectuosa, los picos de dopamina refuerzan el comportamiento de permanecer e investigar esa región.
3. **Validación de Rendimiento**:
   - Verificar que el cálculo de 10-20 cerebros de mosca en NumPy con representaciones esparsas tome menos de $2\text{ ms}$ por tick, garantizando 60 FPS fluidos.
