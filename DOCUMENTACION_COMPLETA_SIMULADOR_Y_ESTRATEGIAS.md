# Documentación Técnica y Analítico-Matemática del Simulador Swarm 2D

*Documento de referencia técnica y fundamento analítico-matemático para la Tesis de Maestría:*
> **"Estrategia basada en Enjambre de Robots Inteligentes para Inspección No Destructiva de Baldosas"**  
> Autor: **Jorge Luis Fadul Flores**  
> Programa: Maestría en Ciencias de la Ingeniería Mecánica (Especialidad en Automatización y Robótica)  
> Institución: Universidad Tecnológica de Panamá (UTP)  
> Asesor: Dr. Humberto Rodríguez  

---

## 1. Introducción y Propósito del Simulador

El **Simulador Swarm 2D** es la plataforma de experimentación computacional desarrollada para evaluar la coordinación colectiva, exploración espacial y diagnóstico no destructivo de baldosas mediante enjambres de robots móviles.

El objetivo principal de la simulación es validar cuantitativamente la eficiencia, cobertura espacial, precisión diagnóstica y consumo energético del enjambre bajo distintas estrategias de control y coordinación en entornos cerrados con presencia de obstáculos.

El simulador resuelve el problema en dos dimensiones mediante un modelo discreto-continuo:
1. **Capa de Cinemática y Física Continua:** Control de movimiento diferencial de robots, evitación reactiva de colisiones en el espacio continuo ($x, y, \theta \in \mathbb{R}$).
2. **Capa de Percepción y Grilla Discreta:** Mapeo espacial sobre una grilla regular de celdas de tamaño $\Delta g \times \Delta g$, donde se registra el estado de exploración, presencia de paredes y diagnóstico de salud de las baldosas.

---

## 2. Arquitectura General del Sistema

El simulador se estructuró bajo una arquitectura modular y desacoplada en Python, utilizando **Pygame** para el renderizado gráfico y **Tkinter** para la interfaz de control y telemetría en tiempo real.

```
simulador-swarm-2d/
├── main.py                 # Punto de entrada principal (Bucle a 20 Hz / dt = 0.05s)
├── config/
│   └── default.yaml        # Parámetros físicos, geométricos y ponderaciones
├── src/
│   ├── core/
│   │   ├── config.py       # Conversor de unidades (cm <-> px) y cargador de parametros
│   │   ├── mundo.py        # Verdad terrena (grid), mapa explorado, raycasting (LOS) y física
│   │   ├── simulacion.py   # Orquestador de estados (Edición, Ejecución, Transición de Fase)
│   │   └── energia.py      # Modelo dinámico de consumo de batería
│   ├── robots/
│   │   └── robot.py        # Modelo cinemático, estados del robot y consumo de tareas
│   ├── algorithms/
│   │   ├── exploration.py  # Algoritmos de exploración (Frontier, Sample, Hybrid, Boustrophedon, Voronoi)
│   │   ├── inspeccion.py   # Algoritmos de inspección (Voronoi Geodésico + Triangulación Recursiva)
│   │   └── pathfinding.py  # Planificación de rutas A* Ponderado por Conocimiento
│   └── rl/
│       ├── env.py / controller.py # Entorno Gym/PettingZoo y controladores de Reinforcement Learning (PPO)
```

### 2.1 Representación del Dominio Espacial

El mapa se modela como una matriz bidimensional $\Omega$ de dimensiones $R \times C$, donde cada celda $(r,c)$ representa una baldosa o región del suelo.

Existen tres matrices principales de estado:
1. **`grid` (Verdad Terrena Global):**  
   $$\text{grid}[r,c] = \begin{cases} 0 & \text{Celda libre (Baldosa)} \\ 1 & \text{Obstáculo / Pared} \end{cases}$$
2. **`explored` (Conocimiento Acumulado del Enjambre):**  
   $$\text{explored}[r,c] = \begin{cases} 0 & \text{Desconocida (Sin visibilidad previo)} \\ 1 & \text{SCOUTED (Avistada mediante Sensor con Línea de Vista)} \\ 2 & \text{VISITED (Visitada físicamente por un robot)} \\ 3 & \text{DISCOVERED (Obstáculo o pared confirmada por LOS o impacto)} \end{cases}$$
3. **`inspec` y `resultado` (Diagnóstico Acústico de Baldosas):**  
   $$\text{inspec}[r,c] \in \{0, 1\}, \quad \text{resultado}[r,c] = \begin{cases} -1 & \text{No inspeccionada} \\ 0 & \text{Baldosa Sana (Adherida)} \\ 1 & \text{Baldosa Defectuosa (Desprendida)} \end{cases}$$

### 2.2 Percepción Realista por Línea de Vista (LOS)

Ningún robot posee conocimiento previo del mapa $\text{grid}$. La actualización del mapa explorado se realiza mediante un algoritmo de trazado de rayos (Raycasting / Bresenham) con un radio de detección $r_{\text{det}}$.

Dado un robot en la posición continua $(x_k, y_k)$, una celda $(r,c)$ con centro en $(x_c, y_c)$ dentro de $\|(x_k, y_k) - (x_c, y_c)\|_2 \le r_{\text{det}}$ es marcada como vista únicamente si el segmento recto entre $(x_k, y_k)$ y $(x_c, y_c)$ no interseca ninguna celda con $\text{grid}[r', c'] = 1$:

$$\text{LOS}(\mathbf{p}_1, \mathbf{p}_2) = \prod_{(r,c) \in \text{Bresenham}(\mathbf{p}_1, \mathbf{p}_2)} (1 - \text{grid}[r,c]) \in \{0, 1\}$$

---

## 3. Modos de Operación y Fases de Simulación

El simulador admite dos modos de operación principales:

1. **Modo Dos Etapas (2E):**  
   - **Etapa 1 (Exploración Espacial):** El enjambre prioriza descubrir el 100% de las celdas libres y delimitar las paredes del mapa.  
   - **Transición Dinámica:** Al alcanzar el umbral de exploración ($\ge 99\%$) o agotarse las celdas frontera, el sistema conmuta automáticamente a la Etapa 2.  
   - **Etapa 2 (Inspección Geométrica Adaptativa):** Se congela el mapa explorado y los robots ejecutan inspección acústica (tap-testing) guiada por una partición de Voronoi geodésica y triangulación recursiva con subasta orientada por defectos.

2. **Modo Híbrido (1E-H - Single Stage):**  
   - Los robots realizan la exploración del mapa y la inspección acústica de las baldosas de manera simultánea en una sola fase continua.

---

## 4. Análisis Lógico, Analítico y Matemático de las Estrategias

En esta sección se detalla el funcionamiento lógico, las formulaciones matemáticas de selección/costo y las características analíticas de cada estrategia implementada.

---

### 4.1 Random Walk (Paseo Aleatorio Estocástico)

#### Lógica y Descripción
Es el algoritmo de referencia (baseline) sin coordinación ni memoria compartida. Cada robot opera de forma autónoma según un proceso estocástico de dirección aleatoria con temporizadores de cambio de rumbo.

#### Formulaciones Matemáticas
- **Intervalo de Re-orientación:** El tiempo de caminata lineal $T_{\text{walk}}$ se muestrea de una distribución uniforme continua:
  $$T_{\text{walk}} \sim \mathcal{U}(t_{\text{min}}, t_{\text{max}})$$
- **Dirección Objetivo:** El nuevo ángulo de avance $\theta_{\text{target}}$ se genera como:
  $$\theta_{\text{target}} \sim \mathcal{U}(0, 2\pi)$$
- **Ecuaciones de Movimiento:**
  $$\begin{aligned}
  \dot{x}(t) &= v_{\text{target}} \cos \theta(t) \\
  \dot{y}(t) &= v_{\text{target}} \sin \theta(t) \\
  \dot{\theta}(t) &= \text{sat}\left( \text{wrap}(\theta_{\text{target}} - \theta(t)), -\omega_{\text{max}}, \omega_{\text{max}} \right)
  \end{aligned}$$

#### Características Principales
- **Complejidad Computacional:** $\mathcal{O}(1)$ por paso.
- **Ventajas:** Cero requerimiento de memoria, comunicación o cómputo centralizado.
- **Desventajas:** Alta tasa de solapamiento (redundancia), tiempo de cobertura exponencial frente al área, propenso a atraparse en esquinas.

---

### 4.2 Exploration Frontier-Based (Yamauchi)

#### Lógica y Descripción
Basada en el trabajo seminal de Yamauchi (1997). Define como **frontera** a toda celda conocida y libre que se encuentre adyacente a al menos una celda desconocida. Los robots se desplazan hacia el centroide de estas fronteras para expandir el mapa.

#### Formulaciones Matemáticas
1. **Detección Morfológica de Fronteras:**  
   Sea $M_{\text{free}} = (\text{grid} == 0)$ y $M_{\text{known}} = (\text{explored} \neq 0)$. La máscara booleana de fronteras $F[r,c]$ se calcula mediante operaciones matriciales con la 8-vecindad:
   $$F[r,c] = M_{\text{free}}[r,c] \land M_{\text{known}}[r,c] \land \left( \bigvee_{(dr,dc) \in \mathcal{N}_8} \neg M_{\text{known}}[r+dr, c+dc] \right)$$

2. **Agrupamiento y Subdivisión de Entidades:**  
   Celdas frontera adyacentes se agrupan en regiones 8-conectadas $\mathcal{R}_k$. Para evitar que fronteras gigantes actúen como un único objetivo distante, las regiones con $|\mathcal{R}_k| > S_{\text{max}}$ se dividen recursivamente mediante *Flood-Fill BFS* en entidades acotadas de tamaño máximo $S_{\text{max}} = \lfloor \Delta g / 2 \rfloor$.

3. **Centroide Navegable de Entidad:**  
   El objetivo espacial $\mathbf{p}_{\text{target}} = (x^*, y^*)$ de una entidad $\mathcal{R}_k$ corresponde a la celda de la entidad con la menor distancia euclídea al punto medio geométrico:
   $$(\bar{r}, \bar{c}) = \left( \frac{1}{|\mathcal{R}_k|} \sum_{i \in \mathcal{R}_k} r_i, \; \frac{1}{|\mathcal{R}_k|} \sum_{i \in \mathcal{R}_k} c_i \right)$$
   $$(r^*, c^*) = \arg\min_{(r,c) \in \mathcal{R}_k} \left( (r - \bar{r})^2 + (c - \bar{c})^2 \right)$$

4. **Función de Costo y Subasta Dinámica:**  
   El costo $J(R_i, \mathcal{R}_k)$ de asignar la entidad $\mathcal{R}_k$ al robot $R_i$ se define analíticamente como:
   $$J(R_i, \mathcal{R}_k) = \frac{\|\mathbf{x}_{R_i} - \mathbf{p}_{\mathcal{R}_k}\|_2}{|\mathcal{R}_k| + 1}$$
   - *Lista de Prioridad Global:* Se evalúan todos los pares $(R_i, \mathcal{R}_k)$ ordenados de forma ascendente según $J$. Cada entidad se asigna **en exclusiva** a un robot.
   - *Umbral de Reasignación en Tránsito ($\gamma \approx 0.7$):* Un robot en movimiento solo cambiará su meta actual $\mathcal{R}_{\text{actual}}$ por una nueva $\mathcal{R}_{\text{nueva}}$ si:
     $$J(R_i, \mathcal{R}_{\text{nueva}}) < \gamma \cdot J(R_i, \mathcal{R}_{\text{actual}})$$

#### Características Principales
- **Complejidad Computacional:** $\mathcal{O}(N_{\text{fronteras}} \log N_{\text{fronteras}} + N_{\text{robots}} \cdot N_{\text{fronteras}})$.
- **Ventajas:** Exploración sistemática dirigida a los bordes del mapa conocido.
- **Desventajas:** Ignora la ganancia real de visibilidad (puede enviar robots a fronteras estrechas que revelan muy poca área).

---

### 4.3 Sample-Based / Next-Best-View (González-Baños)

#### Lógica y Descripción
Inspirada en el enfoque Next-Best-View (NBV) de González-Baños (2002). En lugar de ir directamente a la frontera, muestrea múltiples posiciones candidatas en el espacio libre cercano y selecciona la que maximice una función de utilidad basada en la **ganancia de información esperada** ajustada por el costo de desplazamiento.

#### Formulaciones Matemáticas
1. **Muestreo de Candidatos:** Se eligen al azar celdas frontera y se genera una posición candidata $(x_q, y_q)$ dentro del radio de exploración:
   $$(x_q, y_q) = (x_0 + d \cos\phi, \; y_0 + d \sin\phi), \quad d \sim \mathcal{U}(0, 0.8 r_{\text{det}}), \; \phi \sim \mathcal{U}(0, 2\pi)$$

2. **Ganancia de Información Esperada $I(x_q, y_q)$:**  
   Conteo analítico de celdas no exploradas que quedarían visibles desde $(x_q, y_q)$ aplicando la condición de línea de vista:
   $$I(x_q, y_q) = \sum_{(r,c) \in \mathcal{B}(x_q, y_q, r_{\text{det}})} \mathbb{I}\Big( \text{explored}[r,c] == 0 \; \land \; \text{LOS}\big((x_q, y_q), \mathbf{p}_{r,c}\big) == 1 \Big)$$

3. **Función de Utilidad $U(x_q, y_q)$:**
   $$U(x_q, y_q) = I(x_q, y_q) - w_{\text{cost}} \cdot \min_{i} \|\mathbf{x}_{R_i} - (x_q, y_q)\|_2$$
   donde $w_{\text{cost}}$ es el peso de penalización por distancia (configurado típicamente en $0.02$).

#### Características Principales
- **Ventajas:** Maximiza la cantidad de información nueva revelada por cada movimiento.
- **Desventajas:** Requiere un alto número de trazados de rayos (Raycasting), incrementando el costo computacional.

---

### 4.4 Hybrid Based (Charrow et al. / Combinado)

#### Lógica y Descripción
Combina la estructura geométrica de las entidades frontera (Frontier-Based) con la evaluación estocástica de información y asignación dinámica exclusiva (Sample-Based). 

#### Formulaciones Matemáticas
- Las fronteras se dividen en entidades acotadas $|\mathcal{R}_k| \le S_{\text{max}}$.
- Se evalúa la celda centroide navegable de cada entidad con el costo normalizado:
  $$J(R_i, \mathcal{R}_k) = \frac{\|\mathbf{x}_{R_i} - \mathbf{p}_{\mathcal{R}_k}\|_2}{|\mathcal{R}_k| + 1}$$
- Los objetivos se asignan mediante la subasta dinámica centralizada por lista de prioridad y exclusividad por objetivo.

---

### 4.5 Boustrophedon Cellular Decomposition (Barrido en Zigzag)

#### Lógica y Descripción
Basado en la técnica de descomposición celular Boustrophedon (Choset et al.). Planifica un patrón de barrido metódico en zigzag a lo largo de franjas horizontales o verticales.

#### Formulaciones Matemáticas
1. **Espaciado de Franjas ($\Delta y$):**
   $$\Delta y = \max(r_{\text{det}}, \Delta g)$$
2. **Segmentación Dinámica por Obstáculos Conocidos:**  
   Dada una línea de barrido en la fila $r$, las paredes descubiertas dividen la fila en tramos libres contiguos $\mathcal{T}_m$:
   $$\mathcal{T}_m = \{ c \in [c_{\text{inicio}}, c_{\text{fin}}] \mid \text{grid}[r,c] == 0 \;\land\; \text{explored}[r,c] \neq \text{DISCOVERED} \}$$
3. **Generación de Waypoints:** Cada tramo se sub-divide en waypoints separados por una distancia horizontal $\Delta x = \Delta y$, alternando la dirección de recorrido (izquierda a derecha / derecha a izquierda) entre franjas consecutivas.

---

### 4.6 Voronoi Geodésico y Triangulación Recursiva por Figuras (Estrategia de Inspección en 2E)

Esta es la estrategia central para la etapa de **Inspección Acústica No Destructiva (Fase 3)**. Garantiza una partición geométricamente óptima del espacio y una búsqueda focalizada guiada por la presencia de defectos.

```
+-------------------------------------------------------------------------+
|                  ETAPA DE INSPECCIÓN GEOMÉTRICA (2E)                   |
+-------------------------------------------------------------------------+
|  1. Voronoi Geodésico Centroidal (Gen 0) --> Partición en N regiones     |
|  2. Triangulación Homogénea (Gen 1)     --> Abanico desde centroide       |
|  3. Subdivisión Ternaria (Gen 2 -> Gen n)--> Subtriángulos por centroide   |
|  4. Subasta por Interés con Barrera Local Exponential:                  |
|                                                                         |
|            Interés = Base / [ (gen + 1) * (d_A)^alpha * 2^R ]           |
+-------------------------------------------------------------------------+
```

#### A. Voronoi Geodésico Centroidal (Generación 0)
1. **Sembrado Inicial por Farthest-Point Sampling:** Para evitar que las posiciones instantáneas de los robots dejen regiones desequilibradas, se eligen $N$ sitios iniciales $s_1, s_2, \dots, s_N$ sobre el mapa explorado libre maximizando la distancia geodésica secuencial:
   $$s_k = \arg\max_{\mathbf{p} \in \Omega_{\text{free}}} \min_{j < k} d_g(\mathbf{p}, s_j)$$
2. **BFS Multi-Fuente Geodésico:** Se realiza un recorrido BFS simultáneo sobre el grafo de celdas libres. Cada celda $(r,c)$ se asigna al sitio $s_i$ de menor distancia de camino libre (sin atravesar paredes):
   $$V_i = \{ (r,c) \in \Omega_{\text{free}} \mid d_g((r,c), s_i) \le d_g((r,c), s_j) \quad \forall j \neq i \}$$
3. **Iteración de Lloyd:** Se recalculan los sitios $s_i$ como los centroides geométricos de sus regiones $V_i$ hasta la convergencia.

#### B. Triangulación Recursiva Homogénea por Figuras (Gen 1 a Gen $n$)
- **Generación 1 (Abanico desde Centroide):** Se extrae el polígono del contorno exterior de la región $V_i$, denotado por sus vértices $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_m$. Se construyen triángulos de primer nivel conectando el centroide $M_i$ con cada arista contigua:
  $$T_{1,j} = \triangle(M_i, \mathbf{v}_j, \mathbf{v}_{j+1})$$
- **Generación 2 a Gen $n$ (Subdivisión Ternaria Recursiva):** Cada triángulo $T = \triangle(\mathbf{a}, \mathbf{b}, \mathbf{c})$ de generación $g$ genera 3 subtriángulos de generación $g+1$ usando su centroide geométrico $\mathbf{C}_T = \frac{\mathbf{a}+\mathbf{b}+\mathbf{c}}{3}$ como vértice central de ramificación:
  $$T_{\text{hijo}, 1} = \triangle(\mathbf{a}, \mathbf{b}, \mathbf{C}_T), \quad T_{\text{hijo}, 2} = \triangle(\mathbf{b}, \mathbf{c}, \mathbf{C}_T), \quad T_{\text{hijo}, 3} = \triangle(\mathbf{c}, \mathbf{a}, \mathbf{C}_T)$$
- *Criterio de Parada:* La recursión se detiene si $\mathbf{C}_T$ cae en una celda ya ocupada por un nivel anterior, en una pared o fuera del dominio.

#### C. Modelo Analítico de Subasta con Barrera Local Anti-Hacinamiento
La asignación de figuras/triángulos pendientes a los robots se realiza mediante una subasta de primer precio basada en la función de **Interés**:

$$\text{Interés}(T) = \frac{\text{Interés\_Base}(T)}{(g + 1) \cdot (d_A)^{\alpha} \cdot 2^{R}}$$

Donde:
- **$\text{Interés\_Base}(T)$:** Comienza en $1.0$ y se actualiza según los resultados de inspección de los ancestros directos en el árbol:
  $$\text{Interés\_Base}(T_{\text{hijo}}) = \text{Interés\_Base}(T_{\text{padre}}) \times \begin{cases} K_{\text{defectuoso}} & \text{si el padre fue DEFECTUOSO } (K_{\text{defectuoso}} > 1) \\ K_{\text{sano}} & \text{si el padre fue SANO } (K_{\text{sano}} < 1) \end{cases}$$
- **$g$ (Generación):** Penalización por nivel de profundidad $(g+1)$.
- **$d_A$:** Distancia de navegación A* desde el robot hasta el centroide del triángulo.
- **$\alpha$ (`inspeccion_peso_distancia`):** Exponente de atenuación por distancia ($\alpha = 0.5$ por defecto), garantizando que las regiones defectuosas dominen sobre tareas simplemente cercanas.
- **$2^R$ (Término Repulsivo Exponencial Anti-Hacinamiento):** $R$ representa el número de robots presentes físicamente dentro de la región Voronoi Gen 0 a la que pertenece la tarea en ese instante. Si $R=1$, divide por 2; si $R=2$, divide por 4; si $R=3$, divide por 8. Evita drásticamente la congestión de robots en la misma zona.

---

### 4.7 Reinforcement Learning (PPO - Proximal Policy Optimization)

#### Lógica y Descripción
Entrenado en la Fase 4 para aprendizaje de políticas centralizadas/decentralizadas mediante Aprendizaje por Refuerzo Profundo (Deep RL).

#### Formulaciones Matemáticas
1. **Espacio de Observaciones $S$:** Vector normalizado que codifica:
   - Canal de Baldosas y Paredes: Celdas libres, descubiertas, inspeccionadas y resultados defectuosos.
   - Estado de los Robots: Posición normalizada $(x_i/W, y_i/H)$ y nivel de batería $E_i/100$.
   - Progreso Global: Porcentaje de exploración/inspección y avance temporal $t/t_{\text{max}}$.

2. **Optimización de Política PPO (Clipped Objective):**
   $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
   donde $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ es la razón de probabilidades y $\hat{A}_t$ es el estimador de ventaja (GAE).

3. **Función de Recompensa $R_t$:**
   $$R_t = r_{\text{descubrimiento}} \cdot \Delta N_{\text{vistas}} + r_{\text{inspeccion}} \cdot \Delta N_{\text{inspeccionadas}} + r_{\text{defecto}} \cdot \Delta N_{\text{defectos}} - c_{\text{colision}} \cdot \mathbb{I}(\text{choque}) - c_{\text{energia}} \cdot \sum \Delta E_i$$

---

## 5. Cuadro Comparativo Cuantitativo y Cualitativo de Estrategias

| Estrategia | Dominio / Fase | Complejidad Computacional | Rendimiento en Cobertura | Eficiencia Energética | Resiliencia a Colisiones | Requiere Mapa Previsto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Walk** | Exploración / Inspección | $\mathcal{O}(1)$ | Muy Bajo | Muy Baja | Media (Reactiva) | No |
| **Frontier-Based** | Exploración | $\mathcal{O}(N \log N)$ | Alto | Alta | Alta | No |
| **Sample-Based** | Exploración | $\mathcal{O}(K \cdot N_{\text{ray}})$ | Muy Alto | Media | Alta | No |
| **Hybrid Based** | Exploración | $\mathcal{O}(N_{\text{entidades}} \cdot N_{\text{robots}})$ | Muy Alto | Alta | Alta | No |
| **Boustrophedon** | Exploración / Inspección | $\mathcal{O}(N_{\text{filas}})$ | Alto (Sistemático) | Muy Alta | Media | No (Adaptativo) |
| **Voronoi + Triangulación** | Inspección Acústica | $\mathcal{O}(N_{\text{figuras}} \log N)$ | Máximo (Dirigido) | Máxima | Alta (Exponencial $2^R$) | No (Usa mapa explorado) |
| **RL PPO (V3)** | Híbrido / Exploración | $\mathcal{O}(1)$ (Inferencia Red) | Adaptativo / Alto | Alta | Muy Alta | No |

---

## 6. Métricas de Evaluación y Registro de Telemetría

El simulador mide cuantitativamente el desempeño del enjambre mediante el módulo `src/core/telemetria.py` y la interfaz `StatsWindow`:

1. **Porcentaje de Cobertura Espacial ($C_{\text{exp}}$):**
   $$C_{\text{exp}}(t) = \frac{\sum_{r,c} \mathbb{I}(\text{explored}[r,c] \neq 0)}{\sum_{r,c} \mathbb{I}(\text{grid}[r,c] == 0)} \times 100\%$$
2. **Porcentaje de Inspección Diagnóstica ($C_{\text{insp}}$):**
   $$C_{\text{insp}}(t) = \frac{\sum_{r,c} \mathbb{I}(\text{inspec}[r,c] == 1)}{\sum_{r,c} \mathbb{I}(\text{grid}[r,c] == 0)} \times 100\%$$
3. **Eficiencia Temporal y Tasa de Descubrimiento:** Tiempo simulado total $T_{\text{sim}}$ en segundos y pasos totales $N_{\text{pasos}}$ a $20\text{ Hz}$.
4. **Consumo de Batería Acumulado ($E_{\text{total}}$):** Integración continua de la potencia consumida por motores, sensores y actuadores de percusión en $\text{mWh}$.
5. **Tasa de Colisiones:** Contador de choques entre robots o con obstáculos rígidos.

---

## 7. Conclusión e Integración con la Tesis de Maestría

Este documento constituye la memoria descriptiva y matemática del simulador computacional desarrollado para el **Capítulo 6 de la Tesis de Maestría**. La rigurosidad en las formulaciones de visibilidad (Raycasting LOS), planificación de movimiento (A* ponderado), descomposiciones geométricas (Voronoi Geodésico y Triangulación Recursiva) y subastas dinámicas orientadas por probabilidad de defecto sustentan cuantitativamente las contribuciones de la tesis en robótica de enjambre aplicada a la inspección no destructiva de infraestructura civil.
