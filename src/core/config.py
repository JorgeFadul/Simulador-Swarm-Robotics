import math
import yaml
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

# --- Listas de estrategias por categoria ---
ESTRATEGIAS_EXPLORACION = (
    "Random Walk", "Frontier Based", "Sample Based",
    "Hybrid Based", "Boustrophedon",
)
ESTRATEGIAS_INSPECCION = (
    "Random Walk", "Voronoi", "Boustrophedon Inspeccion",
    "Cerebro Mosca (Dopamina)", "RL Inspeccion V3",
)
ESTRATEGIAS_HIBRIDO = (
    "Random Walk Hibrido", "RL Hibrido V3",
    "Cerebro Mosca (Dopamina)",
)

# Estrategias ocultas de los dropdowns (codigo y modelos intactos, solo
# compatibilidad con configs viejos). No usar para simulaciones nuevas.
ESTRATEGIAS_OCULTAS = (
    "RL Exploracion", "RL Exploracion V3",
    "RL Inspeccion",
    "RL Hibrido", "RL Hibrido V2",
)

# Mapeo de respaldo para configs viejos que traen un nombre oculto.
MAPEO_ESTRATEGIAS_OCULTAS = {
    "RL Exploracion": ("dos_etapas", "exploracion", "Frontier Based"),
    "RL Exploracion V3": ("dos_etapas", "exploracion", "Frontier Based"),
    "RL Inspeccion": ("dos_etapas", "inspeccion", "Voronoi"),
    "RL Hibrido": ("hibrido", "hibrido", "Random Walk Hibrido"),
    "RL Hibrido V2": ("hibrido", "hibrido", "Random Walk Hibrido"),
}



# Compatibilidad: todas las estrategias validas (visibles + ocultas legacy)
ESTRATEGIAS_VALIDAS = (ESTRATEGIAS_EXPLORACION + ESTRATEGIAS_INSPECCION
                       + ESTRATEGIAS_HIBRIDO + ESTRATEGIAS_OCULTAS)


def resolver_estrategia(modo_fase, categoria, nombre):
    """Devuelve un nombre de estrategia visible.

    Si `nombre` esta oculto, lo mapea a su reemplazo y avisa por consola.
    `categoria` es "exploracion", "inspeccion" o "hibrido".
    """
    if nombre in MAPEO_ESTRATEGIAS_OCULTAS:
        modo_nuevo, cat_nueva, reemplazo = MAPEO_ESTRATEGIAS_OCULTAS[nombre]
        print(f"[Estrategia] '{nombre}' esta oculta; usando '{reemplazo}'.")
        return modo_nuevo, reemplazo
    return modo_fase, nombre

# Numero maximo de generaciones de triangulacion visibles/ocultables en el GUI
MAX_GEN_VISIBLE = 10


class Config:
    def __init__(self, path=None):
        path = path or DEFAULT_PATH
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        self.grid_size: int = raw["grid_size"]
        self.map_width: int = raw["map_width"]
        self.map_height: int = raw["map_height"]
        self.fps: int = raw["fps"]
        self.num_robots: int = raw["num_robots"]

        self.pixels_per_cm: float = float(raw["pixels_per_cm"])
        ppcm = self.pixels_per_cm

        self.robot_radius_cm: float = float(raw["robot_radius_cm"])
        self.robot_radius_px: float = self.robot_radius_cm * ppcm
        self.detection_radius_cm: float = float(raw["detection_radius_cm"])
        self.detection_radius_px: float = self.detection_radius_cm * ppcm
        self.robot_max_speed_cm_s: float = float(raw["robot_max_speed_cm_s"])
        self.robot_max_speed_px_s: float = self.robot_max_speed_cm_s * ppcm

        self.random_walk_interval_min: float = float(raw["random_walk_interval_min"])
        self.random_walk_interval_max: float = float(raw["random_walk_interval_max"])
        self.exploration_threshold: float = float(raw.get("exploration_threshold", 0.99))
        self.exploration_radius_multiplier: float = float(raw.get("exploration_radius_multiplier", 1.5))
        self.robot_turn_speed_rad_s: float = float(raw.get("robot_turn_speed_rad_s", math.pi))

        # Radio de vecindad para la obs RL v3: multiplicador del radio del robot.
        self.rl_vecindad_radio_mult: float = float(raw.get("rl_vecindad_radio_mult", 5.0))
        self.rl_vecindad_radio_px: float = self.rl_vecindad_radio_mult * self.robot_radius_px

        # --- Modos de simulacion y estrategias ---
        # Modo (dos opciones):
        #   "dos_etapas" -> simulacion de dos etapas: exploracion -> inspeccion
        #                   (transicion dinamica). Usa estrategia_exploracion
        #                   (fase de exploracion) y estrategia_inspeccion
        #                   (fase de inspeccion).
        #   "hibrido"    -> simulacion de una sola etapa: explora e inspecciona
        #                   a la vez. Usa estrategia_hibrido.
        self.modo_fase: str = str(raw.get("modo_fase", "dos_etapas"))
        self._normalizar_modo()
        self.estrategia_exploracion: str = str(raw.get("estrategia_exploracion", "Frontier Based"))
        self.estrategia_inspeccion: str = str(raw.get("estrategia_inspeccion", "Voronoi"))
        self.estrategia_hibrido: str = str(raw.get("estrategia_hibrido", "Random Walk Hibrido"))

        # Compatibilidad: strategy se deriva del modo activo
        self.strategy: str = self._estrategia_activa()

        self.exploration_goal: float = float(raw.get("exploration_goal", 1.0))
        self.plan_interval_s: float = float(raw.get("plan_interval_s", 1.0))
        self.goal_reached_radius_px: float = float(raw.get("goal_reached_radius_px", 15.0))
        self.goal_timeout_s: float = float(raw.get("goal_timeout_s", 10.0))
        self.frontier_min_size: int = int(raw.get("frontier_min_size", 3))
        self.frontier_max_size: int = int(
            raw.get("frontier_max_size", 0) or (self.grid_size // 2))
        self.reasignar_umbral_costo: float = float(raw.get("reasignar_umbral_costo", 0.7))
        self.sample_candidates: int = int(raw.get("sample_candidates", 40))
        self.sample_per_cluster: int = int(raw.get("sample_per_cluster", 8))
        self.sample_cost_weight: float = float(raw.get("sample_cost_weight", 0.02))
        self.sample_targets: int = int(raw.get("sample_targets", 12))

        self.dirigido_avance_min: float = float(raw.get("dirigido_avance_min", 0.45))
        self.dirigido_peso_evasion_imminente: float = float(raw.get("dirigido_peso_evasion_imminente", 0.5))
        self.dirigido_peso_evasion_fuerte: float = float(raw.get("dirigido_peso_evasion_fuerte", 0.4))
        self.dirigido_peso_evasion_suave: float = float(raw.get("dirigido_peso_evasion_suave", 0.8))
        self.dirigido_umbral_giro_deg: float = float(raw.get("dirigido_umbral_giro_deg", 5.0))
        # Puertas de la repulsion sobre el rumbo (ver default.yaml). La puerta
        # suave en 0.0 hace que con dot >= 0 (obstaculo a un lado o detras) el
        # rumbo NO se desvie: permite correr junto a paredes.
        self.dirigido_gate_fuerte_dot: float = float(raw.get("dirigido_gate_fuerte_dot", -0.5))
        self.dirigido_gate_suave_dot: float = float(raw.get("dirigido_gate_suave_dot", 0.0))
        # Giro cerrado (una rueda quieta): el radio de giro nunca es menor que
        # el radio real del robot (v/omega >= R). `dirigido_giro_kp` suaviza la
        # correccion de rumbo (proporcional, no bang-bang) y
        # `dirigido_pivote_umbral_deg` fija el giro grande a partir del cual,
        # si el arco no es accesible por obstaculos frontales, se pivota en el
        # centro (v=0).
        self.dirigido_giro_kp: float = float(raw.get("dirigido_giro_kp", 6.0))
        self.dirigido_pivote_umbral_deg: float = float(raw.get("dirigido_pivote_umbral_deg", 35.0))

        self.walls: list[dict] = raw.get("walls", [])

        # Modelo de energia (bateria)
        self.masa_kg: float = float(raw.get("masa_kg", 0.35))
        self.radio_rueda_m: float = float(raw.get("radio_rueda_m", 0.025))
        self.ancho_trocha_m: float = float(raw.get("ancho_trocha_m", 0.12))
        self.eficiencia_transmision: float = float(raw.get("eficiencia_transmision", 0.6))
        self.coef_rodadura: float = float(raw.get("coef_rodadura", 0.015))
        self.masa_rueda_kg: float = float(raw.get("masa_rueda_kg", 0.02))
        self.num_ruedas_motor: int = int(raw.get("num_ruedas_motor", 2))
        self.bateria_capacidad_j: float = float(raw.get("bateria_capacidad_j", 1500.0))
        self.energia_inspeccion_j: float = float(raw.get("energia_inspeccion_j", 0.5))

        # Modo headless (CLI)
        self.headless_max_pasos: int = int(raw.get("headless_max_pasos", 60000))

        # Inspeccion de baldosas (resultado binario)
        self.prob_defecto: float = float(raw.get("prob_defecto", 0.2))
        self.inspeccionar_al_visitar: bool = bool(raw.get("inspeccionar_al_visitar", True))
        self.inspeccion_individual: bool = bool(raw.get("inspeccion_individual", False))
        self.ruido_inspeccion_sigma: float = float(raw.get("ruido_inspeccion_sigma", 0.0))

        # Defectos por recursion probabilistica de blobs
        self.defecto_num_blobs: int = int(raw.get("defecto_num_blobs", 7))
        self.defecto_prob_ortogonal: float = float(raw.get("defecto_prob_ortogonal", 0.90))
        self.defecto_prob_diagonal: float = float(raw.get("defecto_prob_diagonal", 0.75))
        self.defecto_decaimiento: float = float(raw.get("defecto_decaimiento", 0.80))
        self.defecto_semilla_global: int | None = raw.get("defecto_semilla_global")
        self.inspeccion_pausa_s: float = float(raw.get("inspeccion_pausa_s", 1.0))

        # --- Fase 3: Voronoi recursivo + Market Auction (etapa de Inspeccion) ---
        # Obsoletos (documentados, se conservan por compatibilidad de YAMLs):
        self.auction_defectivo_factor: float = float(raw.get("auction_defectivo_factor", 0.15))
        self.auction_sano_factor: float = float(raw.get("auction_sano_factor", 0.25))
        self.auction_profundidad_p: float = float(raw.get("auction_profundidad_p", 0.97))
        # Nueva semantica: orden estricto padre->hijos, peso creciente por
        # generacion y ajuste dinamico por resultado de inspeccion.
        # Gate de elegibilidad en la subasta de inspeccion (tarea_elegible):
        # True = orden estricto padre->hijo (un hijo se subasta solo al estar
        # inspeccionado su padre directo); False = sin gate, toda tarea pendiente
        # es subastable independientemente de su nivel.
        self.inspeccion_padre_antes_hijo: bool = bool(raw.get("inspeccion_padre_antes_hijo", True))
        # Legacy: ya no infla el interes por generacion (el descuento
        # generacional se aplica como `/(generacion+1)` en `_interes_total`).
        # Se conserva por compatibilidad de YAMLs.
        self.inspeccion_peso_generacion: float = float(raw.get("inspeccion_peso_generacion", 1.5))
        self.inspeccion_factor_sano: float = float(raw.get("inspeccion_factor_sano", 0.6))
        self.inspeccion_factor_defectivo: float = float(raw.get("inspeccion_factor_defectivo", 2.0))
        # Interes de mercado:
        #   interes = interes_inspeccion / ((generacion+1) * distancia_A**alpha * 2^R)
        # donde R = robots dentro de la region Gen0 de la tarea (termino
        # dinamico anti-hacinamiento exponencial, recalculado en cada
        # evaluacion: 1 robot divide entre 2, 2 entre 4, 3 entre 8...).
        # El interes de inspeccion parte de 1 y cambia SOLO por el resultado de
        # los ancestros inspeccionados de la rama (sana lo baja / defectuosa lo
        # sube, acumulando el factor en cada generacion). La cuota generacional
        # `(generacion+1)` reduce el interes base en cada nivel del arbol y
        # `distancia_A**alpha` quita peso a la cercania: `inspeccion_peso_distancia`
        # es el exponente alpha (default 0.5 = equilibrio medio: una zona con
        # fallas a 20 s vence a una sana a 5 s).
        self.inspeccion_peso_distancia: float = float(raw.get("inspeccion_peso_distancia", 0.5))
        # Cola de tareas fallidas: una tarea que un robot no logro alcanzar
        # (timeout de objetivo) se aparca por cooldown y no se vuelve a ofrecer
        # hasta haber fallado `fallos_para_perdonar` veces, para no bloquear la
        # generacion activa de su region con objetivos fisicamente inalcanzables.
        self.inspeccion_fallos_para_perdonar: int = int(raw.get("inspeccion_fallos_para_perdonar", 2))
        self.inspeccion_cooldown_fallo_s: float = float(raw.get("inspeccion_cooldown_fallo_s", 25.0))

        # Transicion (para modo hibrido: umbral de descubrimiento)
        trans = raw.get("transicion", {}) or {}
        self.transicion_umbral_descubierto: float = float(
            trans.get("umbral_descubierto", 0.99))
        self.transicion_tiempo_sin_scouteo_s: float = float(
            trans.get("tiempo_sin_scouteo_s", 60.0))

        # --- RL Training ---
        rl = raw.get("rl", {}) or {}
        rw = rl.get("reward", {}) or {}
        self.rl_reward = {
            "reward_nueva": float(rw.get("reward_nueva", 1.0)),
            "reward_redundancia": float(rw.get("reward_redundancia", -0.1)),
            "reward_colision_rr": float(rw.get("reward_colision_rr", -0.3)),
            "reward_colision_pared": float(rw.get("reward_colision_pared", -0.1)),
            "reward_tiempo": float(rw.get("reward_tiempo", -0.001)),
            "reward_nueva_global": float(rw.get("reward_nueva_global", 0.5)),
            "reward_tiempo_global": float(rw.get("reward_tiempo_global", -0.001)),
            "alpha": float(rw.get("alpha", 0.6)),
            "bonus_completar": float(rw.get("bonus_completar", 50.0)),
            "castigo_timeout": float(rw.get("castigo_timeout", -50.0)),
        }
        # --- RL Hibrido (control en tiempo real en la GUI) ---
        hib = rl.get("hibrido", {}) or {}
        self.rl_hibrido_model: str = str(hib.get("model", "models/rl_20x20_10r/final"))
        self.rl_hibrido_map_known: bool = bool(hib.get("map_known", True))
        self.rl_hibrido_auto_assign: bool = bool(hib.get("auto_assign", True))
        self.rl_hibrido_num_robots: int = int(hib.get("num_robots", 10))

        hib_v2 = rl.get("hibrido_v2", {}) or {}
        self.rl_hibrido_v2_model: str = str(hib_v2.get("model", "models/rl_v2/final"))
        self.rl_hibrido_v2_map_known: bool = bool(hib_v2.get("map_known", True))
        self.rl_hibrido_v2_auto_assign: bool = bool(hib_v2.get("auto_assign", True))
        self.rl_hibrido_v2_num_robots: int = int(hib_v2.get("num_robots", 10))

        self.rl_curriculum = rl.get("curriculum", [])
        ppo = rl.get("ppo", {}) or {}
        self.rl_ppo = {
            "learning_rate": float(ppo.get("learning_rate", 3e-4)),
            "n_steps": int(ppo.get("n_steps", 2048)),
            "batch_size": int(ppo.get("batch_size", 64)),
            "n_epochs": int(ppo.get("n_epochs", 10)),
            "gamma": float(ppo.get("gamma", 0.99)),
            "gae_lambda": float(ppo.get("gae_lambda", 0.95)),
            "clip_range": float(ppo.get("clip_range", 0.2)),
            "ent_coef": float(ppo.get("ent_coef", 0.01)),
            "vf_coef": float(ppo.get("vf_coef", 0.5)),
            "net_arch": ppo.get("net_arch", [256, 256]),
        }

        # --- Cerebro de Mosca (bioinspirado) ---
        fly = raw.get("fly", {}) or {}
        self.fly_kc: int = int(fly.get("kc", 1000))
        self.fly_weights_path: str = str(fly.get("weights_path", "models/bio/fly_swarm_latest.npz"))
        self.fly_mutacion: float = float(fly.get("mutacion", 0.0))
        self.fly_patch_size: int = int(fly.get("patch_size", 15))
        # Bandera en tiempo de ejecucion: True mientras el planner activo sea el
        # Cerebro de Mosca. Permite que la percepcion (pausa de 1s al inspeccionar)
        # funcione tambien en la fase de exploracion del modo dos_etapas.
        self.fly_auto_inspeccion: bool = False

        self.cols = self.map_width // self.grid_size
        self.rows = self.map_height // self.grid_size

    def _estrategia_activa(self, etapa=None):
        """Devuelve la estrategia activa segun el modo de simulacion.

        - modo "dos_etapas": la estrategia depende de la etapa actual
          (`etapa` "exploracion" o "inspeccion"). Si `etapa` no se pasa,
          asume la fase de exploracion.
        - modo "hibrido": una sola etapa, usa la estrategia hibrida.
        """
        if self.modo_fase == "dos_etapas":
            if etapa == "inspeccion":
                return self.estrategia_inspeccion
            return self.estrategia_exploracion
        elif self.modo_fase == "hibrido":
            return self.estrategia_hibrido
        return self.estrategia_exploracion

    def es_dos_etapas(self):
        """True si el modo es de dos etapas (exploracion -> inspeccion)."""
        return self.modo_fase == "dos_etapas"

    def es_hibrido(self):
        """True si el modo activo es hibrido (una sola etapa)."""
        return self.modo_fase == "hibrido"

    def sync_strategy(self):
        """Sincroniza self.strategy con el modo activo (llamar tras cambiar modo)."""
        self.strategy = self._estrategia_activa()

    def _normalizar_modo(self):
        """Convierte el valor de modo_fase a los 2 modos actuales.

        Modos antiguos ("exploracion", "inspeccion") dejan de existir como
        modos independientes: pasan a la simulacion de dos etapas.
        """
        if self.modo_fase not in ("dos_etapas", "hibrido"):
            self.modo_fase = "dos_etapas"

    def cargar_desde_archivo(self, path="sim_config.txt"):
        """Superpone valores guardados (clave=valor) sobre la configuracion."""
        p = Path(path)
        if not p.exists():
            return
        data = {}
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()

        # Nuevo sistema de modos
        if "modo_fase" in data:
            self.modo_fase = data["modo_fase"]
        self._normalizar_modo()
        if "estrategia_exploracion" in data:
            self.estrategia_exploracion = data["estrategia_exploracion"]
        if "estrategia_inspeccion" in data:
            self.estrategia_inspeccion = data["estrategia_inspeccion"]
        if "estrategia_hibrido" in data:
            self.estrategia_hibrido = data["estrategia_hibrido"]
        # Nombres ocultos (configs viejos): mapear a reemplazos visibles.
        for attr, cat in (("estrategia_exploracion", "exploracion"),
                          ("estrategia_inspeccion", "inspeccion"),
                          ("estrategia_hibrido", "hibrido")):
            nombre = getattr(self, attr)
            if nombre in MAPEO_ESTRATEGIAS_OCULTAS:
                modo_nuevo, reemplazo = resolver_estrategia(
                    self.modo_fase, cat, nombre)
                self.modo_fase = modo_nuevo
                setattr(self, attr, reemplazo)
        if "inspeccion_individual" in data:
            self.inspeccion_individual = data["inspeccion_individual"].lower() in ("true", "1", "yes")
        if "inspeccion_padre_antes_hijo" in data:
            self.inspeccion_padre_antes_hijo = data["inspeccion_padre_antes_hijo"].lower() in ("true", "1", "yes")

        # Compatibilidad: si viene "strategy" viejo, mapear al modo correspondiente
        if "strategy" in data:
            old = data["strategy"]
            if old in MAPEO_ESTRATEGIAS_OCULTAS:
                modo_nuevo, cat_nueva, reemplazo = MAPEO_ESTRATEGIAS_OCULTAS[old]
                print(f"[Estrategia] '{old}' esta oculta; usando '{reemplazo}'.")
                self.modo_fase = modo_nuevo
                if modo_nuevo == "hibrido":
                    self.estrategia_hibrido = reemplazo
                elif cat_nueva == "inspeccion":
                    self.estrategia_inspeccion = reemplazo
                else:
                    self.estrategia_exploracion = reemplazo
            elif old in ESTRATEGIAS_HIBRIDO:
                self.modo_fase = "hibrido"
                self.estrategia_hibrido = old
            elif old in ESTRATEGIAS_EXPLORACION:
                self.modo_fase = "dos_etapas"
                self.estrategia_exploracion = old
            elif old in ESTRATEGIAS_INSPECCION:
                self.modo_fase = "dos_etapas"
                self.estrategia_inspeccion = old
        self._normalizar_modo()

        self.sync_strategy()

        for key in ("robot_radius_cm", "robot_max_speed_cm_s",
                    "robot_turn_speed_rad_s", "detection_radius_cm",
                    "exploration_radius_multiplier",
                    "random_walk_interval_min", "random_walk_interval_max",
                    "exploration_goal", "plan_interval_s",
                    "goal_reached_radius_px", "goal_timeout_s",
                    "sample_cost_weight",
                    "reasignar_umbral_costo",
                    "dirigido_avance_min",
                    "dirigido_peso_evasion_imminente",
                    "dirigido_peso_evasion_fuerte",
                    "dirigido_peso_evasion_suave",
                    "dirigido_umbral_giro_deg",
                    "dirigido_gate_fuerte_dot",
                    "dirigido_gate_suave_dot",
                    "dirigido_giro_kp",
                    "dirigido_pivote_umbral_deg",
                    "auction_defectivo_factor",
                    "auction_sano_factor",
                    "auction_profundidad_p",
                    "inspeccion_peso_generacion",
                    "inspeccion_factor_sano",
                    "inspeccion_factor_defectivo",
                    "inspeccion_peso_distancia",
                    "inspeccion_cooldown_fallo_s",
                    "defecto_prob_ortogonal",
                    "defecto_prob_diagonal",
                    "defecto_decaimiento"):
            if key in data:
                try:
                    setattr(self, key, float(data[key]))
                except ValueError:
                    continue

        for key in ("frontier_min_size", "frontier_max_size", "sample_candidates",
                    "sample_per_cluster", "sample_targets", "defecto_num_blobs",
                    "inspeccion_fallos_para_perdonar"):
            if key in data:
                try:
                    setattr(self, key, int(data[key]))
                except ValueError:
                    continue

        ppcm = self.pixels_per_cm
        self.robot_radius_px = self.robot_radius_cm * ppcm
        self.detection_radius_px = self.detection_radius_cm * ppcm
        self.robot_max_speed_px_s = self.robot_max_speed_cm_s * ppcm
