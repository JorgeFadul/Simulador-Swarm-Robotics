import math
import yaml
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

# --- Listas de estrategias por categoria ---
ESTRATEGIAS_EXPLORACION = (
    "Random Walk", "Frontier Based", "Sample Based",
    "Hybrid Based", "Boustrophedon", "RL Exploracion",
)
ESTRATEGIAS_INSPECCION = (
    "Random Walk", "Voronoi", "Boustrophedon Inspeccion", "RL Inspeccion",
)
ESTRATEGIAS_HIBRIDO = (
    "Random Walk Hibrido", "RL Hibrido",
)

# Compatibilidad: todas las estrategias validas
ESTRATEGIAS_VALIDAS = ESTRATEGIAS_EXPLORACION + ESTRATEGIAS_INSPECCION + ESTRATEGIAS_HIBRIDO


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

        # --- Nuevo sistema de modos y estrategias ---
        self.modo_fase: str = str(raw.get("modo_fase", "exploracion"))
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
        self.sample_candidates: int = int(raw.get("sample_candidates", 40))
        self.sample_per_cluster: int = int(raw.get("sample_per_cluster", 8))
        self.sample_cost_weight: float = float(raw.get("sample_cost_weight", 0.02))
        self.sample_targets: int = int(raw.get("sample_targets", 12))

        self.dirigido_avance_min: float = float(raw.get("dirigido_avance_min", 0.45))
        self.dirigido_peso_evasion_imminente: float = float(raw.get("dirigido_peso_evasion_imminente", 0.5))
        self.dirigido_peso_evasion_fuerte: float = float(raw.get("dirigido_peso_evasion_fuerte", 0.4))
        self.dirigido_peso_evasion_suave: float = float(raw.get("dirigido_peso_evasion_suave", 0.8))
        self.dirigido_umbral_giro_deg: float = float(raw.get("dirigido_umbral_giro_deg", 5.0))

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
        self.defecto_blobs: list = list(raw.get("defecto_blobs", []))
        self.inspeccion_pausa_s: float = float(raw.get("inspeccion_pausa_s", 1.0))

        # --- Fase 3: Voronoi recursivo + Market Auction (etapa de Inspeccion) ---
        self.auction_defectivo_factor: float = float(raw.get("auction_defectivo_factor", 0.15))
        self.auction_sano_factor: float = float(raw.get("auction_sano_factor", 0.25))
        self.auction_profundidad_p: float = float(raw.get("auction_profundidad_p", 0.97))

        # Transicion (para modo hibrido: umbral de descubrimiento)
        trans = raw.get("transicion", {}) or {}
        self.transicion_umbral_descubierto: float = float(
            trans.get("umbral_descubierto", 0.99))
        self.transicion_tiempo_sin_scouteo_s: float = float(
            trans.get("tiempo_sin_scouteo_s", 60.0))

        self.cols = self.map_width // self.grid_size
        self.rows = self.map_height // self.grid_size

    def _estrategia_activa(self):
        """Devuelve la estrategia activa segun el modo de fase."""
        if self.modo_fase == "exploracion":
            return self.estrategia_exploracion
        elif self.modo_fase == "inspeccion":
            return self.estrategia_inspeccion
        elif self.modo_fase == "hibrido":
            return self.estrategia_hibrido
        return self.estrategia_exploracion

    def es_exploracion(self):
        """True si el modo activo es exploracion."""
        return self.modo_fase == "exploracion"

    def es_inspeccion(self):
        """True si el modo activo es inspeccion."""
        return self.modo_fase == "inspeccion"

    def es_hibrido(self):
        """True si el modo activo es hibrido."""
        return self.modo_fase == "hibrido"

    def es_dos_etapas(self):
        """Deprecated: siempre False (compatibilidad con codigo existente)."""
        return False

    def modo_estrategia(self, estrategia=None):
        """Deprecated: devuelve 'H' (compatibilidad con codigo existente)."""
        return "H"

    def sync_strategy(self):
        """Sincroniza self.strategy con el modo activo (llamar tras cambiar modo)."""
        self.strategy = self._estrategia_activa()

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
        if "estrategia_exploracion" in data:
            self.estrategia_exploracion = data["estrategia_exploracion"]
        if "estrategia_inspeccion" in data:
            self.estrategia_inspeccion = data["estrategia_inspeccion"]
        if "estrategia_hibrido" in data:
            self.estrategia_hibrido = data["estrategia_hibrido"]
        if "inspeccion_individual" in data:
            self.inspeccion_individual = data["inspeccion_individual"].lower() in ("true", "1", "yes")

        # Compatibilidad: si viene "strategy" viejo, mapear al modo correspondiente
        if "strategy" in data:
            old = data["strategy"]
            if old in ESTRATEGIAS_EXPLORACION:
                self.modo_fase = "exploracion"
                self.estrategia_exploracion = old
            elif old in ESTRATEGIAS_INSPECCION:
                self.modo_fase = "inspeccion"
                self.estrategia_inspeccion = old
            elif old in ESTRATEGIAS_HIBRIDO:
                self.modo_fase = "hibrido"
                self.estrategia_hibrido = old

        self.sync_strategy()

        for key in ("robot_radius_cm", "robot_max_speed_cm_s",
                    "robot_turn_speed_rad_s", "detection_radius_cm",
                    "exploration_radius_multiplier",
                    "random_walk_interval_min", "random_walk_interval_max",
                    "exploration_goal", "plan_interval_s",
                    "goal_reached_radius_px", "goal_timeout_s",
                    "sample_cost_weight",
                    "dirigido_avance_min",
                    "dirigido_peso_evasion_imminente",
                    "dirigido_peso_evasion_fuerte",
                    "dirigido_peso_evasion_suave",
                    "dirigido_umbral_giro_deg",
                    "auction_defectivo_factor",
                    "auction_sano_factor",
                    "auction_profundidad_p"):
            if key in data:
                try:
                    setattr(self, key, float(data[key]))
                except ValueError:
                    continue

        for key in ("frontier_min_size", "sample_candidates",
                    "sample_per_cluster", "sample_targets"):
            if key in data:
                try:
                    setattr(self, key, int(data[key]))
                except ValueError:
                    continue

        ppcm = self.pixels_per_cm
        self.robot_radius_px = self.robot_radius_cm * ppcm
        self.detection_radius_px = self.detection_radius_cm * ppcm
        self.robot_max_speed_px_s = self.robot_max_speed_cm_s * ppcm
