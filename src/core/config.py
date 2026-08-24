import math
import yaml
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


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
        self.strategy: str = str(raw.get("strategy", "Random Walk"))

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

        self.cols = self.map_width // self.grid_size
        self.rows = self.map_height // self.grid_size

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

        if "strategy" in data:
            self.strategy = data["strategy"]

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
                    "dirigido_umbral_giro_deg"):
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