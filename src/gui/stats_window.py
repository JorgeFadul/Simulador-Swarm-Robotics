import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from ..core.mundo import SCOUTED, VISITED, DISCOVERED
from ..core.config import (
    ESTRATEGIAS_EXPLORACION, ESTRATEGIAS_INSPECCION, ESTRATEGIAS_HIBRIDO,
    MAX_GEN_VISIBLE,
)

COL_SCOUTED = (0, 100, 0)
COL_VISITED = (0, 200, 0)
COL_DISCOVERED = (200, 0, 0)
COL_FONDO = (20, 20, 30)
COL_MURO_NO_VISTO = (40, 20, 20)

_PARAM_LABELS = {
    "robot_radius_cm": "Radio (cm)",
    "robot_max_speed_cm_s": "Vel. maxima (cm/s)",
    "robot_turn_speed_rad_s": "Vel. giro (rad/s)",
    "detection_radius_cm": "Radio deteccion (cm)",
    "exploration_radius_multiplier": "Multipl. exploracion",
    "random_walk_interval_min": "Walk min (s)",
    "random_walk_interval_max": "Walk max (s)",
    "exploration_goal": "Meta descubrimiento",
    "plan_interval_s": "Intervalo plan (s)",
    "goal_reached_radius_px": "Radio llegada (px)",
    "goal_timeout_s": "Timeout objetivo (s)",
    "frontier_min_size": "Frontera min (celdas)",
    "frontier_max_size": "Frontera max (celdas)",
    "reasignar_umbral_costo": "Umbral reasig. costo",
    "sample_candidates": "Candidatos muestreo",
    "sample_per_cluster": "Puntos por region",
    "sample_cost_weight": "Peso del costo",
    "sample_targets": "Max objetivos",
    "dirigido_avance_min": "Avance min al girar",
    "dirigido_peso_evasion_imminente": "Peso evasion inminente",
    "dirigido_peso_evasion_fuerte": "Peso evasion fuerte",
    "dirigido_peso_evasion_suave": "Peso evasion suave",
    "dirigido_umbral_giro_deg": "Umbral giro (deg)",
    "dirigido_giro_kp": "Ganancia giro proporcional",
    "dirigido_pivote_umbral_deg": "Umbral pivote (deg)",
}


class StatsWindow:
    def __init__(self, parent, simulacion):
        self.sim = simulacion
        self._alive = True

        self.window = tk.Toplevel(parent)
        self.window.title("Estadisticas de Exploracion")
        self.window.geometry("520x680")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=4, pady=4)

        frame_ind = ttk.Frame(notebook)
        notebook.add(frame_ind, text="Individual")
        self._build_individual_tab(frame_ind)

        frame_cob = ttk.Frame(notebook)
        notebook.add(frame_cob, text="Cobertura")
        self._build_cobertura_tab(frame_cob)

        frame_global = ttk.Frame(notebook)
        notebook.add(frame_global, text="Global")
        self._build_global_tab(frame_global)

        frame_ctrl = ttk.Frame(notebook)
        notebook.add(frame_ctrl, text="Controles")
        self._build_controls_tab(frame_ctrl)

        frame_estr = ttk.Frame(notebook)
        notebook.add(frame_estr, text="Estrategias")
        self._build_strategy_tab(frame_estr)

        self._img_tk = None
        self._img_robot_tk = None

    def _on_close(self):
        self._alive = False
        self.window.destroy()

    def _build_individual_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(3, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(top, text="Robot:").pack(side="left")
        self.robot_var = tk.StringVar()
        self.robot_menu = ttk.Combobox(top, textvariable=self.robot_var,
                                       state="readonly", width=10)
        self.robot_menu.pack(side="left", padx=8)

        self.canvas_ind = tk.Canvas(parent, bg="#111118", height=240)
        self.canvas_ind.grid(row=1, column=0, sticky="ew", padx=8)

        bat_frame = ttk.Frame(parent)
        bat_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(6, 2))
        ttk.Label(bat_frame, text="Bateria:").pack(side="left")
        self.bateria_bar = ttk.Progressbar(bat_frame, maximum=100.0, length=180)
        self.bateria_bar.pack(side="left", padx=8)
        self.bateria_label = ttk.Label(bat_frame, text="--", width=30, anchor="w")
        self.bateria_label.pack(side="left")

        self.canvas_robot_map = tk.Canvas(parent, bg="#111118", height=200)
        self.canvas_robot_map.grid(row=3, column=0, sticky="nsew", padx=8,
                                   pady=(0, 8))

    def _build_cobertura_tab(self, parent):
        self.canvas_cob = tk.Canvas(parent, bg="#111118")
        self.canvas_cob.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_global_tab(self, parent):
        self.text_global = tk.Text(parent, bg="#111118", fg="#cccccc",
                                   font=("Consolas", 11), relief="flat", padx=12, pady=12)
        self.text_global.pack(fill="both", expand=True, padx=8, pady=8)
        self.text_global.config(state="disabled")

    def _build_controls_tab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        self._scroll_canvas = canvas

        f = ttk.Frame(scroll_frame, padding=12)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="Herramienta:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.tool_var = tk.StringVar(value="robot")
        def _cambiar_tool(*_):
            self.sim.herramienta = self.tool_var.get()
        self.tool_var.trace_add("write", _cambiar_tool)
        tool_frame = ttk.Frame(f)
        tool_frame.pack(fill="x", pady=4)
        for val, lbl in [("robot", "Robot"), ("pared", "Pared"), ("borrar", "Borrar")]:
            ttk.Radiobutton(tool_frame, text=lbl, variable=self.tool_var, value=val).pack(side="left", padx=4)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(f, text="Tamano grid (px):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.grid_size_var = tk.StringVar(value="20")
        gs_frame = ttk.Frame(f)
        gs_frame.pack(fill="x", pady=4)
        ttk.Entry(gs_frame, textvariable=self.grid_size_var, width=6).pack(side="left")
        ttk.Button(gs_frame, text="Aplicar", command=self._aplicar_grid_size).pack(side="left", padx=8)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        # --- Poblacion ---
        ttk.Label(f, text="Poblacion:", font=("Arial", 10, "bold")).pack(anchor="w")
        pop_frame = ttk.Frame(f)
        pop_frame.pack(fill="x", pady=4)
        self.pop_count_var = tk.StringVar(value="10")
        ttk.Label(pop_frame, text="N:").pack(side="left")
        ttk.Entry(pop_frame, textvariable=self.pop_count_var, width=6).pack(side="left", padx=4)
        ttk.Button(pop_frame, text="Poblar", command=self._poblar).pack(side="left", padx=4)
        ttk.Button(pop_frame, text="Limpiar Todo", command=self._limpiar).pack(side="left", padx=4)
        ttk.Button(pop_frame, text="Reset Robots", command=self._reset_robots).pack(side="left", padx=4)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        # --- Parametros editables ---
        self._param_vars = {}
        self._build_param_section(f, "Parametros del Robot",
            ["robot_radius_cm", "robot_max_speed_cm_s", "robot_turn_speed_rad_s"])
        self._build_param_section(f, "Sensores",
            ["detection_radius_cm", "exploration_radius_multiplier"])
        self._build_param_section(f, "Comportamiento",
            ["random_walk_interval_min", "random_walk_interval_max"])

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(f, text="Visualizacion de planificacion:",
                  font=("Arial", 10, "bold")).pack(anchor="w")
        vis_frame = ttk.Frame(f)
        vis_frame.pack(fill="x", pady=4)
        self.ver_fronteras_var = tk.BooleanVar(value=self.sim.ver_fronteras)
        self.ver_objetivos_var = tk.BooleanVar(value=self.sim.ver_objetivos)
        self.ver_intencion_var = tk.BooleanVar(value=self.sim.ver_intencion)
        self.ver_defectos_var = tk.BooleanVar(value=self.sim.ver_defectos)
        self.ver_voronoi_var = tk.BooleanVar(value=self.sim.ver_voronoi)
        self.ver_triangulacion_var = tk.BooleanVar(value=self.sim.ver_triangulacion)
        ttk.Checkbutton(vis_frame, text="Fronteras", variable=self.ver_fronteras_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)
        ttk.Checkbutton(vis_frame, text="Objetivos", variable=self.ver_objetivos_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)
        ttk.Checkbutton(vis_frame, text="Intencion", variable=self.ver_intencion_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)
        ttk.Checkbutton(vis_frame, text="Defectos", variable=self.ver_defectos_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)
        ttk.Checkbutton(vis_frame, text="Voronoi", variable=self.ver_voronoi_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)
        ttk.Checkbutton(vis_frame, text="Triangulacion", variable=self.ver_triangulacion_var,
                        command=self._aplicar_visual).pack(side="left", padx=2)

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=(8, 2))
        ttk.Label(f, text="Generaciones de triangulacion (ver/ocultar):",
                  font=("Arial", 10, "bold")).pack(anchor="w")
        gen_frame = ttk.Frame(f)
        gen_frame.pack(fill="x", pady=4)
        self.ver_gen_vars = {}
        for g in range(1, MAX_GEN_VISIBLE + 1):
            var = tk.BooleanVar(value=self.sim.ver_generaciones.get(g, True))
            self.ver_gen_vars[g] = var
            ttk.Checkbutton(gen_frame, text=f"G{g}", variable=var,
                            command=self._aplicar_visual).grid(
                row=(g - 1) // 5, column=(g - 1) % 5, padx=2, pady=2, sticky="w")

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        # --- Etapa y transicion (modo dos etapas) ---
        ttk.Label(f, text="Etapa:", font=("Arial", 10, "bold")).pack(anchor="w")
        etapa_frame = ttk.Frame(f)
        etapa_frame.pack(fill="x", pady=4)
        self.etapa_label = ttk.Label(etapa_frame, text="Exploracion",
                                     font=("Arial", 10, "bold"), foreground="#88aaff")
        self.etapa_label.pack(side="left", padx=(0, 10))
        self.force_boton = ttk.Button(etapa_frame, text="Transicionar a Inspeccion",
                                      command=self._forzar_inspeccion, state="disabled")
        self.force_boton.pack(side="left")
        ttk.Label(f, text="Solo disponible en modo 'Dos etapas' (transicion automatica "
                          "Exploracion -> Inspeccion).",
                  font=("Arial", 8)).pack(anchor="w", pady=(2, 0))
        self._actualizar_force_boton()

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        # --- Simulacion ---
        ttk.Label(f, text="Simulacion:", font=("Arial", 10, "bold")).pack(anchor="w")
        sim_frame = ttk.Frame(f)
        sim_frame.pack(fill="x", pady=4)
        ttk.Button(sim_frame, text="Iniciar", command=self._iniciar_sim).pack(side="left", padx=2)
        ttk.Button(sim_frame, text="Pausar", command=self._pausar_sim).pack(side="left", padx=2)
        ttk.Button(sim_frame, text="Detener", command=self._detener_sim).pack(side="left", padx=2)

        self.estado_label = ttk.Label(f, text="Estado: EDIT", foreground="#8888ff")
        self.estado_label.pack(anchor="w", pady=(4, 0))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(f, text="Guardar / Cargar config:", font=("Arial", 10, "bold")).pack(anchor="w")
        cfg_frame = ttk.Frame(f)
        cfg_frame.pack(fill="x", pady=4)
        ttk.Button(cfg_frame, text="Guardar config", command=self._guardar_config).pack(side="left", padx=4)
        ttk.Button(cfg_frame, text="Cargar config", command=self._cargar_config).pack(side="left", padx=4)
        self.config_msg = ttk.Label(f, text="", font=("Arial", 8))
        self.config_msg.pack(anchor="w")

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(f, text="Atajos de teclado:", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(f, text="R: Robot   P: Pared   B: Borrar   ESC: Volver a editor",
                  font=("Arial", 9)).pack(anchor="w", pady=2)
        ttk.Label(f, text="V: Ver Voronoi (toggle)",
                  font=("Arial", 9)).pack(anchor="w", pady=2)

        self._sync_params_from_cfg()

    def _build_strategy_tab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

        f = ttk.Frame(scroll_frame, padding=12)
        f.pack(fill="both", expand=True)

        # --- Selector de tipo de simulacion (2 opciones) ---
        ttk.Label(f, text="Tipo de simulacion:", font=("Arial", 11, "bold")).pack(anchor="w")
        modo_frame = ttk.Frame(f)
        modo_frame.pack(fill="x", pady=4)
        self._modo_display = {
            "dos_etapas": "Dos etapas: explora -> inspecciona",
            "hibrido": "Hibrida (una etapa: explora + inspecciona)",
        }
        self._modo_interno = {v: k for k, v in self._modo_display.items()}
        self.modo_var = tk.StringVar(
            value=self._modo_display.get(self.sim.cfg.modo_fase, self.sim.cfg.modo_fase))
        self.modo_combo = ttk.Combobox(
            modo_frame, textvariable=self.modo_var,
            values=list(self._modo_display.values()),
            state="readonly", width=32)
        self.modo_combo.pack(side="left")
        self.modo_combo.bind("<<ComboboxSelected>>", self._cambiar_modo)

        # Label informativa del modo
        self.modo_info_label = ttk.Label(modo_frame, text="", font=("Arial", 9))
        self.modo_info_label.pack(side="left", padx=10)
        self._actualizar_modo_info()

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)

        # --- Estrategia de Exploracion ---
        self._exp_frame = ttk.LabelFrame(f, text="Estrategia de Exploracion",
                                          padding=8)
        self._exp_frame.pack(fill="x", pady=4)
        self.strategy_exp_var = tk.StringVar(value=self.sim.cfg.estrategia_exploracion)
        self.strategy_exp_combo = ttk.Combobox(
            self._exp_frame, textvariable=self.strategy_exp_var,
            values=list(ESTRATEGIAS_EXPLORACION),
            state="readonly", width=30)
        self.strategy_exp_combo.pack(side="left")
        self.strategy_exp_combo.bind("<<ComboboxSelected>>", self._cambiar_estrategia_exp)

        # --- Estrategia de Inspeccion ---
        self._insp_frame = ttk.LabelFrame(f, text="Estrategia de Inspeccion",
                                           padding=8)
        self._insp_frame.pack(fill="x", pady=4)
        self.strategy_insp_var = tk.StringVar(value=self.sim.cfg.estrategia_inspeccion)
        self.strategy_insp_combo = ttk.Combobox(
            self._insp_frame, textvariable=self.strategy_insp_var,
            values=list(ESTRATEGIAS_INSPECCION),
            state="readonly", width=30)
        self.strategy_insp_combo.pack(side="left")
        self.strategy_insp_combo.bind("<<ComboboxSelected>>", self._cambiar_estrategia_insp)

        # --- Estrategia Hibrida ---
        self._hib_frame = ttk.LabelFrame(f, text="Estrategia Hibrida",
                                          padding=8)
        self._hib_frame.pack(fill="x", pady=4)
        self.strategy_hib_var = tk.StringVar(value=self.sim.cfg.estrategia_hibrido)
        self.strategy_hib_combo = ttk.Combobox(
            self._hib_frame, textvariable=self.strategy_hib_var,
            values=list(ESTRATEGIAS_HIBRIDO),
            state="readonly", width=30)
        self.strategy_hib_combo.pack(side="left")
        self.strategy_hib_combo.bind("<<ComboboxSelected>>", self._cambiar_estrategia_hib)

        # Aplicar estado inicial de estrategias segun el modo
        self._actualizar_estado_dropdowns()
        self._actualizar_etiquetas_estrategias()

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(f, text="Los parametros se aplican a las estrategias que usan "
                          "planificador (Frontier, Sample, Hybrid Based, Boustrophedon).",
                  font=("Arial", 8)).pack(anchor="w", pady=(0, 4))

        self._build_param_section(f, "Objetivo global",
            ["exploration_goal"])
        self._build_param_section(f, "Planificacion",
            ["plan_interval_s", "goal_reached_radius_px", "goal_timeout_s"])
        self._build_param_section(f, "Fronteras (Frontier / Hybrid Based)",
            ["frontier_min_size", "frontier_max_size", "reasignar_umbral_costo"])
        self._build_param_section(f, "Muestreo (Sample / Hybrid Based)",
            ["sample_candidates", "sample_per_cluster", "sample_cost_weight", "sample_targets"])
        self._build_param_section(f, "Movimiento dirigido (Frontier / Sample / Hybrid Based)",
            ["dirigido_avance_min", "dirigido_umbral_giro_deg",
             "dirigido_giro_kp", "dirigido_pivote_umbral_deg"])
        self._build_param_section(f, "Evitacion reactiva (dirigido)",
            ["dirigido_peso_evasion_imminente",
             "dirigido_peso_evasion_fuerte",
             "dirigido_peso_evasion_suave"])

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(f, text="Random Walk no usa planificador; los robots se mueven "
                          "aleatoriamente.", font=("Arial", 8)).pack(anchor="w")

        self._sync_params_from_cfg()

    def _modo_actual(self):
        """Devuelve el valor interno del modo segun el combo (display)."""
        display = self.modo_var.get()
        return self._modo_interno.get(display, self.sim.cfg.modo_fase)

    def _cambiar_modo(self, *_):
        """Cambia el tipo de simulacion y actualiza la estrategia activa."""
        self.sim.cfg.modo_fase = self._modo_actual()
        self.sim.cfg.sync_strategy()
        self._actualizar_estado_dropdowns()
        self._actualizar_modo_info()
        self._actualizar_force_boton()
        if self.sim.estado == "running":
            self.sim.planner = self.sim._crear_planner()

    def _cambiar_estrategia_exp(self, *_):
        self.sim.cfg.estrategia_exploracion = self.strategy_exp_var.get()
        self.sim.cfg.sync_strategy()
        if self.sim.estado == "running" and self.sim.cfg.modo_fase == "dos_etapas":
            self.sim.planner = self.sim._crear_planner()

    def _cambiar_estrategia_insp(self, *_):
        self.sim.cfg.estrategia_inspeccion = self.strategy_insp_var.get()
        self.sim.cfg.sync_strategy()

    def _cambiar_estrategia_hib(self, *_):
        self.sim.cfg.estrategia_hibrido = self.strategy_hib_var.get()
        self.sim.cfg.sync_strategy()
        if self.sim.estado == "running" and self.sim.cfg.modo_fase == "hibrido":
            self.sim.planner = self.sim._crear_planner()

    def _actualizar_estado_dropdowns(self):
        """Habilita/deshabilita los dropdowns de estrategia segun el modo.

        Los marcos de estrategia siempre estan visibles; solo cambia el
        estado de cada combobox:
        - "dos_etapas": exploracion e inspeccion seleccionables; hibrida
          desactivada.
        - "hibrido":    solo hibrida seleccionable; exploracion e inspeccion
          desactivadas.
        """
        if self.sim.cfg.modo_fase == "dos_etapas":
            self.strategy_exp_combo.config(state="readonly")
            self.strategy_insp_combo.config(state="readonly")
            self.strategy_hib_combo.config(state="disabled")
        else:
            self.strategy_exp_combo.config(state="disabled")
            self.strategy_insp_combo.config(state="disabled")
            self.strategy_hib_combo.config(state="readonly")

    def _actualizar_modo_info(self):
        """Actualiza la label informativa del modo."""
        if not hasattr(self, "modo_info_label"):
            return
        modo = self.sim.cfg.modo_fase
        if modo == "dos_etapas":
            txt = "Exploracion -> Inspeccion (transicion dinamica)"
            fg = "#ffcc44"
        else:
            txt = "Una sola etapa: explora e inspecciona a la vez"
            fg = "#d0a0ff"
        self.modo_info_label.config(text=txt, foreground=fg)

    def _actualizar_etiquetas_estrategias(self):
        """Actualiza las etiquetas de los marcos de estrategia segun el modo."""
        if not hasattr(self, "_exp_frame"):
            return
        if self.sim.cfg.modo_fase == "dos_etapas":
            self._exp_frame.config(text="Estrategia de Exploracion (fase 1)")
            self._insp_frame.config(text="Estrategia de Inspeccion (fase 2)")
            self._hib_frame.config(text="Estrategia Hibrida (solo en modo Hibrida)")
        else:
            self._exp_frame.config(text="Estrategia de Exploracion (solo en Dos etapas)")
            self._insp_frame.config(text="Estrategia de Inspeccion (solo en Dos etapas)")
            self._hib_frame.config(text="Estrategia Hibrida (una etapa)")

    def _build_param_section(self, parent, title, keys):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text=title + ":", font=("Arial", 10, "bold")).pack(anchor="w")
        sub = ttk.Frame(parent)
        sub.pack(fill="x", pady=4)
        for i, key in enumerate(keys):
            label = _PARAM_LABELS.get(key, key)
            var = tk.StringVar()
            self._param_vars[key] = var
            row_f = ttk.Frame(sub)
            row_f.pack(fill="x", pady=1)
            ttk.Label(row_f, text=label, width=22, anchor="w").pack(side="left")
            e = ttk.Entry(row_f, textvariable=var, width=8)
            e.pack(side="left")
            var.trace_add("write", lambda *_, k=key: self._param_changed(k))

    def _param_changed(self, key):
        try:
            val = float(self._param_vars[key].get())
        except ValueError:
            return
        cfg = self.sim.cfg
        if key == "robot_radius_cm":
            cfg.robot_radius_cm = val
            cfg.robot_radius_px = val * cfg.pixels_per_cm
        elif key == "robot_max_speed_cm_s":
            cfg.robot_max_speed_cm_s = val
            cfg.robot_max_speed_px_s = val * cfg.pixels_per_cm
        elif key == "detection_radius_cm":
            cfg.detection_radius_cm = val
            cfg.detection_radius_px = val * cfg.pixels_per_cm
        elif key == "robot_turn_speed_rad_s":
            cfg.robot_turn_speed_rad_s = val
        elif key == "exploration_radius_multiplier":
            cfg.exploration_radius_multiplier = val
        elif key == "random_walk_interval_min":
            cfg.random_walk_interval_min = val
        elif key == "random_walk_interval_max":
            cfg.random_walk_interval_max = val
        elif key == "exploration_goal":
            cfg.exploration_goal = val
        elif key == "plan_interval_s":
            cfg.plan_interval_s = val
        elif key == "goal_reached_radius_px":
            cfg.goal_reached_radius_px = val
        elif key == "goal_timeout_s":
            cfg.goal_timeout_s = val
        elif key == "frontier_min_size":
            cfg.frontier_min_size = int(val)
        elif key == "frontier_max_size":
            cfg.frontier_max_size = int(val)
        elif key == "reasignar_umbral_costo":
            cfg.reasignar_umbral_costo = val
        elif key == "sample_candidates":
            cfg.sample_candidates = int(val)
        elif key == "sample_per_cluster":
            cfg.sample_per_cluster = int(val)
        elif key == "sample_cost_weight":
            cfg.sample_cost_weight = val
        elif key == "sample_targets":
            cfg.sample_targets = int(val)
        elif key == "dirigido_avance_min":
            cfg.dirigido_avance_min = val
        elif key == "dirigido_peso_evasion_imminente":
            cfg.dirigido_peso_evasion_imminente = val
        elif key == "dirigido_peso_evasion_fuerte":
            cfg.dirigido_peso_evasion_fuerte = val
        elif key == "dirigido_peso_evasion_suave":
            cfg.dirigido_peso_evasion_suave = val
        elif key == "dirigido_umbral_giro_deg":
            cfg.dirigido_umbral_giro_deg = val
        elif key == "dirigido_giro_kp":
            cfg.dirigido_giro_kp = val
        elif key == "dirigido_pivote_umbral_deg":
            cfg.dirigido_pivote_umbral_deg = val

    def _sync_params_from_cfg(self):
        cfg = self.sim.cfg
        mapping = {
            "robot_radius_cm": str(cfg.robot_radius_cm),
            "robot_max_speed_cm_s": str(cfg.robot_max_speed_cm_s),
            "robot_turn_speed_rad_s": str(cfg.robot_turn_speed_rad_s),
            "detection_radius_cm": str(cfg.detection_radius_cm),
            "exploration_radius_multiplier": str(cfg.exploration_radius_multiplier),
            "random_walk_interval_min": str(cfg.random_walk_interval_min),
            "random_walk_interval_max": str(cfg.random_walk_interval_max),
            "exploration_goal": str(cfg.exploration_goal),
            "plan_interval_s": str(cfg.plan_interval_s),
            "goal_reached_radius_px": str(cfg.goal_reached_radius_px),
            "goal_timeout_s": str(cfg.goal_timeout_s),
            "frontier_min_size": str(cfg.frontier_min_size),
            "frontier_max_size": str(cfg.frontier_max_size),
            "reasignar_umbral_costo": str(cfg.reasignar_umbral_costo),
            "sample_candidates": str(cfg.sample_candidates),
            "sample_per_cluster": str(cfg.sample_per_cluster),
            "sample_cost_weight": str(cfg.sample_cost_weight),
            "sample_targets": str(cfg.sample_targets),
            "dirigido_avance_min": str(cfg.dirigido_avance_min),
            "dirigido_peso_evasion_imminente": str(cfg.dirigido_peso_evasion_imminente),
            "dirigido_peso_evasion_fuerte": str(cfg.dirigido_peso_evasion_fuerte),
            "dirigido_peso_evasion_suave": str(cfg.dirigido_peso_evasion_suave),
            "dirigido_umbral_giro_deg": str(cfg.dirigido_umbral_giro_deg),
            "dirigido_giro_kp": str(cfg.dirigido_giro_kp),
            "dirigido_pivote_umbral_deg": str(cfg.dirigido_pivote_umbral_deg),
        }
        for key, val in mapping.items():
            if key in self._param_vars:
                self._param_vars[key].set(val)

    def _guardar_config(self):
        params_path = "sim_config.txt"
        try:
            cfg = self.sim.cfg
            lines = [
                f"robot_radius_cm={cfg.robot_radius_cm}",
                f"robot_max_speed_cm_s={cfg.robot_max_speed_cm_s}",
                f"robot_turn_speed_rad_s={cfg.robot_turn_speed_rad_s}",
                f"detection_radius_cm={cfg.detection_radius_cm}",
                f"exploration_radius_multiplier={cfg.exploration_radius_multiplier}",
                f"random_walk_interval_min={cfg.random_walk_interval_min}",
                f"random_walk_interval_max={cfg.random_walk_interval_max}",
                f"exploration_goal={cfg.exploration_goal}",
                f"plan_interval_s={cfg.plan_interval_s}",
                f"goal_reached_radius_px={cfg.goal_reached_radius_px}",
                f"goal_timeout_s={cfg.goal_timeout_s}",
                f"frontier_min_size={cfg.frontier_min_size}",
                f"frontier_max_size={cfg.frontier_max_size}",
                f"reasignar_umbral_costo={cfg.reasignar_umbral_costo}",
                f"sample_candidates={cfg.sample_candidates}",
                f"sample_per_cluster={cfg.sample_per_cluster}",
                f"sample_cost_weight={cfg.sample_cost_weight}",
                f"sample_targets={cfg.sample_targets}",
                f"dirigido_avance_min={cfg.dirigido_avance_min}",
                f"dirigido_peso_evasion_imminente={cfg.dirigido_peso_evasion_imminente}",
                f"dirigido_peso_evasion_fuerte={cfg.dirigido_peso_evasion_fuerte}",
                f"dirigido_peso_evasion_suave={cfg.dirigido_peso_evasion_suave}",
                f"dirigido_umbral_giro_deg={cfg.dirigido_umbral_giro_deg}",
                f"dirigido_giro_kp={cfg.dirigido_giro_kp}",
                f"dirigido_pivote_umbral_deg={cfg.dirigido_pivote_umbral_deg}",
                f"modo_fase={cfg.modo_fase}",
                f"estrategia_exploracion={cfg.estrategia_exploracion}",
                f"estrategia_inspeccion={cfg.estrategia_inspeccion}",
                f"estrategia_hibrido={cfg.estrategia_hibrido}",
            ]
            with open(params_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            self.config_msg.config(text=f"Guardado en {params_path}", foreground="#8f8")
        except Exception as e:
            self.config_msg.config(text=f"Error al guardar: {e}", foreground="#f88")

    def _cargar_config(self):
        params_path = "sim_config.txt"
        try:
            with open(params_path, "r") as f:
                data = {}
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        data[k.strip()] = v.strip()
            for key, var in self._param_vars.items():
                if key in data:
                    var.set(data[key])

            # Cargar modo y estrategias
            if "modo_fase" in data:
                self.sim.cfg.modo_fase = data["modo_fase"]
                self.sim.cfg._normalizar_modo()
                self.modo_var.set(self._modo_display.get(
                    self.sim.cfg.modo_fase, self.sim.cfg.modo_fase))
            if "estrategia_exploracion" in data:
                self.sim.cfg.estrategia_exploracion = data["estrategia_exploracion"]
                self.strategy_exp_var.set(data["estrategia_exploracion"])
            if "estrategia_inspeccion" in data:
                self.sim.cfg.estrategia_inspeccion = data["estrategia_inspeccion"]
                self.strategy_insp_var.set(data["estrategia_inspeccion"])
            if "estrategia_hibrido" in data:
                self.sim.cfg.estrategia_hibrido = data["estrategia_hibrido"]
                self.strategy_hib_var.set(data["estrategia_hibrido"])

            self.sim.cfg.sync_strategy()
            self._actualizar_estado_dropdowns()
            self._actualizar_modo_info()
            self._actualizar_etiquetas_estrategias()
            self._actualizar_force_boton()

            if self.sim.estado == "running":
                self.sim.planner = self.sim._crear_planner()
            self.config_msg.config(text=f"Cargado desde {params_path}", foreground="#8f8")
        except FileNotFoundError:
            self.config_msg.config(text=f"No encontrado: {params_path}", foreground="#f88")
        except Exception as e:
            self.config_msg.config(text=f"Error al cargar: {e}", foreground="#f88")

    def _aplicar_grid_size(self):
        try:
            val = int(self.grid_size_var.get())
            if val >= 10 and val <= 100:
                self.sim.cfg.grid_size = val
                self.sim.cfg.cols = self.sim.cfg.map_width // val
                self.sim.cfg.rows = self.sim.cfg.map_height // val
                self.sim._reconstruir_mundo()
                self._sync_params_from_cfg()
        except ValueError:
            pass

    def _poblar(self):
        try:
            n = int(self.pop_count_var.get())
        except ValueError:
            return
        self.sim.poblar_robots(n)

    def _limpiar(self):
        self.sim._reconstruir_mundo()

    def _reset_robots(self):
        self.sim.mundo.robots.clear()
        self.sim.mundo.reiniciar_exploracion()

    def _iniciar_sim(self):
        self.sim.iniciar()

    def _aplicar_visual(self):
        self.sim.ver_fronteras = self.ver_fronteras_var.get()
        self.sim.ver_objetivos = self.ver_objetivos_var.get()
        self.sim.ver_intencion = self.ver_intencion_var.get()
        if hasattr(self, "ver_defectos_var"):
            self.sim.ver_defectos = self.ver_defectos_var.get()
        if hasattr(self, "ver_voronoi_var"):
            self.sim.ver_voronoi = self.ver_voronoi_var.get()
        if hasattr(self, "ver_triangulacion_var"):
            self.sim.ver_triangulacion = self.ver_triangulacion_var.get()
        if hasattr(self, "ver_gen_vars"):
            for g, var in self.ver_gen_vars.items():
                self.sim.ver_generaciones[g] = var.get()

    def _forzar_inspeccion(self):
        self.sim.forzar_transicion_inspeccion()
        self._actualizar_force_boton()

    def _actualizar_force_boton(self):
        """Habilita el boton de transicion solo en modo dos_etapas."""
        if not hasattr(self, "force_boton"):
            return
        es_dos_etapas = self.sim.cfg.modo_fase == "dos_etapas"
        en_marcha = self.sim.estado == "running" and not self.sim.transicionado
        estado = "normal" if (es_dos_etapas and en_marcha) else "disabled"
        self.force_boton.config(state=estado)

    def _pausar_sim(self):
        if self.sim.estado == "running":
            self.sim.estado = "paused"

    def _detener_sim(self):
        self.sim.estado = "edicion"
        self.sim._reconstruir_mundo()

    def actualizar(self):
        if not self._alive:
            return
        try:
            self._do_actualizar()
        except tk.TclError:
            self._alive = False

    def _do_actualizar(self):
        m = self.sim.mundo
        if m is None:
            return
        self.estado_label.config(text=f"Estado: {self.sim.estado.upper()}")
        if self.sim.estado == "completado":
            self.estado_label.config(foreground="#ffcc00")
        else:
            self.estado_label.config(foreground="#8888ff")
        if self.tool_var.get() != self.sim.herramienta:
            self.tool_var.set(self.sim.herramienta)
        self._actualizar_lista_robots()
        self._actualizar_individual()
        self._actualizar_individual_map()
        self._actualizar_cobertura()
        self._actualizar_global()
        self._actualizar_modo_info()
        self._actualizar_force_boton()
        self._actualizar_etapa_label()

    def _actualizar_etapa_label(self):
        if not hasattr(self, "etapa_label"):
            return
        if self.sim.cfg.modo_fase == "hibrido":
            self.etapa_label.config(text="Hibrida (una etapa)", foreground="#d0a0ff")
            return
        if self.sim.transicionado:
            texto = "Inspeccion"
            fg = "#8affa0"
        else:
            texto = "Exploracion"
            fg = "#88aaff"
        self.etapa_label.config(text=texto, foreground=fg)

    def _actualizar_lista_robots(self):
        robots = self.sim.mundo.robots
        opciones = [f"Robot {i}" for i in range(len(robots))]
        self.robot_menu["values"] = opciones
        if opciones and not self.robot_var.get():
            self.robot_var.set(opciones[0])
        elif not opciones:
            self.robot_var.set("")

    def _actualizar_individual(self):
        self.canvas_ind.delete("all")
        cw = self.canvas_ind.winfo_width() or 480
        h = self.canvas_ind.winfo_height() or 200
        robots = self.sim.mundo.robots
        sel = self.robot_var.get()
        if not sel or not robots:
            self.bateria_bar["value"] = 0.0
            self.bateria_label.config(text="--")
            self.canvas_ind.create_text(cw // 2, h // 2, text="Sin datos",
                                        fill="#666", font=("Arial", 14))
            return
        idx = int(sel.split()[1])
        if idx >= len(robots):
            self.bateria_bar["value"] = 0.0
            self.bateria_label.config(text="--")
            return
        r = robots[idx]
        valores = [
            ("Pisadas", r.visitadas_robot, COL_VISITED),
            ("Scouteadas", r.scouteadas_robot, COL_SCOUTED),
            ("Muros", r.paredes_robot, COL_DISCOVERED),
            ("Tapadas", r.energia.picos_arranque, (255, 200, 50)),
        ]
        max_val = max((v for _, v, _ in valores), default=1)
        if max_val == 0:
            max_val = 1
        self.canvas_ind.create_text(cw // 2, 14, text=sel, fill="#fff",
                                    font=("Arial", 12, "bold"))
        bar_w = 70
        spacing = 18
        total_w = len(valores) * bar_w + (len(valores) - 1) * spacing
        x_start = (cw - total_w) // 2
        graph_h = h - 50
        y_base = h - 20
        for i, (label, val, color) in enumerate(valores):
            x = x_start + i * (bar_w + spacing)
            bar_h = int((val / max_val) * graph_h) if max_val > 0 else 0
            if bar_h > 0:
                self.canvas_ind.create_rectangle(x, y_base - bar_h, x + bar_w, y_base,
                                                 fill=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                                 outline="")
            self.canvas_ind.create_text(x + bar_w // 2, y_base + 10, text=label,
                                        fill="#aaa", font=("Arial", 8))
            self.canvas_ind.create_text(x + bar_w // 2, y_base - bar_h - 8, text=str(val),
                                        fill="#fff", font=("Arial", 9, "bold"))

        # Bateria del robot seleccionado
        self.bateria_bar["value"] = r.energia.porcentaje_bateria
        info_extra = ""
        planner = getattr(self.sim, "planner", None)
        if planner and hasattr(planner, "cerebros") and id(r) in planner.cerebros:
            cerebro = planner.cerebros[id(r)]
            info_extra = f" | Dopamina: {cerebro.dopamina_actual:+.2f}"

        self.bateria_label.config(
            text=f"{r.energia.porcentaje_bateria:.1f}%  ({r.energia.energia_consumida_total:.1f} J){info_extra}")


    def _actualizar_individual_map(self):
        self.canvas_robot_map.delete("all")
        cw = self.canvas_robot_map.winfo_width() or 480
        ch = self.canvas_robot_map.winfo_height() or 200
        robots = self.sim.mundo.robots
        sel = self.robot_var.get()
        if not sel or not robots:
            return
        idx = int(sel.split()[1])
        if idx >= len(robots):
            return
        robot = robots[idx]
        m = self.sim.mundo
        rows, cols = m.rows, m.cols
        scale = min((cw - 20) // cols, ch - 10)
        if scale < 1:
            return
        img_w = cols * scale
        img_h = rows * scale
        arr = np.zeros((rows, cols, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                val = robot.explored_robot[r, c]
                if m.grid[r, c] == 0:
                    if val == SCOUTED:
                        arr[r, c] = COL_SCOUTED
                    elif val == VISITED:
                        arr[r, c] = COL_VISITED
                    else:
                        arr[r, c] = COL_FONDO
                else:
                    if val == DISCOVERED:
                        arr[r, c] = COL_DISCOVERED
                    else:
                        arr[r, c] = COL_MURO_NO_VISTO
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
        img = Image.fromarray(arr, "RGB")
        self._img_robot_tk = ImageTk.PhotoImage(img)
        self.canvas_robot_map.create_image(cw // 2, ch // 2,
                                           image=self._img_robot_tk, anchor="center")

    def _actualizar_cobertura(self):
        self.canvas_cob.delete("all")
        cw = self.canvas_cob.winfo_width() or 480
        ch = self.canvas_cob.winfo_height() or 400
        m = self.sim.mundo
        rows, cols = m.rows, m.cols
        scale = min(cw // cols, ch // rows)
        if scale < 1:
            return
        arr = np.zeros((rows, cols, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                if m.grid[r, c] == 0:
                    if m.explored[r, c] == 0:
                        arr[r, c] = COL_FONDO
                    elif m.explored[r, c] == 1:
                        arr[r, c] = COL_SCOUTED
                    elif m.explored[r, c] == 2:
                        arr[r, c] = COL_VISITED
                    else:
                        arr[r, c] = COL_FONDO
                else:
                    if m.explored[r, c] == 3:
                        arr[r, c] = COL_DISCOVERED
                    else:
                        arr[r, c] = COL_MURO_NO_VISTO
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
        img = Image.fromarray(arr, "RGB")
        self._img_tk = ImageTk.PhotoImage(img)
        self.canvas_cob.create_image(cw // 2, ch // 2, image=self._img_tk, anchor="center")

    def _actualizar_global(self):
        m = self.sim.mundo
        total = m.rows * m.cols
        libres = m.total_libres
        paredes = m.total_paredes
        v = m._visitadas
        s = m._scouteadas
        d = m._paredes_vistas
        pct_libres = 100.0 * (v + s) / libres if libres else 0
        pct_paredes = 100.0 * d / paredes if paredes else 0
        pct_total = 100.0 * (v + s + d) / total if total else 0
        tiempo = self.sim.tiempo_simulado
        fps = self.sim.fps_actual
        estrategia = self.sim.cfg._estrategia_activa(self.sim.etapa)
        modo = self.sim.cfg.modo_fase
        if self.sim.planner is not None and hasattr(self.sim.planner, "objetivos_pendientes"):
            pendientes = self.sim.planner.objetivos_pendientes
        else:
            pendientes = 0
        if self.sim.estado == "completado":
            resultado = (f"COMPLETADO: mapa al {pct_total:.1f}% en {self.sim.tiempo_final:.2f} s\n"
                         f"(no es necesario visitar todas las casillas)\n")
        else:
            resultado = f"Objetivos pendientes: {pendientes}\n"

        energia_total = sum(r.energia.energia_consumida_total for r in m.robots)
        energia_mov = sum(r.energia.consumida_movimiento for r in m.robots)
        energia_ins = sum(r.energia.consumida_inspeccion for r in m.robots)
        picos = sum(r.energia.picos_arranque for r in m.robots)
        bateria_media = (sum(r.energia.porcentaje_bateria for r in m.robots) / len(m.robots)
                         if m.robots else 0.0)

        # Conteos de celdas (vista / visitada / inspeccionada / defectuosa)
        vistas_c = m._scouteadas + m._visitadas
        ins = m._inspeccionadas
        def_c = m._defectuosas
        pct_vistas = 100.0 * vistas_c / libres if libres else 0.0
        pct_visit = 100.0 * v / libres if libres else 0.0
        pct_ins = 100.0 * ins / libres if libres else 0.0
        pct_def = 100.0 * def_c / ins if ins else 0.0

        por_robot = []
        for i, r in enumerate(m.robots):
            pr = 100.0 * r.inspeccionadas_robot / libres if libres else 0.0
            por_robot.append(f"  R{i}: vistas={r.scouteadas_robot} "
                             f"visit={r.visitadas_robot} "
                             f"inspec={r.inspeccionadas_robot}({pr:.1f}%) "
                             f"defect={r.defectuosas_robot}")
        por_robot_txt = "\n".join(por_robot) if por_robot else "  (sin robots)"

        # Para modo dos_etapas, mostrar ambas estrategias
        if modo == "dos_etapas":
            estrategia_display = ("exp:{} / ins:{}".format(
                self.sim.cfg.estrategia_exploracion,
                self.sim.cfg.estrategia_inspeccion))
        else:
            estrategia_display = estrategia

        etapa_display = ("HIBRIDA (una etapa)" if modo == "hibrido"
                     else self.sim.etapa.upper())
        texto = (
            f"Estrategia:        {estrategia_display} [modo: {modo}]\n"
            f"Etapa:             {etapa_display}\n"
            f"Tiempo simulado:   {tiempo:.2f} s\n"
            f"FPS:               {fps:.1f}\n"
            f"Pasos totales:     {self.sim.pasos_totales}\n"
            f"---\n"
            f"Celdas totales:    {total}\n"
            f"Libres:            {libres}\n"
            f"Paredes:           {paredes}\n"
            f"---\n"
            f"Celdas vistas:     {vistas_c}  ({pct_vistas:.1f}% libres)\n"
            f"Celdas visitadas:  {v}  ({pct_visit:.1f}% libres)\n"
            f"Celdas inspecc.:   {ins}  ({pct_ins:.1f}% libres)\n"
            f"Celdas defect.:    {def_c}  ({pct_def:.1f}% inspeccionadas)\n"
            f"---\n"
            f"Muros descubiertos:{d}  ({pct_paredes:.1f}% paredes)\n"
            f"Cobertura libre:   {pct_libres:.1f}%\n"
            f"Cobertura paredes: {pct_paredes:.1f}%\n"
            f"Mapa descubierto:  {pct_total:.1f}%\n"
            f"---\n"
            f"Robots:            {len(m.robots)}\n"
            f"Por robot:\n{por_robot_txt}\n"
            f"Energia enjambre:  {energia_total:.1f} J  (mov {energia_mov:.1f} + inspeccion {energia_ins:.1f})\n"
            f"Bateria media:     {bateria_media:.1f}%  (picos de arranque: {picos})\n"
            f"---\n"
            f"{resultado}"
        )
        self.text_global.config(state="normal")
        self.text_global.delete("1.0", "end")
        self.text_global.insert("1.0", texto)
        self.text_global.config(state="disabled")
