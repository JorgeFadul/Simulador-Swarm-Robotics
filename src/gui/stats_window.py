import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from ..core.mundo import SCOUTED, VISITED, DISCOVERED

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

        self._img_tk = None
        self._img_robot_tk = None

    def _on_close(self):
        self._alive = False
        self.window.destroy()

    def _build_individual_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Robot:").pack(side="left")
        self.robot_var = tk.StringVar()
        self.robot_menu = ttk.Combobox(top, textvariable=self.robot_var, state="readonly", width=10)
        self.robot_menu.pack(side="left", padx=8)

        self.canvas_ind = tk.Canvas(parent, bg="#111118", height=220)
        self.canvas_ind.pack(fill="x", padx=8)

        self.canvas_robot_map = tk.Canvas(parent, bg="#111118", height=220)
        self.canvas_robot_map.pack(fill="both", expand=True, padx=8, pady=(0, 8))

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
        ttk.Label(f, text="Estrategia:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.strategy_var = tk.StringVar(value=self.sim.cfg.strategy)
        strat_frame = ttk.Frame(f)
        strat_frame.pack(fill="x", pady=4)
        ttk.Combobox(strat_frame, textvariable=self.strategy_var,
                      values=["Random Walk", "Exploracion Sistematica", "Barrido"],
                      state="readonly", width=30).pack(side="left")

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

        self._sync_params_from_cfg()

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
                f"strategy={self.strategy_var.get()}",
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
            if "strategy" in data:
                self.strategy_var.set(data["strategy"])
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
        self.sim.estado = "running"

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
        if self.tool_var.get() != self.sim.herramienta:
            self.tool_var.set(self.sim.herramienta)
        self._actualizar_lista_robots()
        self._actualizar_individual()
        self._actualizar_individual_map()
        self._actualizar_cobertura()
        self._actualizar_global()

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
            self.canvas_ind.create_text(cw // 2, h // 2, text="Sin datos",
                                        fill="#666", font=("Arial", 14))
            return
        idx = int(sel.split()[1])
        if idx >= len(robots):
            return
        r = robots[idx]
        valores = [
            ("Pisadas", r.visitadas_robot, COL_VISITED),
            ("Scouteadas", r.scouteadas_robot, COL_SCOUTED),
            ("Muros", r.paredes_robot, COL_DISCOVERED),
        ]
        max_val = max((v for _, v, _ in valores), default=1)
        if max_val == 0:
            max_val = 1
        self.canvas_ind.create_text(cw // 2, 14, text=sel, fill="#fff",
                                    font=("Arial", 12, "bold"))
        bar_w = 70
        spacing = 25
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
        tiempo = self.sim.tiempo_simulado
        fps = self.sim.fps_actual
        texto = (
            f"Tiempo simulado:   {tiempo:.2f} s\n"
            f"FPS:               {fps:.1f}\n"
            f"Pasos totales:     {self.sim.pasos_totales}\n"
            f"---\n"
            f"Celdas totales:    {total}\n"
            f"Libres:            {libres}\n"
            f"Paredes:           {paredes}\n"
            f"---\n"
            f"Visitadas:         {v}  ({100*v/libres:.1f}% libres)\n"
            f"Scouteadas:        {s}  ({100*s/libres:.1f}% libres)\n"
            f"Muros descubiertos:{d}  ({pct_paredes:.1f}% paredes)\n"
            f"---\n"
            f"Cobertura libre:   {pct_libres:.1f}%\n"
            f"Cobertura paredes: {pct_paredes:.1f}%\n"
            f"Cobertura total:   {100.0*(v+s+d)/total:.1f}%\n"
            f"---\n"
            f"Robots:            {len(m.robots)}\n"
        )
        self.text_global.config(state="normal")
        self.text_global.delete("1.0", "end")
        self.text_global.insert("1.0", texto)
        self.text_global.config(state="disabled")
