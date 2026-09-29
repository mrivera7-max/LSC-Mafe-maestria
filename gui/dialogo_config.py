"""
Diálogo de configuración del sistema.
"""

import tkinter as tk
from tkinter import ttk


class DialogoConfig(tk.Toplevel):
    """Ventana modal de configuración."""

    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.title("Configuración — LSC Bridge")
        self.resizable(False, False)
        self.grab_set()  # Modal

        self._bg = "#1a1a2e"
        self._panel = "#16213e"
        self._texto = "#e2e8f0"
        self._texto_sec = "#94a3b8"
        self.configure(bg=self._bg)

        self._construir()
        self._centrar()

    def _construir(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=16)

        # Pestaña: Cámara
        tab_cam = tk.Frame(notebook, bg=self._panel, padx=16, pady=16)
        notebook.add(tab_cam, text="  Cámara  ")
        self._campo(tab_cam, "Índice de cámara:", "camara_idx", 0)
        self._campo(tab_cam, "Resolución ancho:", "camara_ancho", 1)
        self._campo(tab_cam, "Resolución alto:", "camara_alto", 2)
        self._campo(tab_cam, "FPS objetivo:", "camara_fps", 3)

        # Pestaña: Modelo
        tab_mod = tk.Frame(notebook, bg=self._panel, padx=16, pady=16)
        notebook.add(tab_mod, text="  Modelo  ")
        self._campo(tab_mod, "Umbral de confianza (0-1):", "umbral_confianza", 0)
        self._campo(tab_mod, "Número máximo de manos:", "num_manos", 1)
        self._campo(tab_mod, "Ruta al modelo (.pkl):", "modelo_pesos", 2)

        # Pestaña: Robot G1 (solo si el robot está habilitado)
        if getattr(self.config, "usar_robot", False):
            tab_rob = tk.Frame(notebook, bg=self._panel, padx=16, pady=16)
            notebook.add(tab_rob, text="  Robot G1  ")
            self._campo(tab_rob, "IP del robot:", "robot_ip", 0)
            self._campo(tab_rob, "Puerto SDK:", "robot_puerto", 1)
            self._campo(tab_rob, "Timeout (seg):", "robot_timeout", 2)
            self._campo(tab_rob, "Velocidad máx. (0-1):", "robot_velocidad_max", 3)

        # Botones
        frame_btn = tk.Frame(self, bg=self._bg)
        frame_btn.pack(fill="x", padx=16, pady=(0, 16))

        tk.Button(frame_btn, text="Guardar", bg="#22c55e", fg="#052e16",
                  relief="flat", font=("Segoe UI", 10, "bold"),
                  padx=16, pady=6, cursor="hand2",
                  command=self._guardar).pack(side="right", padx=(8, 0))

        tk.Button(frame_btn, text="Cancelar", bg="#374151", fg=self._texto,
                  relief="flat", font=("Segoe UI", 10),
                  padx=16, pady=6, cursor="hand2",
                  command=self.destroy).pack(side="right")

    def _campo(self, parent, etiqueta: str, atributo: str, fila: int):
        tk.Label(parent, text=etiqueta, bg=self._panel, fg=self._texto_sec,
                 font=("Segoe UI", 9)).grid(row=fila, column=0, sticky="w", pady=6, padx=(0, 16))

        var = tk.StringVar(value=str(getattr(self.config, atributo, "")))
        entry = tk.Entry(parent, textvariable=var, bg="#0d1117", fg=self._texto,
                         insertbackground=self._texto, relief="flat",
                         font=("Segoe UI", 10), width=28)
        entry.grid(row=fila, column=1, sticky="ew", pady=6)
        parent.columnconfigure(1, weight=1)
        setattr(self, f"_var_{atributo}", (var, atributo))

    def _guardar(self):
        for nombre, (var, atributo) in {
            k: v for k, v in self.__dict__.items()
            if k.startswith("_var_")
        }.items():
            val_str = var.get()
            actual = getattr(self.config, atributo)
            try:
                if isinstance(actual, int):
                    setattr(self.config, atributo, int(val_str))
                elif isinstance(actual, float):
                    setattr(self.config, atributo, float(val_str))
                else:
                    setattr(self.config, atributo, val_str)
            except (ValueError, TypeError):
                pass

        self.config.guardar()
        self.destroy()

    def _centrar(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = self.master.winfo_x() + (self.master.winfo_width() - w) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - h) // 2
        self.geometry(f"+{x}+{y}")
