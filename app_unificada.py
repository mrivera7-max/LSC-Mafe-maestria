"""
LSC UDI — Ventana unificada (cámara + voz/texto en una sola aplicación)
========================================================================
Reúne los dos sistemas en una sola ventana con pestañas:

  ┌─────────────────────────────────────────────────┐
  │  [ Cámara → Seña ]   [ Voz/Texto → Seña ]        │  <- pestañas
  ├─────────────────────────────────────────────────┤
  │                                                   │
  │   (contenido de la pestaña activa)                │
  │                                                   │
  └─────────────────────────────────────────────────┘

El avatar 3D (Three.js/WebGL) sigue abriéndose en el navegador: Tkinter
no puede renderizar WebGL dentro de una ventana. La pestaña de voz lo
lanza igual que antes, mediante el servidor HTTP/WebSocket.

Uso:
    python app_unificada.py             # ambas pestañas (sin robot en esta versión)
    python app_unificada.py --robot     # habilita el Unitree G1 (usar_robot)
    python app_unificada.py --v2        # reconocedor v2 (recomendado)
    python app_unificada.py --comparar  # selector RF / MLP / MobileNetV2 + informes
"""

import argparse
import logging
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import configurar_logger
from utils.config import Configuracion
from gui.ventana_principal import VentanaPrincipal
from voz_a_sena.gui_principal import VentanaVozASena

log = logging.getLogger("lsc_udi.app")


class AppUnificada:
    """Contenedor único: una ventana raíz, un Notebook, dos sistemas."""

    TITULO = "LSC UDI — Lengua de Señas Colombiana"
    ANCHO_MIN = 1100
    ALTO_MIN = 700

    def __init__(self, config):
        self.config = config
        self._raiz = None
        self.panel_camara = None   # instancia de VentanaPrincipal
        self.panel_voz = None      # instancia de VentanaVozASena

    def ejecutar(self):
        self._raiz = tk.Tk()
        self._raiz.title(self.TITULO)
        self._raiz.minsize(self.ANCHO_MIN, self.ALTO_MIN)
        self._raiz.configure(bg="#1a1a2e")
        self._raiz.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        # ── Notebook (pestañas) ───────────────────────────────────
        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("TNotebook", background="#1a1a2e", borderwidth=0)
        estilo.configure("TNotebook.Tab", padding=(20, 10),
                         font=("Segoe UI", 10, "bold"))
        estilo.map("TNotebook.Tab",
                   background=[("selected", "#0f3460"), ("!selected", "#16213e")],
                   foreground=[("selected", "#e2e8f0"), ("!selected", "#94a3b8")])

        notebook = ttk.Notebook(self._raiz)
        notebook.pack(fill="both", expand=True)

        tab_cam = ttk.Frame(notebook)
        tab_voz = ttk.Frame(notebook)
        notebook.add(tab_cam, text="  📷  Cámara → Seña  ")
        notebook.add(tab_voz, text="  🎤  Voz / Texto → Seña  ")

        # ── Montar cada sistema dentro de su pestaña ──────────────
        # Cada clase construye su UI sobre el frame que le pasamos,
        # en vez de crear su propio tk.Tk(). Comparten la misma raíz.
        self.panel_camara = VentanaPrincipal(self.config)
        self.panel_camara.montar_en(tab_cam, self._raiz)

        self.panel_voz = VentanaVozASena()
        self.panel_voz.montar_en(tab_voz, self._raiz)
        self.panel_voz.robot = self.panel_camara._robot  # compartir el conector

        log.info("Ventana unificada iniciada (2 pestañas)")
        self._raiz.mainloop()

    def _al_cerrar(self):
        log.info("Cerrando LSC UDI...")
        # Cerrar limpiamente ambos sistemas
        try:
            if self.panel_camara:
                self.panel_camara.cerrar()
        except Exception as e:
            log.warning(f"Error al cerrar cámara: {e}")
        try:
            if self.panel_voz:
                self.panel_voz.cerrar()
        except Exception as e:
            log.warning(f"Error al cerrar voz: {e}")
        self._raiz.quit()
        self._raiz.destroy()


def parsear_args():
    p = argparse.ArgumentParser(description="LSC UDI — aplicación unificada")
    p.add_argument("--robot", action="store_true",
                   help="Habilitar el robot Unitree G1 (botón, estado y envío de señas)")
    p.add_argument("--sin-robot", action="store_true",
                   help="Forzar ejecución sin robot (por defecto en esta versión)")
    p.add_argument("--v2", action="store_true",
                   help="Usar reconocedor v2 (secuencial mano+cara, recomendado)")
    p.add_argument("--comparar", action="store_true",
                   help="Modo comparación: cambiar entre RF / MLP / MobileNetV2 y generar informes")
    p.add_argument("--camara", type=int, default=0, help="Índice de cámara")
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--log-nivel", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO")
    return p.parse_args()


def main():
    args = parsear_args()
    configurar_logger(nivel=args.log_nivel)
    log.info("=" * 60)
    log.info("  LSC UDI — Aplicación unificada (cámara + voz)")
    log.info("=" * 60)

    config = Configuracion(args.config)
    config.camara_idx = args.camara
    config.usar_robot = (args.robot or config.usar_robot) and not args.sin_robot
    config.robot_activo = config.usar_robot
    config.usar_v2 = args.v2 or config.usar_v2
    config.usar_comparativo = args.comparar or config.usar_comparativo

    app = AppUnificada(config)
    app.ejecutar()


if __name__ == "__main__":
    main()
