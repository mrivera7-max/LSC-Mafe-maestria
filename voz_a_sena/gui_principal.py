"""
Interfaz grafica del sistema Voz/Texto -> Sena LSC.

Permite escribir texto o usar el microfono, y muestra la traduccion
mientras se transmite al visor 3D y al robot G1 (si está conectado).
"""

import logging
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import configurar_logger
from voz_a_sena.servidor import ServidorVozASena, PUERTO_HTTP

log = logging.getLogger("voz_a_sena.gui")


class VentanaVozASena:
    TITULO = "LSC UDI — Voz/Texto a Seña"

    def __init__(self):
        self.servidor = ServidorVozASena()
        self.robot = None  # ConectorG1 compartido; lo asigna app_unificada.py
        self._raiz = None
        self._contenedor = None
        self._escuchando = False
        self._servidor_iniciado = False

    def ejecutar(self):
        """Modo autónomo: crea su propia ventana y mainloop."""
        self._raiz = tk.Tk()
        self._raiz.title(self.TITULO)
        self._raiz.geometry("640x560")
        self._raiz.configure(bg="#1a1a2e")
        self._raiz.protocol("WM_DELETE_WINDOW", self._cerrar)

        self._contenedor = self._raiz
        self._construir_ui()

        log.info("Iniciando servidores HTTP y WebSocket...")
        self.servidor.iniciar_servidores()
        self._servidor_iniciado = True

        self._raiz.after(800, self._abrir_visor)
        self._raiz.mainloop()

    def montar_en(self, contenedor, raiz):
        """Construye la UI de voz dentro de un contenedor externo (pestaña),
        compartiendo la raíz de la app unificada. No crea tk.Tk() ni mainloop.

        El servidor HTTP/WebSocket se arranca aquí, pero el visor 3D NO se
        abre solo: se abre cuando el usuario pulsa el botón (evita lanzar el
        navegador al iniciar la app). El avatar sigue en el navegador porque
        es WebGL y Tkinter no lo puede incrustar.
        """
        self._raiz = raiz
        self._contenedor = contenedor
        self._construir_ui()

        log.info("Iniciando servidores HTTP y WebSocket (pestaña voz)...")
        self.servidor.iniciar_servidores()
        self._servidor_iniciado = True
        self._agregar_log("Servidor listo. Pulsa «↗ Abrir visor 3D» para ver el avatar.")
        log.info("Panel de voz montado en pestaña")

    def cerrar(self):
        """Detiene el servidor sin destruir la raíz."""
        try:
            detener = getattr(self.servidor, "detener_servidores", None)
            if callable(detener):
                detener()
        except Exception:
            pass

    def _abrir_visor(self):
        url = f"http://localhost:{PUERTO_HTTP}"
        log.info(f"Abriendo visor 3D en {url}")
        webbrowser.open(url)
        self._agregar_log(f"Visor 3D abierto en: {url}")

    def _construir_ui(self):
        c_fondo = "#1a1a2e"
        c_panel = "#16213e"
        c_texto = "#e2e8f0"
        c_acento = "#2563eb"
        c_verde = "#22c55e"

        cont = self._contenedor

        # Encabezado
        encabezado = tk.Frame(cont, bg=c_acento, height=60)
        encabezado.pack(fill="x")
        tk.Label(encabezado, text="◈ LSC UDI", bg=c_acento, fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=16, pady=14)
        tk.Label(encabezado, text="Voz / Texto  →  Seña", bg=c_acento,
                 fg="#bfdbfe", font=("Segoe UI", 10)).pack(side="left", pady=14)

        boton_visor = tk.Button(
            encabezado, text="↗ Abrir visor 3D", bg="white", fg=c_acento,
            relief="flat", font=("Segoe UI", 9, "bold"), padx=10,
            cursor="hand2", command=self._abrir_visor,
        )
        boton_visor.pack(side="right", padx=16, pady=14)

        # Panel de entrada de texto
        panel_texto = tk.Frame(cont, bg=c_panel, padx=20, pady=16)
        panel_texto.pack(fill="x", padx=16, pady=(16, 8))

        tk.Label(panel_texto, text="ESCRIBE UN TEXTO PARA TRADUCIR", bg=c_panel,
                 fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        frame_entrada = tk.Frame(panel_texto, bg=c_panel)
        frame_entrada.pack(fill="x", pady=(8, 0))

        self._entrada_texto = tk.Entry(
            frame_entrada, bg="#0d1117", fg=c_texto, insertbackground=c_texto,
            relief="flat", font=("Segoe UI", 12), width=40,
        )
        self._entrada_texto.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self._entrada_texto.bind("<Return>", lambda e: self._enviar_texto())

        tk.Button(
            frame_entrada, text="Traducir", bg=c_acento, fg="white",
            relief="flat", font=("Segoe UI", 10, "bold"), padx=16,
            cursor="hand2", command=self._enviar_texto,
        ).pack(side="left")

        # Panel de voz
        panel_voz = tk.Frame(cont, bg=c_panel, padx=20, pady=16)
        panel_voz.pack(fill="x", padx=16, pady=8)

        tk.Label(panel_voz, text="O USA TU VOZ", bg=c_panel,
                 fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self._boton_voz = tk.Button(
            panel_voz, text="🎤  Mantener presionado para hablar",
            bg=c_verde, fg="#052e16", relief="flat",
            font=("Segoe UI", 11, "bold"), pady=10,
            cursor="hand2", command=self._iniciar_escucha,
        )
        self._boton_voz.pack(fill="x", pady=(8, 0))

        # Resultado de la ultima traduccion
        panel_resultado = tk.Frame(cont, bg=c_panel, padx=20, pady=16)
        panel_resultado.pack(fill="x", padx=16, pady=8)

        tk.Label(panel_resultado, text="ULTIMA TRADUCCION", bg=c_panel,
                 fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self._lbl_secuencia = tk.Label(
            panel_resultado, text="—", bg=c_panel, fg=c_verde,
            font=("Segoe UI", 16, "bold"), wraplength=560, justify="left",
        )
        self._lbl_secuencia.pack(anchor="w", pady=(6, 0))

        self._lbl_no_reconocidas = tk.Label(
            panel_resultado, text="", bg=c_panel, fg="#94a3b8",
            font=("Segoe UI", 9), wraplength=560, justify="left",
        )
        self._lbl_no_reconocidas.pack(anchor="w", pady=(4, 0))

        self._frame_ensenar = tk.Frame(panel_resultado, bg=c_panel)
        self._frame_ensenar.pack(fill="x", pady=(8, 0))
        self._texto_pendiente_ensenar = ""

        # Log de actividad
        panel_log = tk.Frame(cont, bg=c_panel, padx=20, pady=16)
        panel_log.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        tk.Label(panel_log, text="ACTIVIDAD", bg=c_panel,
                 fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self._log_texto = scrolledtext.ScrolledText(
            panel_log, bg="#0d1117", fg="#c2c0b6", font=("Consolas", 9),
            relief="flat", wrap="word", height=8,
        )
        self._log_texto.pack(fill="both", expand=True, pady=(8, 0))

    def _enviar_texto(self):
        texto = self._entrada_texto.get().strip()
        if not texto:
            return
        self._entrada_texto.delete(0, "end")
        self._agregar_log(f"Texto enviado: \"{texto}\"")
        threading.Thread(target=self._procesar_texto, args=(texto,), daemon=True).start()

    def _procesar_texto(self, texto: str):
        resultado = self.servidor.procesar_texto(texto)
        self._raiz.after(0, self._mostrar_resultado, resultado)

    def _iniciar_escucha(self):
        if self._escuchando:
            return
        self._escuchando = True
        self._boton_voz.configure(text="🎤  Escuchando...", bg="#ef4444")
        self._agregar_log("Escuchando microfono...")
        threading.Thread(target=self._procesar_voz, daemon=True).start()

    def _procesar_voz(self):
        resultado = self.servidor.procesar_voz()
        self._raiz.after(0, self._finalizar_escucha, resultado)

    def _finalizar_escucha(self, resultado: dict):
        self._escuchando = False
        self._boton_voz.configure(text="🎤  Mantener presionado para hablar", bg="#22c55e")

        if resultado.get("texto_reconocido"):
            self._agregar_log(f"Voz reconocida: \"{resultado['texto_reconocido']}\"")

        self._mostrar_resultado(resultado)

    def _mostrar_resultado(self, resultado: dict):
        # Limpiar el area de "ensenar" antes de redibujar
        for widget in self._frame_ensenar.winfo_children():
            widget.destroy()

        if resultado.get("exito"):
            señas = [p["sena"] for p in resultado["pasos"]]
            self._lbl_secuencia.configure(text="  →  ".join(señas))
            self._agregar_log(f"Secuencia generada: {' -> '.join(señas)}")

            # Enviar también al robot G1 (simulador o real) si está conectado
            if self.robot is not None and self.robot.conectado:
                threading.Thread(target=self._enviar_al_robot, args=(señas,),
                                 daemon=True).start()
            elif self.robot is not None:
                self._agregar_log("Robot G1 no conectado: pulsa 'Conectar G1' en la pestaña Cámara")

            no_reconocidas = resultado.get("no_reconocidas", [])
            if no_reconocidas:
                self._lbl_no_reconocidas.configure(
                    text=f"Palabras no traducidas: {', '.join(no_reconocidas)}"
                )
                self._mostrar_boton_ensenar(" ".join(no_reconocidas))
            else:
                self._lbl_no_reconocidas.configure(text="")
        else:
            self._lbl_secuencia.configure(text="(sin senas reconocidas)")
            mensaje = resultado.get("mensaje", "Error desconocido")
            self._lbl_no_reconocidas.configure(text=mensaje)
            self._agregar_log(f"[ERROR] {mensaje}")

            no_reconocidas = resultado.get("no_reconocidas", [])
            if no_reconocidas:
                self._mostrar_boton_ensenar(" ".join(no_reconocidas))

    def _enviar_al_robot(self, señas):
        """Ejecuta la secuencia en el G1, una seña tras otra (hilo aparte)."""
        for s in señas:
            self.robot.enviar_seña(s)
        self._raiz.after(0, self._agregar_log,
                         f"Robot G1: ejecutada {' -> '.join(señas)}")

    def _mostrar_boton_ensenar(self, frase_pendiente: str):
        """Muestra el boton 'Ensenar' para mapear una frase no reconocida a una sena."""
        self._texto_pendiente_ensenar = frase_pendiente
        tk.Button(
            self._frame_ensenar,
            text=f"+ Ensenar \"{frase_pendiente}\"",
            bg="#7c3aed", fg="white", relief="flat",
            font=("Segoe UI", 9, "bold"), padx=10, pady=4,
            cursor="hand2", command=self._abrir_dialogo_ensenar,
        ).pack(anchor="w")

    def _abrir_dialogo_ensenar(self):
        """Abre una ventana pequena para elegir a que sena mapear la frase."""
        from voz_a_sena.traductor_texto import SEÑAS_VALIDAS

        dialogo = tk.Toplevel(self._raiz)
        dialogo.title("Ensenar nuevo sinonimo")
        dialogo.configure(bg="#16213e")
        dialogo.resizable(False, False)
        dialogo.grab_set()

        tk.Label(
            dialogo, text=f'Cuando diga o escriba:\n"{self._texto_pendiente_ensenar}"',
            bg="#16213e", fg="#e2e8f0", font=("Segoe UI", 11, "bold"),
            justify="left", padx=20, pady=(20, 10),
        ).pack(anchor="w")

        tk.Label(
            dialogo, text="Quiero que se traduzca a la sena:",
            bg="#16213e", fg="#94a3b8", font=("Segoe UI", 10),
            padx=20,
        ).pack(anchor="w")

        frame_botones = tk.Frame(dialogo, bg="#16213e", padx=20, pady=12)
        frame_botones.pack()

        for i, sena in enumerate(SEÑAS_VALIDAS):
            fila, col = divmod(i, 4)
            tk.Button(
                frame_botones, text=sena, bg="#2563eb", fg="white",
                relief="flat", font=("Segoe UI", 10, "bold"),
                padx=14, pady=8, cursor="hand2", width=10,
                command=lambda s=sena: self._confirmar_ensenanza(s, dialogo),
            ).grid(row=fila, column=col, padx=4, pady=4)

        tk.Button(
            dialogo, text="Cancelar", bg="#374151", fg="#e2e8f0",
            relief="flat", font=("Segoe UI", 9), padx=12, pady=6,
            cursor="hand2", command=dialogo.destroy,
        ).pack(pady=(0, 16))

    def _confirmar_ensenanza(self, sena: str, dialogo: tk.Toplevel):
        frase = self._texto_pendiente_ensenar
        exito = self.servidor.traductor.agregar_sinonimo(frase, sena)
        if exito:
            self._agregar_log(f"Aprendido: \"{frase}\" -> {sena}")
        dialogo.destroy()

    def _agregar_log(self, texto: str):
        import time
        hora = time.strftime("%H:%M:%S")
        self._log_texto.insert("end", f"{hora}  {texto}\n")
        self._log_texto.see("end")

    def _cerrar(self):
        log.info("Cerrando aplicacion...")
        self._raiz.quit()
        self._raiz.destroy()


def main():
    configurar_logger()
    log.info("=" * 60)
    log.info("  LSC UDI — Voz/Texto a Sena (Robot G1)")
    log.info("=" * 60)

    app = VentanaVozASena()
    app.ejecutar()


if __name__ == "__main__":
    main()