"""
Reconocedor en vivo con cambio de modelo en caliente (RF / MLP / MobileNetV2).

Misma interfaz que ReconocedorLSCv2 (iniciar, detener, procesar_frame,
dibujar_landmarks, fps, activo) para que la GUI lo use sin cambios de flujo.

Para que la comparación sea justa, los tres modelos comparten:
  * el mismo extractor MediaPipe y el mismo criterio de "frame con mano"
  * la misma ventana deslizante (VENTANA_FRAMES)
  * la misma votación (7 predicciones, 4 votos, conf. mínima de voto 0.55)
  * el mismo umbral final (config.umbral_confianza)
Solo cambia lo que entra al clasificador: vector de landmarks o frames RGB.

Cada frame y cada detección quedan registrados en una SesionComparacion,
de la que sale el informe HTML/CSV.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from comparacion import MODELOS, NOMBRES_LEGIBLES
from comparacion.modelos import cargar_modelo, muestrear_indices
from comparacion.sesion import SesionComparacion
from models.extractor_v2 import VENTANA_FRAMES, ExtractorSecuencial
from models.reconocedor_v2 import TRADUCCIONES_LSC, SenaDetectadaV2

log = logging.getLogger("lsc_bridge.reconocimiento_comparativo")

RAIZ = Path(__file__).resolve().parent.parent


class ReconocedorComparativo:
    VOTOS_VENTANA = 7
    MIN_VOTOS = 4
    MIN_CONF_VOTO = 0.55
    COOLDOWN_FRAMES = 15

    def __init__(self, config):
        self.config = config
        d = Path(getattr(config, "dir_modelos_comparacion", "data/modelos_comparacion"))
        self.dir_modelos = d if d.is_absolute() else RAIZ / d
        self._extractor = ExtractorSecuencial()
        self._modelos = {}           # cache nombre -> (modelo, meta)
        self._lock = threading.Lock()
        self._nombre = None
        self._modelo = None
        self._meta = None
        self._activo = False
        self._t_inicio = time.time()
        self._reiniciar_buffers()
        self._fps_hist = deque(maxlen=30)
        self._ultimo_t = 0.0
        self.sesion = SesionComparacion(self._parametros())
        self.on_modelo_cambiado = None   # callback(nombre, ok, mensaje)

    # ── Utilidades ───────────────────────────────────────────────
    def _parametros(self):
        return {"ventana_frames": VENTANA_FRAMES, "votos_ventana": self.VOTOS_VENTANA,
                "min_votos": self.MIN_VOTOS, "min_conf_voto": self.MIN_CONF_VOTO,
                "umbral_confianza": self.config.umbral_confianza, "cooldown_frames": self.COOLDOWN_FRAMES,
                "camara": f"{self.config.camara_ancho}x{self.config.camara_alto}@{self.config.camara_fps}"}

    def _reiniciar_buffers(self):
        self._buf_feats = deque(maxlen=VENTANA_FRAMES)
        self._buf_emb = deque(maxlen=VENTANA_FRAMES)
        self._votos = deque(maxlen=self.VOTOS_VENTANA)
        self._desde_emision = 0

    def modelos_disponibles(self):
        return [m for m in MODELOS if (self.dir_modelos / m / "meta.json").exists()]

    @property
    def modelo_activo(self):
        return self._nombre

    @property
    def clases(self):
        return self._meta["clases"] if self._meta else list(TRADUCCIONES_LSC)

    # ── Ciclo de vida ────────────────────────────────────────────
    def iniciar(self) -> bool:
        if not self._extractor.iniciar():
            return False
        if self._modelo is None:
            disp = self.modelos_disponibles()
            if not disp:
                log.error(f"No hay modelos en {self.dir_modelos}. Ejecuta: "
                          "python -m comparacion.entrenar_comparacion")
            else:
                pref = getattr(self.config, "modelo_activo", "mlp")
                ok, msg = self.cambiar_modelo(pref if pref in disp else disp[0])
                if not ok:
                    log.error(msg)
        elif self._nombre:
            self.sesion.cambiar_modelo(self._nombre, self._meta)
        self._activo = True
        return True

    def detener(self):
        self._extractor.detener()
        self._activo = False

    def cambiar_modelo(self, nombre: str):
        """Carga (con caché) y activa un modelo. Seguro de llamar desde cualquier hilo."""
        if nombre not in MODELOS:
            return False, f"Modelo desconocido: {nombre}"
        try:
            if nombre not in self._modelos:
                ruta = self.dir_modelos / nombre
                if not (ruta / "meta.json").exists():
                    return False, f"{NOMBRES_LEGIBLES[nombre]} no está entrenado (falta {ruta})"
                self._modelos[nombre] = cargar_modelo(ruta, hilos=getattr(self.config, "hilos_inferencia", 1))
            modelo, meta = self._modelos[nombre]
        except ImportError as e:
            return False, f"Falta una dependencia para {nombre}: {e} (pip install -r requirements-cnn.txt)"
        except Exception as e:
            log.exception("Error cargando modelo")
            return False, f"Error cargando {nombre}: {e}"
        with self._lock:
            self._nombre, self._modelo, self._meta = nombre, modelo, meta
            self._reiniciar_buffers()   # no mezclar ventanas/votos entre modelos
        self.config.modelo_activo = nombre
        self.sesion.cambiar_modelo(nombre, meta)
        log.info(f"Modelo activo: {NOMBRES_LEGIBLES[nombre]}")
        if self.on_modelo_cambiado:
            self.on_modelo_cambiado(nombre, True, "")
        return True, ""

    def fijar_esperada(self, seña: Optional[str]):
        self.sesion.fijar_esperada(seña)

    def generar_informe(self, reiniciar=True):
        """Escribe el informe de la sesión actual. Devuelve la ruta del HTML o None."""
        d = Path(getattr(self.config, "dir_informes", "informes"))
        d = d if d.is_absolute() else RAIZ / d
        ruta = self.sesion.generar_informe(d / "sesiones", self.clases)
        if reiniciar and ruta is not None:
            esperada = self.sesion._esperada
            self.sesion = SesionComparacion(self._parametros())
            if self._nombre:
                self.sesion.cambiar_modelo(self._nombre, self._meta)
            if esperada:
                self.sesion.fijar_esperada(esperada)
        return ruta

    # ── Procesamiento ────────────────────────────────────────────
    def procesar_frame(self, frame_bgr: np.ndarray) -> Optional[SenaDetectadaV2]:
        if not self._activo:
            return None
        ahora = time.time()
        if self._ultimo_t > 0 and ahora > self._ultimo_t:
            self._fps_hist.append(1.0 / (ahora - self._ultimo_t))
        self._ultimo_t = ahora

        t0 = time.perf_counter()
        ff = self._extractor.procesar_frame(frame_bgr, int((ahora - self._t_inicio) * 1000))
        t_mp = (time.perf_counter() - t0) * 1000
        manos = bool(ff and ff.manos_presentes)

        with self._lock:
            modelo, nombre = self._modelo, self._nombre
            self._desde_emision += 1
            if modelo is None or not manos:
                self.sesion.registrar_frame(manos, t_mp, None)
                return None

            t1 = time.perf_counter()
            self._buf_feats.append(ff)
            if modelo.entrada == "imagenes":
                self._buf_emb.append(modelo.backbone.embeber([frame_bgr])[0])

            nombre_pred, conf = None, 0.0
            if len(self._buf_feats) >= VENTANA_FRAMES:
                if modelo.entrada == "imagenes":
                    k = self._meta.get("k_frames") or 8
                    emb = np.stack(self._buf_emb)[muestrear_indices(len(self._buf_emb), k)].mean(0)
                    proba = modelo.predict_proba(emb[None, :])[0]
                else:
                    vec = self._extractor.agregar_secuencia(list(self._buf_feats))
                    proba = modelo.predict_proba(vec[None, :])[0]
                i = int(np.argmax(proba))
                nombre_pred, conf = self._meta["clases"][i], float(proba[i])
            t_mod = (time.perf_counter() - t1) * 1000
            self.sesion.registrar_frame(True, t_mp, t_mod)

            if nombre_pred is None:
                return None
            if conf >= self.MIN_CONF_VOTO:
                self._votos.append((nombre_pred, conf))
            if self._desde_emision < self.COOLDOWN_FRAMES or len(self._votos) < self.MIN_VOTOS:
                return None
            votos, acum = {}, {}
            for n, c in self._votos:
                votos[n] = votos.get(n, 0) + 1
                acum[n] = acum.get(n, 0.0) + c
            gan = max(votos, key=votos.get)
            conf_media = acum[gan] / votos[gan]
            if votos[gan] < self.MIN_VOTOS or conf_media < self.config.umbral_confianza:
                return None
            self._desde_emision = 0
            self._votos.clear()

        self.sesion.registrar_deteccion(gan, conf_media)
        return SenaDetectadaV2(nombre=gan, traduccion=TRADUCCIONES_LSC.get(gan, gan),
                               confianza=conf_media, fps=self.fps)

    def dibujar_landmarks(self, frame, sena=None):
        import cv2
        h, w = frame.shape[:2]
        prog = len(self._buf_feats) / VENTANA_FRAMES
        cv2.rectangle(frame, (10, h - 20), (10 + int(200 * prog), h - 10), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, h - 20), (210, h - 10), (255, 255, 255), 1)
        etiqueta = NOMBRES_LEGIBLES.get(self._nombre, "sin modelo")
        cv2.putText(frame, f"Modelo: {etiqueta}", (w - 330, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if sena:
            txt = f"{sena.nombre}  {sena.confianza*100:.0f}%"
            cv2.rectangle(frame, (10, 10), (len(txt) * 16 + 20, 58), (0, 0, 0), -1)
            cv2.putText(frame, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 100), 2, cv2.LINE_AA)
        if getattr(self.config, "gui_mostrar_fps", True):
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 255), 1, cv2.LINE_AA)
        return frame

    @property
    def fps(self) -> float:
        return round(float(np.mean(self._fps_hist)), 1) if self._fps_hist else 0.0

    @property
    def activo(self) -> bool:
        return self._activo
