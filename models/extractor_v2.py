"""
Extractor de features para señas dinámicas (LSC v2).

Combina:
  - HandLandmarker (21 puntos x 2 manos x 3 coords = 126)
  - FaceLandmarker (puntos clave de boca/labios, 8 puntos x 3 coords = 24)
  - Secuencia temporal de N frames (resume con estadísticos: inicio, fin, máx movimiento)

Vector de features por frame (capturador): 126 (manos) + 24 (boca) + 1 (distancia
índice derecho-boca) + 1 (rostro presente) = 152
Vector de features por secuencia (agregar_secuencia): mean(der, izq, boca) 150 +
std(der, izq) 126 + delta(der) 63 + dist(mean, min, std) 3 = 342

Esto permite diferenciar:
  - Señas estáticas (Hola, Gracias, Bien, Mal) — el agregado captura la postura fija.
  - Señas con movimiento repetido (Si) — el std captura la oscilación.
  - Señas cerca de la cara (Silencio) — la distancia mano-boca lo indica directamente.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("lsc_bridge.features_v2")

MEDIAPIPE_OK = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError as e:
    log.warning(f"MediaPipe no disponible: {e}")

# Puntos clave de la boca en el FaceLandmarker (468 landmarks, modelo MediaPipe Face Mesh)
# Referencia: contorno exterior e interior de labios
PUNTOS_BOCA = [61, 291, 0, 17, 78, 308, 13, 14]  # 8 puntos representativos de labios

# Rutas ancladas a la raíz del proyecto (carpeta que contiene 'models/' y 'data/'),
# NO al directorio de trabajo. Esto evita que el extractor falle cuando corre desde
# un hilo (p.ej. el bucle de cámara) o cuando se ejecuta desde otra carpeta.
_RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELO_MANO = str(_RAIZ_PROYECTO / "data" / "hand_landmarker.task")
RUTA_MODELO_CARA = str(_RAIZ_PROYECTO / "data" / "face_landmarker.task")

# Tamaño de la ventana temporal (frames) para una "muestra" de seña dinámica
VENTANA_FRAMES = 20


@dataclass
class FrameFeatures:
    """Features de un solo frame: manos + boca + distancia."""
    mano_der: np.ndarray   # (63,)
    mano_izq: np.ndarray   # (63,)
    boca: np.ndarray       # (24,) — 8 puntos x 3 coords, relativos al centro de la cara
    dist_mano_boca: float  # distancia normalizada mano derecha -> centro de boca
    manos_presentes: bool
    cara_presente: bool


class ExtractorSecuencial:
    """
    Extrae features de mano + cara para cada frame, y permite agregar
    una secuencia completa (ventana de tiempo) en un solo vector.
    """

    def __init__(self, ruta_modelo_mano: str = RUTA_MODELO_MANO,
                 ruta_modelo_cara: str = RUTA_MODELO_CARA):
        self._mano_detector = None
        self._cara_detector = None
        self._ruta_mano = ruta_modelo_mano
        self._ruta_cara = ruta_modelo_cara
        self._activo = False

    def iniciar(self) -> bool:
        if not MEDIAPIPE_OK:
            log.error("MediaPipe no disponible")
            return False

        if not Path(self._ruta_mano).exists():
            log.error(f"Modelo de mano no encontrado: {self._ruta_mano}")
            return False

        try:
            BaseOptions = mp_python.BaseOptions

            # HandLandmarker
            opciones_mano = mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._ruta_mano),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mano_detector = mp_vision.HandLandmarker.create_from_options(opciones_mano)

            # FaceLandmarker (opcional — si no está el modelo, seguimos sin cara)
            if Path(self._ruta_cara).exists():
                opciones_cara = mp_vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self._ruta_cara),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._cara_detector = mp_vision.FaceLandmarker.create_from_options(opciones_cara)
                log.info("FaceLandmarker iniciado correctamente")
            else:
                log.warning(
                    f"Modelo de cara no encontrado en '{self._ruta_cara}'. "
                    "Se continuará SOLO con detección de manos (sin distancia mano-boca). "
                    "Descárgalo con: curl -o data/face_landmarker.task "
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/1/face_landmarker.task"
                )

            self._activo = True
            return True

        except Exception as e:
            log.error(f"Error iniciando detectores: {e}")
            return False

    def detener(self):
        if self._mano_detector:
            try:
                self._mano_detector.close()
            except Exception:
                pass
        if self._cara_detector:
            try:
                self._cara_detector.close()
            except Exception:
                pass
        self._activo = False

    # ── Procesamiento por frame ──────────────────────────────────

    def procesar_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[FrameFeatures]:
        """Extrae features de mano + cara de un solo frame."""
        if not self._activo:
            return None

        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Manos
        mano_der = np.zeros(63)
        mano_izq = np.zeros(63)
        manos_presentes = False
        try:
            resultado_mano = self._mano_detector.detect_for_video(mp_image, timestamp_ms)
            if resultado_mano.hand_landmarks:
                manos_presentes = True
                for idx, lm_lista in enumerate(resultado_mano.hand_landmarks):
                    lado = resultado_mano.handedness[idx][0].category_name if idx < len(resultado_mano.handedness) else "Right"
                    vec = self._normalizar_mano(lm_lista)
                    if lado == "Left":
                        mano_izq = vec
                    else:
                        mano_der = vec
        except Exception as e:
            log.debug(f"Error detectando mano: {e}")

        # Cara (opcional)
        boca = np.zeros(24)
        cara_presente = False
        centro_boca_xy = None
        if self._cara_detector:
            try:
                resultado_cara = self._cara_detector.detect_for_video(mp_image, timestamp_ms)
                if resultado_cara.face_landmarks:
                    cara_presente = True
                    lm_cara = resultado_cara.face_landmarks[0]
                    puntos_boca_raw = [lm_cara[i] for i in PUNTOS_BOCA]
                    arr = np.array([[p.x, p.y, p.z] for p in puntos_boca_raw])
                    centro_boca_xy = arr[:, :2].mean(axis=0)
                    # Normalizar boca respecto a su propio centro
                    centro = arr.mean(axis=0)
                    arr_norm = arr - centro
                    boca = arr_norm.flatten()
            except Exception as e:
                log.debug(f"Error detectando cara: {e}")

        # Distancia mano derecha -> boca (si ambas presentes)
        dist_mano_boca = 1.0  # valor "lejos" por defecto
        if manos_presentes and centro_boca_xy is not None:
            # Usamos el landmark 8 (punta del índice) de la mano derecha como referencia
            punta_indice = mano_der[8*3:8*3+2]  # x,y del índice (antes de normalizar fue absoluto)
            # Nota: mano_der ya está normalizada respecto a la muñeca, así que usamos
            # la posición absoluta capturada antes de normalizar. Para simplicidad,
            # recalculamos aquí con la posición cruda si está disponible.
            dist_mano_boca = self._distancia_aproximada(resultado_mano if manos_presentes else None, centro_boca_xy)

        return FrameFeatures(
            mano_der=mano_der,
            mano_izq=mano_izq,
            boca=boca,
            dist_mano_boca=dist_mano_boca,
            manos_presentes=manos_presentes,
            cara_presente=cara_presente,
        )

    def _normalizar_mano(self, landmarks_mp) -> np.ndarray:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_mp])
        arr -= arr[0]
        escala = np.linalg.norm(arr[9]) + 1e-7
        arr /= escala
        return arr.flatten()

    def _distancia_aproximada(self, resultado_mano, centro_boca_xy) -> float:
        """Calcula distancia normalizada entre la punta del índice derecho y la boca."""
        if resultado_mano is None or not resultado_mano.hand_landmarks:
            return 1.0
        try:
            for idx, lm_lista in enumerate(resultado_mano.hand_landmarks):
                lado = resultado_mano.handedness[idx][0].category_name if idx < len(resultado_mano.handedness) else "Right"
                if lado != "Left":  # mano derecha (o cualquiera si no hay clasificación)
                    punta = lm_lista[8]  # punta del índice
                    dx = punta.x - centro_boca_xy[0]
                    dy = punta.y - centro_boca_xy[1]
                    return float(np.sqrt(dx**2 + dy**2))
        except Exception:
            pass
        return 1.0

    # ── Agregación de secuencia ──────────────────────────────────

    def agregar_secuencia(self, frames: list) -> np.ndarray:
        """
        Combina una lista de FrameFeatures en un solo vector de features
        para la secuencia completa.

        Features agregados:
          - mean(mano_der), mean(mano_izq), mean(boca)   -> postura promedio
          - std(mano_der), std(mano_izq)                  -> cuánto se mueve
          - mano_der[ultimo] - mano_der[primero]           -> delta de movimiento
          - mean(dist_mano_boca), min(dist_mano_boca)      -> cercanía a la boca
        """
        if not frames:
            return np.zeros(342)

        manos_der = np.array([f.mano_der for f in frames])   # (N, 63)
        manos_izq = np.array([f.mano_izq for f in frames])   # (N, 63)
        bocas = np.array([f.boca for f in frames])            # (N, 24)
        distancias = np.array([f.dist_mano_boca for f in frames])  # (N,)

        mean_der = manos_der.mean(axis=0)
        mean_izq = manos_izq.mean(axis=0)
        mean_boca = bocas.mean(axis=0)

        std_der = manos_der.std(axis=0)
        std_izq = manos_izq.std(axis=0)

        delta_der = manos_der[-1] - manos_der[0]

        dist_mean = np.array([distancias.mean()])
        dist_min = np.array([distancias.min()])
        dist_std = np.array([distancias.std()])

        vector = np.concatenate([
            mean_der, mean_izq, mean_boca,      # 63+63+24 = 150
            std_der, std_izq,                    # 63+63 = 126
            delta_der,                            # 63
            dist_mean, dist_min, dist_std,       # 3
        ])
        # Total: 150 + 126 + 63 + 3 = 342
        return vector

    @property
    def dimension_features(self) -> int:
        """Dimensión del vector agregado (debe coincidir con agregar_secuencia)."""
        return 150 + 126 + 63 + 3  # = 342

    @property
    def tiene_face_landmarker(self) -> bool:
        return self._cara_detector is not None