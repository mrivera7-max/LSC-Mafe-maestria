"""
Captura DUAL para la comparación de modelos (RF / MLP / MobileNetV2).

Cada toma guarda, a partir de los MISMOS frames:
  * el vector agregado de landmarks (342) — idéntico al de capturar_secuencias.py
  * las features por frame (N, 152)        — para modelos temporales futuros
  * los frames RGB (.jpg)                   — entrada de MobileNetV2
  * meta.json con sujeto, tiempos de MediaPipe, resolución, fecha

Solo se conservan los frames con manos detectadas, que es exactamente lo que
el reconocedor en vivo mete en su ventana. Así entrenamiento y uso coinciden.

Uso:
    python models/capturar_dual.py --sujeto S01
    python models/capturar_dual.py --sujeto S02 --sena Hola --muestras 30

Controles:
    ESPACIO  grabar una toma (0.7 s)
    N        siguiente seña
    B        borrar la última toma guardada (toma mala)
    Q        salir
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from models.extractor_v2 import ExtractorSecuencial  # noqa: E402

SEÑAS = ["Hola", "Gracias", "Si", "No", "Bien", "Mal", "Silencio"]
DURACION_SEGUNDOS = 0.7
MIN_FRAMES_MANO = 8          # tomas con menos frames con mano se descartan
ANCHO_GUARDADO = 640         # los jpg se guardan a 640 px de ancho (MobileNet usa 224 de alto)
CALIDAD_JPG = 90


def vector_frame(ff) -> np.ndarray:
    """Features de un frame: mano_der(63) + mano_izq(63) + boca(24) + dist(1) + cara(1) = 152."""
    return np.concatenate([ff.mano_der, ff.mano_izq, ff.boca,
                           [ff.dist_mano_boca], [float(ff.cara_presente)]])


def siguiente_indice(dir_seña: Path, sujeto: str) -> int:
    existentes = [int(p.name.split("_")[-1]) for p in dir_seña.glob(f"{sujeto}_*") if p.is_dir()]
    return max(existentes) + 1 if existentes else 0


def guardar_toma(extractor, tomas, dir_seña, sujeto, seña, idx, cap_info):
    """tomas: lista de (frame_bgr, FrameFeatures, t_mediapipe_ms) SOLO con mano."""
    destino = dir_seña / f"{sujeto}_{idx:04d}"
    (destino / "frames").mkdir(parents=True, exist_ok=True)

    feats = [ff for _, ff, _ in tomas]
    np.save(destino / "landmarks.npy", extractor.agregar_secuencia(feats))
    np.save(destino / "landmarks_frames.npy", np.stack([vector_frame(f) for f in feats]))

    for i, (frame, _, _) in enumerate(tomas):
        h, w = frame.shape[:2]
        esc = ANCHO_GUARDADO / w
        peq = cv2.resize(frame, (ANCHO_GUARDADO, int(round(h * esc))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(destino / "frames" / f"{i:03d}.jpg"), peq,
                    [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])

    t_mp = [t for _, _, t in tomas]
    meta = {
        "clase": seña, "sujeto": sujeto, "indice": idx,
        "n_frames": len(tomas),
        "t_mediapipe_ms_medio": float(np.mean(t_mp)),
        "cara_presente_frac": float(np.mean([f.cara_presente for f in feats])),
        "duracion_s": DURACION_SEGUNDOS,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        **cap_info,
    }
    (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def capturar(sujeto, señas, muestras, camara, salida):
    extractor = ExtractorSecuencial()
    if not extractor.iniciar():
        print("[ERROR] No se pudo iniciar MediaPipe. Verifica data/hand_landmarker.task y face_landmarker.task")
        return
    cap = cv2.VideoCapture(camara)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la cámara {camara}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_info = {"camara_idx": camara,
                "resolucion": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}

    print(f"\nCaptura DUAL — sujeto {sujeto} — {muestras} tomas por seña")
    print("ESPACIO grabar · N siguiente · B borrar última · Q salir\n")
    if not extractor.tiene_face_landmarker:
        print("[AVISO] Sin FaceLandmarker: la distancia mano-boca quedará en 1.0\n")

    t0_global = time.time()
    i_seña = 0
    ultima = None
    while i_seña < len(señas):
        seña = señas[i_seña]
        dir_seña = salida / seña
        dir_seña.mkdir(parents=True, exist_ok=True)
        hechas = len([p for p in dir_seña.glob(f"{sujeto}_*") if p.is_dir()])
        grabando, tomas, t_ini = False, [], 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t_a = time.perf_counter()
            ff = extractor.procesar_frame(frame, int((time.time() - t0_global) * 1000))
            t_mp = (time.perf_counter() - t_a) * 1000
            manos = bool(ff and ff.manos_presentes)

            if grabando:
                if manos:
                    tomas.append((frame.copy(), ff, t_mp))
                if time.time() - t_ini >= DURACION_SEGUNDOS:
                    grabando = False
                    if len(tomas) >= MIN_FRAMES_MANO:
                        idx = siguiente_indice(dir_seña, sujeto)
                        ultima = guardar_toma(extractor, tomas, dir_seña, sujeto, seña, idx, cap_info)
                        hechas += 1
                        print(f"  [{seña}] toma {hechas}/{muestras} ({len(tomas)} frames con mano)")
                    else:
                        print(f"  [AVISO] solo {len(tomas)} frames con mano (< {MIN_FRAMES_MANO}); toma descartada")
                    tomas = []

            vis = frame.copy()
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.putText(vis, f"Sujeto {sujeto} | Sena: {seña}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(vis, f"Tomas: {hechas}/{muestras}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(vis, "Manos OK" if manos else "Sin manos", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if manos else (0, 100, 255), 2)
            if grabando:
                cv2.circle(vis, (w - 30, 70), 10, (0, 0, 255), -1)
            cv2.rectangle(vis, (0, h - 8), (int(min(hechas / muestras, 1) * w), h), (0, 200, 0), -1)
            cv2.imshow("Captura DUAL LSC", vis)

            if hechas >= muestras and not grabando:
                print(f"OK {seña}: {hechas} tomas\n")
                break
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                i_seña = len(señas)
                break
            if tecla == ord(" ") and not grabando:
                grabando, tomas, t_ini = True, [], time.time()
            elif tecla == ord("n"):
                break
            elif tecla == ord("b") and ultima is not None and ultima.exists():
                shutil.rmtree(ultima)
                print(f"  borrada {ultima.name}")
                ultima = None
                hechas -= 1
        i_seña += 1

    cap.release()
    cv2.destroyAllWindows()
    extractor.detener()
    print(f"\nDatos en {salida}")
    print("Siguiente paso: python -m comparacion.entrenar_comparacion --datos data/dual")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Captura dual (landmarks + frames) para la comparación")
    p.add_argument("--sujeto", required=True, help="ID del sujeto/señante, p. ej. S01 (necesario para LOSO)")
    p.add_argument("--sena", default=None, help="Capturar solo esta seña")
    p.add_argument("--muestras", type=int, default=60)
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--salida", default="data/dual")
    a = p.parse_args()
    if "_" in a.sujeto:
        sys.exit("El ID de sujeto no debe contener '_' (p. ej. S01)")
    if a.sena and a.sena not in SEÑAS:
        sys.exit(f"'{a.sena}' no está en {SEÑAS}")
    sal = Path(a.salida)
    sal = sal if sal.is_absolute() else RAIZ / sal
    capturar(a.sujeto, [a.sena] if a.sena else SEÑAS, a.muestras, a.camara, sal)
