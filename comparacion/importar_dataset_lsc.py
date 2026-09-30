"""
Importa la carpeta `dataset_lsc` del software de captura de estudiantes
(LSC_Captura_Dataset_GUI) al formato dual que usan RF, MLP y MobileNetV2.

Estructura de origen (según el instructivo UDI-2026):

    dataset_lsc/
      hola/
        E01_P001/
          sesion_01/
            hola_E01_P001_s01_derecha_luzbuena_fondoclaro_001.jpg
            ...
            metadata.csv
      gracias/ si/ no/ ...
      metadata_global.csv

Decisiones:
  * Cada imagen es una muestra (el software captura fotos una a una, seña estable).
  * Los landmarks se recalculan con MediaPipe SOBRE LA MISMA imagen → los tres
    modelos ven exactamente las mismas muestras.
  * Sujeto = estudiante + participante (E01P001): P001 de E01 y P001 de E02 son
    personas distintas.
  * Grupo = sujeto + sesión. Las 100 fotos de una sesión son casi duplicadas;
    la validación debe separar por grupo (--validacion sesion o loso) para no
    inflar la accuracy.
  * mano / iluminación / fondo se leen del nombre del archivo y quedan en
    meta.json → el informe calcula accuracy por condición.

Uso:
    python -m comparacion.importar_dataset_lsc --origen C:/ruta/dataset_lsc --simular
    python -m comparacion.importar_dataset_lsc --origen C:/ruta/dataset_lsc
    python -m comparacion.importar_dataset_lsc --origen C:/ruta/dataset_lsc --senas hola si no bien mal gracias silencio
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from models.capturar_dual import ANCHO_GUARDADO, CALIDAD_JPG, vector_frame  # noqa: E402
from models.extractor_v2 import ExtractorSecuencial  # noqa: E402

# Nombre en el software de captura -> nombre de clase en el sistema
MAPA_SENAS = {"hola": "Hola", "gracias": "Gracias", "si": "Si", "no": "No",
              "bien": "Bien", "mal": "Mal", "silencio": "Silencio"}
EXT = {".jpg", ".jpeg", ".png"}
PATRON_SESION = re.compile(r"^sesion_(\d+)$", re.I)
PATRON_SUJETO = re.compile(r"^(E\d+)_(P\d+)$", re.I)


def parsear_nombre(nombre: str):
    """hola_E01_P001_s01_derecha_luzbuena_fondoclaro_001 -> dict (se lee de derecha a izquierda)."""
    t = nombre.split("_")
    if len(t) < 8:
        return {}
    return {"indice_foto": t[-1], "fondo": t[-2].replace("fondo", ""), "iluminacion": t[-3].replace("luz", ""),
            "mano": t[-4], "sesion": t[-5], "participante": t[-6], "estudiante": t[-7],
            "sena_archivo": "_".join(t[:-7])}


def descubrir(origen: Path, senas_filtro):
    """[(clase, sujeto, sesion, [rutas])] — una entrada por sesión."""
    sesiones = []
    for d_sena in sorted(p for p in origen.iterdir() if p.is_dir()):
        nombre = d_sena.name.lower()
        if senas_filtro and nombre not in senas_filtro:
            continue
        clase = MAPA_SENAS.get(nombre, nombre.capitalize())
        for d_suj in sorted(p for p in d_sena.iterdir() if p.is_dir()):
            m = PATRON_SUJETO.match(d_suj.name)
            if not m:
                print(f"  [omitida] carpeta con nombre no estándar: {d_suj}")
                continue
            sujeto = (m.group(1) + m.group(2)).upper()
            for d_ses in sorted(p for p in d_suj.iterdir() if p.is_dir()):
                ms = PATRON_SESION.match(d_ses.name)
                if not ms:
                    continue
                rutas = sorted(p for p in d_ses.iterdir() if p.suffix.lower() in EXT)
                if rutas:
                    sesiones.append((clase, sujeto, f"s{int(ms.group(1)):02d}", rutas))
    return sesiones


def main():
    ap = argparse.ArgumentParser(description="Importar dataset_lsc (software de captura de estudiantes)")
    ap.add_argument("--origen", required=True, help="Carpeta dataset_lsc")
    ap.add_argument("--salida", default="data/dual")
    ap.add_argument("--senas", nargs="+", default=list(MAPA_SENAS),
                    help="Señas a importar (nombres del software, minúscula). Por defecto las 7 del proyecto")
    ap.add_argument("--max-por-sesion", type=int, default=None,
                    help="Submuestrear N fotos por sesión (uniforme) para reducir casi-duplicados")
    ap.add_argument("--espejo", action="store_true", help="Voltear horizontalmente antes de procesar")
    ap.add_argument("--simular", action="store_true")
    a = ap.parse_args()

    origen = Path(a.origen)
    salida = Path(a.salida)
    salida = salida if salida.is_absolute() else RAIZ / salida
    senas = {s.lower() for s in a.senas}

    sesiones = descubrir(origen, senas)
    if not sesiones:
        sys.exit(f"No encontré la estructura dataset_lsc/<seña>/E##_P###/sesion_##/*.jpg en {origen}")

    # Resumen
    tabla = defaultdict(lambda: [0, 0])
    for clase, sujeto, _, rutas in sesiones:
        tabla[(clase, sujeto)][0] += 1
        tabla[(clase, sujeto)][1] += len(rutas)
    sujetos = sorted({s for _, s, _, _ in sesiones})
    clases = sorted({c for c, _, _, _ in sesiones})
    print(f"\n{len(sesiones)} sesiones · {len(sujetos)} participantes · {len(clases)} señas\n")
    print("participante  " + "  ".join(f"{c:>8s}" for c in clases))
    for s in sujetos:
        print(f"{s:12s}  " + "  ".join(f"{tabla[(c, s)][1]:8d}" for c in clases))
    faltan = [(c, s) for c in clases for s in sujetos if (c, s) not in tabla]
    if faltan:
        print(f"\n[AVISO] {len(faltan)} combinaciones seña×participante sin datos (dataset desbalanceado): "
              + ", ".join(f"{s}/{c}" for c, s in faltan[:10]) + (" ..." if len(faltan) > 10 else ""))
    no_mapeadas = [c for c in clases if c.lower() not in MAPA_SENAS]
    if no_mapeadas:
        print(f"[AVISO] Señas fuera de las 7 del sistema: {no_mapeadas}")
    if a.simular:
        return

    extractor = ExtractorSecuencial()
    if not extractor.iniciar():
        sys.exit("No se pudo iniciar MediaPipe (revisa data/hand_landmarker.task)")
    logging.getLogger("lsc_bridge.features_v2").setLevel(logging.ERROR)

    ok, sin_mano = 0, Counter()
    cara = []
    t0_total = time.time()
    for n_ses, (clase, sujeto, sesion, rutas) in enumerate(sesiones, 1):
        if a.max_por_sesion and len(rutas) > a.max_por_sesion:
            idx = np.round(np.linspace(0, len(rutas) - 1, a.max_por_sesion)).astype(int)
            rutas = [rutas[i] for i in idx]
        extractor.detener()
        extractor.iniciar()          # detectores nuevos por sesión
        d_clase = salida / clase
        d_clase.mkdir(parents=True, exist_ok=True)
        existentes = [int(p.name.split("_")[-1]) for p in d_clase.glob(f"{sujeto}_*") if p.is_dir()]
        sig = max(existentes) + 1 if existentes else 0
        grupo = f"{sujeto}-{sesion}"

        for i, r in enumerate(rutas):
            frame = cv2.imdecode(np.fromfile(str(r), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if a.espejo:
                frame = cv2.flip(frame, 1)
            t = time.perf_counter()
            ff = extractor.procesar_frame(frame, i * 33)
            t_mp = (time.perf_counter() - t) * 1000
            if not (ff and ff.manos_presentes):
                sin_mano[grupo] += 1
                continue
            destino = d_clase / f"{sujeto}_{sig:04d}"
            sig += 1
            (destino / "frames").mkdir(parents=True)
            np.save(destino / "landmarks.npy", extractor.agregar_secuencia([ff]))
            np.save(destino / "landmarks_frames.npy", vector_frame(ff)[None, :])
            h, w = frame.shape[:2]
            if w > ANCHO_GUARDADO:
                frame = cv2.resize(frame, (ANCHO_GUARDADO, int(round(h * ANCHO_GUARDADO / w))),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(destino / "frames" / "000.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])
            cara.append(ff.cara_presente)
            meta = {"clase": clase, "sujeto": sujeto, "grupo": grupo, "n_frames": 1,
                    "origen": str(r), "t_mediapipe_ms_medio": t_mp,
                    "cara_presente_frac": float(ff.cara_presente),
                    "condiciones": {k: v for k, v in parsear_nombre(r.stem).items()
                                    if k in ("mano", "iluminacion", "fondo", "sesion")},
                    "fuente": "dataset_lsc (foto a foto)", "espejo_aplicado": a.espejo,
                    "fecha_importacion": datetime.now().isoformat(timespec="seconds")}
            (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            ok += 1
        print(f"  [{n_ses}/{len(sesiones)}] {clase:9s} {grupo}: "
              f"{len(rutas) - sin_mano[grupo]}/{len(rutas)} con mano")
    extractor.detener()

    print(f"\nImportadas {ok} imágenes en {time.time() - t0_total:.0f} s → {salida}")
    if sin_mano:
        print(f"Descartadas sin mano detectada: {sum(sin_mano.values())}")
    if cara:
        fr = 100 * np.mean(cara)
        print(f"Rostro detectado en el {fr:.0f}% de las imágenes.")
        if fr < 50:
            print("[AVISO] La mayoría de imágenes no tiene rostro: la distancia mano-boca (clave para "
                  "Silencio y Gracias) queda sin información, y en la app en vivo el rostro SÍ aparece.")
    print("\nSiguiente: python -m comparacion.entrenar_comparacion --datos data/dual --validacion sesion")


if __name__ == "__main__":
    main()
