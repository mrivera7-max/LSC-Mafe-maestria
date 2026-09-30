"""
Importa una base de datos de IMÁGENES al formato dual que usan los tres modelos.

Las imágenes son la fuente de verdad: los landmarks (.npy) se recalculan con
MediaPipe sobre esas mismas imágenes, así RF, MLP y MobileNetV2 ven
exactamente las mismas muestras.

Estructuras de origen aceptadas (jpg / jpeg / png):

  A) Una carpeta por toma (RECOMENDADO — conserva el movimiento de la seña)
        origen/Hola/toma_001/frame_000.jpg, frame_001.jpg, ...
        origen/Hola/toma_002/...

  B) Imágenes sueltas: cada imagen es una toma de 1 frame (solo postura estática)
        origen/Hola/img_001.jpg

  Con --por-sujeto se agrega un nivel de sujeto arriba:
        origen/S01/Hola/toma_001/...   origen/S02/Hola/...

Uso:
    python -m comparacion.importar_imagenes --origen D:/fotos_lsc --sujeto S01
    python -m comparacion.importar_imagenes --origen D:/fotos_lsc --por-sujeto
    python -m comparacion.importar_imagenes --origen D:/fotos_lsc --sujeto S01 --simular

Salida: data/dual/<seña>/<sujeto>_<idx>/{landmarks.npy, landmarks_frames.npy, frames/, meta.json}
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from models.capturar_dual import ANCHO_GUARDADO, CALIDAD_JPG, vector_frame  # noqa: E402
from models.extractor_v2 import ExtractorSecuencial  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def orden_natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def imagenes(carpeta: Path):
    return sorted([p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXT], key=orden_natural)


def descubrir(origen: Path, sujeto_fijo, por_sujeto):
    """Devuelve [(sujeto, seña, nombre_toma, [rutas])]."""
    raices = [(d.name, d) for d in sorted(origen.iterdir()) if d.is_dir()] if por_sujeto else [(sujeto_fijo, origen)]
    tomas = []
    for sujeto, base in raices:
        for d_seña in sorted(p for p in base.iterdir() if p.is_dir()):
            subtomas = sorted([p for p in d_seña.iterdir() if p.is_dir()], key=orden_natural)
            if subtomas:                                   # estructura A
                for t in subtomas:
                    rutas = imagenes(t)
                    if rutas:
                        tomas.append((sujeto, d_seña.name, t.name, rutas))
            for img in imagenes(d_seña):                    # estructura B
                tomas.append((sujeto, d_seña.name, img.stem, [img]))
    return tomas


def procesar_toma(extractor, rutas, espejo=False):
    """MediaPipe sobre cada imagen. Devuelve [(frame, FrameFeatures, t_ms)] solo con mano."""
    extractor.detener()
    extractor.iniciar()      # detectores nuevos por toma: sin arrastre de tracking entre tomas
    salida = []
    for i, r in enumerate(rutas):
        frame = cv2.imdecode(np.fromfile(str(r), dtype=np.uint8), cv2.IMREAD_COLOR)  # admite rutas con tildes
        if frame is None:
            continue
        if espejo:
            frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()
        ff = extractor.procesar_frame(frame, i * 33)
        t_ms = (time.perf_counter() - t0) * 1000
        if ff and ff.manos_presentes:
            salida.append((frame, ff, t_ms))
    return salida


def main():
    ap = argparse.ArgumentParser(description="Importar base de imágenes al formato dual")
    ap.add_argument("--origen", required=True)
    ap.add_argument("--sujeto", help="ID del sujeto si todo el origen es de una persona (p. ej. S01)")
    ap.add_argument("--por-sujeto", action="store_true", help="origen/<sujeto>/<seña>/...")
    ap.add_argument("--salida", default="data/dual")
    ap.add_argument("--min-frames", type=int, default=8,
                    help="Mínimo de frames con mano por toma (estructura A). Imágenes sueltas: 1")
    ap.add_argument("--espejo", action="store_true",
                    help="Voltear horizontalmente (si las imágenes NO vienen en modo espejo como la app)")
    ap.add_argument("--simular", action="store_true", help="Solo listar lo que se importaría")
    a = ap.parse_args()

    if not a.por_sujeto and not a.sujeto:
        sys.exit("Indica --sujeto S01 o usa --por-sujeto")
    if a.sujeto and "_" in a.sujeto:
        sys.exit("El ID de sujeto no debe contener '_'")
    origen = Path(a.origen)
    salida = Path(a.salida)
    salida = salida if salida.is_absolute() else RAIZ / salida

    tomas = descubrir(origen, a.sujeto, a.por_sujeto)
    if not tomas:
        sys.exit(f"No encontré imágenes con la estructura esperada en {origen}")
    resumen = {}
    for s, c, _, r in tomas:
        resumen.setdefault((s, c), [0, 0])
        resumen[(s, c)][0] += 1
        resumen[(s, c)][1] += len(r)
    print(f"{len(tomas)} tomas encontradas:")
    for (s, c), (n, f) in sorted(resumen.items()):
        print(f"  {s:6s} {c:10s} {n:4d} tomas  ({f / n:.1f} imágenes/toma)")
    sueltas = sum(1 for t in tomas if len(t[3]) == 1)
    if sueltas:
        print(f"\n[AVISO] {sueltas} tomas son imágenes sueltas (1 frame). Sirven para postura estática, "
              "pero no capturan movimiento (Sí/No) y no coinciden con la ventana de 20 frames de la app.")
    if a.simular:
        return

    extractor = ExtractorSecuencial()
    if not extractor.iniciar():
        sys.exit("No se pudo iniciar MediaPipe (revisa data/hand_landmarker.task)")
    import logging
    logging.getLogger("lsc_bridge.features_v2").setLevel(logging.ERROR)  # no repetir avisos por toma
    ok = desc = 0
    for n, (sujeto, seña, nombre, rutas) in enumerate(tomas, 1):
        if "_" in sujeto:
            sujeto = sujeto.replace("_", "")
        minimo = 1 if len(rutas) == 1 else a.min_frames
        res = procesar_toma(extractor, rutas, a.espejo)
        if len(res) < minimo:
            desc += 1
            print(f"  [descartada] {sujeto}/{seña}/{nombre}: {len(res)}/{len(rutas)} imágenes con mano")
            continue
        d_seña = salida / seña
        d_seña.mkdir(parents=True, exist_ok=True)
        existentes = [int(p.name.split("_")[-1]) for p in d_seña.glob(f"{sujeto}_*") if p.is_dir()]
        idx = max(existentes) + 1 if existentes else 0
        destino = d_seña / f"{sujeto}_{idx:04d}"
        (destino / "frames").mkdir(parents=True)
        feats = [ff for _, ff, _ in res]
        np.save(destino / "landmarks.npy", extractor.agregar_secuencia(feats))
        np.save(destino / "landmarks_frames.npy", np.stack([vector_frame(f) for f in feats]))
        for i, (frame, _, _) in enumerate(res):
            h, w = frame.shape[:2]
            if w > ANCHO_GUARDADO:
                frame = cv2.resize(frame, (ANCHO_GUARDADO, int(round(h * ANCHO_GUARDADO / w))),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(destino / "frames" / f"{i:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])
        meta = {"clase": seña, "sujeto": sujeto, "indice": idx, "n_frames": len(res),
                "n_imagenes_origen": len(rutas), "origen": str(rutas[0].parent),
                "t_mediapipe_ms_medio": float(np.mean([t for _, _, t in res])),
                "cara_presente_frac": float(np.mean([f.cara_presente for f in feats])),
                "importado": True, "espejo_aplicado": a.espejo,
                "fecha": datetime.now().isoformat(timespec="seconds")}
        (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1
        if n % 25 == 0:
            print(f"  {n}/{len(tomas)}")
    extractor.detener()
    print(f"\nImportadas: {ok}   Descartadas: {desc}   → {salida}")
    print("Siguiente: python -m comparacion.entrenar_comparacion --datos data/dual")


if __name__ == "__main__":
    main()
