"""
Carga del dataset para la comparación.

Formato dual (models/capturar_dual.py) — una carpeta por toma:

    data/dual/<clase>/<sujeto>_<idx:04d>/
        landmarks.npy          (342,)  vector agregado, idéntico al de v2
        landmarks_frames.npy   (N,152) features por frame (para modelos temporales futuros)
        frames/000.jpg ...     N frames RGB de la MISMA toma (solo frames con mano)
        meta.json

Formato solo-landmarks (data/sequences/<clase>/*.npy): permite comparar RF y MLP
con el dataset ya existente; MobileNetV2 queda excluido (no hay imágenes).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class Muestra:
    clase: str
    sujeto: str
    ruta_landmarks: Path
    dir_frames: Optional[Path] = None
    meta: dict = field(default_factory=dict)

    def frames(self) -> List[Path]:
        if self.dir_frames is None:
            return []
        return sorted(self.dir_frames.glob("*.jpg"))


@dataclass
class Dataset:
    muestras: List[Muestra]
    clases: List[str]
    formato: str  # "dual" | "solo_landmarks"
    directorio: Path

    @property
    def y(self) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.clases)}
        return np.array([idx[m.clase] for m in self.muestras])

    @property
    def grupos(self) -> np.ndarray:
        return np.array([m.sujeto for m in self.muestras])

    def X_landmarks(self) -> np.ndarray:
        return np.stack([np.load(m.ruta_landmarks) for m in self.muestras])

    @property
    def tiene_imagenes(self) -> bool:
        return self.formato == "dual" and all(m.dir_frames and m.frames() for m in self.muestras)

    def resumen(self) -> dict:
        por_clase = {c: 0 for c in self.clases}
        for m in self.muestras:
            por_clase[m.clase] += 1
        return {
            "formato": self.formato,
            "directorio": str(self.directorio),
            "n_muestras": len(self.muestras),
            "n_clases": len(self.clases),
            "por_clase": por_clase,
            "sujetos": sorted(set(self.grupos.tolist())),
        }


def _resolver(ruta) -> Path:
    p = Path(ruta)
    return p if p.is_absolute() else RAIZ / p


def cargar_dataset(directorio) -> Dataset:
    d = _resolver(directorio)
    if not d.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {d}")
    clases = sorted(x.name for x in d.iterdir() if x.is_dir())
    if not clases:
        raise FileNotFoundError(f"Sin carpetas de clase en {d}")

    # ¿dual o solo landmarks?
    es_dual = any(next((d / c).glob("*/landmarks.npy"), None) is not None for c in clases)

    muestras = []
    if es_dual:
        for c in clases:
            for toma in sorted((d / c).iterdir()):
                lm = toma / "landmarks.npy"
                if not lm.exists():
                    continue
                meta = {}
                if (toma / "meta.json").exists():
                    meta = json.loads((toma / "meta.json").read_text(encoding="utf-8"))
                muestras.append(Muestra(
                    clase=c, sujeto=meta.get("sujeto", toma.name.split("_")[0]),
                    ruta_landmarks=lm, dir_frames=toma / "frames", meta=meta))
        formato = "dual"
    else:
        for c in clases:
            for f in sorted((d / c).glob("*.npy")):
                muestras.append(Muestra(clase=c, sujeto="desconocido", ruta_landmarks=f))
        formato = "solo_landmarks"

    return Dataset(muestras=muestras, clases=clases, formato=formato, directorio=d)
