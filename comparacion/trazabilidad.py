"""Trazabilidad de experimentos: commit de Git, huella del dataset y bitácora acumulada.

Cumple la función de MLflow sin servidor: cada entrenamiento agrega una fila por
modelo a experimentos/bitacora.csv (versionada en Git) con el código, los datos,
los parámetros y las métricas que produjeron ese resultado.
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

IGNORAR = {"Thumbs.db", "desktop.ini", ".DS_Store"}

COLUMNAS_BITACORA = [
    "id_corrida", "fecha", "commit", "codigo_sin_commit", "version_codigo",
    "dataset", "version_dataset", "huella_dataset", "n_muestras", "n_clases", "n_sujetos",
    "validacion", "particiones", "semilla", "hilos", "k_frames",
    "modelo", "hiperparametros",
    "accuracy_media", "accuracy_std", "f1_macro_media", "f1_macro_std",
    "precision_macro_media", "recall_macro_media", "latencia_mediana_ms",
    "informe", "nota",
]


def _git(raiz: Path, *args, crudo=False) -> str:
    r = subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout if crudo else r.stdout.strip()


def estado_git(raiz) -> dict:
    """Commit actual, rama, etiqueta y si hay cambios sin commit en archivos versionados."""
    raiz = Path(raiz)
    try:
        cambios = _git(raiz, "status", "--porcelain", "--untracked-files=no", crudo=True).rstrip("\n")
        return {
            "commit": _git(raiz, "rev-parse", "HEAD"),
            "rama": _git(raiz, "rev-parse", "--abbrev-ref", "HEAD"),
            "version": _git(raiz, "describe", "--tags", "--always", "--dirty"),
            "sin_commit": bool(cambios),
            "archivos_modificados": [l[3:] for l in cambios.splitlines()][:50],
        }
    except Exception as e:  # sin git instalado o fuera de un repositorio
        return {"commit": "no disponible", "rama": "", "version": "", "sin_commit": None,
                "archivos_modificados": [], "error": str(e)}


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def huella_dataset(directorio) -> dict:
    """SHA-256 de las líneas 'ruta;sha256' ordenadas de todos los archivos de la carpeta.
    Mismo algoritmo que manifiesto_dataset.py del capturador."""
    raiz = Path(directorio)
    archivos = sorted(
        (p for p in raiz.rglob("*") if p.is_file() and p.name not in IGNORAR
         and not p.name.startswith("MANIFIESTO_")
         and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(raiz).parts)),
        key=lambda p: p.relative_to(raiz).as_posix())
    h = hashlib.sha256()
    for p in archivos:
        h.update(f"{p.relative_to(raiz).as_posix()};{_sha256(p)}\n".encode("utf-8"))
    huella = h.hexdigest()
    return {"huella_sha256": huella, "huella_corta": huella[:12], "archivos": len(archivos)}


def registrar_bitacora(ruta_csv, filas):
    """Agrega filas (dicts con COLUMNAS_BITACORA) al final de la bitácora; crea el encabezado si no existe."""
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    nueva = not ruta_csv.exists() or ruta_csv.stat().st_size == 0
    with ruta_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_BITACORA, extrasaction="ignore")
        if nueva:
            w.writeheader()
        for fila in filas:
            w.writerow({k: (json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v)
                        for k, v in fila.items()})
