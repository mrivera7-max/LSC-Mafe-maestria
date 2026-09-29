"""Registro del entorno de hardware y software (condiciones experimentales)."""

import os
import platform
import socket
import sys
from importlib import metadata


def _version(paquete):
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return "no instalado"


def _cpu_nombre():
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                                 text=True, timeout=5).stdout.split("\n")
            nombres = [l.strip() for l in out if l.strip() and l.strip() != "Name"]
            if nombres:
                return nombres[0]
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for linea in f:
                    if linea.startswith("model name"):
                        return linea.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "desconocido"


def _ram_gb():
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return None


def describir_entorno(hilos=None) -> dict:
    gpu = "no usada (CPU)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} (disponible, NO usada: comparación en CPU)"
    except ImportError:
        pass
    return {
        "equipo": socket.gethostname(),
        "so": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": _cpu_nombre(),
        "nucleos_logicos": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": gpu,
        "hilos_limitados_a": hilos,
        "python": sys.version.split()[0],
        "paquetes": {p: _version(p) for p in (
            "numpy", "scikit-learn", "mediapipe", "opencv-python",
            "torch", "torchvision", "threadpoolctl")},
    }
