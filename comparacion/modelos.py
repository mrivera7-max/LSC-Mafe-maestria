"""
Los tres clasificadores con una interfaz común.

    m = crear_modelo("rf", clases, semilla=42, hilos=4)
    m.fit(X, y)                 # X: (n, 342) landmarks  |  (n, 1280) embeddings
    p = m.predict_proba(X)      # (n, n_clases)
    m.guardar(dir); m2 = cargar_modelo(dir)

MobileNetV2 se usa por transferencia de aprendizaje en modo *extracción de
características*: el backbone preentrenado en ImageNet queda congelado, cada
frame produce un embedding de 1280 dims, los K frames de la toma se promedian
(pooling temporal) y una cabeza densa entrenable clasifica. Es el análogo
directo del vector agregado de landmarks (media/std/delta sobre la ventana).
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("lsc_bridge.comparacion.modelos")

# Hiperparámetros fijos, idénticos en entrenamiento offline y en vivo.
HIPER = {
    "rf": {"n_estimators": 200, "max_depth": None, "escalado": "StandardScaler"},
    "mlp": {"hidden_layer_sizes": [128, 64], "activation": "relu",
            "max_iter": 1000, "early_stopping": True, "validation_fraction": 0.15,
            "escalado": "StandardScaler"},
    "mobilenetv2": {"backbone": "mobilenet_v2 IMAGENET1K_V1 (congelado)",
                    "frames_por_toma": 8, "alto_entrada": 224,
                    "pooling_temporal": "media", "cabeza": [256],
                    "dropout": 0.3, "optimizador": "Adam", "lr": 1e-3,
                    "weight_decay": 1e-4, "epocas_max": 300, "batch": 32,
                    "early_stopping_paciencia": 25, "validation_fraction": 0.15},
}

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def muestrear_indices(n: int, k: int) -> np.ndarray:
    """K índices uniformes en [0, n-1] (repite si n < k)."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.round(np.linspace(0, n - 1, k)).astype(int)


# ── Modelos sobre landmarks (scikit-learn) ────────────────────────

class ModeloSklearn:
    entrada = "landmarks"

    def __init__(self, nombre, clases, semilla=42, hilos=1):
        self.nombre = nombre
        self.clases = list(clases)
        self.semilla = semilla
        self.hilos = hilos
        self._pipe = None

    def _construir(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        if self.nombre == "rf":
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(n_estimators=HIPER["rf"]["n_estimators"],
                                         random_state=self.semilla, n_jobs=self.hilos)
        else:
            from sklearn.neural_network import MLPClassifier
            h = HIPER["mlp"]
            est = MLPClassifier(hidden_layer_sizes=tuple(h["hidden_layer_sizes"]),
                                activation=h["activation"], max_iter=h["max_iter"],
                                early_stopping=h["early_stopping"],
                                validation_fraction=h["validation_fraction"],
                                random_state=self.semilla)
        return Pipeline([("scaler", StandardScaler()), (self.nombre, est)])

    def fit(self, X, y):
        self._pipe = self._construir()
        self._pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        proba = self._pipe.predict_proba(X)
        # Si en un fold faltó alguna clase, re-expandir a n_clases
        out = np.zeros((len(X), len(self.clases)))
        out[:, self._pipe.classes_] = proba
        return out

    def modo_inferencia(self):
        """Un solo hilo por predicción (como en vivo): latencia comparable entre modelos."""
        if self.nombre == "rf":
            self._pipe[-1].n_jobs = 1

    def n_parametros(self) -> int:
        est = self._pipe[-1]
        if self.nombre == "rf":
            return int(sum(t.tree_.node_count for t in est.estimators_))
        return int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))

    def guardar(self, directorio: Path):
        import joblib
        directorio.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, directorio / "modelo.joblib")

    @classmethod
    def cargar(cls, directorio: Path, meta: dict):
        import joblib
        m = cls(meta["modelo"], meta["clases"], meta.get("semilla", 42), 1)
        m._pipe = joblib.load(directorio / "modelo.joblib")
        m.modo_inferencia()
        return m


# ── MobileNetV2 por transferencia de aprendizaje (PyTorch) ────────

def torch_disponible() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


class BackboneMobileNetV2:
    """Extractor congelado: frame BGR -> embedding (1280,)."""

    def __init__(self, pesos_imagenet=True, ruta_estado: Path = None, hilos=None):
        import torch
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        if hilos:
            torch.set_num_threads(int(hilos))
        if ruta_estado is not None and Path(ruta_estado).exists():
            red = mobilenet_v2(weights=None)
            red.load_state_dict(torch.load(ruta_estado, map_location="cpu"))
            self.origen = f"local:{Path(ruta_estado).name}"
        else:
            pesos = MobileNet_V2_Weights.IMAGENET1K_V1 if pesos_imagenet else None
            red = mobilenet_v2(weights=pesos)
            self.origen = "IMAGENET1K_V1" if pesos_imagenet else "SIN_PESOS(prueba)"
        red.eval()
        for p in red.parameters():
            p.requires_grad_(False)
        self._red = red
        self._torch = torch
        self.alto = HIPER["mobilenetv2"]["alto_entrada"]

    def guardar_estado(self, ruta: Path):
        self._torch.save(self._red.state_dict(), ruta)

    def n_parametros(self) -> int:
        return int(sum(p.numel() for p in self._red.parameters()))

    def _preprocesar(self, frames_bgr):
        import cv2
        lote = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            ancho = int(round(w * self.alto / h))
            img = cv2.resize(f, (ancho, self.alto), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - MEDIA_IMAGENET) / STD_IMAGENET
            lote.append(img.transpose(2, 0, 1))
        return self._torch.from_numpy(np.stack(lote))

    def embeber(self, frames_bgr) -> np.ndarray:
        """(n_frames, 1280). Todos los frames deben tener el mismo tamaño."""
        if len(frames_bgr) == 0:
            return np.zeros((0, 1280), dtype=np.float32)
        with self._torch.inference_mode():
            x = self._preprocesar(frames_bgr)
            f = self._red.features(x)
            f = self._torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        return f.numpy().astype(np.float32)


class ModeloMobileNetV2:
    """Cabeza entrenable sobre el embedding promediado de K frames."""
    entrada = "imagenes"

    def __init__(self, clases, semilla=42, hilos=1):
        self.nombre = "mobilenetv2"
        self.clases = list(clases)
        self.semilla = semilla
        self.hilos = hilos
        self._cabeza = None
        self.backbone = None   # se asigna al cargar para uso en vivo
        self.epocas_usadas = None

    def _construir(self):
        import torch.nn as nn
        h = HIPER["mobilenetv2"]
        return nn.Sequential(
            nn.Linear(1280, h["cabeza"][0]), nn.ReLU(), nn.Dropout(h["dropout"]),
            nn.Linear(h["cabeza"][0], len(self.clases)),
        )

    def fit(self, X, y):
        import torch
        h = HIPER["mobilenetv2"]
        torch.manual_seed(self.semilla)
        rng = np.random.default_rng(self.semilla)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Partición de validación estratificada para early stopping (igual que el MLP)
        from sklearn.model_selection import train_test_split
        try:
            itr, iva = train_test_split(np.arange(len(y)), test_size=h["validation_fraction"],
                                        stratify=y, random_state=self.semilla)
        except ValueError:
            itr, iva = train_test_split(np.arange(len(y)), test_size=h["validation_fraction"],
                                        random_state=self.semilla)
        Xtr, ytr = torch.from_numpy(X[itr]), torch.from_numpy(y[itr])
        Xva, yva = torch.from_numpy(X[iva]), torch.from_numpy(y[iva])

        self._cabeza = self._construir()
        opt = torch.optim.Adam(self._cabeza.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
        perdida = torch.nn.CrossEntropyLoss()
        mejor, mejor_estado, sin_mejora = -1.0, None, 0
        for epoca in range(h["epocas_max"]):
            self._cabeza.train()
            orden = rng.permutation(len(itr))
            for i in range(0, len(orden), h["batch"]):
                b = orden[i:i + h["batch"]]
                opt.zero_grad()
                loss = perdida(self._cabeza(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            self._cabeza.eval()
            with torch.no_grad():
                acc = (self._cabeza(Xva).argmax(1) == yva).float().mean().item()
            if acc > mejor + 1e-6:
                mejor, sin_mejora = acc, 0
                mejor_estado = {k: v.clone() for k, v in self._cabeza.state_dict().items()}
                self.epocas_usadas = epoca + 1
            else:
                sin_mejora += 1
                if sin_mejora >= h["early_stopping_paciencia"]:
                    break
        self._cabeza.load_state_dict(mejor_estado)
        self._cabeza.eval()
        return self

    def predict_proba(self, X):
        import torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            return torch.softmax(self._cabeza(X), dim=1).numpy()

    def modo_inferencia(self):
        pass

    def n_parametros(self) -> int:
        cab = int(sum(p.numel() for p in self._cabeza.parameters()))
        # backbone MobileNetV2 = 3 504 872 parámetros (congelados)
        return cab + (self.backbone.n_parametros() if self.backbone else 3504872)

    def guardar(self, directorio: Path, backbone: BackboneMobileNetV2 = None):
        import torch
        directorio.mkdir(parents=True, exist_ok=True)
        torch.save(self._cabeza.state_dict(), directorio / "cabeza.pt")
        bb = backbone or self.backbone
        if bb is not None:
            bb.guardar_estado(directorio / "backbone.pt")

    @classmethod
    def cargar(cls, directorio: Path, meta: dict, hilos=None):
        import torch
        m = cls(meta["clases"], meta.get("semilla", 42), 1)
        m._cabeza = m._construir()
        m._cabeza.load_state_dict(torch.load(directorio / "cabeza.pt", map_location="cpu"))
        m._cabeza.eval()
        m.backbone = BackboneMobileNetV2(ruta_estado=directorio / "backbone.pt", hilos=hilos)
        return m


# ── Fábrica / persistencia ────────────────────────────────────────

def crear_modelo(nombre, clases, semilla=42, hilos=1):
    if nombre in ("rf", "mlp"):
        return ModeloSklearn(nombre, clases, semilla, hilos)
    if nombre == "mobilenetv2":
        return ModeloMobileNetV2(clases, semilla, hilos)
    raise ValueError(f"Modelo desconocido: {nombre}")


def tamano_en_disco_mb(directorio: Path) -> float:
    return sum(p.stat().st_size for p in Path(directorio).glob("*") if p.is_file()
               and p.name != "meta.json") / 1e6


def guardar_meta(directorio: Path, meta: dict):
    directorio.mkdir(parents=True, exist_ok=True)
    (directorio / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                          encoding="utf-8")


def cargar_modelo(directorio: Path, hilos=None):
    """Carga un modelo guardado por entrenar_comparacion.py. Devuelve (modelo, meta)."""
    directorio = Path(directorio)
    meta = json.loads((directorio / "meta.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    if meta["modelo"] == "mobilenetv2":
        m = ModeloMobileNetV2.cargar(directorio, meta, hilos=hilos)
    else:
        m = ModeloSklearn.cargar(directorio, meta)
    log.info(f"Modelo '{meta['modelo']}' cargado en {(time.perf_counter()-t0)*1000:.0f} ms")
    return m, meta
