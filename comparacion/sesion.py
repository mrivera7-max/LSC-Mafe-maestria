"""
Registro de una sesión de uso en vivo y su informe comparativo.

Qué se mide por modelo (mientras está activo):
  * rendimiento: FPS, % frames con mano, latencia MediaPipe / modelo / total por frame
  * salida: detecciones emitidas y su confianza
  * exactitud en vivo (solo si se marca la "seña esperada" en la GUI):
      - ensayo = intervalo con una seña esperada fija y un mismo modelo
      - acierto por ensayo      : hubo al menos una detección correcta
      - precisión de emisiones  : detecciones correctas / detecciones emitidas
      - latencia de reconocimiento: tiempo desde el inicio del ensayo hasta la 1.ª correcta
      - matriz de confusión esperada × detectada

El informe incluye además las métricas offline (validación cruzada) guardadas en
el meta.json de cada modelo, y avisa si los modelos NO se entrenaron bajo las
mismas condiciones (distinto dataset, esquema de validación o equipo).
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from comparacion import NOMBRES_LEGIBLES
from comparacion import informe as inf
from comparacion.entorno import describir_entorno


@dataclass
class Ensayo:
    modelo: str
    esperada: str
    t_inicio: float
    t_fin: Optional[float] = None
    detecciones: List[tuple] = field(default_factory=list)  # (t, seña, conf)

    @property
    def t_primera_correcta(self):
        for t, s, _ in self.detecciones:
            if s == self.esperada:
                return t - self.t_inicio
        return None


class SesionComparacion:
    def __init__(self, parametros: dict = None):
        self._lock = threading.Lock()
        self.inicio = time.time()
        self.fecha = datetime.now()
        self.parametros = parametros or {}
        self.frames: List[tuple] = []        # (t, modelo, manos, t_mp, t_modelo|None)
        self.eventos: List[tuple] = []       # (t, tipo, modelo, seña, conf, esperada)
        self.ensayos: List[Ensayo] = []
        self.tiempo_activo: Dict[str, float] = {}
        self.metas: Dict[str, dict] = {}
        self._modelo = None
        self._t_modelo_desde = None
        self._esperada = None
        self._ensayo: Optional[Ensayo] = None

    # ── Eventos ──────────────────────────────────────────────────
    def _t(self):
        return time.time() - self.inicio

    def _cerrar_ensayo(self, t):
        if self._ensayo is not None:
            self._ensayo.t_fin = t
            # Ensayos de menos de 1 s sin detecciones son ruido de la interfaz
            # (p. ej. cambiar de modelo y de seña esperada casi a la vez).
            if t - self._ensayo.t_inicio >= 1.0 or self._ensayo.detecciones:
                self.ensayos.append(self._ensayo)
            self._ensayo = None

    def _abrir_ensayo(self, t):
        if self._modelo and self._esperada:
            self._ensayo = Ensayo(self._modelo, self._esperada, t)

    def cambiar_modelo(self, modelo: str, meta: dict = None):
        with self._lock:
            t = self._t()
            if self._modelo is not None and self._t_modelo_desde is not None:
                self.tiempo_activo[self._modelo] = self.tiempo_activo.get(self._modelo, 0) + t - self._t_modelo_desde
            self._cerrar_ensayo(t)
            self._modelo, self._t_modelo_desde = modelo, t
            if meta:
                self.metas[modelo] = meta
            self.eventos.append((t, "modelo", modelo, "", "", self._esperada or ""))
            self._abrir_ensayo(t)

    def fijar_esperada(self, seña: Optional[str]):
        with self._lock:
            t = self._t()
            self._cerrar_ensayo(t)
            self._esperada = seña or None
            self.eventos.append((t, "esperada", self._modelo or "", seña or "", "", seña or ""))
            self._abrir_ensayo(t)

    def registrar_frame(self, manos: bool, t_mp_ms: float, t_modelo_ms: Optional[float]):
        with self._lock:
            self.frames.append((self._t(), self._modelo, manos, t_mp_ms, t_modelo_ms))

    def registrar_deteccion(self, seña: str, conf: float):
        with self._lock:
            t = self._t()
            self.eventos.append((t, "deteccion", self._modelo, seña, round(conf, 4), self._esperada or ""))
            if self._ensayo is not None:
                self._ensayo.detecciones.append((t, seña, conf))

    def finalizar(self):
        with self._lock:
            t = self._t()
            if self._modelo is not None and self._t_modelo_desde is not None:
                self.tiempo_activo[self._modelo] = self.tiempo_activo.get(self._modelo, 0) + t - self._t_modelo_desde
                self._t_modelo_desde = t
            self._cerrar_ensayo(t)
            self._abrir_ensayo(t)  # por si la sesión continúa

    @property
    def vacia(self) -> bool:
        return len(self.frames) == 0

    # ── Métricas ─────────────────────────────────────────────────
    def _metricas(self, modelo, clases):
        fr = [f for f in self.frames if f[1] == modelo]
        t_act = self.tiempo_activo.get(modelo, 0.0)
        mp = np.array([f[3] for f in fr]) if fr else np.zeros(0)
        tm = np.array([f[4] for f in fr if f[4] is not None])
        tot = np.array([f[3] + (f[4] or 0.0) for f in fr]) if fr else np.zeros(0)
        det = [e for e in self.eventos if e[1] == "deteccion" and e[2] == modelo]
        ens = [e for e in self.ensayos if e.modelo == modelo]
        det_en_ens = [(e.esperada, s) for e in ens for _, s, _ in e.detecciones]
        latencias = [e.t_primera_correcta for e in ens if e.t_primera_correcta is not None]
        idx = {c: i for i, c in enumerate(clases)}
        cm = np.zeros((len(clases), len(clases)), dtype=int)
        for esp, s in det_en_ens:
            if esp in idx and s in idx:
                cm[idx[esp], idx[s]] += 1

        def q(a, p):
            return float(np.percentile(a, p)) if len(a) else float("nan")
        return {
            "t_activo_s": t_act, "frames": len(fr),
            "fps": len(fr) / t_act if t_act > 0 else float("nan"),
            "pct_mano": 100 * np.mean([f[2] for f in fr]) if fr else float("nan"),
            "mp_med": q(mp, 50), "mp_p95": q(mp, 95),
            "mod_med": q(tm, 50), "mod_p95": q(tm, 95),
            "tot_med": q(tot, 50), "tot_p95": q(tot, 95),
            "detecciones": len(det),
            "conf_media": float(np.mean([e[4] for e in det])) if det else float("nan"),
            "ensayos": len(ens),
            "ensayos_acertados": sum(1 for e in ens if e.t_primera_correcta is not None),
            "emisiones_en_ensayo": len(det_en_ens),
            "emisiones_correctas": sum(1 for a, b in det_en_ens if a == b),
            "lat_rec_med_s": float(np.median(latencias)) if latencias else float("nan"),
            "cm": cm,
        }

    def _aviso_homogeneidad(self):
        claves = {}
        for m, meta in self.metas.items():
            ds = meta.get("dataset", {})
            val = meta.get("validacion", {})
            claves[m] = (ds.get("directorio"), ds.get("n_muestras"), val.get("esquema"),
                         meta.get("entorno", {}).get("equipo"))
        if len(set(claves.values())) > 1:
            return ("Los modelos cargados NO se entrenaron bajo las mismas condiciones "
                    "(dataset, esquema de validación o equipo difieren). Reentrena los tres juntos con "
                    "comparacion.entrenar_comparacion para que la comparación sea válida.")
        return None

    # ── Informe ──────────────────────────────────────────────────
    def generar_informe(self, dir_base: Path, clases: List[str]) -> Optional[Path]:
        self.finalizar()
        with self._lock:
            if not self.frames:
                return None
            modelos = [m for m in dict.fromkeys(f[1] for f in self.frames) if m]
            destino = Path(dir_base) / f"sesion_{self.fecha:%Y%m%d_%H%M%S}"
            destino.mkdir(parents=True, exist_ok=True)
            met = {m: self._metricas(m, clases) for m in modelos}
            entorno = describir_entorno()

            # CSV
            inf.escribir_csv(destino / "eventos.csv", ["t_s", "tipo", "modelo", "sena", "confianza", "esperada"],
                             [[f"{e[0]:.3f}"] + list(e[1:]) for e in self.eventos])
            inf.escribir_csv(destino / "frames.csv", ["t_s", "modelo", "manos", "t_mediapipe_ms", "t_modelo_ms"],
                             [[f"{f[0]:.3f}", f[1], int(f[2]), f"{f[3]:.2f}",
                               "" if f[4] is None else f"{f[4]:.2f}"] for f in self.frames])
            inf.escribir_csv(destino / "ensayos.csv",
                             ["modelo", "esperada", "t_inicio_s", "duracion_s", "detecciones",
                              "correctas", "t_primera_correcta_s"],
                             [[e.modelo, e.esperada, f"{e.t_inicio:.2f}", f"{(e.t_fin or e.t_inicio) - e.t_inicio:.2f}",
                               len(e.detecciones), sum(1 for d in e.detecciones if d[1] == e.esperada),
                               "" if e.t_primera_correcta is None else f"{e.t_primera_correcta:.2f}"]
                              for e in self.ensayos])
            cols = ["modelo", "t_activo_s", "frames", "fps", "pct_frames_mano", "mediapipe_med_ms", "mediapipe_p95_ms",
                    "modelo_med_ms", "modelo_p95_ms", "total_med_ms", "total_p95_ms", "detecciones",
                    "confianza_media", "ensayos", "ensayos_acertados", "emisiones_en_ensayo",
                    "emisiones_correctas", "latencia_reconocimiento_med_s", "acc_offline_media", "acc_offline_std"]
            filas = []
            for m in modelos:
                r = met[m]
                acc = self.metas.get(m, {}).get("validacion", {}).get("accuracy", [None, None])
                filas.append([m, r["t_activo_s"], r["frames"], r["fps"], r["pct_mano"], r["mp_med"], r["mp_p95"],
                              r["mod_med"], r["mod_p95"], r["tot_med"], r["tot_p95"], r["detecciones"],
                              r["conf_media"], r["ensayos"], r["ensayos_acertados"], r["emisiones_en_ensayo"],
                              r["emisiones_correctas"], r["lat_rec_med_s"], acc[0], acc[1]])
            inf.escribir_csv(destino / "resumen_modelos.csv", cols, filas)
            (destino / "sesion.json").write_text(json.dumps({
                "fecha": self.fecha.isoformat(timespec="seconds"), "parametros": self.parametros,
                "entorno": entorno, "metas_modelos": self.metas}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")

            # HTML
            f1 = lambda v, d=1: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"  # noqa: E731
            c = []
            aviso = self._aviso_homogeneidad()
            if aviso:
                c.append(f'<div class="aviso">{inf.esc(aviso)}</div>')
            if not any(met[m]["ensayos"] for m in modelos):
                c.append('<div class="aviso">No se marcó “seña esperada” en ningún momento: el informe solo '
                         'compara rendimiento (FPS, latencias, confianza). Para medir exactitud en vivo, '
                         'selecciona la seña esperada antes de ejecutarla.</div>')
            c.append(inf.tiles([
                ("Duración de la sesión", f"{self._t()/60:.1f} min"),
                ("Modelos usados", str(len(modelos))),
                ("Frames procesados", str(len(self.frames))),
                ("Ensayos con seña esperada", str(len(self.ensayos))),
            ]))

            c.append("<h2>Exactitud en vivo</h2>")
            c.append(inf.tabla(
                ["Modelo", "Ensayos", "Acierto por ensayo (%)", "Precisión de emisiones (%)",
                 "Latencia de reconocimiento (s, mediana)", "Accuracy offline CV (%)"],
                [[inf.nombre_modelo(m), met[m]["ensayos"],
                  f1(100 * met[m]["ensayos_acertados"] / met[m]["ensayos"]) if met[m]["ensayos"] else "—",
                  f1(100 * met[m]["emisiones_correctas"] / met[m]["emisiones_en_ensayo"]) if met[m]["emisiones_en_ensayo"] else "—",
                  f1(met[m]["lat_rec_med_s"], 2),
                  (lambda a: "—" if not a or a[0] is None else f"{a[0]*100:.2f} ± {a[1]*100:.2f}")(
                      self.metas.get(m, {}).get("validacion", {}).get("accuracy"))]
                 for m in modelos]))
            ens_items = [(m, 100 * met[m]["ensayos_acertados"] / met[m]["ensayos"], None,
                          f"{met[m]['ensayos_acertados']}/{met[m]['ensayos']} ensayos")
                         for m in modelos if met[m]["ensayos"]]
            if ens_items:
                c.append(inf.barras("Acierto por ensayo en vivo (%)", ens_items, unidad="%", maximo=100))

            c.append("<h2>Rendimiento computacional (mismo equipo, misma sesión)</h2>")
            c.append(inf.tabla(
                ["Modelo", "Tiempo activo (s)", "FPS", "Frames con mano (%)", "MediaPipe med/p95 (ms)",
                 "Modelo med/p95 (ms)", "Total/frame med/p95 (ms)", "Detecciones", "Confianza media"],
                [[inf.nombre_modelo(m), f1(met[m]["t_activo_s"]), f1(met[m]["fps"]), f1(met[m]["pct_mano"]),
                  f"{f1(met[m]['mp_med'])} / {f1(met[m]['mp_p95'])}",
                  f"{f1(met[m]['mod_med'], 2)} / {f1(met[m]['mod_p95'], 2)}",
                  f"{f1(met[m]['tot_med'])} / {f1(met[m]['tot_p95'])}",
                  met[m]["detecciones"], f1(met[m]["conf_media"], 3)] for m in modelos]))
            c.append(inf.barras("Latencia total por frame (ms, mediana)",
                                [(m, met[m]["tot_med"], None, f"p95 = {f1(met[m]['tot_p95'])} ms")
                                 for m in modelos if not np.isnan(met[m]["tot_med"])], unidad=" ms"))
            c.append(inf.barras("FPS efectivos", [(m, met[m]["fps"], None, None) for m in modelos
                                                  if not np.isnan(met[m]["fps"])]))
            c.append('<p class="nota">“Modelo” = costo por frame del clasificador (RF/MLP: agregación + predicción; '
                     'MobileNetV2: embedding del frame + cabeza). Todos los modelos ejecutan también MediaPipe '
                     '(RF/MLP lo necesitan como entrada; MobileNetV2 lo usa para detectar presencia de mano y '
                     'mantener la misma ventana y votación).</p>')

            cms = [m for m in modelos if met[m]["cm"].sum() > 0]
            if cms:
                c.append("<h2>Matrices de confusión en vivo (esperada × detectada)</h2><div class='cms'>")
                for m in cms:
                    c.append(inf.matriz_confusion(met[m]["cm"], clases, inf.nombre_modelo(m)))
                c.append("</div>")

            c.append("<h2>Condiciones</h2>")
            c.append(inf.tabla(["Parámetro", "Valor"], [[inf.esc(k), inf.esc(v)] for k, v in self.parametros.items()]))
            filas_m = []
            for m in modelos:
                meta = self.metas.get(m, {})
                ds = meta.get("dataset", {})
                filas_m.append([inf.nombre_modelo(m), inf.esc(meta.get("fecha", "—")),
                                inf.esc(f"{ds.get('n_muestras', '—')} muestras · {ds.get('formato', '—')}"),
                                inf.esc(meta.get("validacion", {}).get("esquema", "—")),
                                inf.esc(meta.get("entorno", {}).get("equipo", "—"))])
            c.append("<h3>Origen de cada modelo</h3>")
            c.append(inf.tabla(["Modelo", "Entrenado", "Dataset", "Validación", "Equipo"], filas_m))
            c.append("<h3>Equipo de esta sesión</h3>")
            c.append(inf.entorno_html(entorno))
            c.append('<p class="nota">Archivos: resumen_modelos.csv, ensayos.csv, eventos.csv, frames.csv, '
                     'sesion.json (separador “;”).</p>')

            html_ = inf.pagina("Comparación de modelos LSC — sesión en vivo",
                               f"{self.fecha:%Y-%m-%d %H:%M} · {inf.esc(entorno['equipo'])} · "
                               + ", ".join(NOMBRES_LEGIBLES.get(m, m) for m in modelos), "".join(c))
            (destino / "informe.html").write_text(html_, encoding="utf-8")
            return destino / "informe.html"
