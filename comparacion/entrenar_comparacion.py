"""
Entrenamiento y evaluación comparativa de RF, MLP y MobileNetV2
bajo condiciones experimentales homogéneas.

Garantías de homogeneidad:
  * mismas muestras (cada toma aporta su vector de landmarks Y sus frames)
  * mismas particiones (los folds se generan una sola vez y se reutilizan)
  * misma semilla para particiones y modelos
  * mismo límite de hilos de CPU para todos (threadpoolctl + torch)
  * todo en CPU, mismo equipo; el entorno HW/SW queda registrado en el informe

Uso:
    # comparación completa (requiere dataset dual capturado con capturar_dual.py)
    python -m comparacion.entrenar_comparacion --datos data/dual

    # validación inter-sujeto (leave-one-subject-out, requiere >= 2 sujetos)
    python -m comparacion.entrenar_comparacion --datos data/dual --validacion loso

    # solo RF vs MLP con el dataset existente (sin imágenes)
    python -m comparacion.entrenar_comparacion --datos data/sequences --modelos rf mlp

Salidas:
    data/modelos_comparacion/<modelo>/   modelo final (todas las muestras) + meta.json
    informes/entrenamiento_<fecha>/      informe.html + CSV + entorno.json
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comparacion import MODELOS, NOMBRES_LEGIBLES  # noqa: E402
from comparacion import informe as inf  # noqa: E402
from comparacion.datos import RAIZ, cargar_dataset  # noqa: E402
from comparacion.entorno import describir_entorno  # noqa: E402
from comparacion.modelos import (HIPER, BackboneMobileNetV2, crear_modelo,  # noqa: E402
                                 guardar_meta, muestrear_indices, tamano_en_disco_mb,
                                 torch_disponible)

log = logging.getLogger("lsc_bridge.comparacion")


def parsear_args():
    p = argparse.ArgumentParser(description="Comparación RF / MLP / MobileNetV2 (LSC)")
    p.add_argument("--datos", default="data/dual")
    p.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    p.add_argument("--validacion", choices=["kfold", "loso"], default="kfold",
                   help="kfold estratificado o leave-one-subject-out")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--hilos", type=int, default=4, help="Hilos de CPU para TODOS los modelos")
    p.add_argument("--k-frames", type=int, default=HIPER["mobilenetv2"]["frames_por_toma"])
    p.add_argument("--salida-modelos", default="data/modelos_comparacion")
    p.add_argument("--salida-informes", default="informes")
    p.add_argument("--no-guardar-modelos", action="store_true",
                   help="Solo evaluar; no sobrescribir los modelos usados por la app")
    p.add_argument("--sin-pesos-imagenet", action="store_true",
                   help="SOLO PARA PRUEBAS: MobileNetV2 con pesos aleatorios")
    return p.parse_args()


def fijar_semillas(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


def generar_particiones(ds, y, esquema, folds, semilla):
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
    idx = np.arange(len(y))
    if esquema == "loso":
        grupos = ds.grupos
        if len(set(grupos)) < 2:
            raise SystemExit("LOSO necesita al menos 2 sujetos distintos en el dataset.")
        return list(LeaveOneGroupOut().split(idx, y, grupos)), \
            [f"sujeto {grupos[te][0]}" for _, te in LeaveOneGroupOut().split(idx, y, grupos)]
    min_clase = np.bincount(y).min()
    k = min(folds, int(min_clase))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    return list(skf.split(idx, y)), [f"fold {i+1}" for i in range(k)]


def embeddings_mobilenet(ds, backbone, k):
    """Embedding promedio de K frames por toma + tiempo por toma (ms)."""
    import cv2
    X, t_ms = [], []
    for i, m in enumerate(ds.muestras):
        rutas = m.frames()
        sel = [rutas[j] for j in muestrear_indices(len(rutas), k)]
        imgs = [cv2.imread(str(r)) for r in sel]
        t0 = time.perf_counter()
        e = backbone.embeber(imgs).mean(axis=0)
        t_ms.append((time.perf_counter() - t0) * 1000)
        X.append(e)
        if (i + 1) % 50 == 0:
            log.info(f"  embeddings {i+1}/{len(ds.muestras)}")
    return np.stack(X), np.array(t_ms)


def latencia_clasificador_ms(modelo, X, n=50):
    """Latencia de predicción de UNA muestra (como en vivo): mediana y p95."""
    modelo.modo_inferencia()
    t = []
    for x in X[:n]:
        t0 = time.perf_counter()
        modelo.predict_proba(x[None, :])
        t.append((time.perf_counter() - t0) * 1000)
    return np.array(t)


def mcnemar(correcto_a, correcto_b):
    from scipy.stats import binomtest
    b = int(np.sum(correcto_a & ~correcto_b))
    c = int(np.sum(~correcto_a & correcto_b))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def main():
    args = parsear_args()
    from utils.logger import configurar_logger
    configurar_logger()
    fijar_semillas(args.semilla)

    from threadpoolctl import threadpool_limits
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    threadpool_limits(args.hilos)
    try:
        import torch
        torch.set_num_threads(args.hilos)
    except ImportError:
        pass

    ds = cargar_dataset(args.datos)
    y = ds.y
    X_lm = ds.X_landmarks()
    log.info(f"Dataset {ds.formato}: {len(y)} muestras, {len(ds.clases)} clases, "
             f"sujetos={sorted(set(ds.grupos.tolist()))}")

    avisos = []
    modelos = list(args.modelos)
    entradas = {}
    t_embed = None
    backbone = None
    if "rf" in modelos or "mlp" in modelos:
        entradas["landmarks"] = X_lm
    if "mobilenetv2" in modelos:
        if not ds.tiene_imagenes:
            avisos.append("MobileNetV2 excluido: el dataset no tiene frames (usa capturar_dual.py).")
            modelos.remove("mobilenetv2")
        elif not torch_disponible():
            avisos.append("MobileNetV2 excluido: PyTorch/torchvision no instalados (requirements-cnn.txt).")
            modelos.remove("mobilenetv2")
        else:
            log.info("Calculando embeddings MobileNetV2 (backbone congelado)...")
            backbone = BackboneMobileNetV2(pesos_imagenet=not args.sin_pesos_imagenet, hilos=args.hilos)
            if args.sin_pesos_imagenet:
                avisos.append("MobileNetV2 SIN pesos ImageNet (modo prueba): resultados NO válidos.")
            entradas["imagenes"], t_embed = embeddings_mobilenet(ds, backbone, args.k_frames)

    particiones, nombres_part = generar_particiones(ds, y, args.validacion, args.folds, args.semilla)
    log.info(f"Validación {args.validacion}: {len(particiones)} particiones (idénticas para todos)")

    n_cl = len(ds.clases)
    res = {}
    for nombre in modelos:
        X = entradas["imagenes" if nombre == "mobilenetv2" else "landmarks"]
        oof = np.full(len(y), -1)
        oof_conf = np.zeros(len(y))
        folds = []
        lat_todas = []
        for (tr, te), etq in zip(particiones, nombres_part):
            m = crear_modelo(nombre, ds.clases, args.semilla, args.hilos)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            t_fit = time.perf_counter() - t0
            p = m.predict_proba(X[te])
            pred = p.argmax(1)
            oof[te] = pred
            oof_conf[te] = p.max(1)
            lat = latencia_clasificador_ms(m, X[te])
            if nombre == "mobilenetv2":
                lat = lat + np.median(t_embed[te])  # + backbone sobre K frames
            lat_todas.extend(lat.tolist())
            folds.append({
                "particion": etq, "n_test": len(te),
                "accuracy": accuracy_score(y[te], pred),
                "f1_macro": f1_score(y[te], pred, average="macro", labels=range(n_cl), zero_division=0),
                "precision_macro": precision_score(y[te], pred, average="macro", labels=range(n_cl), zero_division=0),
                "recall_macro": recall_score(y[te], pred, average="macro", labels=range(n_cl), zero_division=0),
                "t_entrenamiento_s": t_fit,
            })
            log.info(f"  {nombre:12s} {etq:>14s}: acc={folds[-1]['accuracy']*100:6.2f}%  "
                     f"f1={folds[-1]['f1_macro']:.3f}  fit={t_fit:.2f}s")
        f1_clase = f1_score(y, oof, average=None, labels=range(n_cl), zero_division=0)
        res[nombre] = {
            "folds": folds, "oof": oof, "oof_conf": oof_conf,
            "cm": confusion_matrix(y, oof, labels=range(n_cl)),
            "f1_clase": f1_clase, "lat": np.array(lat_todas),
        }
        for met in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "t_entrenamiento_s"):
            v = np.array([f[met] for f in folds])
            res[nombre][met] = (float(v.mean()), float(v.std()))

    # ── Modelo final (todas las muestras) y persistencia ─────────
    entorno = describir_entorno(args.hilos)
    fecha = datetime.now()
    dir_modelos = RAIZ / args.salida_modelos
    for nombre in modelos:
        X = entradas["imagenes" if nombre == "mobilenetv2" else "landmarks"]
        m = crear_modelo(nombre, ds.clases, args.semilla, args.hilos)
        t0 = time.perf_counter()
        m.fit(X, y)
        res[nombre]["t_final_s"] = time.perf_counter() - t0
        if nombre == "mobilenetv2":
            m.backbone = backbone
        res[nombre]["n_param"] = m.n_parametros()
        destino = dir_modelos / nombre
        if not args.no_guardar_modelos:
            if nombre == "mobilenetv2":
                m.guardar(destino, backbone)
            else:
                m.guardar(destino)
            res[nombre]["mb"] = tamano_en_disco_mb(destino)
            guardar_meta(destino, {
                "modelo": nombre, "nombre": NOMBRES_LEGIBLES[nombre],
                "clases": ds.clases, "semilla": args.semilla,
                "entrada": "imagenes" if nombre == "mobilenetv2" else "landmarks",
                "k_frames": args.k_frames if nombre == "mobilenetv2" else None,
                "hiperparametros": HIPER[nombre],
                "backbone_origen": backbone.origen if nombre == "mobilenetv2" else None,
                "validacion": {"esquema": args.validacion, "particiones": len(particiones),
                               "accuracy": res[nombre]["accuracy"], "f1_macro": res[nombre]["f1_macro"],
                               "latencia_mediana_ms": float(np.median(res[nombre]["lat"]))},
                "dataset": ds.resumen(), "entorno": entorno,
                "fecha": fecha.isoformat(timespec="seconds"),
            })
        else:
            res[nombre]["mb"] = None

    dir_inf = RAIZ / args.salida_informes / f"entrenamiento_{fecha:%Y%m%d_%H%M%S}"
    generar_informe_entrenamiento(dir_inf, ds, y, modelos, res, particiones, nombres_part,
                                  args, entorno, avisos, fecha, t_embed)
    log.info(f"Informe: {dir_inf / 'informe.html'}")
    print(f"\nInforme generado: {dir_inf / 'informe.html'}")


def _pct(mu_sd):
    return f"{mu_sd[0]*100:.2f} ± {mu_sd[1]*100:.2f}"


def generar_informe_entrenamiento(dir_inf, ds, y, modelos, res, particiones, nombres_part,
                                  args, entorno, avisos, fecha, t_embed):
    dir_inf.mkdir(parents=True, exist_ok=True)
    clases = ds.clases

    # ── CSV ──────────────────────────────────────────────────────
    cols = ["modelo", "accuracy_media", "accuracy_std", "f1_macro_media", "f1_macro_std",
            "precision_macro_media", "recall_macro_media", "t_entrenamiento_fold_s",
            "latencia_mediana_ms", "latencia_p95_ms", "n_parametros", "tamano_mb"]
    filas_csv = []
    for n in modelos:
        r = res[n]
        filas_csv.append([n, r["accuracy"][0], r["accuracy"][1], r["f1_macro"][0], r["f1_macro"][1],
                          r["precision_macro"][0], r["recall_macro"][0], r["t_entrenamiento_s"][0],
                          float(np.median(r["lat"])), float(np.percentile(r["lat"], 95)),
                          r["n_param"], r["mb"]])
    inf.escribir_csv(dir_inf / "metricas_resumen.csv", cols, filas_csv)
    inf.escribir_csv(dir_inf / "metricas_por_particion.csv",
                     ["modelo", "particion", "n_test", "accuracy", "f1_macro", "precision_macro",
                      "recall_macro", "t_entrenamiento_s"],
                     [[n] + list(f.values()) for n in modelos for f in res[n]["folds"]])
    inf.escribir_csv(dir_inf / "f1_por_clase.csv", ["modelo"] + clases,
                     [[n] + [float(v) for v in res[n]["f1_clase"]] for n in modelos])
    inf.escribir_csv(dir_inf / "predicciones_oof.csv",
                     ["muestra", "sujeto", "real"] + [f"pred_{n}" for n in modelos],
                     [[str(m.ruta_landmarks.relative_to(ds.directorio)), m.sujeto, clases[y[i]]] +
                      [clases[res[n]["oof"][i]] for n in modelos] for i, m in enumerate(ds.muestras)])
    pares = []
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            a, b = modelos[i], modelos[j]
            ca, cb = res[a]["oof"] == y, res[b]["oof"] == y
            nb, nc, p = mcnemar(ca, cb)
            pares.append([a, b, nb, nc, p])
    # Corrección de Holm
    orden = np.argsort([p[4] for p in pares])
    m_ = len(pares)
    p_holm = [0.0] * m_
    acum = 0.0
    for rango, k in enumerate(orden):
        acum = max(acum, min(1.0, (m_ - rango) * pares[k][4]))
        p_holm[k] = acum
    for k in range(m_):
        pares[k].append(p_holm[k])
    inf.escribir_csv(dir_inf / "mcnemar.csv",
                     ["modelo_a", "modelo_b", "a_bien_b_mal", "a_mal_b_bien", "p_exacto", "p_holm"], pares)
    (dir_inf / "entorno.json").write_text(json.dumps({
        "entorno": entorno, "dataset": ds.resumen(), "hiperparametros": {n: HIPER[n] for n in modelos},
        "validacion": args.validacion, "particiones": nombres_part, "semilla": args.semilla,
        "hilos": args.hilos, "k_frames": args.k_frames}, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── HTML ─────────────────────────────────────────────────────
    cuerpo = []
    for a in avisos:
        cuerpo.append(f'<div class="aviso">{inf.esc(a)}</div>')

    mejor_acc = max(modelos, key=lambda n: res[n]["accuracy"][0])
    mas_rapido = min(modelos, key=lambda n: np.median(res[n]["lat"]))
    cuerpo.append(inf.tiles([
        ("Muestras / clases", f"{len(y)} / {len(clases)}"),
        ("Validación", f"{'LOSO' if args.validacion == 'loso' else 'K-fold'} · {len(particiones)} part."),
        ("Mayor accuracy", f"{NOMBRES_LEGIBLES[mejor_acc]} · {res[mejor_acc]['accuracy'][0]*100:.1f}%"),
        ("Menor latencia", f"{NOMBRES_LEGIBLES[mas_rapido]} · {np.median(res[mas_rapido]['lat']):.1f} ms"),
    ]))

    cuerpo.append("<h2>Resumen de métricas</h2>")
    filas = []
    for n in modelos:
        r = res[n]
        filas.append([inf.nombre_modelo(n), _pct(r["accuracy"]), _pct(r["f1_macro"]),
                      f"{r['precision_macro'][0]*100:.2f}", f"{r['recall_macro'][0]*100:.2f}",
                      f"{r['t_entrenamiento_s'][0]:.2f}",
                      f"{np.median(r['lat']):.2f} / {np.percentile(r['lat'], 95):.2f}",
                      f"{r['n_param']:,}".replace(",", " "),
                      "—" if r["mb"] is None else f"{r['mb']:.2f}"])
    cuerpo.append(inf.tabla(["Modelo", "Accuracy (%)", "F1 macro (%)", "Precisión (%)", "Recall (%)",
                             "Entren./partición (s)", "Latencia med./p95 (ms)", "Parámetros", "Disco (MB)"],
                            filas))
    cuerpo.append('<p class="nota">Media ± desviación estándar entre particiones. '
                  'Parámetros: RF = nodos totales del bosque; MLP = pesos + sesgos; MobileNetV2 = backbone + cabeza.</p>')

    cuerpo.append(inf.barras("Accuracy por modelo (%)",
                             [(n, res[n]["accuracy"][0]*100, res[n]["accuracy"][1]*100, None) for n in modelos],
                             unidad="%", maximo=100, decimales=2, nota="Bigote = ± 1 desviación estándar entre particiones."))
    cuerpo.append(inf.barras("F1 macro por modelo (%)",
                             [(n, res[n]["f1_macro"][0]*100, res[n]["f1_macro"][1]*100, None) for n in modelos],
                             unidad="%", maximo=100, decimales=2))
    cuerpo.append(inf.barras("Latencia de inferencia por muestra (ms, mediana)",
                             [(n, float(np.median(res[n]["lat"])), None,
                               f"p95 = {np.percentile(res[n]['lat'], 95):.2f} ms") for n in modelos],
                             unidad=" ms", decimales=2,
                             nota="RF/MLP: solo el clasificador sobre el vector de 342 features (no incluye MediaPipe). "
                                  f"MobileNetV2: backbone sobre {args.k_frames} frames + cabeza. "
                                  "El costo de extremo a extremo (MediaPipe incluido) se mide en el informe de sesión en vivo."))
    cuerpo.append(inf.barras("Tiempo de entrenamiento por partición (s)",
                             [(n, res[n]["t_entrenamiento_s"][0], res[n]["t_entrenamiento_s"][1], None) for n in modelos],
                             unidad=" s", decimales=2,
                             nota=("MobileNetV2: solo la cabeza; la extracción de embeddings (una vez, "
                                   f"{t_embed.sum()/1000:.1f} s en total) no se repite por partición.")
                             if t_embed is not None else ""))

    cuerpo.append("<h2>Comparación estadística (McNemar exacto, predicciones fuera de fold)</h2>")
    cuerpo.append(inf.tabla(["Modelo A", "Modelo B", "A bien / B mal", "A mal / B bien", "p", "p (Holm)", "¿Diferencia? (α=0.05)"],
                            [[inf.nombre_modelo(a), inf.nombre_modelo(b), nb, nc, f"{p:.4f}", f"{ph:.4f}",
                              "sí" if ph < 0.05 else "no"] for a, b, nb, nc, p, ph in pares]))

    cuerpo.append("<h2>F1 por clase</h2>")
    cuerpo.append(inf.tabla(["Modelo"] + [inf.esc(c) for c in clases],
                            [[inf.nombre_modelo(n)] + [f"{v:.3f}" for v in res[n]["f1_clase"]] for n in modelos]))

    cuerpo.append("<h2>Matrices de confusión (fuera de fold)</h2><div class='cms'>")
    for n in modelos:
        cuerpo.append(inf.matriz_confusion(res[n]["cm"], clases, inf.nombre_modelo(n)))
    cuerpo.append("</div>")

    cuerpo.append("<h2>Resultados por partición</h2>")
    cuerpo.append(inf.tabla(["Modelo", "Partición", "n test", "Accuracy (%)", "F1 macro", "Entren. (s)"],
                            [[inf.nombre_modelo(n), f["particion"], f["n_test"], f"{f['accuracy']*100:.2f}",
                              f"{f['f1_macro']:.3f}", f"{f['t_entrenamiento_s']:.2f}"]
                             for n in modelos for f in res[n]["folds"]]))

    cuerpo.append("<h2>Condiciones experimentales</h2>")
    rs = ds.resumen()
    cuerpo.append(inf.tabla(["Elemento", "Valor"], [
        ["Dataset", inf.esc(f"{rs['directorio']} ({rs['formato']})")],
        ["Muestras por clase", inf.esc(", ".join(f"{k}: {v}" for k, v in rs["por_clase"].items()))],
        ["Sujetos", inf.esc(", ".join(rs["sujetos"]))],
        ["Esquema de validación", inf.esc(f"{args.validacion} — {len(particiones)} particiones, idénticas para los 3 modelos")],
        ["Semilla", args.semilla],
        ["Hilos de CPU (todos los modelos)", args.hilos],
        ["Frames por toma (MobileNetV2)", args.k_frames],
    ]))
    mp_ms = [m.meta.get("t_mediapipe_ms_medio") for m in ds.muestras if m.meta.get("t_mediapipe_ms_medio")]
    if mp_ms:
        cuerpo.append(f'<p class="nota">Costo medio de MediaPipe por frame medido durante la captura: '
                      f'{np.mean(mp_ms):.1f} ms (equipo de captura).</p>')
    cuerpo.append("<h3>Hardware y software</h3>")
    cuerpo.append(inf.entorno_html(entorno))
    cuerpo.append("<h3>Hiperparámetros (fijos)</h3>")
    cuerpo.append(inf.tabla(["Modelo", "Hiperparámetros"],
                            [[inf.nombre_modelo(n), inf.esc(json.dumps(HIPER[n], ensure_ascii=False))] for n in modelos]))
    cuerpo.append('<p class="nota">Archivos: metricas_resumen.csv, metricas_por_particion.csv, f1_por_clase.csv, '
                  'predicciones_oof.csv, mcnemar.csv, entorno.json (separador “;”).</p>')

    html_ = inf.pagina("Comparación de modelos LSC — entrenamiento",
                       f"Generado {fecha:%Y-%m-%d %H:%M} · {inf.esc(entorno['equipo'])} · {inf.esc(entorno['cpu'])}",
                       "".join(cuerpo))
    (dir_inf / "informe.html").write_text(html_, encoding="utf-8")


if __name__ == "__main__":
    main()
