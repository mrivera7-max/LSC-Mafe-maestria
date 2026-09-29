# Comparación de modelos: Random Forest vs MLP vs MobileNetV2

Objetivo: entrenar y comparar tres clasificadores de señas LSC bajo **condiciones
experimentales homogéneas de hardware y software**, y poder cambiar de modelo en
la app para compararlos también en uso real.

| Modelo | Entrada | Implementación |
|---|---|---|
| Random Forest | vector de 342 features MediaPipe (mano + boca, media/std/delta de la ventana) | scikit-learn, 200 árboles |
| MLP | el mismo vector de 342 | scikit-learn, capas (128, 64), early stopping |
| MobileNetV2 (TL) | 8 frames RGB de la misma toma | PyTorch, backbone ImageNet congelado → embedding 1280 promediado en el tiempo → cabeza densa (256) |

## Qué garantiza la homogeneidad

- **Mismas muestras**: cada toma de `capturar_dual.py` guarda, de los *mismos frames*, el vector de landmarks y las imágenes.
- **Mismas particiones**: los folds (o LOSO) se generan una vez y se usan para los tres.
- **Misma semilla** (42) para particiones y modelos.
- **Mismo límite de hilos de CPU** (`--hilos`, por defecto 4) para los tres; todo en CPU aunque haya GPU.
- **Mismo equipo**: el informe registra CPU, RAM, SO, Python y versiones de paquetes. Entrena los tres en la misma corrida.
- **En vivo**: los tres comparten MediaPipe, ventana de 20 frames, votación (7/4/0.55) y umbral. Solo cambia la entrada al clasificador.

## 1. Instalar dependencias de la CNN

```bash
# CPU (recomendado: menor tamaño; la comparación se hace en CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

La primera vez que se entrena MobileNetV2 se descargan los pesos ImageNet (~14 MB) → requiere internet. Después, el backbone queda guardado en `data/modelos_comparacion/mobilenetv2/backbone.pt` y la app no necesita internet.

## 2. Capturar el dataset dual

```bash
python models/capturar_dual.py --sujeto S01            # 7 señas × 60 tomas
python models/capturar_dual.py --sujeto S02 --muestras 60
```

- `--sujeto` es obligatorio: habilita la validación inter-sujeto (LOSO).
- ESPACIO graba 0,7 s · N siguiente seña · B borra la última toma · Q sale.
- Solo se guardan frames con mano detectada (igual que la ventana del reconocedor en vivo). Tomas con < 8 frames con mano se descartan.
- Estructura: `data/dual/<seña>/<sujeto>_<idx>/{landmarks.npy, landmarks_frames.npy, frames/*.jpg, meta.json}` (~20–40 MB por sujeto y seña; no va al repositorio).

## 3. Entrenar y evaluar los tres modelos

```bash
python -m comparacion.entrenar_comparacion --datos data/dual                   # K-fold estratificado (5)
python -m comparacion.entrenar_comparacion --datos data/dual --validacion loso # inter-sujeto (≥ 2 sujetos)
python -m comparacion.entrenar_comparacion --datos data/sequences --modelos rf mlp  # dataset viejo, sin CNN
```

Genera `informes/entrenamiento_<fecha>/informe.html` con:
accuracy, F1 macro, precisión y recall (media ± DE), latencia por muestra (mediana/p95), tiempo de entrenamiento,
parámetros, tamaño en disco, F1 por clase, matrices de confusión fuera de fold, **McNemar exacto con corrección de Holm**
y la ficha de condiciones experimentales. Además CSV (`;`) para Excel/R.

Los modelos finales (entrenados con todas las muestras) quedan en `data/modelos_comparacion/<modelo>/`
con un `meta.json` que incluye sus métricas de validación. `--no-guardar-modelos` evalúa sin sobrescribirlos.

## 4. Usar la app en modo comparación

```bash
python app_unificada.py --comparar            # o "usar_comparativo": true en config.json
```

En el panel derecho aparece **Comparación de modelos**:

- **Modelo**: cambia en caliente entre RF, MLP y MobileNetV2 (se reinician ventana y votación para no mezclar).
- **Seña esperada**: márcala antes de ejecutar cada seña. Convierte la sesión en un experimento con verdad de terreno:
  cada intervalo *modelo + seña esperada* es un **ensayo**.
- **Generar informe**: escribe y abre `informes/sesiones/sesion_<fecha>/informe.html`.
  También se genera solo al **detener la cámara** y al cerrar la app (`informe_auto`).

El informe de sesión compara por modelo:
acierto por ensayo, precisión de emisiones, latencia de reconocimiento (s hasta la 1.ª detección correcta),
matriz esperada × detectada, FPS, % de frames con mano, latencia MediaPipe / modelo / total por frame (mediana y p95),
junto a la accuracy offline de cada modelo. Si los tres modelos no se entrenaron en la misma corrida
(dataset, validación o equipo distintos) el informe lo advierte.

### Protocolo sugerido para la sesión en vivo

1. Mismo equipo, misma cámara, misma iluminación y fondo para los tres modelos.
2. Por cada modelo: para cada seña, marcar "seña esperada", ejecutarla N veces (p. ej. 5), pasar a la siguiente.
3. Alternar el orden de los modelos entre sujetos (contrabalanceo) para no favorecer al último por aprendizaje del señante.

## Limitaciones conocidas

- MobileNetV2 se usa como **extractor congelado** (transferencia por extracción de características). El ajuste fino
  (descongelar los últimos bloques) no está implementado: multiplica el costo en CPU por fold.
- MobileNetV2 también corre MediaPipe en vivo (para el criterio de "frame con mano"), así que su costo total incluye ambos.
- La latencia offline de RF/MLP no incluye MediaPipe; la comparación de extremo a extremo es la del informe de sesión.
