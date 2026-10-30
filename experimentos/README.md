# Bitácora de experimentos

`bitacora.csv` se crea y crece sola: cada ejecución de
`python -m comparacion.entrenar_comparacion` agrega **una fila por modelo**.
Este archivo SÍ se versiona en Git (los informes HTML de `informes/` no).

## Protocolo de una corrida oficial

1. `git status` sin cambios (si hay cambios sin commit, el informe y la bitácora lo marcan como no reproducible).
2. Dataset congelado con su manifiesto (`manifiesto_dataset.py` del capturador) y su versión declarada:

   ```
   python -m comparacion.entrenar_comparacion --datos data/dual --validacion loso --version-dataset v1 --nota "LOSO 3 sujetos"
   ```

3. `git add experimentos/bitacora.csv` y commit con el resultado.

En la tesis, cada cifra se cita con `id_corrida`, `commit`, `huella_dataset` y `semilla`.

## Columnas

| Columna | Contenido |
|---|---|
| id_corrida | Fecha y hora de la corrida (AAAAMMDD_HHMMSS); igual para los modelos de la misma ejecución |
| fecha | Fecha-hora ISO |
| commit | Primeros 12 caracteres del commit de Git del código |
| codigo_sin_commit | True si había cambios sin commit (resultado no reproducible) |
| version_codigo | `git describe` (etiqueta más cercana, p. ej. `maestria-v1.0`) |
| dataset, version_dataset | Carpeta usada y versión declarada con `--version-dataset` |
| huella_dataset | Primeros 12 caracteres del SHA-256 del dataset (mismo algoritmo que el manifiesto) |
| n_muestras, n_clases, n_sujetos | Tamaño del dataset |
| validacion, particiones | Esquema (kfold, sesion, loso) y número de particiones |
| semilla, hilos, k_frames | Semilla global, hilos de CPU y frames por toma (MobileNetV2) |
| modelo, hiperparametros | Modelo y sus hiperparámetros fijos (JSON) |
| accuracy_media, accuracy_std, f1_macro_media, f1_macro_std, precision_macro_media, recall_macro_media | Métricas fuera de fold |
| latencia_mediana_ms | Latencia mediana de inferencia |
| informe | Carpeta del informe HTML y CSV de la corrida |
| nota | Comentario de `--nota` |
