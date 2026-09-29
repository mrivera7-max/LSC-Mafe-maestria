# LSC Bridge 🤝🤖
## Reconocimiento de Lengua de Señas Colombiana para Unitree G1

Sistema completo para detectar señas LSC desde cámara, traducirlas al español
y enviar los comandos de postura al robot humanoide **Unitree G1** en tiempo real.

---

## 📁 Estructura del proyecto

```
lsc_bridge/
├── main.py                     ← Punto de entrada principal
├── requirements.txt            ← Dependencias
├── config.json                 ← Configuración (auto-generada)
│
├── models/
│   ├── reconocedor.py          ← Motor de reconocimiento LSC (MediaPipe + MLP)
│   ├── entrenar.py             ← Script de entrenamiento del clasificador
│   └── capturar_muestras.py    ← Herramienta para construir el dataset
│
├── robot/
│   └── conector_g1.py          ← Integración SDK Unitree G1
│
├── gui/
│   ├── ventana_principal.py    ← Ventana principal (Tkinter + OpenCV)
│   └── dialogo_config.py       ← Diálogo de configuración
│
├── utils/
│   ├── config.py               ← Gestión de configuración
│   ├── logger.py               ← Sistema de logging con colores
│   └── modo_consola.py         ← Modo sin GUI (solo OpenCV)
│
└── data/
    ├── signs/                  ← Dataset de señas por carpeta
    │   ├── Hola/
    │   │   ├── muestra_0001.jpg
    │   │   └── ...
    │   └── Gracias/
    │       └── ...
    └── lsc_model.pkl           ← Modelo entrenado (generado)
```

---

## 🚀 Instalación rápida

### 1. Clonar y preparar entorno

```bash
# Crear entorno virtual (recomendado)
python -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Activar (Linux/macOS)
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. (Opcional) SDK del Unitree G1

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python
pip install -e .
```

---

## ▶️ Ejecutar en VSCode

Abre el proyecto en VSCode. Las configuraciones de debug ya están listas en `.vscode/launch.json`.

Selecciona con **F5** o el panel de Run & Debug:

| Configuración                   | Descripción                              |
|---------------------------------|------------------------------------------|
| `▶ LSC Bridge — GUI completa`   | Interfaz gráfica completa                |
| `▶ LSC Bridge — Modo consola`   | Solo ventana OpenCV, sin Tkinter         |
| `▶ LSC Bridge — Sin robot`      | Prueba sin conectar el G1                |
| `▶ Capturar muestras LSC`       | Captura imágenes para el dataset         |
| `▶ Entrenar clasificador LSC`   | Entrena el modelo con las muestras       |

---

## 📋 Flujo de trabajo completo

### Paso 1 — Prueba sin robot ni modelo

```bash
python main.py --sin-robot
```

El sistema usará heurísticas geométricas básicas. Suficiente para verificar que la cámara y MediaPipe funcionan.

### Paso 2 — Capturar tus propias muestras

```bash
python models/capturar_muestras.py
```

Controles:
- `ESPACIO` — capturar frame
- `A` — captura automática (3 fps)
- `N` — siguiente seña
- `Q` — salir

### Paso 3 — Entrenar el clasificador

```bash
python models/entrenar.py --datos data/signs --salida data/lsc_model.pkl --modelo mlp
```

Modelos disponibles: `mlp` (recomendado), `rf` (random forest), `svm`.

### Paso 4 — Ejecutar con GUI

```bash
python main.py
```

---

## 🧪 Comparación de modelos (RF / MLP / MobileNetV2)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python models/capturar_dual.py --sujeto S01           # landmarks + frames de la misma toma
python -m comparacion.entrenar_comparacion --datos data/dual   # entrena los 3 + informe HTML/CSV
python app_unificada.py --comparar                    # cambiar de modelo en vivo + informe de sesión
```

Detalles y protocolo: [docs/comparacion_modelos.md](docs/comparacion_modelos.md)

---

## 🤖 Conexión al Unitree G1

El robot debe estar encendido y conectado a la misma red (ethernet o WiFi directo).

IP por defecto: `192.168.123.161:8080`

### Modos de operación

| Modo         | Descripción                                           |
|--------------|-------------------------------------------------------|
| `espejo`     | El G1 replica la postura de la seña detectada        |
| `respuesta`  | El G1 responde con un gesto predefinido              |
| `traduccion` | Solo traduce al español, no mueve el robot           |

### Añadir nuevas señas al G1

Edita `robot/conector_g1.py`, sección `POSTURAS_G1`:

```python
"Tu nueva seña": ComandoArticular(
    hombro_der_pitch=-0.5,   # radianes
    codo_der=1.2,
    duracion=1.0,
    seña_origen="Tu nueva seña",
),
```

Los ángulos siguen el convenio de articulaciones del SDK de Unitree G1.

---

## ⚙️ Configuración

Editable en la GUI (botón ⚙) o directamente en `config.json`:

```json
{
  "camara_idx": 0,
  "umbral_confianza": 0.75,
  "robot_ip": "192.168.123.161",
  "robot_modo": "espejo",
  "robot_velocidad_max": 0.5
}
```

---

## 📚 Referencias

- [INSOR — Instituto Nacional para Sordos](https://www.insor.gov.co)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [Unitree G1 SDK Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Diccionario de LSC — FENASCOL](https://www.fenascol.org.co)

---

## 🐛 Solución de problemas

| Problema | Solución |
|----------|----------|
| `No module named 'mediapipe'` | `pip install mediapipe` |
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'PIL'` | `pip install Pillow` |
| Cámara no abre | Cambia `camara_idx` a 1 o 2 en config |
| Robot no conecta | Verifica IP y que el robot esté en modo control remoto |
| Señas no detectadas | Reduce `umbral_confianza` a 0.60 en config |
