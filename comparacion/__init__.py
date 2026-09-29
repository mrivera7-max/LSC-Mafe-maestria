"""
Comparación experimental de clasificadores LSC.

Tres modelos bajo las mismas condiciones (mismas muestras, mismos folds,
misma semilla, mismo límite de hilos, mismo equipo):

  rf           Random Forest sobre el vector de 342 features MediaPipe
  mlp          Red neuronal multicapa (128, 64) sobre el mismo vector
  mobilenetv2  MobileNetV2 (ImageNet) por transferencia de aprendizaje sobre
               los frames RGB de la misma toma
"""

MODELOS = ("rf", "mlp", "mobilenetv2")

NOMBRES_LEGIBLES = {
    "rf": "Random Forest",
    "mlp": "MLP (landmarks)",
    "mobilenetv2": "MobileNetV2 (TL)",
}

# Colores fijos por modelo (el color sigue a la entidad, nunca al rango).
COLORES = {
    "rf": "#2a78d6",
    "mlp": "#eb6834",
    "mobilenetv2": "#1baf7a",
}
