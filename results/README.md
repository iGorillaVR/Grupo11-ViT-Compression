# Resultados exploratorios E2

Conservamos en este directorio la evidencia de nuestra ejecución preliminar E2. Utilizamos una GPU Tesla T4, semilla 42, 3 000 imágenes de entrenamiento, una época y un subconjunto de 1 000 imágenes de prueba. Estos valores corresponden exclusivamente a la etapa exploratoria y no deben confundirse con los resultados E3.

- `resultados_e2.csv`: exactitud Top-1, tamaño serializado del checkpoint, latencia y memoria GPU para FP32, INT8 simulado selectivamente en MHSA o MLP, y FP16 real.
- `teardown_por_bloque.csv`: conteos de parámetros y estimaciones analíticas de FLOPs para los 12 bloques del encoder.
- `train_history.csv`: registro de validación tras el entrenamiento.
- `run_config.json`: configuración del entorno y de los subconjuntos evaluados.

En esta fase, la intervención INT8 fue una simulación sobre los pesos y no un modelo con almacenamiento entero real. Las mediciones de latencia y VRAM son observaciones preliminares, sensibles al estado de la GPU. Las visualizaciones se conservan en el notebook E2 ejecutado.

Para las mediciones que utilizamos en el informe final, remitimos al [directorio E3](e3/) y al [registro de decisiones](../DESIGN_LOG.md).
