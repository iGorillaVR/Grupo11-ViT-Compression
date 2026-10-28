# Resultados exploratorios E2 — Grupo 11

Datos procedentes de la ejecución rápida con Tesla T4 (semilla 42, 3,000 imágenes de entrenamiento, una época, test exploratorio de 1,000 imágenes).

- `resultados_e2.csv`: accuracy Top-1, tamaño de checkpoint serializado, latencia y VRAM para FP32, Fake-INT8 MHSA/MLP y FP16.
- `teardown_por_bloque.csv`: parámetros y FLOPs analíticos aproximados para los 12 encoder blocks.
- `train_history.csv`: validación después de la época.
- `run_config.json`: entorno, tamaño de subconjuntos y parámetros de corrida.

**Precauciones:** resultados exploratorios, no test final de 10,000 imágenes. Fake-INT8 no genera almacenamiento INT8 real. Las figuras PNG originales están en el ZIP recibido por el equipo y aparecen incrustadas en el notebook ejecutado, pero aún no están publicadas como archivos PNG independientes en GitHub. Consultar `DESIGN_LOG.md` para las limitaciones e interpretación de H1.
