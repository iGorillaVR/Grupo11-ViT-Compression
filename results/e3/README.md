# Evidencia de la ejecución E3 (3 noviembre 2026)

Los archivos CSV fueron reconstruidos **literalmente de las salidas visibles** del notebook ejecutado publicado en `notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb` (no se alteraron métricas). `resultados_ab.csv`: baseline/ablaciones, incluyendo FP16. `resultados_cifar10c.csv`: 36 evaluaciones, 2 000 imágenes por corrupción/severidad. `diagnostico_int4.csv` y `diagnostico_int4_clases.csv`: prueba adicional posterior. Las figuras están insertadas como outputs en el notebook.

El ZIP local que emitió Colab no está disponible en Drive para copiarlo al repositorio; estas tablas sirven como respaldo revisable de los resultados observados. Los datos completos y checkpoints siguen en Google Drive del ejecutor, no se versionan binarios grandes. No confundir fake-quant con compresión INT8/INT4 real.
