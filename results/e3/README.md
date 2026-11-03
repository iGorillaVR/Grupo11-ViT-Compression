# Evidencia de la ejecución E3 (3 noviembre 2026)

Los archivos CSV fueron reconstruidos **literalmente de las salidas visibles** del notebook ejecutado publicado en `notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb` (no se alteraron métricas). `resultados_ab.csv`: baseline/ablaciones, incluyendo FP16. `resultados_cifar10c.csv`: 36 evaluaciones, 2 000 imágenes por corrupción/severidad. `diagnostico_int4.csv` y `diagnostico_int4_clases.csv`: prueba adicional posterior. Las figuras están insertadas como outputs en el notebook.

El ZIP local que emitió Colab no está disponible en Drive para copiarlo al repositorio; estas tablas sirven como respaldo revisable de los resultados observados. Los datos completos y checkpoints siguen en Google Drive del ejecutor, no se versionan binarios grandes. No confundir fake-quant con compresión INT8/INT4 real.

## ZIP reproducido
El archivo [`grupo11_e3_resultados_verificados.zip`](grupo11_e3_resultados_verificados.zip) empaqueta los cuatro CSV publicados y el README que estaba disponible al crear el paquete. **No es el ZIP original generado en Colab** ni incluye PNG o checkpoints.

## Archivo original de Colab (añadido posteriormente)

Se recibió y contrastó el ZIP auténtico de la primera exportación de Colab: [`grupo11_e3_resultados_original.zip`](grupo11_e3_resultados_original.zip) (516 033 bytes). Este **sí** contiene las tablas originales del experimento, el teardown, `run_config.json` y cuatro figuras PNG (`ablacion_accuracy.png`, `features_intermedias.png`, `campo_influencia_gradiente.png`, `robustez_severidad.png`). Las métricas de accuracy de `resultados_e2.csv` coinciden con `resultados_ab.csv` publicado; las 36 condiciones y valores de accuracy de `resultados_cifar10c.csv` coinciden con la tabla CSV reconstruida. El ZIP original añade columnas de caída de precisión que no estaban en la reconstrucción.

**Diferencia temporal:** el ZIP se creó antes de la celda de diagnóstico INT4 incorporada al final del notebook. Por ello, **no incluye** `diagnostico_int4.csv` ni `diagnostico_int4_clases.csv`, que permanecen publicados por separado. El archivo `run_config.json` conserva los campos de plantilla `run_type: E3_pending_confirmation` y `github_commit: completar ...`; son metadatos incompletos de exportación, **no** indican que la ejecución haya quedado pendiente. El notebook ejecutado y el commit son la evidencia de esa ejecución.

Para el informe preferir los CSV originales y gráficos de este ZIP y los dos archivos adicionales de diagnóstico. El ZIP `_verificados.zip` anterior se conserva para mantener trazabilidad, pero ya no es necesario como fuente primaria.
