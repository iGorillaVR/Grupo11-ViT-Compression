# Resultados de la ejecución E3

Reunimos aquí los archivos que respaldan los experimentos de nuestro notebook [`Grupo11_E3_ViT_CIFAR10_FINAL.ipynb`](../../notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb), ejecutado el 3 de noviembre de 2026. Distinguimos los resultados exportados directamente desde Colab de las tablas reconstruidas a partir de las salidas visibles.

## Evidencia experimental

- [`grupo11_e3_resultados_original.zip`](grupo11_e3_resultados_original.zip): archivo original exportado durante la ejecución. Contiene tablas de los experimentos, teardown por bloque, `run_config.json` y cuatro gráficos: `ablacion_accuracy.png`, `features_intermedias.png`, `campo_influencia_gradiente.png` y `robustez_severidad.png`.
- `resultados_ab.csv`: tabla de FP32, FP16 y ablaciones INT8/INT4 simuladas, transcrita de las salidas del notebook.
- `resultados_cifar10c.csv`: 36 evaluaciones correspondientes a cuatro variantes, tres corrupciones y tres severidades, con 2 000 imágenes por condición. Contrastamos sus cifras de exactitud con el ZIP original y no encontramos discrepancias.
- `diagnostico_int4.csv` y `diagnostico_int4_clases.csv`: resultados de la prueba complementaria realizada **después** de generar el ZIP original. Registramos la exactitud y la distribución de clases predichas.

Las exactitudes de las seis variantes incluidas en el CSV reconstruido de la Fase B coinciden con las medidas del ZIP original. Este último conserva también campos adicionales y las figuras generadas en la ejecución.

## Trazabilidad

Mantenemos [`grupo11_e3_resultados_verificados.zip`](grupo11_e3_resultados_verificados.zip) como respaldo de la reconstrucción anterior de los CSV. Para consultar los archivos generados originalmente en Colab, priorizamos el ZIP `_original.zip`; para el diagnóstico posterior de INT4 utilizamos los CSV independientes.

En el `run_config.json` original quedaron sin actualizar los campos `run_type: E3_pending_confirmation` y `github_commit: completar ...`. Los identificamos como marcadores de la plantilla de exportación y no como evidencia de una ejecución incompleta; el notebook contiene las salidas finales. Conservamos ese archivo sin modificar para preservar su procedencia.

No incluimos en GitHub el checkpoint entrenado ni los datasets de gran tamaño. Reconocemos que las variantes INT8/INT4 constituyen cuantización simulada y que las medidas de rendimiento requieren controles adicionales para establecer conclusiones generales. Nuestra interpretación detallada se encuentra en [`DESIGN_LOG.md`](../../DESIGN_LOG.md).
