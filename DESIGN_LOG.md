# Design Log — Grupo 11

**Fecha del registro documental:** 28/10/2026. **Estado:** documentación basada en archivos aportados; no representa una ejecución nueva.

| ID | Decisión / hipótesis previa | Justificación | Evidencia y estado |
|---|---|---|---|
| D01 | ViT-B/16 preentrenado ImageNet-1K V1; salida de 10 clases | Transfer learning sobre CIFAR-10 | Implementado en el notebook; salida registrada de 85,806,346 parámetros |
| D02 | Redimensionar a 224×224 y normalizar con estadísticas ImageNet | Compatibilidad con la entrada de pesos preentrenados | Código presente |
| D03 | Entrenar últimos 2 bloques encoder y cabeza | Reducir costo de ajuste | Código presente. Validación guardada: época 1 = 95.12%, época 2 = 94.84%. No acreditar tercera época completada |
| D04 | Teardown: parámetros/FLOPs analíticos, activaciones, mapa de influencia, latencia y VRAM | Caracterizar núcleo antes de intervenir | Celdas presentes; sin resultados guardados en la copia recibida. Mapa por gradiente es proxy local, no campo receptivo estructural exacto |
| H1 | **Antes de ablación:** simular INT8 solo sobre MHSA causará mayor caída Top-1 que simularlo solo sobre MLP | Hipótesis refutable sobre sensibilidad de bloques | Código de comparación preparado, sin salidas experimentales guardadas. Pendiente contrastar, incluso si H1 resulta falsa |
| D05 | Ablación de pesos fake-INT8 con restauración a FP32 | Aislar efecto numérico del redondeo | Implementada, **NO es INT8 real, ni prueba reducción de almacenamiento o latencia** |
| D06 | Comparación FP16 con CUDA cuando exista GPU | Medir una intervención real de tipo de datos | Código condicionado a CUDA, sin resultado acreditado en esta versión |
| D07 | CIFAR-10-C para Fase C | Medir robustez ante shift de dominio | Planificado para E3, no ejecutado en el notebook recibido |

## Registro de ejecución y resultados faltantes

- [ ] Identificar responsable, fecha, commit, semilla, versiones, GPU y batch de cada corrida.
- [ ] Confirmar finalización del entrenamiento y evaluación en test limpio.
- [ ] Obtener métricas y figuras de Fase A, aclarando método de estimación FLOPs.
- [ ] Ejecutar comparación controlada FP32 vs fake-INT8 MHSA vs fake-INT8 MLP y contrastar H1.
- [ ] Documentar tamaños serializados, latencia, VRAM con condiciones de medición, sin presentar fake-INT8 como compresión real.
- [ ] Completar FP16 en hardware adecuado, y posteriormente INT8/INT4 reales y CIFAR-10-C si son factibles.
- [ ] Registrar fallos, resultados negativos y cambios de plan.

## Asistencia de IA

Según el registro de transferencia del equipo, se usó ChatGPT para interpretar requisitos, preparar borradores del notebook, documentación y diapositivas, y revisar salidas parciales. **En esta iteración** ChatGPT organizó y redactó README/Design Log y preparó archivos para GitHub; el notebook se preservó sin modificaciones. El equipo debe revisar y validar todo antes de la defensa individual.

## Procedencia y límites

Los valores de validación provienen de salidas **guardadas** en la copia del notebook. No se han ejecutado experimentos adicionales durante esta preparación. La ausencia de output no demuestra que nunca se ejecutó fuera del archivo; solo indica que esta copia no lo acredita.

## Ejecución exploratoria E2 — evidencia de 28/10/2026

**Origen:** archivo `grupo11_e2_resultados.zip` entregado por el equipo; corresponde al notebook ejecutado `notebooks/Grupo11_E2_ViT_CIFAR10_RAPIDO.ipynb`. Esta sección actualiza el estado previo (donde aún no había outputs de Fase A y B); no sustituye el historial original.

**Configuración registrada:** Tesla T4 (CUDA), PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, semilla 42, entrenamiento 3,000 imágenes, 1 época, evaluación de 750 imágenes de validación y 1,000 de test (subconjuntos exploratorios), batch 16, dos últimos bloques y cabeza entrenables. El test oficial completo contiene 10,000 ejemplos, pero NO se evaluó aquí completo.

**Entrenamiento:** Top-1 validación = 94.1333% (1 época). **Teardown:** cada uno de los 12 bloques presenta 7,087,872 parámetros y FLOPs analíticos aproximados de 2,907,909,120; esta estimación excluye operaciones y no debe confundirse con FLOPs exhaustivos de un profiler.

| Variante | Top-1 test exploratorio | Checkpoint MB | Latencia mediana ms/imagen | VRAM pico MiB |
|---|---:|---:|---:|---:|
| FP32 baseline | 93.6% | 343.285519 | 14.4762 | 639.95 |
| Fake-INT8 MHSA | 93.7% | 343.285519 | 15.0842 | 971.17 |
| Fake-INT8 MLP | 93.6% | 343.285519 | 13.8164 | 969.87 |
| FP16 real | 93.6% | 171.672847 | 2.6687 | 374.46 |

**Contraste H1:** esta ejecución **no respalda H1**: el cambio de precisión con Fake-INT8 MHSA es +0.1 puntos porcentuales frente a FP32 y Fake-INT8 MLP registra 0.0 pp. En 1,000 imágenes, 0.1 pp equivale a una clasificación. Se necesita repetir y evaluar sobre test completo; no concluir robustez ni superioridad a partir de la diferencia mínima.

**Alcance y límites:** la comparación Fake-INT8 cuantiza y de-cuantiza pesos, manteniendo almacenamiento FP32; **no** demuestra compresión INT8 real. Las latencias proceden de mediciones breves y deben replicarse antes de sustentar una aceleración general. FP16 sí reduce tamaño serializado en esta corrida. VRAM pico varía según asignaciones de cada experimento; no es una medición aislada de memoria estrictamente comparable sin limpiar/reiniciar el estado. CIFAR-10-C e INT4 siguen pendientes.

**Archivos verificables:** `results/resultados_e2.csv`, `results/train_history.csv`, `results/teardown_por_bloque.csv`, `results/run_config.json`. Los PNG exportados se conservan en el ZIP original recibido, pero aún no están publicados como ficheros independientes en GitHub. El notebook ejecutado contiene las figuras incrustadas.

**Asistencia IA (actualización):** ChatGPT interpretó los resultados del ZIP, transcribió los datos a la carpeta `results/` y redactó esta síntesis; la ejecución del entrenamiento no fue realizada por ChatGPT.
