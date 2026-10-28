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