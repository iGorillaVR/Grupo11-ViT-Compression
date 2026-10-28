# Compresión con Honestidad en Vision Transformers

**Grupo 11 — Deep Learning (1INF52), PUCP, 2026-2**

**Integrantes:** Mikler Junior Díaz Pérez, Rodrigo Ernesto Roller Cachay y Aaron Manuel Arana Solgorre.

## Pregunta de investigación
¿Cómo cambia el compromiso precisión–tamaño–latencia de ViT-B/16 al aplicar FP16/INT8/INT4, qué bloque es más sensible (MHSA frente a MLP), y cómo se afecta su robustez en CIFAR-10-C?

## Estado verificable del E2 (28 de octubre de 2026)

El notebook adapta `torchvision.models.vit_b_16` preentrenado con ImageNet-1K V1 a 10 clases de CIFAR-10 y prepara fine-tuning de los últimos dos bloques y la cabeza clasificadora. En el archivo recibido hay salidas de validación **95.12% (época 1)** y **94.84% (época 2)**. Estas son métricas de **validación**, no del conjunto de test. El entorno registrado fue **CPU**. La copia no contiene salidas comprobables de benchmarking GPU, FLOPs, mapas de activaciones, resultados de fake-INT8, FP16, INT4 o CIFAR-10-C.

**No interpretar las celdas implementadas como experimentos realizados.** Fake-INT8 simula cuantización y restaura los valores a pesos FP32; no demuestra reducción real de bytes ni aceleración.

## Contenido

- `notebooks/Grupo11_E2_ViT_CIFAR10.ipynb`: notebook fuente del grupo, conservado con sus salidas parciales para trazabilidad.
- `DESIGN_LOG.md`: decisiones, hipótesis y estado de evidencia.
- `requirements.txt`: dependencias utilizadas, sin pretender fijar las versiones del entorno original.
- `docs/`: propuesta E1 y documentos de referencia aportados por el equipo.
- `results/`: solo resultados exportados después de ejecutar y comprobar.

## Reproducción (Google Colab con GPU recomendado)

1. Abrir `notebooks/Grupo11_E2_ViT_CIFAR10.ipynb` en Colab y seleccionar GPU en **Entorno de ejecución > Cambiar tipo de entorno de ejecución**. El notebook necesita internet para descargar pesos y CIFAR-10.
2. Ejecutar en orden. La celda de entorno establece semilla 42 y muestra versiones de PyTorch, torchvision y dispositivo; conservar esas salidas y registrar GPU usada.
3. La celda de datos utiliza CIFAR-10: **15,000** imágenes de entrenamiento (submuestra), **5,000** de validación y **10,000** del conjunto oficial de test. Entrada 224×224 con normalización ImageNet.
4. La celda de entrenamiento usa `TRAIN=True`, `EPOCHS=3` y guarda `grupo11_outputs/vit_cifar10_best.pt`. **No cambiar `TRAIN=False`** a menos que el checkpoint entrenado exista previamente en esa ruta.
5. Ejecutar después el teardown, evaluación FP32 en test, perfiles y ablación simulada por bloque. FP16 real se ejecuta solo si hay CUDA.
6. Guardar los CSV/figuras/configuración exportados desde `grupo11_outputs/` en `results/` y actualizar el Design Log con GPU, tiempos, resultados y errores. No subir checkpoints gigantes al repositorio.

Nota: la ejecución desde cero **no se ha comprobado** en esta preparación del repositorio. La instalación del notebook solo añade pandas/matplotlib y asume torch/torchvision compatibles preinstalados. Revisar la compatibilidad en el entorno de ejecución real antes de informar reproducibilidad completa.

## Plan hacia E3

Completar baseline y ablación con la misma configuración y hardware; diferenciar quantización simulada de empaquetado INT8/INT4 real. Evaluar CIFAR-10-C (15 corrupciones × 5 severidades) y reportar resultados negativos. Los tamaños teóricos del póster E1 no son mediciones del archivo serializado.

## Referencias y uso de IA

- Dosovitskiy et al. (2021), *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
- Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*.
- Documentar en `DESIGN_LOG.md` toda asistencia de IA en diseño, código, análisis y documentación.

Los pesos y datasets pertenecen a sus autores y distribuidores originales; comprobar los términos de uso antes de redistribuirlos.