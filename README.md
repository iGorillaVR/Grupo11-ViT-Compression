# Compresión con honestidad en Vision Transformers

**Grupo 11 — Deep Learning (1INF52), PUCP, ciclo 2026-2**

**Integrantes:** Mikler Junior Díaz Pérez, Rodrigo Ernesto Roller Cachay y Aaron Manuel Arana Solgorre.

## Planteamiento de la investigación

En este proyecto estudiamos el compromiso entre precisión, tamaño y latencia de un Vision Transformer ViT-B/16 adaptado a CIFAR-10. Nos interesa determinar cómo cambia su comportamiento al reducir la precisión numérica y comparar la sensibilidad de sus componentes de atención multi-cabeza (MHSA) y perceptrón multicapa (MLP). Complementamos el análisis con pruebas de robustez frente a corrupciones de CIFAR-10-C.

Formulamos dos hipótesis de trabajo. En primer lugar, planteamos que MHSA sería más sensible que MLP a la cuantización de igual precisión. En segundo lugar, consideramos que una menor precisión numérica podría incrementar la degradación de exactitud bajo perturbaciones de distribución. Ambas se contrastan con los resultados efectivamente obtenidos, sin asumir que deban confirmarse.

## Diseño experimental

Partimos de `torchvision.models.vit_b_16` con pesos ImageNet-1K V1 y sustituimos su capa de clasificación para reconocer las diez clases de CIFAR-10. Ajustamos los dos últimos bloques del encoder y la cabeza clasificadora.

Organizamos el trabajo en tres fases:

1. **Fase A — Caracterización del modelo.** Contabilizamos parámetros, estimamos operaciones por bloque y examinamos representaciones intermedias y un mapa de influencia basado en gradientes.
2. **Fase B — Intervenciones controladas.** Evaluamos FP32, FP16 real y cuantización **simulada** de pesos a INT8 e INT4, aplicada selectivamente a MHSA y MLP. Nuestra simulación redondea y restaura los pesos a FP32; por tanto, no supone almacenamiento entero ni permite afirmar aceleración o reducción de tamaño por INT8/INT4.
3. **Fase C — Robustez.** Comparamos FP32, FP16 y las variantes simuladas INT8 en tres corrupciones de CIFAR-10-C (`gaussian_noise`, `defocus_blur` y `fog`), con severidades 1, 3 y 5 y 2 000 imágenes por condición.

En la corrida E3 empleamos semilla 42, una GPU Tesla T4, 15 000 imágenes de entrenamiento, 5 000 de validación, 10 000 de prueba, batch de 16, tres épocas y tasa de aprendizaje de `3e-5`. No realizamos un estudio con varias semillas ni entrenamos esta versión con las 45 000 imágenes disponibles después de separar validación.

## Resultados principales de E3

| Variante | Precisión Top-1 en CIFAR-10 |
|---|---:|
| FP32 | 95.60 % |
| FP16 real | 95.61 % |
| INT8 simulado en MHSA | 95.58 % |
| INT8 simulado en MLP | 95.56 % |
| INT4 simulado en MHSA | 91.25 % |
| INT4 simulado en MLP | 12.32 % |

Observamos una pérdida particularmente intensa al simular INT4 en MLP, en contraste con lo esperado en nuestra primera hipótesis. Para examinar este resultado efectuamos una comprobación adicional, posterior a la ablación principal, sobre 2 000 imágenes. La exactitud de INT4 MLP fue 13.35 % en dos repeticiones con la misma configuración; además, el 74.7 % de las predicciones se concentró en una clase. Interpretamos este patrón como indicio de colapso de las predicciones bajo la intervención evaluada, sin atribuirle un mecanismo causal demostrado ni presentarlo como validación con semillas independientes.

En la Fase C, la exactitud media del baseline FP32 para las tres corrupciones seleccionadas disminuyó de 93.18 % en severidad 1 a 67.35 % en severidad 5. Las diferencias entre FP32, FP16 y las dos variantes INT8 simuladas fueron pequeñas dentro del subconjunto analizado. No extrapolamos estas observaciones a todas las corrupciones de CIFAR-10-C.

FP16 redujo el tamaño serializado del checkpoint de aproximadamente 343.29 MB a 171.67 MB. Las mediciones de latencia y memoria corresponden a una sesión y requieren pruebas más aisladas para sostener comparaciones concluyentes de rendimiento.

## Organización y reproducción

- [Notebook E3 ejecutado](notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb): contiene código, métricas y visualizaciones de las tres fases y el diagnóstico complementario.
- [Resultados E3](results/e3/): incluimos el archivo ZIP original de Colab, tablas CSV y los archivos del diagnóstico adicional.
- [Registro de decisiones](DESIGN_LOG.md): consignamos el razonamiento experimental, los resultados negativos y las limitaciones.
- [Datasets y fuentes](datasets/README.md): describimos la procedencia de los datos y su almacenamiento persistente.
- [Resultados exploratorios E2](results/): mantenemos las primeras pruebas como antecedentes, separados de E3.
- [Documentación de referencia](docs/README.md): identificamos la propuesta y las indicaciones del curso.

Para reproducir E3 abrimos el notebook en Google Colab con GPU y ejecutamos sus celdas en orden. El cuaderno monta el Google Drive personal de quien realiza la ejecución, reutiliza los pesos originales y los datos descargados cuando están disponibles y guarda el checkpoint en una ruta dependiente de la configuración. Si dicho checkpoint no existe, se realiza el entrenamiento. Los datasets y pesos de gran tamaño no se versionan en GitHub.

El código fija una semilla, pero esto no garantiza resultados numéricamente idénticos en todos los entornos. Registramos las versiones observadas de PyTorch, TorchVision y el dispositivo en el notebook y en el log; la replicación en otra GPU o versión puede producir diferencias.

## Alcance, antecedentes y referencias

Conservamos los notebooks E2 como registro de las iteraciones anteriores. Sus subconjuntos de entrenamiento y prueba difieren de los empleados en E3, por lo que no comparamos sus cifras como si correspondieran a un mismo experimento controlado.

Utilizamos como referencias de base:

- Dosovitskiy et al. (2021). *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
- Hendrycks y Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*.
- Krizhevsky (2009). *Learning Multiple Layers of Features from Tiny Images*.

Documentamos el uso de herramientas de IA en el [registro de decisiones](DESIGN_LOG.md), de acuerdo con los requisitos de transparencia del proyecto. La interpretación académica, la comprobación de los resultados y su defensa corresponden a los integrantes del grupo.
