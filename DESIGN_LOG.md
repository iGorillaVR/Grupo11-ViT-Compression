# Registro de decisiones y experimentos — Grupo 11

**Curso:** Deep Learning (1INF52), PUCP, 2026-2  
**Proyecto:** Compresión con honestidad de un Vision Transformer ViT-B/16

Utilizamos este registro para explicar las decisiones que adoptamos durante las etapas E2 y E3, la evidencia que obtuvimos y las limitaciones que identificamos. Conservamos las diferencias entre la exploración inicial y la ejecución final para evitar interpretar resultados de configuraciones distintas como observaciones directamente comparables.

## 1. Decisiones metodológicas

| Código | Decisión | Fundamento y alcance |
|---|---|---|
| D01 | Seleccionamos ViT-B/16 con pesos ImageNet-1K V1 y una salida de diez clases. | Empleamos aprendizaje por transferencia para adaptar el modelo a CIFAR-10. |
| D02 | Redimensionamos las entradas a 224 × 224 y aplicamos la normalización ImageNet. | Mantuvimos compatibilidad con la configuración de los pesos preentrenados. |
| D03 | Ajustamos los últimos dos bloques del encoder y la cabeza clasificadora. | Buscamos limitar el costo de entrenamiento. |
| D04 | Caracterizamos parámetros, FLOPs aproximados, representaciones intermedias, influencia por gradientes, latencia y VRAM. | Establecimos una referencia anterior a las intervenciones de precisión numérica. |
| H1 | Planteamos que MHSA perdería más exactitud que MLP a igual nivel de cuantización. | Formulamos una hipótesis refutable sobre sensibilidad de componentes; la contrastamos en E2 y E3. |
| D05 | Aplicamos cuantización simulada de pesos a 8 y 4 bits de forma selectiva. | Aislamos efectos de redondeo numérico. La intervención no genera pesos enteros empaquetados. |
| D06 | Evaluamos FP16 real en GPU. | Examinamos una reducción efectiva de precisión y tamaño del checkpoint. |
| D07 | Seleccionamos CIFAR-10-C para el estudio de robustez. | Evaluamos alteraciones del dominio de entrada bajo un protocolo delimitado. |
| H2 | Planteamos que las variantes de menor precisión podrían degradarse más que FP32 bajo corrupciones. | Comparamos la pérdida de exactitud en el conjunto de corrupciones seleccionado. |

## 2. Etapa exploratoria E2 — 28 de octubre de 2026

En esta etapa utilizamos una GPU Tesla T4, semilla 42, PyTorch `2.11.0+cu130` y TorchVision `0.26.0+cu130`. Ajustamos el modelo con 3 000 imágenes durante una época y evaluamos 750 imágenes de validación y 1 000 de prueba; utilizamos un batch de 16. Este diseño tuvo un propósito exploratorio, no constituyó una evaluación sobre el test completo.

Registramos 94.1333 % de precisión Top-1 en validación. Nuestro conteo analítico estimó 7 087 872 parámetros y 2 907 909 120 FLOPs aproximados por bloque del encoder (12 bloques), sin incluir todas las operaciones del modelo.

| Variante | Top-1 de prueba E2 | Tamaño checkpoint (MB) | Latencia mediana (ms/imagen) | VRAM pico (MiB) |
|---|---:|---:|---:|---:|
| FP32 | 93.6 % | 343.285519 | 14.4762 | 639.95 |
| INT8 simulado en MHSA | 93.7 % | 343.285519 | 15.0842 | 971.17 |
| INT8 simulado en MLP | 93.6 % | 343.285519 | 13.8164 | 969.87 |
| FP16 real | 93.6 % | 171.672847 | 2.6687 | 374.46 |

En el conjunto reducido de E2, la variación de una décima porcentual equivale a una imagen clasificada correctamente. Por ello, no consideramos esas diferencias suficientes para establecer una mayor sensibilidad de MHSA. Identificamos, además, que la medición de VRAM podía verse afectada por el orden de ejecución y las asignaciones previas de memoria.

Guardamos los datos de esta etapa en `results/resultados_e2.csv`, `results/train_history.csv`, `results/teardown_por_bloque.csv` y `results/run_config.json`. Conservamos el notebook exploratorio `notebooks/Grupo11_E2_ViT_CIFAR10_RAPIDO.ipynb`.

## 3. Ejecución E3 — 3 de noviembre de 2026

### 3.1. Configuración y línea base

Para esta corrida trabajamos con una GPU Tesla T4, semilla 42, PyTorch `2.11.0+cu130`, TorchVision `0.26.0+cu130`, batch de 16 y tasa de aprendizaje de `3e-5`. Empleamos 15 000 imágenes de entrenamiento, 5 000 de validación y las 10 000 imágenes de prueba oficiales de CIFAR-10. Realizamos tres épocas de ajuste de los dos últimos bloques del encoder y de la cabeza clasificadora.

Obtuvimos **95.64 % de exactitud en la mejor validación** y **95.60 % en la prueba limpia**. Conservamos el checkpoint elegido por validación en Drive, separado del checkpoint exploratorio de E2.

### 3.2. Fase A: caracterización

Contabilizamos **85 806 346 parámetros** en el modelo. Para cada bloque del encoder obtuvimos una estimación analítica de **2 907 909 120 FLOPs**; al sumar los doce bloques, calculamos aproximadamente **34.89 GFLOPs**. Estas cifras no equivalen al cómputo exacto de todas las operaciones, pues excluimos determinados componentes y operaciones del modelo.

También examinamos similitudes de representaciones intermedias y un mapa espacial construido mediante gradientes. Consideramos estas visualizaciones herramientas descriptivas: no constituyen mapas causales de atención ni una medición exacta del campo receptivo teórico.

### 3.3. Fase B: intervención y contraste de H1

| Variante | Top-1 de prueba E3 |
|---|---:|
| FP32 | 95.60 % |
| FP16 real | 95.61 % |
| INT8 simulado en MHSA | 95.58 % |
| INT8 simulado en MLP | 95.56 % |
| INT4 simulado en MHSA | 91.25 % |
| INT4 simulado en MLP | 12.32 % |

En INT8 simulado observamos diferencias de apenas centésimas de punto porcentual respecto a FP32. En cambio, al aplicar INT4 simulado sobre MLP, la exactitud descendió a **12.32 %**, mientras que con INT4 simulado sobre MHSA se mantuvo en **91.25 %**. En esta corrida **no respaldamos H1**; el comportamiento observado fue contrario a nuestra expectativa inicial.

Precisamos que INT8 e INT4 corresponden a redondeo y de-cuantización de pesos que permanecen almacenados en FP32. **No obtuvimos compresión entera real** con esas intervenciones y, por tanto, no inferimos beneficios de tamaño o velocidad. Con FP16 sí observamos que el tamaño serializado del checkpoint pasó de aproximadamente **343.29 MB a 171.67 MB**. Las medidas de latencia y VRAM provinieron de la misma sesión de GPU y no permiten una comparación causal plenamente controlada.

### 3.4. Comprobación complementaria del comportamiento INT4

Después de observar el descenso de exactitud de INT4 MLP, diseñamos una comprobación adicional sobre 2 000 imágenes de prueba limpias. Registramos **96.20 %** para FP32, **91.80 %** para INT4 simulado en MHSA y **13.35 %** para INT4 simulado en MLP.

Repetimos dos veces la evaluación de la misma configuración INT4 MLP y obtuvimos **13.35 %** en ambas. Observamos que las predicciones se concentraron en cinco clases y que **74.7 %** de las imágenes se asignaron a una única clase (índice 8). Consideramos esta concentración compatible con un colapso de las predicciones tras la intervención. No demostramos su causa interna y no presentamos la repetición como una validación con semillas o modelos independientes. **Esta prueba fue posterior al resultado principal**, no una hipótesis pre-registrada.

### 3.5. Fase C: robustez y contraste de H2

Para CIFAR-10-C seleccionamos tres corrupciones (`gaussian_noise`, `defocus_blur` y `fog`), tres niveles de severidad (1, 3 y 5) y **2 000 imágenes por condición**. Evaluamos FP32, FP16 y las intervenciones INT8 simuladas en MHSA y MLP; no incluimos las variantes INT4 en esta fase.

Al promediar las tres corrupciones, el baseline FP32 alcanzó **93.18 %**, **84.07 %** y **67.35 %** de exactitud para severidades 1, 3 y 5, respectivamente. Encontramos diferencias reducidas entre las variantes INT8 simuladas, FP16 y FP32 dentro de este subconjunto. En consecuencia, **no obtuvimos evidencia clara a favor de H2** con las condiciones examinadas. Tampoco extendemos esta conclusión a la totalidad de CIFAR-10-C ni a otros niveles de compresión.

## 4. Evidencia y reproducibilidad

Conservamos el notebook E3 ejecutado, sus salidas, el ZIP original de Colab (`results/e3/grupo11_e3_resultados_original.zip`) y las tablas correspondientes en `results/e3/`. El ZIP original incluye los gráficos y la configuración exportada durante la corrida. Los CSV `diagnostico_int4.csv` y `diagnostico_int4_clases.csv` corresponden a la comprobación posterior y, por tanto, no estaban incluidos en ese ZIP.

En el manifiesto original `run_config.json` quedaron campos de plantilla sin actualizar (`E3_pending_confirmation` y un marcador para el commit). Conservamos ese archivo tal como se generó para preservar su procedencia y remitimos al notebook ejecutado como evidencia de que la corrida terminó.

No versionamos pesos, checkpoints o datasets grandes. Los almacenamos en el Drive de la cuenta utilizada para la ejecución. La semilla fija y el registro de versiones facilitan la repetición del procedimiento, aunque no garantizan equivalencia exacta entre entornos diferentes.

## 5. Limitaciones y líneas de trabajo futuras

Reconocemos las siguientes restricciones: utilizamos un único checkpoint y una sola semilla de entrenamiento; ajustamos únicamente parte del modelo con 15 000 imágenes; la cuantización INT8/INT4 fue simulada y no generó modelos de almacenamiento entero; evaluamos solo tres corrupciones de CIFAR-10-C y 2 000 imágenes por condición; y nuestras estimaciones de FLOPs y mediciones de memoria no cubren un análisis integral aislado.

Como extensiones razonables identificamos la cuantización entera real con un backend compatible, la ampliación del número de semillas y corrupciones, y una caracterización de latencia/VRAM mediante ejecuciones independientes. Estas extensiones **no forman parte de los resultados obtenidos** y no las presentamos como experimentos realizados.

## 6. Uso de herramientas de IA

Durante el desarrollo utilizamos ChatGPT como apoyo para interpretar las instrucciones del proyecto, estructurar y revisar código de los notebooks, proponer pruebas de ablación y diagnóstico, analizar las salidas exportadas y editar documentación. También empleamos esta asistencia para organizar los archivos y redactar versiones preliminares de la presente documentación.

Mantuvimos los resultados numéricos que arrojaron las ejecuciones de Colab y distinguimos los datos originales de las tablas reconstruidas. Las decisiones experimentales, la interpretación de los resultados y su defensa académica son responsabilidad de nuestro grupo. Registramos esta asistencia para cumplir los criterios de transparencia del curso.
