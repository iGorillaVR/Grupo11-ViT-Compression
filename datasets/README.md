# Fuentes y gestión de datasets

En nuestro estudio utilizamos CIFAR-10 para el ajuste y la evaluación del clasificador, y CIFAR-10-C para analizar su comportamiento ante alteraciones de distribución. No incluimos estos conjuntos de datos como archivos binarios dentro del repositorio debido a su tamaño. En su lugar, documentamos las fuentes originales y automatizamos su obtención y persistencia en Google Drive.

## Procedencia de los datos

- **CIFAR-10:** Krizhevsky (2009). Fuente: https://www.cs.toronto.edu/~kriz/cifar.html. Lo obtenemos mediante TorchVision, que incorpora su propia comprobación de integridad.
- **CIFAR-10-C:** Hendrycks y Dietterich (2019). Registro: https://zenodo.org/records/2535967. DOI: `10.5281/zenodo.2535967`.
- **Archivo de CIFAR-10-C:** https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1.
- **MD5 esperado del archivo TAR:** `56bf5dcef84df0e2308c6dcbcbbd8499`.

## Almacenamiento y reutilización

Al ejecutar `notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb` en Colab, montamos nuestro Google Drive y conservamos CIFAR-10 en `MyDrive/Grupo11_ViT_datasets/cifar10`. Guardamos CIFAR-10-C dentro de `MyDrive/Grupo11_ViT_datasets/` y extraemos únicamente las corrupciones incluidas en el protocolo. Cuando los archivos requeridos ya existen, evitamos repetir las descargas correspondientes.

La primera descarga de CIFAR-10-C requiere espacio para el TAR, de aproximadamente 2.9 GB, y para los archivos `.npy` extraídos. La disponibilidad del conjunto depende del acceso a internet y de la capacidad de almacenamiento de la cuenta utilizada.

## Subconjunto evaluado

En nuestra corrida E3 seleccionamos tres corrupciones: `gaussian_noise`, `defocus_blur` y `fog`. Evaluamos las severidades 1, 3 y 5 con 2 000 imágenes por condición. Redimensionamos las imágenes de 32 × 32 a 224 × 224 y aplicamos la normalización utilizada para los pesos ImageNet. El notebook permite ampliar la cantidad de imágenes a 10 000 por severidad, pero **no afirmamos haber ejecutado ese protocolo ampliado**.

Respetamos la atribución de los conjuntos de datos originales y evitamos redistribuir archivos que no necesitamos incorporar al repositorio.
