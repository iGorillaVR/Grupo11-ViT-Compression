# Datos del proyecto Grupo 11 — E3

Este directorio **no contiene los datasets binarios**: CIFAR-10 y CIFAR-10-C son demasiado grandes para GitHub ordinario (CIFAR-10-C.tar ocupa ~2.9 GB y supera el límite de 100 MB por archivo). Los datasets se obtienen de sus fuentes oficiales y se conservan automáticamente en el Google Drive de cada integrante al ejecutar el notebook E3.

## Fuentes
- **CIFAR-10 (Krizhevsky):** https://www.cs.toronto.edu/~kriz/cifar.html . Lo descarga TorchVision desde la fuente original con verificación de integridad propia.
- **CIFAR-10-C (Hendrycks y Dietterich, ICLR 2019):** https://zenodo.org/records/2535967 , DOI: 10.5281/zenodo.2535967 .
- Archivo CIFAR-10-C.tar: https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1
- MD5 original del TAR: `56bf5dcef84df0e2308c6dcbcbbd8499`.

## Descarga y persistencia
Abrir `notebooks/Grupo11_E3_ViT_CIFAR10_FINAL.ipynb` con GPU en Colab. El notebook monta Drive; descarga y almacena CIFAR-10 en `MyDrive/Grupo11_ViT_datasets/cifar10`, y CIFAR-10-C en `MyDrive/Grupo11_ViT_datasets/`. Para ahorrar espacio de extracción, extrae **solo los archivos de las corrupciones elegidas** del TAR, sin alterar los datos originales. Si ya están guardados, no se vuelve a descargar.

**Importante:** ejecutar por primera vez CIFAR-10-C requiere ~2.9 GB para el TAR y espacio adicional para los `.npy` extraídos. Si los archivos no están disponibles o la conexión falla, la Fase C no podrá ejecutarse. No se afirma que estén incluidos como datos binarios en este repositorio.

## Protocolo C inicial
Severidades 1,3,5; corrupciones gaussian_noise, defocus_blur, fog; 2 000 imágenes por combinación; imagen original redimensionada de 32×32 a 224×224 y normalizada como ImageNet. Configuraciones explícitas en el notebook para ampliar a 10 000 imágenes por combinación.

Referencias: Hendrycks, D. & Dietterich, T. (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR; Krizhevsky, A. (2009), *Learning Multiple Layers of Features from Tiny Images*. Respetar las atribuciones y condiciones de las fuentes.
