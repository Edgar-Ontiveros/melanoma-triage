# Datos: ISIC 2020 (conjunto de entrenamiento)

## Qué es

Conjunto de entrenamiento del *SIIM-ISIC Melanoma Classification Challenge 2020*: 33,126
imágenes dermatoscópicas de 2,056 pacientes, con 584 melanomas confirmados por histopatología
(prevalencia 1.76 %). Cada imagen corresponde a una lesión distinta; la curación garantiza una
sola imagen por lesión y varias lesiones por paciente. Las etiquetas benignas provienen de
histopatología, seguimiento clínico, consenso de expertos o imagen de cuerpo completo.

| Cantidad | Publicado | Verificado localmente (2026-09-04) |
|---|---|---|
| Imágenes | 33,126 | 33,126 |
| Registros con `target = 1` | 584 | 584 |
| Pacientes únicos | 2,056 | 2,056 |
| Prevalencia | 1.76 % | 1.76 % |

## Licencia y atribución

Los datos se distribuyen bajo **Creative Commons Attribution-NonCommercial 4.0
International (CC-BY-NC 4.0)**: <https://creativecommons.org/licenses/by-nc/4.0/>.

Atribución requerida, que debe acompañar a cualquier uso de los datos, de modelos entrenados
con ellos y de artefactos derivados (checkpoints, ONNX, API, figuras):

> International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset.
> International Skin Imaging Collaboration. https://doi.org/10.34970/2020-ds01 (2020).
>
> Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Gillis, L., Caffery, B., Chousakos, E.,
> Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B.,
> Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J.,
> Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J. & Soyer, P. A patient-centric
> dataset of images and metadata for identifying melanomas using clinical context.
> *Sci Data* 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z

Las imágenes fueron aportadas por: Hospital Clínic de Barcelona, Medical University of Vienna,
Memorial Sloan Kettering Cancer Center, Melanoma Institute Australia, University of
Queensland y University of Athens Medical School.

**Consecuencia para este proyecto:** cualquier peso entrenado sobre estos datos hereda la
restricción no comercial. El despliegue de la API y cualquier demostración tienen que ser no
comerciales y mantener la atribución anterior.

## Fuente de descarga y procedencia

**Fuente elegida: la distribución oficial del ISIC Challenge**, servida desde el bucket S3 del
organizador (`https://isic-challenge-data.s3.amazonaws.com/2020/`), enlazada desde
<https://challenge.isic-archive.com/data/#2020>.

| Archivo | Bytes | ETag S3 | SHA256 |
|---|---|---|---|
| `ISIC_2020_Training_JPEG.zip` | 24,707,698,022 | `013145107654177e7dd94091fdf27e9c-2946` | tamaño verificado byte a byte; SHA256 por imagen en el manifiesto |
| `ISIC_2020_Training_GroundTruth_v2.csv` | 2,387,418 | `f894624ff25df80143501a2702d4d9fe` | `a1c7c97c59e5f49232dfe3b6f6817efaaee5dbf9bcb4ce584c1889e641be5c4e` |
| `ISIC_2020_Training_GroundTruth.csv` | 2,056,020 | `86657c7009b003543b377b62a8745b4c` | `98c68a10eb669215e3458ad193a24a582c974a4401e539dbbf099b318584264b` |
| `ISIC_2020_Training_Duplicates.csv` | 11,500 | `62992a2252b492095a54dbc5ce0d5b3e` | `ddfd6c2d42430bd3581f3b2116bdb1433462a7109c715a02bf7f51115fcfabce` |

Por qué esta fuente y no la copia de Kaggle:

1. Es la publicación citable (DOI 10.34970/2020-ds01), sin cuenta ni token, con tamaño y
   ETag verificables por HTTP. La copia de Kaggle (`siim-isic-melanoma-classification`,
   carpeta `jpeg/train/`) se derivó de ella para la competencia.
2. La igualdad byte a byte entre ambas copias **no se asume: se verifica**. El manifiesto
   (`data/manifests/isic2020.csv`) guarda el SHA256 de cada archivo original, y
   `notebooks/kaggle_setup.ipynb` compara 200 imágenes de la copia de Kaggle contra ese
   manifiesto y falla ruidosamente si algún hash no coincide. El resultado de esa
   verificación se anota abajo.
3. Kaggle no ofrece un ZIP con hash publicado; descargar de ahí requeriría credenciales y
   tampoco eliminaría la necesidad de verificar.

Se usó `GroundTruth_v2.csv` (octubre de 2020) porque agrega `lesion_id` y corrige valores de
la primera versión; los conteos oficiales coinciden en ambas.

**Resultado de la verificación local vs. Kaggle (200 imágenes):** corrida el **2026-09-04**
en Kaggle con `notebooks/kaggle_setup.ipynb`, competencia `siim-isic-melanoma-classification`
y dataset privado `edgaronti26/melanoma-isic2020-splits`. Salida completa:

```
{'train': 22911, 'val': 4963, 'test': 5252}
manifiesto y splits íntegros
muestra: 200  faltantes: 0  hashes distintos: 0
OK: 200/200 imágenes de Kaggle coinciden byte a byte con el manifiesto
```

Conclusión: la copia de Kaggle y la descarga oficial de ISIC son byte-idénticas en la muestra;
la cadena de procedencia laptop → Kaggle queda verificada.

## Duplicados

ISIC publica `ISIC_2020_Training_Duplicates.csv` con 425 pares de imágenes duplicadas
(850 imágenes), todos del mismo paciente. La deduplicación propia encuentra 433 grupos
byte-idénticos (los 425 oficiales más 8) y, con pHash idéntico, 18 grupos cruzados que se
tratan como una sola unidad al partir. Detalles y justificación del umbral en
`reports/dedup_report.md`.

## Archivos derivados versionados

| Archivo | Contenido |
|---|---|
| `data/manifests/isic2020.csv` | Una fila por imagen: hashes original y redimensionado, rutas, dimensiones, metadatos, etiqueta |
| `data/splits/{train,val,test}.txt` | `image_id` por línea, ordenados; SHA256 en el README y en `data/splits/SHA256SUMS` |
| `reports/*.md` | Manifiesto, deduplicación, splits, redimensionado y EDA |

Las imágenes originales (`data/raw/`) y redimensionadas (`data/processed/`) **no** se
versionan ni se suben a Kaggle: en Kaggle se adjunta la distribución oficial de la
competencia y el dataset privado
[`edgaronti26/melanoma-isic2020-splits`](https://www.kaggle.com/datasets/edgaronti26/melanoma-isic2020-splits)
(manifiesto, tres splits y `SHA256SUMS`, 10 MB; publicado el 2026-09-04).

## Presupuesto de acceso al conjunto de prueba

`data/splits/test.txt` se lee únicamente desde `scripts/evaluate_test.py` (F4), **dos veces
en todo el proyecto**. Cada acceso se registra en `logs/test_set_access.log`. La prueba
`tests/test_test_split_isolation.py` impide que cualquier módulo de `src/` lo referencie.
