# Manifiesto ISIC 2020 — reporte de generación

Generado por `scripts/build_manifest.py` a partir de `data/raw/isic2020/ISIC_2020_Training_GroundTruth_v2.csv` y
`data/raw/isic2020/train`. Salida: `data/manifests/isic2020.csv`.

## Verificación contra la publicación oficial (F1.1)

| cantidad       |   esperado |   obtenido | ok   |
|:---------------|-----------:|-----------:|:-----|
| images         |   33126    |   33126    | True |
| positives      |     584    |     584    | True |
| patients       |    2056    |    2056    | True |
| prevalence_pct |       1.76 |       1.76 | True |

Fuente: DOI 10.34970/2020-ds01; Rotemberg et al., *Sci Data* 8, 34 (2021).

## Archivos

- Imágenes: 33,126
- Tamaño total en disco (originales): 25.76 GB
- Duplicados exactos por SHA256: 433

## Faltantes por columna (F1.2)

| columna          |   faltantes |   pct |
|:-----------------|------------:|------:|
| image_id         |           0 |  0    |
| sha256_original  |           0 |  0    |
| relpath_original |           0 |  0    |
| width            |           0 |  0    |
| height           |           0 |  0    |
| bytes            |           0 |  0    |
| patient_id       |           0 |  0    |
| lesion_id        |           0 |  0    |
| sex              |          65 |  0.2  |
| age_approx       |          68 |  0.21 |
| anatom_site      |         527 |  1.59 |
| target           |           0 |  0    |
| diagnosis        |           0 |  0    |
