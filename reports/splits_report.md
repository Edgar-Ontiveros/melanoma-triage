# Splits ISIC 2020 — conteos alcanzados

Generado por `scripts/make_splits.py`. Semilla `20260904`, fracciones {'train': 0.7, 'val': 0.15, 'test': 0.15} (`configs/data/isic2020.yaml`).

Unidad de agrupación: `patient_id`. Estratificación a nivel paciente por «tiene al menos un melanoma». Reparto voraz por déficit: el estrato positivo se balancea por número de melanomas y el negativo por número de imágenes.

- Pacientes: 2,056; unidades de split tras fusionar grupos cruzados: 2,033 (18 grupos cruzados, 23 pacientes fusionados)
- Imágenes excluidas del conjunto de prueba por duplicados cruzados: 0 (la fusión de pacientes en unidades hace innecesaria la exclusión)
- Pacientes en más de un split: 0

| split   |   pacientes |   imagenes |   melanomas |   prevalencia_pct |
|:--------|------------:|-----------:|------------:|------------------:|
| train   |         948 |      22911 |         410 |              1.79 |
| val     |         568 |       4963 |          88 |              1.77 |
| test    |         540 |       5252 |          86 |              1.64 |

Restricción: el conjunto de prueba debe tener ≥ 80 melanomas. Obtenidos: **86**.

## SHA256 de los archivos de split

```
701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c  train.txt
b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1  val.txt
7dedecdd0ebf1e8997eb4be12fe71af04475de2bd1b0da5d6fde8440a7742316  test.txt
```
