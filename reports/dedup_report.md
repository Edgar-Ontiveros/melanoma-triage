# Deduplicación ISIC 2020

Generado por `scripts/dedup_report.py`. Manifiesto: `data/manifests/isic2020.csv` (33,126 imágenes).
pHash 8x8 (64 bits), decodificación reducida a 256 px, umbral de Hamming **0** (`configs/data/isic2020.yaml`, `data.phash.threshold`).
Tiempos: hashing 2.2 min (12 procesos), 548,649,375 pares de distancias en 89 s.

## Pasada 1 — duplicados exactos (SHA256)

- Grupos con archivos byte-idénticos: **433** (intra-paciente 433, cruzados 0)
- ISIC_1188141, ISIC_6351451
- ISIC_2171314, ISIC_5791131
- ISIC_2986014, ISIC_8428997
- ISIC_0544886, ISIC_2127378
- ISIC_5941314, ISIC_8611845
- ISIC_5922139, ISIC_6740634
- ISIC_7365445, ISIC_7791770
- ISIC_2406262, ISIC_9393085
- ISIC_3526268, ISIC_3791309
- ISIC_2748736, ISIC_5481688
- ISIC_7654562, ISIC_8951006
- ISIC_6270846, ISIC_7628257
- ISIC_7644150, ISIC_8010113
- ISIC_3980006, ISIC_9354947
- ISIC_7445800, ISIC_7733258
- ISIC_5892680, ISIC_8755969
- ISIC_2792248, ISIC_9165411
- ISIC_2526675, ISIC_7743266
- ISIC_0277354, ISIC_4436810
- ISIC_2033106, ISIC_8022865
- ISIC_6696536, ISIC_6856100
- ISIC_2850155, ISIC_8546599
- ISIC_4590786, ISIC_9133072
- ISIC_3062039, ISIC_3734497
- ISIC_1350659, ISIC_8194371
- ISIC_3757997, ISIC_6676883
- ISIC_3456673, ISIC_5123837
- ISIC_0604088, ISIC_6346327
- ISIC_1594422, ISIC_2449313
- ISIC_5169650, ISIC_7385195
- … y 403 grupos más (ver `reports/dedup_pairs.csv`)

## Pasada 2 — casi-duplicados (pHash)

- Pares con distancia ≤ 0: **464**
- Grupos (componentes conexas): **452**, que abarcan 909 imágenes
  - Intra-paciente (se conservan, sin riesgo de fuga): **434**
  - Cruzados (dos o más pacientes; F1.4 los fusiona en una sola unidad de split): **18**
- Pares con al menos un melanoma: 3

### Distribución de distancias por pares

![histograma](figures/phash_distance_hist.png)

|   distancia |   pares |   acumulado |
|------------:|--------:|------------:|
|           0 |     464 |         464 |
|           1 |       0 |         464 |
|           2 |    1066 |        1530 |
|           3 |       0 |        1530 |
|           4 |   11228 |       12758 |
|           5 |       0 |       12758 |
|           6 |   57833 |       70591 |
|           7 |       0 |       70591 |
|           8 |  195295 |      265886 |
|           9 |       0 |      265886 |
|          10 |  518227 |      784113 |
|          11 |       0 |      784113 |
|          12 | 1197449 |     1981562 |
|          13 |       0 |     1981562 |
|          14 | 2546735 |     4528297 |
|          15 |       0 |     4528297 |
|          16 | 5100682 |     9628979 |

Total de pares evaluados: 548,649,375. Mediana de la distancia: 30.

### Contraste con la lista oficial `ISIC_2020_Training_Duplicates.csv`

- Pares oficiales: **425** (cruzados según `patient_id`: 0)
- Recuperados por pHash con umbral 0: **425** (100.0 %)
- Distancia pHash de los pares oficiales: mín 0, mediana 0, p90 0, máx 0
- Pares encontrados que NO están en la lista oficial: 39

### Grupos cruzados

- ISIC_1010416, ISIC_3935308
- ISIC_1039396, ISIC_9030907
- ISIC_1496111, ISIC_7912183
- ISIC_2647198, ISIC_7644573
- ISIC_2670475, ISIC_4676156
- ISIC_2690597, ISIC_7411913
- ISIC_2697895, ISIC_4591526, ISIC_5950041, ISIC_6625344
- ISIC_3996990, ISIC_9117456
- ISIC_4007215, ISIC_4026569
- ISIC_4107250, ISIC_6081985, ISIC_7145969, ISIC_8576068
- ISIC_4199101, ISIC_5707761
- ISIC_4382472, ISIC_5234835
- ISIC_4409939, ISIC_8154864, ISIC_9227253
- ISIC_4790284, ISIC_7069853
- ISIC_6260969, ISIC_6304525
- ISIC_6597839, ISIC_7164076
- ISIC_6974005, ISIC_9167290
- ISIC_8085141, ISIC_8692006

### Hojas de contactos

Hasta 20 pares por categoría, ordenados por distancia ascendente.

- Intra-paciente: `reports/figures/dedup_intra.jpg` (20 pares)
- Cruzados: `reports/figures/dedup_cross.jpg` (20 pares)

## Justificación del umbral

Pares hasta la distancia máxima reportada, desglosados. `oficiales` son pares de la lista de ISIC; `cruzados` son pares entre pacientes distintos:

|   distance |   pares |   byte_identicos |   mismo_paciente |   oficiales |   cruzados |   cruzados_acum |   oficiales_acum |
|-----------:|--------:|-----------------:|-----------------:|------------:|-----------:|----------------:|-----------------:|
|          0 |     464 |              433 |              434 |         425 |         30 |              30 |              425 |
|          2 |    1066 |                0 |               12 |           0 |       1054 |            1084 |              425 |
|          4 |   11228 |                0 |               93 |           0 |      11135 |           12219 |              425 |
|          6 |   57833 |                0 |              326 |           0 |      57507 |           69726 |              425 |
|          8 |  195295 |                0 |              960 |           0 |     194335 |          264061 |              425 |
|         10 |  518227 |                0 |             2101 |           0 |     516126 |          780187 |              425 |
|         12 | 1197449 |                0 |             4306 |           0 |    1193143 |         1973330 |              425 |
|         14 | 2546735 |                0 |             7786 |           0 |    2538949 |         4512279 |              425 |
|         16 | 5100682 |                0 |            13019 |           0 |    5087663 |         9599942 |              425 |

Lectura de la tabla con los datos reales (2026-09-04):

- Los 425 pares oficiales están a distancia **0** y además son **byte-idénticos**; la pasada 1 (SHA256) ya los recupera al 100 %.
- No hay ningún hueco en la distribución: a partir de la distancia 2 aparecen cientos de pares cruzados y a distancia 8 cientos de miles, ninguno oficial. Con umbral 8 los grupos abarcaban 13,997 imágenes y 451 grupos cruzados, es decir, casi la mitad del dataset quedaba encadenada en una sola unidad de split.
- Inspección visual de muestras de pares cruzados a distancias 0, 2, 4 y 6: son lesiones distintas (mancha oscura centrada sobre piel clara). pHash resume una miniatura de 32x32 en frecuencias bajas y esa composición es la misma en casi todas las imágenes dermatoscópicas, así que la distancia no separa lesión repetida de lesión parecida.

**Decisión:** `data.phash.threshold = 0`. Solo cuentan como casi-duplicados los pares con pHash idéntico. Los grupos cruzados que quedan a esa distancia son falsos positivos visuales, pero fusionar sus pacientes en una misma unidad de split no cuesta nada y elimina la duda; por eso se conservan como cruzados.

## Punto de verificación

La literatura reporta ~425 duplicados benignos (32,542 benignas, 32,120 sin duplicados). La lista oficial trae 425 pares; SHA256 encuentra 433 grupos byte-idénticos y pHash con umbral 0 encuentra 464 pares y recupera 425 de los oficiales.

## Pasada 3 — embeddings

**No se ejecuta.** Motivos: (1) la curación de ISIC 2020 garantiza una imagen por lesión y `lesion_id` solo se repite en los pares oficiales, que ya están cubiertos; (2) la pasada 2 muestra que en este dataset cualquier medida global de parecido produce miles de pares cruzados entre lesiones distintas, y la similitud coseno de embeddings tendría el mismo problema de precisión sin una verdad de referencia para calibrarla; (3) el costo (~3 h de CPU) no compraría un guardarraíl que los splits agrupados por paciente no den ya.
