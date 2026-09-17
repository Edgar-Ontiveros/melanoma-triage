# F2 — Líneas base sobre validación: B0 (metadatos) vs B1 (ResNet50 a 224 px)

Generado por `scripts/f2_report.py`. Todas las métricas son sobre `val.txt` (4,963 imágenes,
88 melanomas, 568 pacientes, prevalencia 1.77 %); intervalos por bootstrap de 2,000
remuestreos **a nivel paciente**. el split de prueba no se tocó. La exactitud no se reporta.

Referencia: un clasificador aleatorio tiene AUC-ROC 0.5 y AUPRC ≈ prevalencia (0.0177).

## Tabla comparativa

| modelo | corrida | AUC-ROC [IC 95 %] | AUPRC [IC 95 %] | sens@spec 0.90 | spec@sens 0.90 | mejor época / corridas | colapso |
|:--|:--|:--|:--|--:|--:|:--|:--|
| **B0** metadatos (regresión logística) | `b0` commit `c1600c0` | 0.620 [0.552, 0.691] | 0.044 [0.028, 0.079] | 0.284 | 0.139 | — | — |
| **B1** ResNet50-224, pos_weight (3 semillas) | _pendiente: corrida de Kaggle no copiada a `reports/runs/`_ | | | | | | |

## B1: media y rango sobre semillas

_pendiente: corrida de Kaggle no copiada a `reports/runs/`_

## pos_weight vs. muestreo ponderado (misma semilla)

_pendiente: corrida de Kaggle no copiada a `reports/runs/`_

## Lectura

B0 es el piso: cualquier modelo de imágenes por debajo de estos números tiene un bug. La comparación con B1 se completa cuando lleguen las corridas de Kaggle.

Detalle de B0 en `reports/b0_metadata.md`; de cada corrida en `reports/runs/<run_name>/summary.md`.
