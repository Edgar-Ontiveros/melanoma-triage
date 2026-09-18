# F2 — Líneas base sobre validación: B0 (metadatos) vs B1 (ResNet50 a 224 px)

Generado por `scripts/f2_report.py`. Todas las métricas son sobre `val.txt` (4,963 imágenes,
88 melanomas, 568 pacientes, prevalencia 1.77 %); intervalos por bootstrap de 2,000
remuestreos **a nivel paciente**. el split de prueba no se tocó. La exactitud no se reporta.

Referencia: un clasificador aleatorio tiene AUC-ROC 0.5 y AUPRC ≈ prevalencia (0.0177).

## Tabla comparativa

| modelo | corrida | AUC-ROC [IC 95 %] | AUPRC [IC 95 %] | sens@spec 0.90 | spec@sens 0.90 | mejor época / corridas | colapso |
|:--|:--|:--|:--|--:|--:|:--|:--|
| **B0** metadatos (regresión logística) | `b0` commit `c1600c0` | 0.620 [0.552, 0.691] | 0.044 [0.028, 0.079] | 0.284 | 0.139 | — | — |
| **B1** ResNet50-224, pos_weight, semilla 0 | `resnet50-224-s0-f431f11` | 0.778 [0.729, 0.822] | 0.077 [0.048, 0.130] | 0.386 | 0.418 | 5 / 11 | no |
| **B1** ResNet50-224, pos_weight, semilla 1 | `resnet50-224-s1-bcdbbf9` | 0.774 [0.723, 0.819] | 0.076 [0.049, 0.128] | 0.364 | 0.460 | 6 / 12 | no |
| **B1** ResNet50-224, pos_weight, semilla 2 | `resnet50-224-s2-bcdbbf9` | 0.782 [0.730, 0.830] | 0.079 [0.050, 0.138] | 0.455 | 0.395 | 7 / 13 | no |
| **B1** ResNet50-224, muestreo ponderado, semilla 0 | `resnet50-224-s0-da55e98` | 0.768 [0.704, 0.824] | 0.115 [0.071, 0.187] | 0.534 | 0.191 | 6 / 12 | no |

## B1: media y rango sobre semillas

- AUC-ROC: 0.778 (rango 0.774–0.782, n=3)
- AUPRC: 0.077 (rango 0.076–0.079, n=3)
- Semillas: [0, 1, 2]

## pos_weight vs. muestreo ponderado (misma semilla)

| estrategia | AUC-ROC [IC] | AUPRC [IC] | colapso |
|:--|:--|:--|:--|
| pos_weight = 54.9 | 0.778 [0.729, 0.822] | 0.077 [0.048, 0.130] | no |
| muestreo ponderado | 0.768 [0.704, 0.824] | 0.115 [0.071, 0.187] | no |

**Recomendación para F3:** muestreo ponderado (mayor AUPRC puntual: 0.115 vs 0.077). Los intervalos se traslapan: la diferencia no es concluyente con una semilla; se elige por AUPRC puntual y simplicidad, y F3 puede revisarlo.

## Lectura

B1 supera a B0 en AUPRC (0.077 vs 0.044) — las imágenes aportan información que los metadatos no tienen.

Detalle de B0 en `reports/b0_metadata.md`; de cada corrida en `reports/runs/<run_name>/summary.md`.
