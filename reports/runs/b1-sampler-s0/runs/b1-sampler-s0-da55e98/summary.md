# resnet50-224-s0-da55e98

commit `da55e98` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.768 | [0.704, 0.824] |
| AUPRC (primaria) | 0.115 | [0.071, 0.187] |
| Sensibilidad a especificidad 0.90 | 0.534 | [0.407, 0.647] |
| Sensibilidad a especificidad 0.95 | 0.364 | [0.259, 0.467] |
| Especificidad a sensibilidad 0.90 | 0.191 | [0.111, 0.501] |
| Especificidad a sensibilidad 0.95 | 0.113 | [0.026, 0.248] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0002 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 932 | 3943 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.191 |
| VPP a la prevalencia observada (0.0177) | 0.020 |
| VPN a la prevalencia observada | 0.991 |
