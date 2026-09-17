# resnet50-224-s0-bcdbbf9

commit `bcdbbf9` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.785 | [0.739, 0.828] |
| AUPRC (primaria) | 0.085 | [0.045, 0.139] |
| Sensibilidad a especificidad 0.90 | 0.364 | [0.250, 0.464] |
| Sensibilidad a especificidad 0.95 | 0.216 | [0.138, 0.312] |
| Especificidad a sensibilidad 0.90 | 0.468 | [0.344, 0.630] |
| Especificidad a sensibilidad 0.95 | 0.342 | [0.271, 0.483] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.3698 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2282 | 2593 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.468 |
| VPP a la prevalencia observada (0.0177) | 0.030 |
| VPN a la prevalencia observada | 0.997 |
