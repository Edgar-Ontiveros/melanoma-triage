# resnet50-224-s0-8da1f44

commit `8da1f44` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.683 | [0.622, 0.734] |
| AUPRC (primaria) | 0.043 | [0.029, 0.074] |
| Sensibilidad a especificidad 0.90 | 0.273 | [0.163, 0.363] |
| Sensibilidad a especificidad 0.95 | 0.148 | [0.078, 0.242] |
| Especificidad a sensibilidad 0.90 | 0.281 | [0.201, 0.374] |
| Especificidad a sensibilidad 0.95 | 0.208 | [0.060, 0.301] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.3418 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1371 | 3504 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.281 |
| VPP a la prevalencia observada (0.0177) | 0.022 |
| VPN a la prevalencia observada | 0.994 |
