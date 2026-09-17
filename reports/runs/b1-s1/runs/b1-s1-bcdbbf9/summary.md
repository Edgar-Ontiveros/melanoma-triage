# resnet50-224-s1-bcdbbf9

commit `bcdbbf9` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.774 | [0.723, 0.819] |
| AUPRC (primaria) | 0.076 | [0.049, 0.128] |
| Sensibilidad a especificidad 0.90 | 0.364 | [0.253, 0.472] |
| Sensibilidad a especificidad 0.95 | 0.261 | [0.175, 0.356] |
| Especificidad a sensibilidad 0.90 | 0.460 | [0.301, 0.562] |
| Especificidad a sensibilidad 0.95 | 0.314 | [0.147, 0.491] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.1680 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2242 | 2633 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.460 |
| VPP a la prevalencia observada (0.0177) | 0.029 |
| VPN a la prevalencia observada | 0.996 |
