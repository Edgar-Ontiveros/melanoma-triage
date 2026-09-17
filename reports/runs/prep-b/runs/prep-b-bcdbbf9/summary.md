# resnet50-224-s0-bcdbbf9

commit `bcdbbf9` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.713 | [0.662, 0.764] |
| AUPRC (primaria) | 0.040 | [0.028, 0.064] |
| Sensibilidad a especificidad 0.90 | 0.295 | [0.195, 0.396] |
| Sensibilidad a especificidad 0.95 | 0.170 | [0.085, 0.275] |
| Especificidad a sensibilidad 0.90 | 0.381 | [0.337, 0.484] |
| Especificidad a sensibilidad 0.95 | 0.359 | [0.195, 0.427] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.5237 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1858 | 3017 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.381 |
| VPP a la prevalencia observada (0.0177) | 0.026 |
| VPN a la prevalencia observada | 0.996 |
