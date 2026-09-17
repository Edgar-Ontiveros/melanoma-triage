# resnet50-224-s2-bcdbbf9

commit `bcdbbf9` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.782 | [0.730, 0.830] |
| AUPRC (primaria) | 0.079 | [0.050, 0.138] |
| Sensibilidad a especificidad 0.90 | 0.455 | [0.340, 0.577] |
| Sensibilidad a especificidad 0.95 | 0.284 | [0.170, 0.378] |
| Especificidad a sensibilidad 0.90 | 0.395 | [0.253, 0.610] |
| Especificidad a sensibilidad 0.95 | 0.262 | [0.131, 0.482] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0940 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1925 | 2950 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.395 |
| VPP a la prevalencia observada (0.0177) | 0.026 |
| VPN a la prevalencia observada | 0.996 |
