# resnet50-224-s0-f431f11

commit `f431f11` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.778 | [0.729, 0.822] |
| AUPRC (primaria) | 0.077 | [0.048, 0.130] |
| Sensibilidad a especificidad 0.90 | 0.386 | [0.275, 0.483] |
| Sensibilidad a especificidad 0.95 | 0.239 | [0.145, 0.330] |
| Especificidad a sensibilidad 0.90 | 0.418 | [0.358, 0.552] |
| Especificidad a sensibilidad 0.95 | 0.363 | [0.242, 0.453] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.2469 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2037 | 2838 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.418 |
| VPP a la prevalencia observada (0.0177) | 0.027 |
| VPN a la prevalencia observada | 0.996 |
