# tf_efficientnetv2_s-384-s0-d294744

commit `d294744` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.838 | [0.785, 0.886] |
| AUPRC (primaria) | 0.164 | [0.105, 0.247] |
| Sensibilidad a especificidad 0.90 | 0.614 | [0.506, 0.728] |
| Sensibilidad a especificidad 0.95 | 0.466 | [0.350, 0.594] |
| Especificidad a sensibilidad 0.90 | 0.435 | [0.220, 0.706] |
| Especificidad a sensibilidad 0.95 | 0.224 | [0.115, 0.457] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0035 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2121 | 2754 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.435 |
| VPP a la prevalencia observada (0.0177) | 0.028 |
| VPN a la prevalencia observada | 0.996 |
