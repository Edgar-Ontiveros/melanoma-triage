# tf_efficientnetv2_s-384-s1-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.810 | [0.755, 0.857] |
| AUPRC (primaria) | 0.133 | [0.084, 0.207] |
| Sensibilidad a especificidad 0.90 | 0.523 | [0.405, 0.628] |
| Sensibilidad a especificidad 0.95 | 0.432 | [0.312, 0.541] |
| Especificidad a sensibilidad 0.90 | 0.365 | [0.238, 0.635] |
| Especificidad a sensibilidad 0.95 | 0.243 | [0.129, 0.420] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0019 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1781 | 3094 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.365 |
| VPP a la prevalencia observada (0.0177) | 0.025 |
| VPN a la prevalencia observada | 0.996 |
