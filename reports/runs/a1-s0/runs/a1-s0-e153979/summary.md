# tf_efficientnetv2_s-224-s0-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.834 | [0.786, 0.875] |
| AUPRC (primaria) | 0.132 | [0.085, 0.203] |
| Sensibilidad a especificidad 0.90 | 0.614 | [0.500, 0.705] |
| Sensibilidad a especificidad 0.95 | 0.386 | [0.256, 0.500] |
| Especificidad a sensibilidad 0.90 | 0.525 | [0.393, 0.670] |
| Especificidad a sensibilidad 0.95 | 0.399 | [0.069, 0.550] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0699 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2560 | 2315 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.525 |
| VPP a la prevalencia observada (0.0177) | 0.033 |
| VPN a la prevalencia observada | 0.997 |
