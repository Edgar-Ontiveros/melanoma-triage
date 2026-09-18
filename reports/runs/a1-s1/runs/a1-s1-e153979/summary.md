# tf_efficientnetv2_s-224-s1-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.824 | [0.776, 0.869] |
| AUPRC (primaria) | 0.126 | [0.083, 0.202] |
| Sensibilidad a especificidad 0.90 | 0.568 | [0.450, 0.675] |
| Sensibilidad a especificidad 0.95 | 0.420 | [0.291, 0.522] |
| Especificidad a sensibilidad 0.90 | 0.437 | [0.390, 0.654] |
| Especificidad a sensibilidad 0.95 | 0.392 | [0.251, 0.454] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0059 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2132 | 2743 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.437 |
| VPP a la prevalencia observada (0.0177) | 0.028 |
| VPN a la prevalencia observada | 0.996 |
