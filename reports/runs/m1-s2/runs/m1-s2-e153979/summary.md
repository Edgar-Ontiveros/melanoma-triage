# tf_efficientnetv2_s-384-s2-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.841 | [0.798, 0.879] |
| AUPRC (primaria) | 0.128 | [0.084, 0.203] |
| Sensibilidad a especificidad 0.90 | 0.545 | [0.430, 0.652] |
| Sensibilidad a especificidad 0.95 | 0.420 | [0.315, 0.536] |
| Especificidad a sensibilidad 0.90 | 0.560 | [0.394, 0.661] |
| Especificidad a sensibilidad 0.95 | 0.400 | [0.208, 0.579] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0577 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2732 | 2143 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.560 |
| VPP a la prevalencia observada (0.0177) | 0.036 |
| VPN a la prevalencia observada | 0.997 |
