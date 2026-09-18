# convnext_tiny-384-s2-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.828 | [0.780, 0.872] |
| AUPRC (primaria) | 0.163 | [0.100, 0.249] |
| Sensibilidad a especificidad 0.90 | 0.557 | [0.437, 0.662] |
| Sensibilidad a especificidad 0.95 | 0.386 | [0.280, 0.490] |
| Especificidad a sensibilidad 0.90 | 0.541 | [0.316, 0.662] |
| Especificidad a sensibilidad 0.95 | 0.311 | [0.174, 0.580] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0021 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2637 | 2238 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.541 |
| VPP a la prevalencia observada (0.0177) | 0.035 |
| VPN a la prevalencia observada | 0.997 |
