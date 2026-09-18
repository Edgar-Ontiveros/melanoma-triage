# convnext_tiny-384-s1-e153979

commit `e153979` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.832 | [0.786, 0.874] |
| AUPRC (primaria) | 0.148 | [0.091, 0.232] |
| Sensibilidad a especificidad 0.90 | 0.511 | [0.407, 0.643] |
| Sensibilidad a especificidad 0.95 | 0.409 | [0.289, 0.521] |
| Especificidad a sensibilidad 0.90 | 0.501 | [0.330, 0.633] |
| Especificidad a sensibilidad 0.95 | 0.328 | [0.194, 0.532] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0017 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2444 | 2431 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.501 |
| VPP a la prevalencia observada (0.0177) | 0.032 |
| VPN a la prevalencia observada | 0.997 |
