# resnet50-224-s0-da55e98

commit `da55e98` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.775 | [0.724, 0.818] |
| AUPRC (primaria) | 0.053 | [0.039, 0.082] |
| Sensibilidad a especificidad 0.90 | 0.330 | [0.221, 0.419] |
| Sensibilidad a especificidad 0.95 | 0.193 | [0.115, 0.282] |
| Especificidad a sensibilidad 0.90 | 0.438 | [0.327, 0.543] |
| Especificidad a sensibilidad 0.95 | 0.333 | [0.227, 0.457] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.3478 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 2137 | 2738 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.438 |
| VPP a la prevalencia observada (0.0177) | 0.028 |
| VPN a la prevalencia observada | 0.996 |
