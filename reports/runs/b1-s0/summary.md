<!-- extraído del log del kernel melanoma-f2-train-b1-s0 v3: la API no pudo listar la salida (33 mil archivos copiados bajo /kaggle/working) -->
# resnet50-224-s0-58524d2

commit `58524d2` · splits {'train': '701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c', 'val': 'b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1'}

| métrica | valor | IC 95 % (bootstrap por paciente) |
|:--|--:|:--|
| AUC-ROC | 0.794 | [0.742, 0.842] |
| AUPRC (primaria) | 0.091 | [0.058, 0.151] |
| Sensibilidad a especificidad 0.90 | 0.455 | [0.342, 0.560] |
| Sensibilidad a especificidad 0.95 | 0.330 | [0.227, 0.439] |
| Especificidad a sensibilidad 0.90 | 0.367 | [0.257, 0.595] |
| Especificidad a sensibilidad 0.95 | 0.257 | [0.114, 0.479] |
| Prevalencia (AUPRC de un clasificador aleatorio) | 0.0177 | |

Umbral: 0.0421 (política: sensibilidad>=0.9; la selección definitiva de umbral es de F4).

| | pred. benigno | pred. melanoma |
|:--|--:|--:|
| real benigno | 1787 | 3088 |
| real melanoma | 8 | 80 |

| métrica en el umbral | valor |
|:--|--:|
| Sensibilidad | 0.909 |
| Especificidad | 0.367 |
| VPP a la prevalencia observada (0.0177) | 0.025 |
| VPN a la prevalencia observada | 0.996 |
