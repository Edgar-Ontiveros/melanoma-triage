# F2.6 — Experimento de preprocesamiento: qué pasa cuando la escala de entrada es 255× incorrecta

Generado por `scripts/f2_report.py` a partir de las corridas `prep_a`, `prep_b` y `prep_c`
(`configs/experiment/prep_*.yaml`). Mismo backbone (ResNet50 preentrenado, 224 px), misma
semilla, mismos datos, mismos hiperparámetros; solo cambia `data.preprocess_scale`.

## Qué afirma la v1 de la tesis y qué se puede probar

La versión anterior reporta que EfficientNetB3 con capas congeladas obtuvo 50.75 % de
exactitud y concluye que «las capas congeladas impidieron un aprendizaje adecuado». Congelar
el backbone y entrenar solo la cabeza es una práctica estándar que produce resultados muy por
encima del azar con pesos de ImageNet, así que esa conclusión no explica el número. La
hipótesis alternativa es un desajuste de preprocesamiento.

- **Se puede probar aquí:** que un desajuste de escala de 255× entre lo que el modelo espera y lo
  que recibe produce desempeño de azar (condiciones B y C).
- **No se puede probar sin volver a correr el código original de la v1:** que ese fue exactamente el
  error. Se puede mostrar que es consistente con el resultado y que es la explicación más probable.
- **Es verificable documentalmente:** que en Keras el reescalado de EfficientNet vive dentro del
  grafo del modelo. La documentación oficial (keras.io, *EfficientNet B0 to B7*, nota bajo cada
  función `EfficientNetB0`…`EfficientNetB7`) dice textualmente:

  > Note: each Keras Application expects a specific kind of input preprocessing. For EfficientNet,
  > input preprocessing is included as part of the model (as a `Rescaling` layer), and thus
  > `keras.applications.efficientnet.preprocess_input` is actually a pass-through function.
  > EfficientNet models expect their inputs to be float tensors of pixels with values in the
  > `[0-255]` range.

  y sobre `preprocess_input` (*EfficientNet preprocessing utilities*): «A placeholder method for
  backward compatibility. The preprocessing logic has been included in the efficientnet model
  implementation. Users are no longer required to call this method to normalize the input data.»
  Fuentes: <https://keras.io/api/applications/efficientnet/> (consultado el 2026-09-17).

  Consecuencia: si un pipeline entrega a EfficientNet imágenes ya divididas por 255 (en [0, 1],
  como es habitual con `ImageDataGenerator(rescale=1./255)` o con `preprocess_input` de otras
  familias), el modelo vuelve a reescalar internamente y recibe valores en [0, 1/255]: toda la
  imagen queda comprimida en un rango donde el contraste entre píxeles es ~255 veces menor del
  que vieron los pesos preentrenados.

## Mecanismo

Con normalización `(x/255 − media)/desv`, la entrada a la primera convolución tiene media ≈ 0 y
desviación ≈ 1 por canal. Si la imagen llega 255× más pequeña (condición B), `x/255` es ≈ 0 para
todo píxel y la entrada se vuelve casi la constante `−media/desv`: las activaciones son
prácticamente idénticas para toda imagen, la cabeza solo puede aprender el sesgo y el modelo
colapsa a predecir la prevalencia (AUC ≈ 0.5). Con capas congeladas el efecto es total, porque
el backbone no puede reaprender la escala; con fine-tuning completo puede recuperarse en parte,
y por eso el experimento reporta también la curva por época. Si la imagen llega 255× más grande
(condición C), las activaciones se saturan o explotan (valores del orden de cientos en la
entrada), el gradiente es inestable y la pérdida diverge o el modelo colapsa.

En este repositorio el error se inyecta con `data.preprocess_scale` (`melanoma.data.transforms`):
`divide255` multiplica por 255 el rango de píxel que asume la normalización (la señal útil queda
÷255) y `multiply255` lo divide (señal ×255). La condición A usa el `data_config` de timm sin tocar.

## Resultados

| condición | preprocesamiento | AUC-ROC final [IC] | AUPRC final [IC] | colapso (épocas) | std de prob. última época |
|:--|:--|:--|:--|:--|--:|
| A — correcto | | _pendiente: corrida de Kaggle no copiada a `reports/runs/`_ | | | |
| B — ÷255 de más | | _pendiente: corrida de Kaggle no copiada a `reports/runs/`_ | | | |
| C — ×255 de más | | _pendiente: corrida de Kaggle no copiada a `reports/runs/`_ | | | |

_Las curvas se generan cuando las tres corridas estén en `reports/runs/`._

## Párrafo para la tesis (corrige la afirmación de la v1)

_Se redacta con los números de las tres condiciones cuando estén disponibles. El borrador sin números: la v1 atribuyó el 50.75 % a las capas congeladas; la documentación de Keras muestra que EfficientNet reescala internamente y espera [0, 255]; el experimento A/B/C muestra que un desajuste de 255× basta para producir desempeño de azar; no se afirma que ese fue exactamente el error de la v1, solo que es la explicación más probable y consistente._
