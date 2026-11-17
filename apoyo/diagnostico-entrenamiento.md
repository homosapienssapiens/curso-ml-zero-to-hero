# 🩺 Diagnóstico del entrenamiento

**Síntoma → causa probable → qué probar → dónde se explica.** Reúne las tablas "Errores comunes y señales de alerta" de todo el curso. Las dos lecciones base son la [3.7 — Curvas de aprendizaje](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) (ML clásico y cualquier modelo entrenado por pasos) y la [9.9 — Recetario de diagnóstico](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb) (redes neuronales).

Material de apoyo: [Glosario](glosario.md) · [Chuleta de hiperparámetros](chuleta-hiperparametros.md) · [¿Qué modelo uso?](arbol-de-decision-modelos.md) · [Recursos](recursos.md) · [Volver al README](../README.md)

## Índice

0. [Antes de empezar: tres reglas](#0-antes-de-empezar-tres-reglas)
1. [Leer las curvas de aprendizaje](#1-leer-las-curvas-de-aprendizaje)
2. [Sobreajuste, subajuste y resultados "demasiado buenos"](#2-sobreajuste-subajuste-y-resultados-demasiado-buenos)
3. [Datos y preparación](#3-datos-y-preparación)
4. [Evaluación y métricas](#4-evaluación-y-métricas)
5. [Modelos clásicos](#5-modelos-clásicos)
6. [Ensambles y ajuste de hiperparámetros](#6-ensambles-y-ajuste-de-hiperparámetros)
7. [No supervisado](#7-no-supervisado)
8. [Redes neuronales: el recetario](#8-redes-neuronales-el-recetario)
9. [Visión, secuencias y Transformers](#9-visión-secuencias-y-transformers)
10. [Generativos y LLMs](#10-generativos-y-llms)
11. [Aprendizaje por refuerzo](#11-aprendizaje-por-refuerzo)
12. [Producción](#12-producción)

---

## 0. Antes de empezar: tres reglas

1. **Síntoma no es causa.** "La pérdida no baja" tiene al menos cinco causas distintas. Observa los signos vitales antes de recetar ([9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb)).
2. **Cambia una sola cosa a la vez** y compara contra una línea base ([4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb)). Repite con 3–5 semillas antes de creer diferencias pequeñas ([3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb), [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb)).
3. **Dibuja siempre las dos curvas** (entrenamiento y validación), con la pérdida en escala logarítmica ([3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb)).

### ✅ Checklist antes de entrenar en serio (redes) — [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb)

1. Mira unos cuantos **ejemplos con su etiqueta**. ¿Coinciden?
2. ¿Las entradas están **normalizadas**? (media ≈ 0 y desviación ≈ 1, o al menos entre 0 y 1).
3. ¿La pérdida del primer lote es ≈ **ln(n_clases)**? (2 clases ≈ 0.69 · 10 clases ≈ 2.30 · 100 clases ≈ 4.61).
4. ¿La red **sobreajusta un solo lote** de 64 ejemplos? Si no puede, hay un error de código, datos o pérdida.
5. ¿La red devuelve **logits** y usas la pérdida de PyTorch adecuada?
6. ¿El bucle tiene `zero_grad()` → `backward()` → `step()`, y `train()`/`eval()` en su sitio?
7. Empieza con **Adam, lr = 1e-3** y una red pequeña. Complica solo cuando lo simple funcione.

---

## 1. Leer las curvas de aprendizaje

Vale para cualquier modelo entrenado por pasos: regresión con descenso de gradiente, boosting, redes.

| Lo que ves en la curva | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Ambas bajan con codo y se aplanan juntas, cerca | ✅ Sano | Parar cuando se aplane la validación | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) |
| Sube y llega a `inf` / `nan` | Tasa de aprendizaje **muy alta** | Dividir la tasa entre 3–10 | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb), [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Baja en zigzag, sube y baja sin asentarse | Tasa **algo alta** o lote muy pequeño | Bajar la tasa; subir el lote | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Baja casi en línea recta y sigue bajando al terminar | Tasa **baja** o **pocas épocas** | Subir la tasa (barrido) o entrenar más | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Plana desde el principio | Tasa diminuta, datos sin escalar o error de signo en el código | Barrido de tasa; escalar; revisar `w = w − tasa × gradiente` | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb), [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Plana mucho tiempo y luego cae de golpe | **Meseta** o punto de silla | Paciencia, tasa algo mayor, momento/Adam | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb) |
| Se estanca en valores distintos según la semilla | **Mínimo local** | Varios inicios, mini-batch, momento | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb) |
| Nunca se estabiliza al final | Lote pequeño con tasa fija | Decaimiento de la tasa o subir el lote | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| Patrón de escalones o sierra que se repite cada época | Datos ordenados **sin barajar** | `shuffle=True` | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| Cambiaste el lote y dejó de funcionar | La tasa buena depende del lote | Repetir el barrido de tasa | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Entrenamiento baja, validación baja y luego **sube** | **Sobreajuste** | Early stopping; más datos; regularización; modelo más simple | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb), [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Ambas planas y **altas**, juntas | **Subajuste** | Modelo más flexible; mejores características; menos regularización | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb), [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| Tiembla mucho | Ruido de lotes chicos | Suavizar con promedio móvil para leerla; subir el lote | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) |
| Validación **siempre mejor** que entrenamiento | Fuga o datos repetidos entre conjuntos, o dropout/aumento activos solo al entrenar | Revisar la división ([2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb)); medir ambas en modo `eval()` | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Curva error vs. **cantidad de datos**: la brecha se cierra al añadir datos | Varianza: más datos **sí** ayudarán | Conseguir más datos | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb), [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb) |
| Curva vs. datos: ambas altas y planas | Sesgo: más datos **no** ayudarán | Modelo más flexible o mejores características | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb), [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb) |

---

## 2. Sobreajuste, subajuste y resultados "demasiado buenos"

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Error de entrenamiento muy bajo y validación mucho peor | **Sobreajuste** (varianza alta) | Más datos · modelo más simple (bajar `max_depth`, subir k…) · regularización · early stopping · quitar columnas inútiles | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb), [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb), [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Error alto en entrenamiento **y** validación, parecidos | **Subajuste** (sesgo alto) | Modelo más flexible · mejores características · menos regularización · entrenar más | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb), [2.4](../nivel-1-cero/m02-datos/2.4-ingenieria-de-caracteristicas.ipynb) |
| Validación mejor que entrenamiento, o ~100 % en un problema difícil | **Fuga de datos**, validación pequeña o fácil | Buscar columnas que "contienen" la respuesta; revisar la división | [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb), [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| "Si consigo más datos, mejorará" | Solo es cierto con varianza alta | Mira la curva vs. datos antes de pagar | [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb), [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) |
| Persigues error 0 en validación | Hay **ruido irreducible**: es un piso | Acepta el piso; con más ruido, modelos más simples | [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb) |
| Agregaste columnas y el modelo empeoró | Maldición de la dimensionalidad | Quitar columnas inútiles; seleccionar dentro del pipeline | [2.8](../nivel-1-cero/m02-datos/2.8-maldicion-de-la-dimensionalidad.ipynb) |

---

## 3. Datos y preparación

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Un "promedio" sin sentido (código postal, ID) | Categoría guardada como número | Convertirla en categoría | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb) |
| Números guardados como texto (`"1,200"`, `"$500"`) | Comas o símbolos | Limpiar y convertir | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb), [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| `ValueError: Input contains NaN` | Datos faltantes | `SimpleImputer` (dentro del pipeline) | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| `dropna()` te borró medio conjunto | Huecos repartidos en muchas columnas | Imputar en lugar de borrar | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| Más categorías de las que existen | Mayúsculas, espacios, sinónimos | `.str.strip().str.lower()` y un diccionario de equivalencias | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| Borraste "atípicos" y el modelo ya no detecta fraudes | Eran casos raros **reales** | Borrar solo errores | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| KNN, SVM, K-means o PCA funcionan muy mal | Columnas **sin escalar** | `StandardScaler` dentro de un pipeline | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb), [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb), [8.1](../nivel-3-practicante/m08-no-supervisado/8.1-estructura-sin-etiquetas.ipynb) |
| El modelo cree que "Oro < Plata" | `OrdinalEncoder` con orden alfabético | Dar el orden con `categories=` | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| El sistema se cae con una categoría nueva | `handle_unknown="error"` | `handle_unknown="ignore"` | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| Miles de columnas tras el one-hot | Columna con muchísimas categorías | Agrupar raras, `TargetEncoder`, o preguntarte si sirve | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb), [2.8](../nivel-1-cero/m02-datos/2.8-maldicion-de-la-dimensionalidad.ipynb) |
| Una característica nueva hace al modelo "casi perfecto" | Contiene la respuesta (fuga) | ¿La tendrías **antes** de saber el resultado? | [2.4](../nivel-1-cero/m02-datos/2.4-ingenieria-de-caracteristicas.ipynb), [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb) |
| Resultados "demasiado buenos" tras escalar, imputar, seleccionar o re-muestrear | El paso se ajustó con **todos** los datos (fuga) | Meter todos los pasos que aprenden en un `Pipeline` | [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb), [7.5](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.5-pipelines-sin-fugas.ipynb) |
| El modelo acierta "demasiado" | Duplicados repartidos entre entrenamiento y prueba | Quitar duplicados **antes** de dividir | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb), [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb) |
| Resultados espectaculares con datos en el tiempo | Se barajaron datos temporales | División temporal / `TimeSeriesSplit` | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb), [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb), [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb) |
| Exactitud altísima con varias filas por persona/tienda | Fuga por **grupos** | `GroupKFold` con `groups=` | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb), [7.5](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.5-pipelines-sin-fugas.ipynb) |
| La prueba no tiene casos de la clase rara | División sin estratificar | `stratify=y` | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb), [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| Se tiró el 95 % de los datos al equilibrar | Submuestreo con pocos datos | Pesos de clase o mover el umbral | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |

---

## 4. Evaluación y métricas

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Métricas espectaculares (R² ≈ 1, MAE ≈ 0) | Mediste en entrenamiento, o hay fuga | Medir en datos no vistos | [4.1](../nivel-2-aprendiz/m04-evaluacion/4.1-metricas-de-regresion.ipynb) |
| R² negativo en prueba | Peor que predecir el promedio | Revisar sobreajuste; comparar con líneas base | [4.1](../nivel-2-aprendiz/m04-evaluacion/4.1-metricas-de-regresion.ipynb), [4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb) |
| RMSE mucho mayor que el MAE | Unos pocos errores enormes | Mirar esos casos: ¿atípicos reales o datos sucios? | [4.1](../nivel-2-aprendiz/m04-evaluacion/4.1-metricas-de-regresion.ipynb) |
| MAPE gigantesco | Valores reales cerca de 0 | MAE o RMSE | [4.1](../nivel-2-aprendiz/m04-evaluacion/4.1-metricas-de-regresion.ipynb) |
| 98 % de exactitud con una clase rara | Paradoja de la exactitud | Comparar con "siempre la mayoritaria"; mirar precisión y recall | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb), [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| Recall altísimo y nadie lo cuestiona | El modelo dice "sí" a casi todo | Reportar siempre la pareja precisión + recall | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| `weighted avg` alto, pero una clase rara va mal | El promedio ponderado esconde clases pequeñas | Mirar `macro avg` y el reporte por clase | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| AUC por debajo de 0.5 | Etiquetas o columna de `predict_proba` invertidas | Usar `[:, 1]` para la clase 1 | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb) |
| AUC 0.95 pero "demasiadas alarmas" | Positivos raros: la ROC se ve optimista | Curva precisión-recall y precisión al umbral elegido | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb) |
| El modelo casi nunca dice "sí" | Umbral 0.5 con datos desbalanceados | Bajar el umbral según el costo | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb), [4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb) |
| "70 %" no acierta 7 de cada 10 | Probabilidades sin calibrar | Curva de calibración; `CalibratedClassifierCV` | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb), [4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb) |
| Mejor exactitud y el negocio pierde dinero | La métrica no refleja el costo de los errores | Matriz de costos y costo esperado | [4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb) |
| Umbral, early stopping o hiperparámetros elegidos mirando la prueba | Fuga: el resultado final es optimista | Decidir con validación; la prueba, **una vez** al final | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb), [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb), [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| El modelo le gana a la línea base por 0.5 puntos | Mejora dentro del ruido | Mirar la variación entre pliegues | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb), [4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb) |
| "Mejora un 70 %" contra el promedio, en una serie con ciclos | Línea base demasiado débil | Estacional ingenua o la regla actual del negocio | [4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb) |

---

## 5. Modelos clásicos

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Los residuos forman una curva o un embudo | La relación no es una recta | Transformar características u objetivo; modelo no lineal | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| Coeficientes enormes con signos raros | Multicolinealidad | Quitar una de las columnas o usar `Ridge` | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| "Esta variable es la más importante porque su coeficiente es el mayor" | Unidades distintas | Estandarizar antes de comparar | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb), [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| `ConvergenceWarning` | Datos sin escalar o `max_iter` bajo | Escalar en un pipeline; subir `max_iter` | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| Probabilidades de exactamente 0 o 1 | Clases separables o `C` muy grande | Bajar `C` | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| KNN: 100 % en entrenamiento y mucho menos en prueba | k muy pequeño | Subir k con validación cruzada | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb) |
| KNN o árbol predicen siempre la clase mayoritaria | k grande o clases desbalanceadas | Bajar k; `class_weight="balanced"`; métricas por clase | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb), [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| KNN lentísimo en producción | Compara con todos los datos | Menos datos, índices aproximados u otro modelo | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb) |
| Árbol: entrenamiento ≈ 100 % y prueba mucho menor | Árbol sin límites | Bajar `max_depth`, subir `min_samples_leaf`, `ccp_alpha` | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| El árbol cambia por completo con datos casi iguales | Inestabilidad natural | Random Forest | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb), [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Un "ID de cliente" sale como la variable más importante | Memoriza identificadores (posible fuga) | Quitar identificadores | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb), [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Predicciones planas fuera del rango visto | Árboles y KNN **no extrapolan** | Modelo lineal, o predecir el cambio | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb), [6.6](../nivel-3-practicante/m06-ensambles/6.6-por-que-los-arboles-dominan-lo-tabular.ipynb) |
| SVM: horas de entrenamiento | `SVC` escala mal con muchas filas | `LinearSVC`, submuestrear u otro modelo (> ~50,000 filas) | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| SVM: `predict_proba` da error | No calcula probabilidades por defecto | `decision_function` o `CalibratedClassifierCV` | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| Naive Bayes: una palabra rara manda la probabilidad a 0 | `alpha = 0` | `alpha ≥ 0.1` | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |
| `ValueError: Negative values in data` | `MultinomialNB` con datos escalados | No escalar los conteos; `GaussianNB` para continuos | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |
| Lasso elimina una variable que sabes importante | Variables sin escalar | `StandardScaler` antes del modelo | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| Subiste `C` "para regularizar más" y sobreajusta más | Confundir `C` con `alpha` | `C` grande = **menos** regularización | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |

---

## 6. Ensambles y ajuste de hiperparámetros

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| El ensamble rinde igual que su mejor miembro | Poca diversidad | Datos, columnas o familias de modelos distintas | [6.1](../nivel-3-practicante/m06-ensambles/6.1-por-que-combinar-modelos.ipynb), [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| El ensamble rinde **peor** que su mejor miembro | Miembros malos votando igual | Quitar al débil, pesos, voto suave o stacking | [6.1](../nivel-3-practicante/m06-ensambles/6.1-por-que-combinar-modelos.ipynb), [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| "Mi bosque tiene 100 % en entrenamiento: ¡sobreajuste!" | Es normal con árboles profundos | Juzgar con OOB o validación cruzada | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Bajas `n_estimators` del bosque "para no sobreajustar" | Mito: más árboles no sobreajustan | 200–500 y ajustar `min_samples_leaf`/`max_features` | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb), [7.4](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.4-priorizar-hiperparametros.ipynb) |
| Boosting: la validación empeora al sumar árboles | Demasiados árboles para ese `learning_rate` | Parada temprana; bajar `learning_rate` | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb) |
| Boosting rinde mal con etiquetas ruidosas | Insiste en los errores de etiqueta | Limpiar datos, `subsample` < 1, pérdida robusta | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb) |
| "XGBoost me dio peor que Random Forest" | Valores por defecto y sin parada temprana | Parada temprana + ajustar `learning_rate`, profundidad, `min_child_*` | [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| LightGBM sobreajusta con pocos miles de filas | Crecer por hojas genera ramas profundas | Bajar `num_leaves`, subir `min_child_samples` | [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Validación excelente y prueba decepcionante con boosting | La prueba se usó como `eval_set` de la parada temprana | Validación sacada del entrenamiento | [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb), [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Stacking perfecto en entrenamiento, malo en prueba | Meta-modelo entrenado con predicciones "vistas" | Predicciones fuera de pliegue (`StackingClassifier`) | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| El modelo rinde peor en prueba que su `best_score_` | Optimismo por selección | Reportar la prueba; validación cruzada anidada | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb), [7.5](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.5-pipelines-sin-fugas.ipynb) |
| El mejor valor está en el borde del rango | Rango demasiado estrecho | Ampliar el rango en esa dirección | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| "Probé `C` = 1, 2, 3, 4, 5 y todos dan igual" | Escala lineal para una perilla logarítmica | `loguniform` o 0.01, 0.1, 1, 10… | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb), [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| La búsqueda tarda horas | Cuadrícula con muchas perillas | Búsqueda aleatoria o *successive halving*; 2–3 perillas críticas | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb), [7.4](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.4-priorizar-hiperparametros.ipynb) |
| Un mes ajustando por 0.002 de AUC | Rendimientos decrecientes | Volver a los datos: nuevas variables | [6.6](../nivel-3-practicante/m06-ensambles/6.6-por-que-los-arboles-dominan-lo-tabular.ipynb), [7.4](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.4-priorizar-hiperparametros.ipynb) |
| `Invalid parameter` en la búsqueda | Nombre mal escrito en el pipeline | `pipe.get_params().keys()` | [7.5](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.5-pipelines-sin-fugas.ipynb) |
| "La variable X es la más importante, así que causa el resultado" | Importancia ≠ causalidad | Hablar de "el modelo se apoya en X" | [7.6](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.6-interpretabilidad.ipynb) |
| `feature_importances_` señala una columna tipo ID | Sesgo de la importancia por impureza | Importancia por permutación sobre datos no vistos | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb), [7.6](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.6-interpretabilidad.ipynb) |

---

## 7. No supervisado

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Los grupos se separan solo por una columna | Columnas sin escalar | `StandardScaler` antes de agrupar | [8.1](../nivel-3-practicante/m08-no-supervisado/8.1-estructura-sin-etiquetas.ipynb), [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| Los grupos cambian por completo al repetir | Grupos inestables o inexistentes; sin semilla | Fijar semilla, subir `n_init`, probar submuestras | [8.1](../nivel-3-practicante/m08-no-supervisado/8.1-estructura-sin-etiquetas.ipynb), [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| "Encontramos 7 segmentos" y nadie sabe qué hacer | Grupos sin utilidad de negocio | Perfilar cada grupo; si no cambia una decisión, sobran | [8.1](../nivel-3-practicante/m08-no-supervisado/8.1-estructura-sin-etiquetas.ipynb) |
| Elegiste k = 10 porque la inercia era la más baja | La inercia **siempre** baja al subir k | Codo + silueta + negocio | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| K-means corta formas curvas por la mitad | Supone grupos redondos y parecidos | DBSCAN o `GaussianMixture` | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb), [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| DBSCAN marca todo como ruido / un solo grupo | `eps` demasiado pequeño / demasiado grande | Gráfica de k-distancias | [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| DBSCAN no sirve con densidades distintas | Un solo `eps` no alcanza | HDBSCAN | [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| PC1 explica el 99 % y es una sola columna | Datos sin escalar | `StandardScaler` antes de PCA | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| El modelo empeora mucho tras PCA | Se descartó una dirección poco variable pero predictiva | Más componentes o sin PCA | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| Lees tamaños o distancias entre islas de t-SNE | t-SNE deforma tamaños y distancias globales | Medir en los datos originales; varias perplejidades | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| Las alarmas de anomalías son siempre "los más grandes" | Una columna enorme domina | Escalar, logaritmos, normalizar por tamaño | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| Demasiadas alarmas para el equipo | `contamination` alto | Fijarla según la capacidad de revisión | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| El recomendador solo muestra lo popular | Sesgo de popularidad / burbuja de filtro | Diversidad, exploración, penalizar lo popular | [8.7](../nivel-3-practicante/m08-no-supervisado/8.7-sistemas-de-recomendacion.ipynb) |

---

## 8. Redes neuronales: el recetario

La tabla completa de la [Lección 9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), seguida de los errores de código más frecuentes del [Módulo 9](../nivel-4-avanzado/m09-redes-neuronales/).

| Síntoma | Huella en las curvas | Causa probable | Qué probar primero | Lección |
|---|---|---|---|---|
| **No baja** | Plana en ≈ ln(n_clases) desde el principio | lr demasiado baja · gradientes que desaparecen · red "muerta" tras un lr alto | Subir lr ×10 · ReLU + He · revisar inicialización | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| **No baja** | Primer lote ≫ ln(n_clases) | Entradas sin normalizar o pesos iniciales enormes | Escalar entradas; He/Xavier | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| **No baja** | Salta a valores enormes y luego queda plana | lr demasiado alta | Bajar lr ×10, *LR range test*, warmup | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| **No baja** | Baja y se estanca en un "piso" alto | Softmax doble · activación de salida incorrecta | Quitar la softmax, pasar logits crudos a la pérdida | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| **No baja** (ni con un lote) | No memoriza 64 ejemplos | Error de código: pérdida equivocada, datos y etiquetas desalineados, `step()` olvidado | Sobreajustar un lote; imprimir ejemplos con su etiqueta | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb) |
| **NaN / inf** | De golpe, a veces tras unas épocas | log(0) casero · lr alta · gradientes que explotan | Pérdidas de PyTorch sobre logits · bajar lr · `clip_grad_norm_` | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| **Errática** | Sube y baja sin control | `zero_grad()` olvidado · lr alta · lotes muy pequeños | Revisar el bucle · bajar lr · subir el lote | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.4](../nivel-4-avanzado/m09-redes-neuronales/9.4-retropropagacion.ipynb) |
| **Valida peor que entrena** | Train → 0, validación en forma de "U" | Sobreajuste | Más datos/aumento, dropout, weight decay, early stopping | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| **Valida peor que entrena** | Validación **peor que el azar** | Etiquetas desalineadas o validación de otra distribución | Revisar cómo se unieron datos y etiquetas | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb) |
| **Validación inestable** | El mismo modelo da números distintos | `eval()` olvidado con dropout/BatchNorm | `red.eval()` al medir, `red.train()` al entrenar | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| **Sospechosamente perfecto** | ~100 % en validación en un problema difícil | Fuga de datos | Buscar columnas o ejemplos que contienen la respuesta | [9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb) |

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| No pasa de cierto acierto aunque entrenes más | Problema no separable linealmente (XOR) | Añadir capas ocultas | [9.1](../nivel-4-avanzado/m09-redes-neuronales/9.1-la-neurona-artificial.ipynb), [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb) |
| Red profunda que rinde igual que una logística | Faltan activaciones entre capas | `nn.ReLU()` tras cada `nn.Linear` oculta | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb) |
| `mat1 and mat2 shapes cannot be multiplied` | Tamaños de capas que no encajan | `nn.Linear(a, b)` seguida de `nn.Linear(b, c)` | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb), [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| La regresión nunca predice negativos | ReLU o sigmoide en la capa de salida | Salida sin activación | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb), [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| Muchas neuronas dan siempre 0 | ReLU muertas (a menudo por lr alto) | Bajar lr, Leaky ReLU | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| `Target … is out of bounds` | Etiquetas fuera de 0…K−1 | Una neurona de salida por clase | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| `Trying to backward through the graph a second time` | `backward()` dos veces sobre el mismo cálculo | Recalcular la pérdida en cada iteración | [9.4](../nivel-4-avanzado/m09-redes-neuronales/9.4-retropropagacion.ipynb) |
| La memoria crece época tras época | Guardas `perdida` con su grafo | Guardar `perdida.item()` | [9.4](../nivel-4-avanzado/m09-redes-neuronales/9.4-retropropagacion.ipynb) |
| Evaluar es lento y gasta memoria | Sin `torch.no_grad()` | `with torch.no_grad():` | [9.4](../nivel-4-avanzado/m09-redes-neuronales/9.4-retropropagacion.ipynb) |
| Cambiaste de optimizador y "ya no funciona" | Cada optimizador tiene su escala de lr | Volver a buscar `lr` | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Adam entrena genial pero valida peor que SGD | Adam a veces generaliza peor | AdamW con `weight_decay`, o SGD+momentum con *scheduler* | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| `weight_decay` en Adam no hace nada | Adam mezcla el decay con su escala | `AdamW` | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Al reanudar el entrenamiento la pérdida salta | No guardaste el estado del optimizador | Guardar también `opt.state_dict()` | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Las primeras capas casi no cambian | Gradiente que desaparece | Medir la norma del gradiente por capa; ReLU + He, BatchNorm | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| Varias neuronas con pesos idénticos | Inicialización constante | Inicialización aleatoria | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| Predecir **un** ejemplo con BatchNorm da error o rarezas | BatchNorm en modo `train` | `red.eval()` antes de predecir | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |

---

## 9. Visión, secuencias y Transformers

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Colores "rotos" al aclarar una imagen | Desbordamiento de `uint8` | Pasar a `float` y `np.clip` | [10.1](../nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb) |
| `imshow` falla o muestra algo raro | Formato `C×H×W` en lugar de `H×W×C` | `tensor.permute(1, 2, 0)` | [10.1](../nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb) |
| *Expected 4D input* | Falta la dimensión de lote | `img.unsqueeze(0)` | [10.1](../nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb), [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| La imagen se encoge hasta desaparecer | Sin padding, *stride* grande o demasiados poolings | `padding = kernel_size // 2`; menos poolings | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb), [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| Buena en prueba, mala con objetos movidos o girados | Datos de entrenamiento siempre centrados | Aumento de datos | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Red muy profunda peor que una menos profunda | Problema de degradación | Conexiones residuales (ResNet) | [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb) |
| Sobreajuste brutal con pocas imágenes | Millones de parámetros entrenados desde cero | Transfer learning | [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb), [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| El modelo preentrenado rinde mucho peor de lo esperado | Falta la normalización de ImageNet | `pesos.transforms()` | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| Tras unas iteraciones de fine-tuning la exactitud se desploma | lr demasiado alto: olvido catastrófico | lr 10–100× menor; entrenar primero la cabeza | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| Varias cajas sobre el mismo objeto | Falta NMS o umbral alto | NMS con IoU ≈ 0.5 | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| Segmentación con "buena exactitud" pero inútil | El fondo domina | Medir IoU/Dice por clase | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| Clasifica bien por la razón equivocada (mira el fondo) | Atajos espurios en los datos | Revisar con CAM; diversificar los datos | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| Error bajísimo en validación de una serie, desastre en producción | Se barajó la serie | `TimeSeriesSplit` | [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb) |
| No le gana a "mañana = hoy" | El modelo no aporta | Compararse con líneas base ingenuas y estacionales | [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb), [4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb) |
| RNN: pérdida a `nan` o picos enormes | Gradientes que explotan | `clip_grad_norm_`, bajar lr | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb), [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| RNN ignora lo que pasó hace muchos pasos | Gradientes que desaparecen | LSTM/GRU o atención | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb), [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| Errores de forma `(lote, pasos, rasgos)` | `batch_first` mal puesto | `batch_first=True` y revisar `.shape` | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| `too many values to unpack` con LSTM | `nn.LSTM` devuelve `(salidas, (h, c))` | Desempaquetar las dos memorias | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| Frases cortas perfectas, largas pésimas | Cuello de botella del contexto | Atención | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb), [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb) |
| Buena validación, mala generación real | Validaste con teacher forcing (*exposure bias*) | Validar generando; bajar el teacher forcing | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |
| La atención mira "todo a una palabra" y no aprende | Falta dividir entre √d | `scaled_dot_product_attention` | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| Pérdida casi 0 al entrenar, pero genera basura | Falta la **máscara causal** | `is_causal=True` | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb), [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| `embed_dim must be divisible by num_heads` | `d_model` no divisible | 64/4, 128/8, 768/12… | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Trata igual "el perro mordió al hombre" y al revés | Falta la codificación posicional | Sumar embeddings de posición | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| La pérdida explota al apilar capas | Faltan residuales o LayerNorm, o lr alta | Revisar `x + …`, LayerNorm, warmup | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| "El modelo grande perdió contra el pequeño" | Sin datos ni cómputo para el grande | Escalar N, D y C juntos | [12.5](../nivel-4-avanzado/m12-transformers/12.5-leyes-de-escalamiento.ipynb) |

---

## 10. Generativos y LLMs

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Las muestras copian ejemplos del entrenamiento | El generador memorizó | Modelo menos flexible; buscar el vecino real más cercano | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb), [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| Muestras "promedio" borrosas | Modelo demasiado simple o VAE | Más capacidad; GAN o difusión para nitidez | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb), [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| Reconstrucción perfecta pero el código no sirve | Cuello de botella demasiado ancho | Bajar `dim_latente` o añadir ruido | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| Muestrear al azar da garabatos | Autoencoder normal o β muy bajo | VAE con β ≈ 1 | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| Todas las muestras de la GAN se parecen | Colapso de modos | Igualar tasas, más `dim_ruido`, WGAN | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| El discriminador gana y el generador no mejora | Detective demasiado fuerte | Bajar `lr_d`, detective más pequeño | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| "Mi pérdida del generador es la más baja" | La pérdida de una GAN no se lee como en supervisado | Mirar las muestras | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| Difusión: manchas "promedio" | Pocos pasos de muestreo | Más pasos o mejor muestreador | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| La imagen no obedece al texto / colores saturados | Escala de guía baja / alta | Ajustar la *guidance scale* | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| La pérdida inicial del LLM está lejos de ln(vocabulario) | Datos o etiquetas mal desplazados | La respuesta debe ser el **siguiente** token | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| El LLM se vuelve adulador o larguísimo tras RLHF | *Reward hacking* | Penalización KL, mejores datos de preferencia | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb), [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |
| Frases que se repiten en bucle | Decodificación codiciosa o temperatura muy baja | Muestrear, `repetition_penalty ≈ 1.2`, `no_repeat_ngram_size=3` | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Texto sin sentido | Temperatura alta sin filtro de cola | Bajar temperatura, `top_p = 0.9` | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| No puedes reproducir el resultado de ayer | Muestreo sin semilla | Fijar semilla o temperatura 0 | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Inventa datos incluso con temperatura 0 | La temperatura no arregla el conocimiento | RAG | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb), [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| LoRA no mejora nada | `r`/`alpha` pequeños, lr bajo o módulos irrelevantes | Subir `r`/`alpha`, lr 1e-4–3e-4, más módulos | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| El modelo cuantizado escribe sin sentido | Pocos bits o escala por tensor con atípicos | 8 bits; escala por canal o grupo | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| Con RAG sigue respondiendo mal | Falla la recuperación o la generación | Mirar qué documento llegó al prompt antes de tocar nada | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| Un documento "da órdenes" al modelo | Inyección de instrucciones | Delimitar datos, permisos mínimos, revisión humana | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb), [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| El agente da vueltas sin terminar | Sin límite de pasos | `max_pasos`, presupuesto, errores claros | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| Cita fuentes que no existen | Alucinación | Verificar cada fuente; RAG con documentos reales | [13.9](../nivel-5-heroe/m13-generativos-llms/13.9-alucinaciones-y-limites.ipynb) |
| Olvida lo dicho al principio del chat | Se superó la ventana de contexto | Resumir, repetir lo importante | [13.9](../nivel-5-heroe/m13-generativos-llms/13.9-alucinaciones-y-limites.ipynb) |

---

## 11. Aprendizaje por refuerzo

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Hace algo raro que da muchos puntos pero no resuelve la tarea | *Reward hacking* | Revisar la recompensa; mirar episodios | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| El agente nunca aprende nada | Recompensa demasiado escasa | Recompensas intermedias con cuidado, más exploración | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb), [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| Termina los episodios lo antes posible, aunque sea mal | Costo por paso demasiado alto | Bajar el castigo por paso | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| Repite siempre la misma acción mediocre | Sin exploración o ε decae muy rápido | Subir ε, decaer más lento, valores iniciales optimistas | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| Se estanca por debajo de lo posible | ε fijo demasiado alto | Decaer ε | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| Ignora metas lejanas | γ demasiado bajo | γ 0.95–0.99 | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| La curva baja y luego oscila | α demasiado alto | Bajar α o hacerlo decaer | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| (DQN) Los Q crecen sin control | Objetivos que se mueven | Red objetivo, bajar lr, recortar recompensas | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| La recompensa sube y de pronto se desploma | Pasos de política demasiado grandes | Bajar `learning_rate`, `clip_range` o `n_epochs` | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| La política se vuelve determinista muy pronto | Se acabó la exploración | Subir `ent_coef` | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| Resultados muy distintos con otra semilla | El RL es muy sensible al azar | 3–5 semillas; reportar media y dispersión | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| Excelente en el simulador, malo en la realidad | Brecha simulador-realidad | Aleatorización de dominio, ajuste con datos reales | [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |
| 6 meses de RL y no supera a una regla simple | RL no era la herramienta | Comparar siempre con una línea base simple | [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |

---

## 12. Producción

| Síntoma | Causa probable | Qué probar | Lección |
|---|---|---|---|
| Rinde mucho peor que en el notebook desde el día 1 | *Training-serving skew* | Guardar el pipeline completo; validar entradas | [15.1](../nivel-5-heroe/m15-produccion-y-etica/15.1-del-notebook-a-produccion.ipynb) |
| Error al cargar el modelo en el servidor | Versiones de librerías distintas | Fijar versiones; contenedores | [15.1](../nivel-5-heroe/m15-produccion-y-etica/15.1-del-notebook-a-produccion.ipynb) |
| Una versión nueva rompe todo | Despliegue de golpe | Sombra, canario, plan de *rollback* | [15.1](../nivel-5-heroe/m15-produccion-y-etica/15.1-del-notebook-a-produccion.ipynb), [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| El negocio se queja, pero el tablero está en verde | Deriva de concepto: solo se vigilan entradas | Monitorear desempeño con etiquetas reales | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Alertas cada mes que nadie revisa | Valores p con muchos datos | PSI o tamaño de efecto; priorizar variables | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Alarmas cada diciembre | Estacionalidad | Comparar con el mismo período del año anterior | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Tras reentrenar automáticamente, todo empeora | Se reentrenó con datos corruptos | Validar datos antes; canario; *rollback* | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| No puedes reproducir un modelo de hace meses | Sin versionado de datos, semillas ni librerías | Registro de modelos + versionado de datos | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| Buena exactitud global, quejas de un grupo | Métricas agregadas ocultan diferencias | Auditar por grupo | [15.4](../nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb) |
| "No usamos el género, así que es justo" | Variables *proxy* | Medir métricas por grupo | [15.4](../nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb) |
| 100 % en entrenamiento con datos sensibles | Riesgo de filtrar información | Regularizar, privacidad diferencial | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| Una variable "sin sentido" es la más importante | Fuga o *proxy* | Revisar cómo se creó la variable | [15.6](../nivel-5-heroe/m15-produccion-y-etica/15.6-explicabilidad-y-regulacion.ipynb) |
| El cliente recibe "razones" que no puede cambiar | Explicaciones no accionables | Contrafactuales con variables accionables | [15.6](../nivel-5-heroe/m15-produccion-y-etica/15.6-explicabilidad-y-regulacion.ipynb) |
