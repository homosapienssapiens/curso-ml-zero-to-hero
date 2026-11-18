# 🎛️ Chuleta de hiperparámetros

La tabla maestra del curso: para cada familia de modelos, **qué pasa si subes o bajas cada perilla** y **con qué valor empezar**. Todo sale de las tablas "Hiperparámetros y perillas" de las lecciones; la columna *Lección* te lleva a la explicación completa.

> ⚠️ Los valores de partida son **puntos de partida**, no recetas: el valor bueno se elige con validación cruzada ([4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb)) y la métrica del negocio ([4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb)). Busca en **escala logarítmica** las perillas que se mueven por órdenes de magnitud (`C`, `alpha`, `lr`) ([7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb)).

Material de apoyo: [Glosario](glosario.md) · [¿Qué modelo uso?](arbol-de-decision-modelos.md) · [Diagnóstico del entrenamiento](diagnostico-entrenamiento.md) · [Recursos](recursos.md) · [Volver al README](../README.md)

## Índice

1. [Preparación de datos y validación](#1-preparación-de-datos-y-validación)
2. [Entrenamiento por descenso de gradiente](#2-entrenamiento-por-descenso-de-gradiente)
3. [Modelos clásicos supervisados](#3-modelos-clásicos-supervisados)
4. [Ensambles](#4-ensambles)
5. [Regularización y early stopping](#5-regularización-y-early-stopping)
6. [No supervisado](#6-no-supervisado)
7. [Redes neuronales](#7-redes-neuronales)
8. [Visión (CNN)](#8-visión-cnn)
9. [Secuencias](#9-secuencias)
10. [Transformers](#10-transformers)
11. [Generativos y LLMs](#11-generativos-y-llms)
12. [Aprendizaje por refuerzo](#12-aprendizaje-por-refuerzo)
13. [Producción](#13-producción)
14. [Equivalencias entre librerías](#14-equivalencias-entre-librerías)
15. [¿Qué ajusto primero?](#15-qué-ajusto-primero)

**Regla de oro** (la verás en casi todas las filas): más complejidad → menos sesgo y más riesgo de **sobreajuste**; menos complejidad → más estable y más riesgo de **subajuste** ([4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb), [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb)).

---

## 1. Preparación de datos y validación

| Herramienta | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| `OneHotEncoder` | `handle_unknown` | — | — | `"ignore"` (con `"error"`, una categoría nueva rompe el sistema) | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| `OneHotEncoder` | `min_frequency` | Menos columnas, se pierde detalle | Más columnas | `None`, o ~1 % de los datos si hay muchas categorías | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| `TargetEncoder` | `smooth` | Más prudente con categorías raras | Confía más en cada categoría (riesgo de sobreajuste) | `"auto"`; 5–20 si hay categorías con 1–2 ejemplos | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| `OrdinalEncoder` | `categories` | — | — | Siempre explícito (si no, el orden es alfabético) | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| `train_test_split` | `test_size` | Medición más fiable, menos datos para aprender | Más datos para aprender, medición más ruidosa | 0.2 (0.1 con muchísimos datos) | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| `train_test_split` | `stratify` / `shuffle` | — | — | `stratify=y` en clasificación; `shuffle=False` con series de tiempo | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| Desbalance | `class_weight` | Más recall, menos precisión | Menos recall, más precisión | `"balanced"` o `{0: 1, 1: w}` | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| Desbalance | Proporción de re-muestreo | Más equilibrio, más recall (y más riesgo de memorizar copias) | Más cerca de la proporción real | No hace falta llegar a 50/50 | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| Clasificador | Umbral de decisión | Menos alarmas: ⬆️ precisión, ⬇️ recall | Más alarmas: ⬆️ recall, ⬇️ precisión | 0.5 por convención; el bueno sale del costo de cada error, con validación | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb), [4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb) |
| Validación cruzada | `n_splits` (k) | Cada modelo entrena con más datos; más lento | Más rápido; estimación algo pesimista | 5 (10 con pocos datos) | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb) |
| `TimeSeriesSplit` | `gap` | Simula un retraso real en la llegada de datos | Más datos para entrenar | 0, o el retraso real de tu sistema | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb) |
| `TimeSeriesSplit` | `max_train_size` | Usa más historia | Solo historia reciente (útil si el mundo cambia rápido) | Sin límite | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb) |
| `GroupKFold` | `groups` | — | — | El ID de persona/cliente/máquina si se repite | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb), [7.5](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.5-pipelines-sin-fugas.ipynb) |
| Series de tiempo | Ventana (*lags*) | Ve ciclos largos, pero hay menos filas y más ruido | Más simple, "ciego" a los ciclos | Al menos un ciclo completo (7 días, 24 h…) | [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb) |
| Series de tiempo | Horizonte | Problema más difícil, error mayor | Más fácil | El que pida el negocio | [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb) |

## 2. Entrenamiento por descenso de gradiente

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Cualquiera entrenado con gradiente | **Tasa de aprendizaje** (`lr`, `eta0`, `learning_rate_init`) | Más rápido… hasta zigzaguear y **divergir** (`nan`) | Estable pero lentísimo; se queda a medio camino | Barrido 1e-4 → 1. SGD: 0.01 · Adam: 0.001 | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb), [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Cualquiera | Épocas | Más cerca del fondo; riesgo de **sobreajuste** | Termina antes de llegar (**subajuste**) | Hasta que la validación deje de bajar (early stopping) | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb), [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) |
| Cualquiera | Tamaño de lote (`batch_size`) | Pasos suaves, menos pasos por época, más memoria; puede pedir subir la tasa | Pasos ruidosos, más pasos; puede pedir bajar la tasa | 32–256 | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb), [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| Cualquiera | Iteraciones (pasos) | — (no se elige aparte) | — | `épocas × ⌈N / lote⌉` | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Cualquiera | `shuffle` | — | Desactivado con datos ordenados: escalones y sesgo | `True` (salvo series de tiempo) | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| Cualquiera | Decaimiento de la tasa (*schedule*) | Se asienta antes, puede frenar antes de llegar | Sigue "bailando" | El *scheduler* por defecto de tu librería | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Cualquiera | Momento (`momentum`, `beta1`) | Cruza mesetas, pero puede pasarse de largo | Descenso normal | 0.9 | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb), [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Cualquiera | Semilla / nº de reinicios (`random_state`, `n_init`) | Más probable un buen valle; más tiempo | Más rápido, más riesgo de quedarse en una hondonada | 3–5 semillas en redes; `n_init=10` en K-means | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb) |

## 3. Modelos clásicos supervisados

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Regresión lineal / polinomial | `degree` (`PolynomialFeatures`) | Más curva → **sobreajuste** | Más rígida → **subajuste** | 1; 2–3 si ves curvatura | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| `Ridge` / `Lasso` / `ElasticNet` | `alpha` | Coeficientes más pequeños → **subajuste** | Más libre → **sobreajuste** | 1.0; buscar en 0.001–1000 (log) | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb), [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| `ElasticNet` | `l1_ratio` | Más Lasso: más coeficientes en 0 | Más Ridge: reparte el peso | 0.5 | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| Regresión logística | `C` (**inverso** de la regularización) | **Menos** castigo: probabilidades extremas, **sobreajuste** | Más castigo: probabilidades cerca de 0.5, **subajuste** | 1.0; probar 0.01–100 (log) | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| Regresión logística | `max_iter` | Más tiempo para converger | `ConvergenceWarning` | 100; súbelo a 1000 si sale el aviso | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| Regresión logística | `l1_ratio` | Más coeficientes exactamente en 0 | Castigo suave a todos | 0 (L2) | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| Regresión logística / árbol / SVM | `class_weight` | — | — | `None`; `"balanced"` si la clase que importa es rara | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb), [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb), [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| KNN | `n_neighbors` (k) | Frontera suave → **subajuste** | Frontera irregular → **sobreajuste** | 5; probar 1–50 (impar con 2 clases) | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb) |
| KNN | `weights` / `metric` | — | — | `uniform` o `distance`; euclidiana (`p=2`), Manhattan (`p=1`) con muchas columnas | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb) |
| Árbol de decisión | `max_depth` | Más complejo → **sobreajuste** | Más simple → **subajuste** | 3–10 | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| Árbol de decisión | `min_samples_leaf` | Más simple, hojas más confiables | Hojas de 1 ejemplo (memoriza) | 1–10 (o ~1 % de los datos) | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| Árbol de decisión | `min_samples_split` | Más simple | Más complejo | 2–20 | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| Árbol de decisión | `max_leaf_nodes` | Más complejo | Más simple | Sin límite / 10–100 | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| Árbol de decisión | `max_features` | Usa más información | Más aleatorio (clave en Random Forest) | Todas | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| Árbol de decisión | `ccp_alpha` (poda) | Más podado y simple | Más grande | 0.0; ajustar con CV | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| SVM | `C` | Calle angosta, intenta acertar todo → **sobreajuste** | Calle ancha, tolera errores → **subajuste** | 1; probar 0.01–1000 (log) | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| SVM | `gamma` (rbf) | Alcance corto: islas alrededor de cada punto → **sobreajuste** | Alcance largo: frontera muy suave → **subajuste** | `"scale"`; probar 0.001–10 | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| SVM | `kernel` / `degree` | Más grado: más curvas | Más simple | `"rbf"` (comparar con `"linear"`); `degree=3` | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| Naive Bayes (Multinomial, Bernoulli) | `alpha` (suavizado) | Más tímido → **subajuste** | Confía en palabras raras → **sobreajuste** | 1.0; probar 0.01–10 | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |
| Naive Bayes (Gaussian) | `var_smoothing` | Campanas más anchas, más tímido | Campanas exactas | 1e-9 (casi nunca se toca) | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |
| Naive Bayes | `priors` | — | — | Fíjalos si en producción la proporción de clases será otra | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |

## 4. Ensambles

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Random Forest | `n_estimators` | Más estable hasta estancarse; más lento. **No sobreajusta** | Resultados más ruidosos | 200–500 (fíjalo, no lo busques) | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb), [7.4](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.4-priorizar-hiperparametros.ipynb) |
| Random Forest | `max_features` | Árboles mejores pero más parecidos | Árboles más distintos pero más débiles | `"sqrt"` (clasif.) · `1.0` o 0.3–0.5 (regresión) | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Random Forest | `min_samples_leaf` | Menos ruido memorizado | Hojas de 1 ejemplo | 1; 3–20 con datos ruidosos | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Random Forest | `max_depth` | Árboles más complejos | Más simples (más sesgo) | `None` | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Random Forest | `bootstrap` / `max_samples` / `oob_score` | — | Menos muestra: árboles más distintos, más rápido | `True` / `None` / `True` para una validación gratis | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| Gradient boosting | `n_estimators` (rondas) | Menos sesgo; demasiados → **sobreajuste** | **Subajuste** | 100–1000 **con parada temprana** | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb) |
| Gradient boosting | `learning_rate` | Aprende rápido y brusco, sobreajusta antes | Necesita más árboles, suele generalizar mejor | 0.05–0.1 | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb), [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `max_depth` / `num_leaves` | Más interacciones, más sobreajuste | Más simple (`1` = tocones) | depth 3 (2–6) en sklearn · 4–8 en XGBoost · leaves 15–63 | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb), [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `min_samples_leaf` / `min_child_samples` | Hojas más confiables, menos sobreajuste | Hojas diminutas | 20 (subir con datos ruidosos) | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb), [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `subsample` | — | Más azar: regulariza y acelera | 0.7–0.9 | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb), [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `colsample_bytree` / `max_features` | — | Árboles más distintos | 0.5–0.9 | [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `reg_lambda` / `l2_regularization` | Hojas más moderadas, menos sobreajuste | Menos freno | 0–10 | [6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| Gradient boosting | `n_iter_no_change` (paciencia) | — | — | 10–50, con `validation_fraction=0.1` | [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb), [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Voting | `voting` / `weights` | — | — | `"soft"` si los miembros dan probabilidades razonables; pesos iguales | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| Stacking / Voting | `estimators` | — | — | 3–6 miembros buenos y **distintos** | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| Stacking | `final_estimator` | Un jefe complejo sobreajusta | — | Uno simple: `LogisticRegression` / `RidgeCV` | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| Stacking | `cv` | Más lento, algo más estable | — | 5 | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |
| Stacking | `passthrough` | El jefe ve también las columnas originales: sobreajusta más fácil | — | `False` | [6.5](../nivel-3-practicante/m06-ensambles/6.5-stacking-y-voting.ipynb) |

## 5. Regularización y early stopping

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Lineales | `alpha` (tarifa por complejidad) | **Subajuste** | **Sobreajuste** | 1.0, buscar 0.001–1000 (log) | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| Logística / SVC | `C ≈ 1/alpha` | **Menos** castigo → sobreajuste | **Más** castigo → subajuste | 1.0, buscar 0.001–1000 (log) | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| Boosting, SGD, MLP | `early_stopping` | — | — | `True`. ⚠️ En `HistGradientBoosting`, `"auto"` solo lo activa con > 10,000 filas | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Boosting, SGD, MLP | `n_iter_no_change` (paciencia) | Menos riesgo de parar por un bache | Riesgo de parar antes de tiempo | 10 (boosting, redes); 10–20 en SGD | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Boosting, SGD, MLP | `validation_fraction` | Vigilancia más fiable, menos datos para aprender | Vigilancia más ruidosa | 0.1–0.2 | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Boosting, SGD, MLP | `max_iter` / `n_estimators` (tope) | Con early stopping, casi no importa | Si es bajo, el tope corta antes que la paciencia | Generoso: 500–1000 | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |
| Boosting, SGD, MLP | `tol` | Para antes | Para después | El valor por defecto | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb) |

**Qué perillas regularizan cada modelo** ([7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb)): lineales `alpha`, `C`, `l1_ratio` · SVM `C`, `gamma` · árbol `max_depth`, `min_samples_leaf`, `ccp_alpha` · boosting `learning_rate`, `max_depth`, `l2_regularization`, `subsample` · redes `alpha`/weight decay, dropout, early stopping.

## 6. No supervisado

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| K-means | `n_clusters` (k) | Grupos más pequeños; la inercia **siempre** baja | Grupos más grandes y mezclados | Codo + silueta + negocio (2–10) | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| K-means | `n_init` | Más estable, más lento | Riesgo de mínimo local | 10 (o `"auto"`) | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| K-means | `init` / `max_iter` | — | Con `max_iter` bajo puede no converger | `"k-means++"` / 300 | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| DBSCAN | `eps` | Grupos que se **fusionan** (al extremo, uno solo) | Grupos que se **rompen** (al extremo, todo ruido) | El codo de la gráfica de k-distancias | [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| DBSCAN | `min_samples` | Exige más densidad: más ruido | Grupos con pocos puntos, menos ruido | ≈ 2 × nº de columnas (mínimo 4–5) | [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| PCA | `n_components` (entero) | Más información, menos compresión | Más compresión, se pierde información | 2–3 para visualizar | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| PCA | `n_components` (0–1) | Más componentes | Menos componentes | 0.90–0.95 de varianza para preprocesar | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| PCA | `whiten` / `svd_solver` | — | — | `False` / `"auto"` (`"randomized"` con datos grandes) | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| t-SNE | `perplexity` | Visión más global | Muy local: aparecen grupos falsos | 30 (probar 5–50) | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| t-SNE | `max_iter` | Mapa más asentado | Mapa a medio hacer, "apelotonado" | 1000 | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| t-SNE | `init` / `random_state` / `metric` | — | — | `"pca"` / fija la semilla / `"cosine"` para texto | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| UMAP | `n_neighbors` | Visión más global | Más local; puede romper grupos | 15 (5–50) | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| UMAP | `min_dist` | Islas más esponjosas | Islas muy compactas | 0.1 | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| Isolation Forest | `contamination` | Más alarmas: atrapas más, más falsas alarmas | Menos alarmas; se escapan anomalías | `"auto"` o la capacidad de revisión | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| Isolation Forest | `n_estimators` | Puntuaciones más estables, más lento | Más ruidosas | 100–300 | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| Isolation Forest | `max_samples` | Ve más detalle, más lento | Más rápido; ayuda con grupos de anomalías juntas | `"auto"` (256) | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| Recomendación (factorización) | `n_components` (factores latentes) | Más matices, empieza a memorizar ruido | Demasiado simple | 10–100, elegir con validación | [8.7](../nivel-3-practicante/m08-no-supervisado/8.7-sistemas-de-recomendacion.ipynb) |
| Recomendación (vecinos) | `k` vecinos | Más estable, menos personal | Más personal, más ruidoso | 20–50 | [8.7](../nivel-3-practicante/m08-no-supervisado/8.7-sistemas-de-recomendacion.ipynb) |
| Recomendación (vecinos) | Mínimo de calificaciones en común | Similitudes más fiables, menos vecinos | Vecinos "parecidos" por casualidad | 3–5 | [8.7](../nivel-3-practicante/m08-no-supervisado/8.7-sistemas-de-recomendacion.ipynb) |

## 7. Redes neuronales

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Red densa (MLP) | Neuronas por capa (ancho) | Más capacidad, más lenta, más **sobreajuste** | **Subajuste** | 32–256 ([9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb)) · 64–512 ([9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb)) | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Red densa | Nº de capas ocultas | Más capacidad, más difícil de entrenar | Más simple y estable | 1–3 para tablas o MNIST | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Red | Activación oculta | — | — | **ReLU** (GELU en Transformers); Leaky ReLU (`negative_slope=0.01`) si hay neuronas muertas | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| Red | Activación de salida | — | — | Ninguna (regresión) · sigmoide (binaria) · softmax (multiclase), **dentro** de la pérdida | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| Optimizador | `lr` | Más rápido hasta oscilar o dar NaN | Lento | Adam `1e-3` · SGD+momentum 0.01–0.1; luego *LR range test* | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb), [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| SGD | `momentum` / `nesterov` | Atraviesa mesetas, se pasa de largo | Más zigzag | 0.9 / `True` | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Adam / AdamW | `betas` | Promedios más suaves y lentos | Más reactivo y ruidoso | `(0.9, 0.999)`, casi nunca se tocan | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| RMSprop | `alpha` (ρ) | Más suave | Más reactivo | 0.99 | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Adam / RMSprop | `eps` | Pasos más parecidos a SGD | — | 1e-8 (no tocar) | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Optimizador | `weight_decay` | Más regularización (riesgo de subajuste) | Más libertad para memorizar | 0 · con **AdamW**: 0.01 (1e-4–0.1) | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Red | *Scheduler* | — | — | Coseno u *one-cycle*; *warmup* ≈ 5–10 % de los pasos en Transformers | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Red | Tamaño de lote | Más rápido por época, gradiente estable, más memoria | Gradiente ruidoso, más pasos | 32–128 | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Red | Épocas | **Sobreajuste** | **Subajuste** | Generosas + early stopping (paciencia 5–10) | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Red | Inicialización | Pesos grandes: activaciones saturadas o que explotan | Señal y gradiente que se apagan | **He** con ReLU, **Xavier** con tanh/sigmoide | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| Red | `max_norm` (recorte de gradiente) | Recorta menos | Pasos muy cortos | 1.0 (RNN y Transformers: 0.5–5) | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| Red | Dropout `p` | Más regularización; muy alto → subajuste | Menos regularización | 0.1–0.3 (redes modernas); 0.5 en densas con pocos datos | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Red | BatchNorm `momentum` | Promedio más "nervioso" | Promedio más lento | 0.1 (PyTorch) | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Red | Intensidad del aumento de datos | Más variedad; demasiado → irreconocible | Menos efecto | Pequeño y realista (±10°, 10 %) | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |

## 8. Visión (CNN)

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Conv2d | `kernel_size` | Patrones más grandes, más parámetros | Patrones locales, más barato | **3** (5–7 en la primera capa de algunas redes) | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| Conv2d | `stride` | Salida más pequeña y rápida, pierde detalle | Salida más grande y detallada | 1 (2 para reducir a la mitad) | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| Conv2d | `padding` | Conserva tamaño y bordes | La salida se encoge en cada capa | `kernel_size // 2` ("same") | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| Conv2d | `out_channels` (filtros) | Más capacidad, más sobreajuste, más lento | Menos capacidad | 16–64 al principio; **duplicar** tras cada reducción | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb), [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| Pooling | `kernel_size` | Reduce más rápido, más tolerancia, menos detalle de posición | Reduce menos | 2 (con `stride=2`) | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| Pooling | Tipo | — | — | Max entre capas; promedio **global** al final | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| CNN | Nº de bloques conv+pool / profundidad | Patrones más abstractos; sin atajos residuales no entrena | Puede no ver el objeto completo | 2–3 bloques (28–32 px); ResNet18–50 para fotos | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb), [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb) |
| CNN | Resolución de entrada | Detecta detalles y objetos pequeños; cómputo ∝ píxeles | Más rápida, pierde detalle | 224 (ImageNet) · 32 (CIFAR) · 640 (YOLO) | [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb), [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| Transfer learning | Capas descongeladas | Más adaptación, más sobreajuste | Más estable y rápido | Solo la cabeza; luego `layer4` si hay datos | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| Transfer learning | `lr` del fine-tuning | **Olvido catastrófico** | Aprende muy despacio | 1e-4 (10–100× menor que desde cero); cabeza 1e-3 | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| Transfer learning | Épocas | Sobreajuste | Subajuste | 3–10 | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| Detector | Umbral de confianza | Menos cajas: ⬆️ precisión, ⬇️ recall | Más cajas y falsas alarmas | 0.5 | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| Detector | Umbral IoU de NMS | Quedan duplicados | Borra objetos pegados | 0.5 (0.45–0.7) | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| ViT | Tamaño de parche | Más rápido, menos detalle | Más detalle, mucho más costo | 16 × 16 | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |

## 9. Secuencias

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Embeddings | `embedding_dim` | Más matices, más parámetros y datos necesarios | Se mezclan significados | 50–300 (word2vec); 768+ en LLMs | [11.2](../nivel-4-avanzado/m11-secuencias/11.2-tokens-y-embeddings.ipynb) |
| Embeddings | Ventana de contexto (word2vec) | Capta **temas** | Capta **función gramatical** y sinónimos | 2–5 | [11.2](../nivel-4-avanzado/m11-secuencias/11.2-tokens-y-embeddings.ipynb) |
| Tokenizador | Tamaño del vocabulario | Secuencias más cortas, tabla más grande | Más pedazos por palabra | 30k–100k subpalabras | [11.2](../nivel-4-avanzado/m11-secuencias/11.2-tokens-y-embeddings.ipynb) |
| RNN | `hidden_size` | Recuerda más; más lento, más sobreajuste | Memoria pobre | 32–256 | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| RNN / LSTM | `num_layers` | Más abstracción, más difícil de entrenar | Más simple | 1–2 (RNN) · 1–3 (LSTM) | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb), [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| RNN | Largo de secuencia (BPTT) | Dependencias más largas; más memoria | Rápido, pero "no ve" lo lejano | 20–200 | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| RNN / LSTM | Recorte de gradiente | Menos protección contra explosiones | Pasos tímidos | 1.0–5.0 | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| RNN | `bidirectional` | Usa el futuro: **no** sirve para pronosticar | Solo pasado | `False` para pronóstico; `True` para clasificar textos completos | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| LSTM / GRU | `hidden_size` | Más capacidad (×4 parámetros en LSTM, ×3 en GRU) | Memoria pobre | 64–512 | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| LSTM | `dropout` entre capas | Menos sobreajuste | Más sobreajuste | 0.1–0.3 (solo si `num_layers` ≥ 2) | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| LSTM | Sesgo inicial de la puerta de olvido | Recuerda más desde el principio | Olvida rápido | 1.0 (Keras por defecto; PyTorch 0) | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| Seq2seq | Proporción de *teacher forcing* | Entrena rápido, más *exposure bias* | Más robusto, más lento | 1.0, bajando poco a poco (0.5) | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |
| Seq2seq | Ancho del haz | Mejores secuencias, más lento | 1 = voraz | 4–10 | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |

## 10. Transformers

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Atención | Escala de los puntajes | Atención casi "dura", gradientes débiles | Atención difusa | Siempre `1/√d` | [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb), [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| Atención | `d` por cabeza | Búsquedas más finas, más cálculo | Más barato | 64 | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| Atención | Máscara | — | — | Causal para generar; ninguna para comprender | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb), [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| Transformer | `d_model` | Más capacidad y memoria | Si es muy chico, no cabe la información | 64–512 (pequeños) · 768 (GPT-2) | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | `n_cabezas` | Más tipos de relaciones, vectores más chicos | Menos variedad | `d_model / 64` (debe dividir a `d_model`) | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | `n_capas` | Razonamientos de más pasos, más difícil de entrenar | Más rápido | 2–6 (pequeños) · 12 (GPT-2) | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | `d_ff` (MLP interno) | Más capacidad | Menos parámetros | 4 × `d_model` | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | Longitud de contexto | Más contexto, costo × n² | Más barato | Lo que pida la tarea (GPT-2: 1,024) | [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb), [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | `dropout` | Menos sobreajuste | Puede sobreajustar | 0.0–0.1 | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Transformer | `lr` + *warmup* | Inestable | Lento | 3e-4 con warmup; 1e-3–3e-3 en modelos diminutos | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| Escalamiento | N (parámetros) | Menos pérdida si hay datos y cómputo | Techo más bajo, más barato de usar | Decídelo junto con D | [12.5](../nivel-4-avanzado/m12-transformers/12.5-leyes-de-escalamiento.ipynb) |
| Escalamiento | D (tokens) | Menos pérdida y memorización | Sobreajuste | D ≈ 20·N (Chinchilla) | [12.5](../nivel-4-avanzado/m12-transformers/12.5-leyes-de-escalamiento.ipynb) |
| Escalamiento | C (cómputo ≈ 6·N·D) | Permite subir N y D a la vez | Obliga a elegir | Repártelo según la frontera eficiente | [12.5](../nivel-4-avanzado/m12-transformers/12.5-leyes-de-escalamiento.ipynb) |

## 11. Generativos y LLMs

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| KDE | `bandwidth` | Muestras borrosas (subajuste) | Copia los datos (sobreajuste) | Elegir con verosimilitud en validación | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb) |
| Mezcla de gaussianas | `n_components` | Más flexible, puede sobreajustar | No capta la forma | Verosimilitud en validación o BIC | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb) |
| Autoencoder / VAE | `dim_latente` | Mejor reconstrucción, menos compresión | Reconstrucción más borrosa | 2 (para ver) · 16–128 (para usar) | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| VAE | `beta` (β) | Mapa ordenado, imágenes más borrosas | Más fiel, el mapa tiene huecos | 1 | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| Autoencoder que quita ruido | `ruido` | Limpia ruido fuerte, pierde detalle | Como un autoencoder normal | 0.2–0.5 | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| GAN | `lr_g` / `lr_d` | Juego inestable | Lento; si difieren mucho, desequilibrio | 1e-4–2e-4 en imágenes, **parecidas entre sí** | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| GAN | `betas` de Adam | β₁ = 0.9: más inercia, oscila | β₁ = 0.5: reacciona al rival | `(0.5, 0.999)` | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| GAN | `dim_ruido` | Más variedad posible | Riesgo de colapso de modos | 100–512 en imágenes | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| Difusión | `T` (pasos al entrenar) | Escalones más finos | Saltos bruscos | 1000 en imágenes | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| Difusión | Pasos de muestreo | Más calidad, más lento | Resultados borrosos o "promedio" | 20–50 (DDIM) | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| Difusión | Escala de guía (CFG) | Más fiel al texto, menos variedad, colores exagerados | Más variedad, menos fiel | 5–8 | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| Difusión | Calendario de ruido | — | — | Coseno | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| LLM (entrenamiento) | `lr` | Diverge | Lentísimo | 1e-4–6e-4 con warmup (LLMs) | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| LLM (entrenamiento) | Longitud de contexto | Dependencias largas, más cómputo | Solo patrones locales | 1,024 (GPT-2) · 100k+ (actuales) | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| LLM (entrenamiento) | Tokens vistos | Mejor, hasta agotar datos (luego memoriza) | Subentrenado | ≈ 20 tokens por parámetro | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| Generación | `temperature` | Más variado → disparates | Más predecible → repeticiones (0 = codicioso) | 0.7 (chat) · 0–0.3 (datos, código, RAG) · 0.9–1.1 (creativo) | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `top_k` | Más opciones | Más conservador (1 = codicioso) | 40–50, o 0 si usas top-p | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `top_p` | Más opciones de la cola | Más conservador | 0.9–0.95 | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `max_new_tokens` | Más largo y caro | Cortes a media frase | Lo que pida la tarea + margen | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `repetition_penalty` | Gramática rara | Bucles (1.0 = nada) | 1.1–1.3 | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `no_repeat_ngram_size` | Menos restrictivo | Muy restrictivo | 0 o 3–4 | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| Generación | `do_sample` / semilla | — | — | `True` para texto natural; fija la semilla para reproducir | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| LoRA | `r` (rango) | Más capacidad y memoria, más sobreajuste | Subajuste si la tarea pide más | 8–16 | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| LoRA | `alpha` | El adaptador pesa más (como subir `lr`) | Pesa menos | α = 2·r (o α = r) | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| LoRA | `target_modules` | Mejor calidad, más parámetros | Más barato | `q`, `k`, `v`, `o`; agregar el MLP si no alcanza | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| LoRA | `lr` / `lora_dropout` | Diverge / más regularización | Lento / menos | 1e-4–3e-4 / 0.05–0.1 | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| Cuantización | Bits por peso | Más fiel, más memoria | Menos memoria; < 4 bits suele romperse | 8 bits casi sin pérdida · 4 bits con buenos métodos | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| Cuantización | Granularidad de la escala | Más fina = menos error | Los atípicos lo arruinan | Por canal o grupos de 64–128 | [13.7](../nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb) |
| Prompting | Nº de ejemplos (*few-shot*) | Formato más fiable, prompt más caro | Menos guía | 2–5 | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| RAG | `k` documentos | Más probable incluir el correcto, más ruido | Contexto limpio, puede faltar el bueno | 3–5 | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| RAG | Tamaño de fragmento | Más contexto, búsqueda menos precisa | Búsqueda precisa, frases sueltas | 200–500 tokens con solapamiento | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| RAG | Umbral de similitud | Más "no lo sé", menos inventos | Responde aunque no haya información | Ajustar con preguntas de prueba | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| Agente | `max_pasos` | Tareas más largas, más costo y bucles | Corta tareas legítimas | 5–10 | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |

## 12. Aprendizaje por refuerzo

| Modelo | Hiperparámetro | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Diseño del entorno | Costo por paso | Positivo: premia alargar el episodio (¡peligro!) | Muy negativo: termina rápido aunque sea mal | −1 o 0 | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| Diseño del entorno | Recompensa final | La meta domina | Puede no compensar el camino | Mayor que el costo de un camino razonable | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| ε-codicioso | `epsilon` (ε) | Descubre antes, desperdicia jugadas | Se atasca en lo conocido | 0.1, o 1.0 que decae a 0.01–0.05 | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb), [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| ε-codicioso | Decaimiento de ε | Explora más tiempo | Deja de explorar antes de tiempo | Llegar a ε mínimo a mitad o al final | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| Bandidos / Q | `q_inicial` | Más optimista: prueba todo | Sin incentivo para probar | 0, o mayor que la mejor recompensa | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| Q-learning, PPO | `gamma` (γ) | Más paciente, aprende más lento | Miope | 0.9–0.99 | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb), [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb), [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| Q-learning | `alfa` (α) | Rápido pero nervioso | Estable pero lento | 0.1 (tabla) · 1e-4–1e-3 (red) | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| DQN | `buffer_size` | Experiencias más variadas pero viejas | Recientes, inestable | 10⁴–10⁶ | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| DQN | `target_update_interval` | Más estable, más lento | Más inestable | 500–10,000 pasos | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| PPO | `learning_rate` | Inestable | Lento | 3e-4 | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| PPO | `clip_range` | Cambios audaces | Cambios tímidos | 0.2 (0.1–0.3) | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| PPO | `n_epochs` | Aprovecha cada lote, se sobreajusta a él | Desperdicia datos | 10 (3–30) | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| PPO | `n_steps` | Ventajas menos ruidosas | Más rondas, más ruido | 2048 | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| PPO | `gae_lambda` | Menos sesgo, más ruido | Menos ruido, más sesgo | 0.95 | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| PPO | `ent_coef` | Explora más | Determinista antes | 0.0–0.01 | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |

## 13. Producción

| Sistema | Perilla | ⬆️ Al subir | ⬇️ Al bajar | Valor de partida | Lección |
|---|---|---|---|---|---|
| Monitoreo | Ventana actual | Menos ruido, detecta tarde | Detecta antes, más falsas alarmas | 1 semana–1 mes (≥ 1,000 casos) | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Monitoreo | Umbral de PSI | Menos alertas | Más alertas | 0.1 (vigilar) · 0.25 (actuar) | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Monitoreo | Umbral de caída de desempeño | Toleras más pérdida | Reentrenas más seguido | AUC −0.03 respecto a validación | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| Reentrenamiento | Frecuencia | Se adapta antes, aprende ruido | Envejece | Lo que permitan las etiquetas (p. ej. mensual) | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| Reentrenamiento | Tamaño de la ventana | Más estable, mezcla mundos viejos | Más actual, más ruido | 3–12 meses | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| Reentrenamiento | Margen campeón/retador | Menos cambios | Cambia por ruido | 0.01 de AUC o prueba estadística | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| Despliegue | % de canario | Detecta antes, afecta a más gente | Más seguro, más lento | 5 % | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| Privacidad | ε de privacidad diferencial | Más exactitud, menos privacidad | Más ruido, más privacidad | 1–10 | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| API | Información devuelta | Más útil, más fácil de atacar o copiar | Más difícil de atacar | Solo lo necesario | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |

---

## 14. Equivalencias entre librerías

### 14.1 Boosting: XGBoost / LightGBM / CatBoost / HistGradientBoosting ([6.4](../nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb))

| Idea | XGBoost | LightGBM | CatBoost | scikit-learn `HistGradientBoosting…` |
|---|---|---|---|---|
| Tasa de aprendizaje | `learning_rate` (= `eta`), def. 0.3 | `learning_rate`, def. 0.1 | `learning_rate`, automático | `learning_rate`, def. 0.1 |
| Número de árboles | `n_estimators`, def. 100 | `n_estimators`, def. 100 | `iterations`, def. 1000 | `max_iter`, def. 100 |
| Tamaño del árbol | `max_depth`, def. 6 | `num_leaves`, def. 31 (+ `max_depth`) | `depth`, def. 6 | `max_leaf_nodes`, def. 31 (+ `max_depth`) |
| Mínimo por hoja | `min_child_weight`, def. 1 | `min_child_samples`, def. 20 | `min_data_in_leaf`, def. 1 | `min_samples_leaf`, def. 20 |
| Fracción de filas por árbol | `subsample`, def. 1 | `subsample` (= `bagging_fraction`), def. 1 | `subsample` | — (no disponible) |
| Fracción de columnas | `colsample_bytree`, def. 1 | `colsample_bytree` (= `feature_fraction`), def. 1 | `rsm` | `max_features` (por división), def. 1.0 |
| Regularización L2 | `reg_lambda`, def. 1 | `reg_lambda`, def. 0 | `l2_leaf_reg`, def. 3 | `l2_regularization`, def. 0 |
| Regularización L1 | `reg_alpha`, def. 0 | `reg_alpha`, def. 0 | — | — |
| Ganancia mínima para dividir | `gamma`, def. 0 | `min_split_gain`, def. 0 | — | — |
| Parada temprana | `early_stopping_rounds` | callback `early_stopping` | `early_stopping_rounds` | `early_stopping`, `n_iter_no_change` (def. 10) |
| Crecimiento del árbol | Por niveles | Por hojas | Simétrico | Por hojas |
| Categorías | `enable_categorical` | Nativo | Nativo (su especialidad) | `categorical_features` |
| Brilla en… | Robustez, ecosistema | Velocidad con datos grandes | Muchas categorías, buenos valores por defecto | Cero instalación, pipelines de sklearn |

> En `GradientBoostingClassifier` (el clásico de sklearn, [6.3](../nivel-3-practicante/m06-ensambles/6.3-boosting-adaboost-y-gradient-boosting.ipynb)) el tamaño del árbol es `max_depth` (def. 3) y la parada temprana se activa con `n_iter_no_change`.

### 14.2 Redes: scikit-learn / Keras / PyTorch

La fila base (tasa, épocas, lote) es la tabla de la [Lección 3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb); el resto completa las perillas de [Módulo 9](../nivel-4-avanzado/m09-redes-neuronales/). El curso usa PyTorch; Keras aparece como referencia.

| Concepto | scikit-learn (`MLPClassifier`, `SGD…`) | Keras | PyTorch | Lección |
|---|---|---|---|---|
| Tasa de aprendizaje | `learning_rate_init` (MLP) · `eta0` (SGD) | `optimizer=Adam(learning_rate=...)` | `torch.optim.Adam(params, lr=...)` | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb), [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Épocas | `max_iter` (en MLP/SGD cuenta **épocas**) | `model.fit(..., epochs=...)` | Tu bucle `for epoca in range(...)` | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Tamaño de lote | `batch_size` (MLP) | `model.fit(..., batch_size=...)` | `DataLoader(..., batch_size=...)` | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| Barajar | `shuffle=True` | `model.fit(..., shuffle=True)` | `DataLoader(..., shuffle=True)` | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| Capas y neuronas | `hidden_layer_sizes=(64, 64)` | `layers.Dense(64)` apiladas | `nn.Linear(entradas, 64)` apiladas | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb) |
| Activación | `activation="relu"` | `Dense(64, activation="relu")` | `nn.ReLU()` | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| Optimizador | `solver="adam"` / `"sgd"` | `keras.optimizers.Adam` / `SGD` | `torch.optim.Adam` / `SGD` | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Momento | `momentum=0.9` (solver `"sgd"`) · `beta_1` (adam) | `SGD(momentum=0.9)` · `Adam(beta_1=...)` | `SGD(..., momentum=0.9)` · `Adam(..., betas=(0.9, 0.999))` | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| Regularización L2 | `alpha` | `kernel_regularizer=regularizers.l2(...)` o `AdamW(weight_decay=...)` | `torch.optim.AdamW(..., weight_decay=...)` | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Dropout | — (no disponible) | `layers.Dropout(rate)` | `nn.Dropout(p)` | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Batch normalization | — (no disponible) | `layers.BatchNormalization(momentum=0.99)` | `nn.BatchNorm1d/2d(momentum=0.1)` | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Early stopping | `early_stopping=True`, `n_iter_no_change`, `validation_fraction` | `callbacks.EarlyStopping(patience=..., restore_best_weights=True)` | Bucle propio con paciencia + `state_dict()` | [7.2](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.2-early-stopping.ipynb), [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| Recorte de gradiente | — | `Adam(clipnorm=1.0)` | `torch.nn.utils.clip_grad_norm_(params, 1.0)` | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| *Scheduler* | `learning_rate="adaptive"` (solo SGD) | `callbacks.LearningRateScheduler` / `optimizers.schedules` | `torch.optim.lr_scheduler` (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`) | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| Inicialización | — (automática) | `kernel_initializer="he_normal"` | `nn.init.kaiming_normal_` / `xavier_normal_` | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| Modo entrenamiento / evaluación | — (automático) | Automático en `fit` / `predict` | `red.train()` / `red.eval()` + `torch.no_grad()` | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |

> ⚠️ El `momentum` de BatchNorm tiene **sentidos opuestos**: en Keras es cuánto se **conserva** del promedio viejo; en PyTorch, cuánto se **toma** del lote nuevo. Keras `0.99` equivale a PyTorch `0.01`, y el valor por defecto de PyTorch (`0.1`) equivale a Keras `0.9`.

---

## 15. ¿Qué ajusto primero?

De la [Lección 7.4](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.4-priorizar-hiperparametros.ipynb): ajusta primero las perillas 🎯, luego las 🔧 si hay presupuesto, y **fija** las 📌.

| Modelo | 🎯 Ajusta primero | 🔧 Luego, si hay presupuesto | 📌 No lo busques: fíjalo |
|---|---|---|---|
| Regresión logística / lineal | `C` o `alpha` | `l1_ratio`, `class_weight` | `max_iter` (súbelo hasta que converja) |
| KNN | `n_neighbors` | `weights`, `metric` | — |
| SVM | `C`, `gamma` | `kernel` | — |
| Árbol de decisión | `max_depth` o `min_samples_leaf` (o `ccp_alpha`) | `max_features` | `criterion` |
| Random Forest | `max_features`, `min_samples_leaf` | `max_depth` | `n_estimators` (200–500), `criterion` |
| Gradient boosting | `learning_rate` (+ early stopping para el nº de árboles) | `max_depth`/`max_leaf_nodes`, `min_samples_leaf` | `subsample`, `colsample`, `l2_regularization`, `max_bins` |
| Red neuronal (MLP) | `learning_rate_init`, `alpha` | `hidden_layer_sizes`, `batch_size` | `max_iter` (con early stopping) |

> Para redes en PyTorch, el recetario de la [Lección 9.9](../nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb) recomienda empezar con **Adam, `lr = 1e-3` y una red pequeña**, y complicar solo cuando lo simple funcione.
