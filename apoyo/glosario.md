# 📖 Glosario ES ↔ EN

Todos los términos clave del curso, en español y en inglés, con una definición de una línea. La última columna enlaza la **primera lección en la que aparece** el término (a veces como adelanto; la explicación a fondo puede llegar más tarde).

> 💡 La documentación, los foros y las librerías están en inglés. Aprende las dos versiones: buscar "*early stopping* scikit-learn" te dará muchas más respuestas que "parada temprana".

Material de apoyo: [Chuleta de hiperparámetros](chuleta-hiperparametros.md) · [¿Qué modelo uso?](arbol-de-decision-modelos.md) · [Diagnóstico del entrenamiento](diagnostico-entrenamiento.md) · [Recursos](recursos.md) · [Volver al README](../README.md)

## Índice

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w)

## A

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Acción** | *action* | En refuerzo, lo que el agente elige hacer en cada estado. | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| **AdaBoost** | *Adaptive Boosting* | Boosting que da más peso, en cada ronda, a los ejemplos que la ronda anterior clasificó mal. | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| **Adam** | *Adaptive Moment Estimation* | Optimizador que combina momento y una tasa adaptativa por parámetro; el punto de partida habitual en redes. | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb) |
| **AdamW** | *AdamW* | Variante de Adam que aplica el weight decay por separado; la forma correcta de regularizar con Adam. | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| **Agente (LLM)** | *LLM agent* | LLM que, en un bucle, decide qué herramienta usar, observa el resultado y continúa hasta terminar la tarea. | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| **Agente (refuerzo)** | *agent* | Quien observa el estado del entorno, elige acciones y recibe recompensas. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Agrupamiento** | *clustering* | Encontrar grupos de ejemplos parecidos sin usar etiquetas. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Ajuste fino** | *fine-tuning* | Seguir entrenando un modelo preentrenado con los datos de tu tarea. | [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb) |
| **Ajuste por instrucciones** | *instruction tuning / SFT* | Ajuste fino de un LLM base con pares instrucción-respuesta para que siga órdenes. | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| **Aleatorización de dominio** | *domain randomization* | Variar al azar el simulador (luz, fricción…) para que el agente funcione en el mundo real. | [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |
| **Alucinación** | *hallucination* | Respuesta de un LLM que suena plausible y segura, pero es falsa. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Aprendizaje auto-supervisado** | *self-supervised learning* | El propio dato fabrica la etiqueta (p. ej., predecir la palabra siguiente). | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Aprendizaje automático** | *Machine Learning (ML)* | Programas que aprenden patrones a partir de ejemplos en lugar de seguir reglas escritas a mano. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Aprendizaje en línea** | *online learning* | Actualizar el modelo ejemplo a ejemplo, a medida que llegan los datos. | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| **Aprendizaje no supervisado** | *unsupervised learning* | Buscar estructura (grupos, direcciones, rarezas) en datos sin etiquetas. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Aprendizaje perezoso** | *lazy learning* | No entrenar nada: guardar los datos y hacer todo el trabajo al predecir (KNN). | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |
| **Aprendizaje por refuerzo** | *reinforcement learning (RL)* | Aprender a decidir por prueba y error a partir de recompensas. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Aprendizaje profundo** | *Deep Learning (DL)* | Machine Learning con redes neuronales de muchas capas. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Aprendizaje semi-supervisado** | *semi-supervised learning* | Aprender con pocas etiquetas y muchos datos sin etiquetar. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Aprendizaje supervisado** | *supervised learning* | Aprender de ejemplos que traen la respuesta correcta (etiqueta). | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Árbol de decisión** | *decision tree* | Modelo que predice con una secuencia de preguntas sí/no sobre las características. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Arquitectura residual** | *residual connection / ResNet* | Atajo que suma la entrada de un bloque a su salida (x + F(x)); permite entrenar redes muy profundas. | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| **Ataque adversario** | *adversarial attack* | Cambio pequeño y deliberado en la entrada que engaña al modelo. | [15.4](../nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb) |
| **Atención** | *attention* | Mecanismo que pondera qué partes de la entrada mirar para producir cada salida. | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| **Atención cruzada** | *cross-attention* | El decodificador consulta las salidas del codificador. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Atención multi-cabeza** | *multi-head attention* | Varias atenciones en paralelo, cada una especializada en un tipo de relación. | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| **AUC** | *Area Under the Curve* | Área bajo la curva ROC: probabilidad de que el modelo puntúe a un positivo por encima de un negativo. | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| **Aumento de datos** | *data augmentation* | Crear variaciones realistas de los ejemplos (girar, recortar…) para regularizar. | [9.8](../nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb) |
| **Auto-atención** | *self-attention* | Cada token consulta a todos los de su misma secuencia mediante consultas, claves y valores. | [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb) |
| **Autoencoder** | *autoencoder* | Red que comprime la entrada a un código pequeño y la reconstruye. | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| **Autojuego** | *self-play* | El agente aprende jugando contra versiones de sí mismo. | [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |

[↑ Índice](#índice)

## B

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Bagging** | *bootstrap aggregating* | Entrenar modelos sobre remuestreos con reemplazo y promediar sus predicciones. | [4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb) |
| **Bandido multibrazo** | *multi-armed bandit* | Elegir repetidamente entre opciones cuyo premio no conoces de antemano. | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| **Barrido de la tasa de aprendizaje** | *LR range test* | Entrenar probando tasas en escala logarítmica para encontrar la región buena. | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| **Basura entra, basura sale** | *garbage in, garbage out* | Con datos malos, ningún modelo da buenos resultados. | [1.1](../nivel-1-cero/m01-que-es-ml/1.1-programacion-tradicional-vs-aprendizaje.ipynb) |
| **Bolsa de palabras** | *bag of words* | Representar un texto por el conteo de sus palabras, ignorando el orden. | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb) |
| **Boosting** | *boosting* | Ensamble secuencial: cada modelo nuevo corrige los errores de los anteriores. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Brecha simulador-realidad** | *sim-to-real gap* | Diferencia entre el simulador donde se entrenó el agente y el mundo real. | [14.5](../nivel-5-heroe/m14-refuerzo/14.5-casos-famosos-alphago-robotica-rlhf.ipynb) |
| **Búsqueda aleatoria** | *random search* | Probar combinaciones de hiperparámetros al azar; suele ganarle a la cuadrícula con el mismo presupuesto. | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| **Búsqueda bayesiana** | *Bayesian optimization* | Elegir el siguiente intento según un modelo de los resultados anteriores (p. ej. Optuna). | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| **Búsqueda de haz** | *beam search* | Generar manteniendo las k secuencias parciales más probables. | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |
| **Búsqueda en cuadrícula** | *grid search* | Probar todas las combinaciones de una lista de valores de hiperparámetros. | [4.6](../nivel-2-aprendiz/m04-evaluacion/4.6-validacion-cruzada.ipynb) |

[↑ Índice](#índice)

## C

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Calendario de ruido** | *noise schedule* | Cómo se reparte el ruido entre los pasos de un modelo de difusión. | [13.4](../nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) |
| **Calibración** | *calibration* | Que "70 % de probabilidad" acierte de verdad unas 7 de cada 10 veces. | [4.3](../nivel-2-aprendiz/m04-evaluacion/4.3-umbrales-roc-y-precision-recall.ipynb) |
| **Campeón vs. retador** | *champion/challenger* | Comparar el modelo nuevo con el vigente, en datos recientes, antes de reemplazarlo. | [15.3](../nivel-5-heroe/m15-produccion-y-etica/15.3-reentrenamiento-y-versionado.ipynb) |
| **Campo receptivo** | *receptive field* | Región de la imagen original que "ve" una neurona de una capa convolucional. | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| **Canario** | *canary release* | Mandar una pequeña parte del tráfico al modelo nuevo antes de desplegarlo del todo. | [15.1](../nivel-5-heroe/m15-produccion-y-etica/15.1-del-notebook-a-produccion.ipynb) |
| **Capa oculta** | *hidden layer* | Capa de neuronas entre la entrada y la salida de una red. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Característica** | *feature* | Columna o dato de entrada que el modelo usa para predecir. | [0.4](../nivel-1-cero/m00-bienvenida/0.4-kit-minimo-python-jupyter-colab.ipynb) |
| **Cargas** | *loadings* | Peso de cada columna original dentro de un componente principal. | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| **Clasificación** | *classification* | Predecir una categoría (¿spam o no?, ¿qué dígito?). | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Codificación one-hot** | *one-hot encoding* | Convertir una categoría en varias columnas 0/1, una por valor. | [1.4](../nivel-1-cero/m01-que-es-ml/1.4-tipos-de-problema.ipynb) |
| **Codificación ordinal** | *ordinal encoding* | Convertir categorías con orden en números que respetan ese orden. | [1.4](../nivel-1-cero/m01-que-es-ml/1.4-tipos-de-problema.ipynb) |
| **Codificación por objetivo** | *target encoding* | Sustituir cada categoría por el promedio de la etiqueta, calculado sin fugas. | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| **Codificación posicional** | *positional encoding* | Información de posición que se suma a los embeddings para que el Transformer sepa el orden. | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| **Codificador / decodificador** | *encoder / decoder* | Parte que resume la entrada / parte que genera la salida. | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| **Coeficiente de determinación** | *R² (R-squared)* | Fracción de la variación explicada: 0 = como predecir el promedio, 1 = perfecto. | [3.2](../nivel-2-aprendiz/m03-como-aprende/3.2-funcion-de-perdida.ipynb) |
| **Colapso de modos** | *mode collapse* | La GAN genera siempre unas pocas variantes parecidas. | [13.3](../nivel-5-heroe/m13-generativos-llms/13.3-gans.ipynb) |
| **Componente principal** | *principal component* | Dirección de máxima varianza encontrada por PCA. | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| **Conjunto de entrenamiento** | *training set* | Datos con los que el modelo ajusta sus parámetros. | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| **Conjunto de prueba** | *test set* | Datos reservados para medir el modelo final una sola vez. | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| **Conjunto de validación** | *validation set* | Datos para elegir modelo e hiperparámetros sin tocar la prueba. | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| **Contaminación** | *contamination* | Fracción esperada de anomalías; fija el umbral de un detector como Isolation Forest. | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| **Contrafactual** | *counterfactual explanation* | "Si tu ingreso fuera X, el crédito se aprobaría": el cambio mínimo que altera la decisión. | [15.6](../nivel-5-heroe/m15-produccion-y-etica/15.6-explicabilidad-y-regulacion.ipynb) |
| **Convolución** | *convolution* | Pasar un filtro pequeño por toda la imagen para detectar un patrón local. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Costo esperado** | *expected cost* | Suma de cada tipo de error por lo que cuesta; la métrica de negocio por excelencia. | [4.8](../nivel-2-aprendiz/m04-evaluacion/4.8-metrica-segun-costo.ipynb) |
| **Cuantización** | *quantization* | Guardar los pesos con menos bits (8, 4) para ahorrar memoria. | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| **Cuello de botella** | *bottleneck* | Representación estrecha por la que debe pasar toda la información (seq2seq, autoencoder). | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Curva de aprendizaje** | *learning curve* | Pérdida o error frente a las épocas (o frente a la cantidad de datos). | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Curva precisión-recall** | *precision-recall curve* | Precisión frente a recall para todos los umbrales; más informativa que la ROC con positivos raros. | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| **Curva ROC** | *ROC curve (Receiver Operating Characteristic)* | Tasa de verdaderos positivos frente a falsos positivos para todos los umbrales. | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |

[↑ Índice](#índice)

## D

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Datos desbalanceados** | *imbalanced data* | Una clase es mucho más rara que la otra (fraude, enfermedad). | [2.6](../nivel-1-cero/m02-datos/2.6-fuga-de-datos.ipynb) |
| **Datos faltantes** | *missing values* | Celdas vacías (NaN) que hay que imputar o tratar. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **Datos tabulares** | *tabular data* | Datos en filas y columnas con significado (clientes, ventas, créditos). | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **DBSCAN** | *Density-Based Spatial Clustering of Applications with Noise* | Agrupamiento por densidad que encuentra formas arbitrarias y marca ruido. | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| **Decodificación** | *decoding* | Regla para elegir cada token al generar texto (codiciosa, muestreo, haz…). | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| **Decodificación codiciosa (voraz)** | *greedy decoding* | Elegir siempre la opción más probable en cada paso. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Dendrograma** | *dendrogram* | Árbol que muestra cómo se van fusionando los grupos en el clustering jerárquico. | [8.3](../nivel-3-practicante/m08-no-supervisado/8.3-jerarquico-y-dbscan.ipynb) |
| **Deriva de concepto** | *concept drift* | Cambia la relación entre entradas y respuesta: el mismo cliente ahora se comporta distinto. | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| **Deriva de datos** | *data drift* | Cambia la distribución de las entradas respecto a la de entrenamiento. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **Descenso de gradiente** | *gradient descent* | Ajustar las perillas dando pasos en la dirección en que más baja la pérdida. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |
| **Descenso de gradiente estocástico** | *stochastic gradient descent (SGD)* | Descenso de gradiente que usa un ejemplo (o un lote pequeño) por paso. | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| **Detección de anomalías** | *anomaly detection* | Encontrar ejemplos raros respecto a lo normal (fraude, fallas). | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Detección de novedades** | *novelty detection* | Entrenar solo con datos normales y vigilar los nuevos. | [8.6](../nivel-3-practicante/m08-no-supervisado/8.6-deteccion-de-anomalias.ipynb) |
| **Detección de objetos** | *object detection* | Encontrar qué objetos hay en una imagen y dónde (cajas). | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| **Difusión (modelo de)** | *diffusion model* | Generador que aprende a quitar ruido paso a paso, partiendo de ruido puro. | [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb) |
| **Discriminativo / generativo** | *discriminative / generative model* | Aprender a decidir p(y\|x) / aprender cómo son los datos p(x) para crear nuevos. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Divergencia KL** | *KL divergence* | Medida de cuánto se aleja una distribución de otra (en VAE y en RLHF). | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| **DPO** | *Direct Preference Optimization* | Ajustar un LLM directamente con pares de respuestas preferidas, sin modelo de recompensa aparte. | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| **DQN** | *Deep Q-Network* | Q-learning en el que una red neuronal estima los valores Q. | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| **Dropout** | *dropout* | Apagar neuronas al azar durante el entrenamiento para regularizar. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |

[↑ Índice](#índice)

## E

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Early stopping** | *early stopping (parada temprana)* | Dejar de entrenar cuando la validación deja de mejorar y quedarse con el mejor punto. | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| **Elastic Net** | *Elastic Net* | Regularización que mezcla L1 y L2 (`l1_ratio`). | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| **Embedding** | *embedding* | Vector de números que representa un token, objeto o usuario de forma que lo parecido quede cerca. | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb) |
| **Ensamble** | *ensemble* | Combinar varios modelos para que sus errores se compensen. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Entropía cruzada** | *cross-entropy (log loss)* | Pérdida de clasificación que castiga mucho la confianza en la respuesta equivocada. | [3.2](../nivel-2-aprendiz/m03-como-aprende/3.2-funcion-de-perdida.ipynb) |
| **Envenenamiento de datos** | *data poisoning* | Meter datos maliciosos en el entrenamiento para alterar el modelo. | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| **Época** | *epoch* | Una pasada completa por todos los datos de entrenamiento. | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |
| **Épsilon-codicioso** | *ε-greedy* | Explorar al azar con probabilidad ε y explotar lo mejor conocido el resto del tiempo. | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| **Equidad** | *fairness* | Que el modelo no perjudique sistemáticamente a ciertos grupos. | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| **Escalado** | *feature scaling / standardization* | Poner las columnas numéricas en escalas comparables (StandardScaler, MinMax…). | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Espacio latente** | *latent space* | Espacio de los códigos internos de un modelo generativo; puntos cercanos generan cosas parecidas. | [13.2](../nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb) |
| **Especificidad** | *specificity* | De los negativos reales, qué fracción el modelo identifica como negativos. | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| **Estado** | *state* | En refuerzo, la situación actual del entorno que observa el agente. | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| **Estado oculto** | *hidden state* | Memoria que una red recurrente lleva de un paso al siguiente. | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| **Estratificación** | *stratification* | Dividir los datos manteniendo la proporción de clases en cada parte. | [2.5](../nivel-1-cero/m02-datos/2.5-division-de-datos.ipynb) |
| **Exactitud** | *accuracy* | Fracción de predicciones correctas; engaña con clases desbalanceadas. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **Explicabilidad** | *explainability* | Poder justificar por qué el modelo tomó una decisión concreta. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Exploración vs. explotación** | *exploration vs. exploitation* | Equilibrio entre probar opciones nuevas y repetir la mejor conocida. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Exposure bias** | *exposure bias* | Al generar, el modelo se encuentra con sus propios errores, que nunca vio al entrenar con teacher forcing. | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |
| **Extrapolar** | *extrapolation* | Predecir fuera del rango de los datos vistos; árboles y KNN no saben hacerlo. | [1.1](../nivel-1-cero/m01-que-es-ml/1.1-programacion-tradicional-vs-aprendizaje.ipynb) |

[↑ Índice](#índice)

## F

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **F1** | *F1-score* | Media armónica de precisión y recall. | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| **Factorización de matrices** | *matrix factorization* | Explicar la tabla usuario × producto con unos pocos "gustos ocultos" (factores latentes). | [8.7](../nivel-3-practicante/m08-no-supervisado/8.7-sistemas-de-recomendacion.ipynb) |
| **Falso positivo / falso negativo** | *false positive / false negative* | Alarma sin motivo / caso real que se escapa. | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| **Fecha de corte** | *knowledge cutoff* | Fecha hasta la que llegan los datos de entrenamiento de un LLM. | [13.9](../nivel-5-heroe/m13-generativos-llms/13.9-alucinaciones-y-limites.ipynb) |
| **Few-shot** | *few-shot prompting* | Poner unos pocos ejemplos resueltos dentro del prompt. | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| **Ficha del modelo** | *model card* | Documento con qué hace el modelo, con qué datos se entrenó, cómo rinde y sus límites. | [15.6](../nivel-5-heroe/m15-produccion-y-etica/15.6-explicabilidad-y-regulacion.ipynb) |
| **Filtrado colaborativo** | *collaborative filtering* | Recomendar según lo que gustó a usuarios parecidos. | [5.3](../nivel-2-aprendiz/m05-modelos-clasicos/5.3-k-vecinos-mas-cercanos.ipynb) |
| **Filtro (kernel)** | *filter / kernel* | Pequeña matriz de pesos que una convolución pasa por la imagen. | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| **Frontera de decisión** | *decision boundary* | Línea o superficie donde el modelo cambia de una clase a otra. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Fuga de datos** | *data leakage* | Información que no tendrás al predecir se cuela en el entrenamiento o la evaluación. | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |
| **Función de activación** | *activation function* | Función no lineal que aplica cada neurona a su suma ponderada (ReLU, sigmoide, tanh…). | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Función de pérdida** | *loss function* | Número que mide cuánto se equivoca el modelo; el entrenamiento intenta bajarlo. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |

[↑ Índice](#índice)

## G

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **GAN** | *Generative Adversarial Network* | Un generador y un discriminador que compiten: falsificador contra detective. | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb) |
| **Gradient boosting** | *gradient boosting* | Boosting en el que cada árbol nuevo predice los errores (residuos) de los anteriores. | [6.1](../nivel-3-practicante/m06-ensambles/6.1-por-que-combinar-modelos.ipynb) |
| **Gradiente** | *gradient* | Dirección e intensidad en que más sube la pérdida; se camina en sentido contrario. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |
| **Gradiente que desaparece** | *vanishing gradient* | La señal de aprendizaje se encoge capa tras capa y las primeras capas no aprenden. | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb) |
| **Gradiente que explota** | *exploding gradient* | La señal de aprendizaje crece capa tras capa hasta dar NaN. | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |
| **Gradientes de política** | *policy gradient* | Familia de RL que ajusta directamente la política hacia lo que dio más recompensa. | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| **GRU** | *Gated Recurrent Unit* | Red recurrente con dos puertas; más ligera que la LSTM. | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |

[↑ Índice](#índice)

## H

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Hiperparámetro** | *hyperparameter* | Perilla que eliges tú antes de entrenar (profundidad, tasa de aprendizaje…). | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Horizonte** | *forecast horizon* | Cuántos pasos hacia adelante se pronostica. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |

[↑ Índice](#índice)

## I

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **IA generativa** | *Generative AI* | Modelos que crean contenido nuevo: texto, imágenes, audio. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Impacto dispar** | *disparate impact* | Cociente entre las tasas de decisión favorable de dos grupos. | [15.4](../nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb) |
| **Importancia por permutación** | *permutation importance* | Cuánto empeora el modelo si barajas una columna; mide en qué se apoya. | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| **Imputación** | *imputation* | Rellenar los datos faltantes (media, mediana, más frecuente…). | [2.2](../nivel-1-cero/m02-datos/2.2-calidad-de-datos.ipynb) |
| **Inercia** | *inertia* | Suma de distancias al cuadrado de cada punto a su centroide en K-means. | [8.2](../nivel-3-practicante/m08-no-supervisado/8.2-k-means.ipynb) |
| **Inferencia de pertenencia** | *membership inference* | Ataque que adivina si un registro estuvo en los datos de entrenamiento. | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| **Ingeniería de características** | *feature engineering* | Crear columnas nuevas y útiles a partir de las que tienes. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Inicialización de pesos** | *weight initialization (He, Xavier)* | Elegir los pesos iniciales con la escala justa para que la señal no se apague ni explote. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Intercepto** | *intercept (bias)* | Valor que predice un modelo lineal cuando todas las entradas valen 0. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |
| **Interpretabilidad** | *interpretability* | Entender qué aprendió el modelo y en qué se apoya. | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| **Inyección de instrucciones** | *prompt injection* | Texto externo que da órdenes al LLM para desviarlo de su tarea. | [13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb) |
| **IoU** | *Intersection over Union* | Solapamiento entre una caja predicha y la real (0 a 1). | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| **Isolation Forest** | *Isolation Forest* | Detector de anomalías: lo raro se aísla con pocos cortes al azar. | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| **Iteración** | *iteration / step* | Una actualización de los parámetros (un lote procesado). | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb) |

[↑ Índice](#índice)

## K

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **K-means** | *K-means* | Agrupamiento que reparte los puntos en k grupos alrededor de centroides. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Kernel (SVM)** | *kernel trick* | Truco que permite a una SVM trazar fronteras curvas midiendo similitudes. | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| **KNN** | *k-nearest neighbors* | Predice copiando a los k ejemplos más parecidos. | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |

[↑ Índice](#índice)

## L

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Lasso / Ridge** | *L1 / L2 regularization* | Castigo a los coeficientes: L1 pone algunos en cero, L2 los encoge a todos. | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |
| **Leyes de escalamiento** | *scaling laws* | Relación predecible entre pérdida, tamaño del modelo, datos y cómputo. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Línea base** | *baseline* | Predicción tonta o regla actual contra la que se compara el modelo. | [0.4](../nivel-1-cero/m00-bienvenida/0.4-kit-minimo-python-jupyter-colab.ipynb) |
| **LLM** | *Large Language Model* | Transformer enorme entrenado para predecir el siguiente token sobre muchísimo texto. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Logits** | *logits* | Puntuaciones crudas de la red antes de la softmax o la sigmoide. | [9.2](../nivel-4-avanzado/m09-redes-neuronales/9.2-capas-y-profundidad.ipynb) |
| **LoRA** | *Low-Rank Adaptation* | Ajuste fino que entrena solo dos matrices pequeñas añadidas a los pesos congelados. | [9.5](../nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb) |
| **Lote** | *batch* | Grupo de ejemplos que se procesa en cada paso de entrenamiento. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |
| **LSTM** | *Long Short-Term Memory* | Red recurrente con puertas (olvido, entrada, salida) que recuerda a largo plazo. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |

[↑ Índice](#índice)

## M

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **MAE** | *Mean Absolute Error* | Error absoluto medio: cuánto te equivocas en promedio, en las unidades del problema. | [1.4](../nivel-1-cero/m01-que-es-ml/1.4-tipos-de-problema.ipynb) |
| **Maldición de la dimensionalidad** | *curse of dimensionality* | Con muchas columnas los datos quedan dispersos y las distancias pierden sentido. | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| **Mapa de características** | *feature map* | Salida de un filtro convolucional: dónde aparece su patrón en la imagen. | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| **MAPE** | *Mean Absolute Percentage Error* | Error porcentual medio; explota cuando los valores reales están cerca de 0. | [4.1](../nivel-2-aprendiz/m04-evaluacion/4.1-metricas-de-regresion.ipynb) |
| **Margen** | *margin* | Ancho de la "calle" entre clases que una SVM intenta maximizar. | [5.5](../nivel-2-aprendiz/m05-modelos-clasicos/5.5-maquinas-de-vectores-de-soporte.ipynb) |
| **Máscara causal** | *causal mask* | Impide que cada token mire a los siguientes; necesaria para generar texto. | [12.2](../nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb) |
| **Matriz de confusión** | *confusion matrix* | Tabla de aciertos y errores por clase (VP, FP, FN, VN). | [4.2](../nivel-2-aprendiz/m04-evaluacion/4.2-metricas-de-clasificacion.ipynb) |
| **Memoria de repetición** | *experience replay* | Almacén de experiencias pasadas del que DQN muestrea al azar para entrenar. | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| **Meseta** | *plateau* | Zona casi plana de la pérdida donde el entrenamiento avanza muy despacio. | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb) |
| **Métrica** | *metric* | Número con el que se juzga el modelo (MAE, recall, AUC…); no tiene por qué ser la pérdida. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Mini-batch** | *mini-batch gradient descent* | Descenso de gradiente con lotes pequeños (32–256): el término medio habitual. | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |
| **Mínimo local** | *local minimum* | Valle de la pérdida que no es el más profundo. | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb) |
| **MLOps** | *Machine Learning Operations* | Prácticas para desplegar, vigilar y mantener modelos en producción. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **Modelo** | *model* | Función aprendida a partir de datos que convierte entradas en predicciones. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Modelo base** | *base model* | LLM solo pre-entrenado: continúa textos, todavía no sigue instrucciones. | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| **Momento** | *momentum* | Inercia en el optimizador: acumula la dirección de los pasos anteriores. | [3.3](../nivel-2-aprendiz/m03-como-aprende/3.3-descenso-de-gradiente.ipynb) |
| **Monitoreo** | *monitoring* | Vigilar entradas, predicciones y desempeño del modelo en producción. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **MSE / RMSE** | *Mean Squared Error / Root MSE* | Error cuadrático medio (y su raíz, en unidades del problema); castiga más los errores grandes. | [3.2](../nivel-2-aprendiz/m03-como-aprende/3.2-funcion-de-perdida.ipynb) |
| **Muestreo (generación)** | *sampling* | Elegir el siguiente token al azar según sus probabilidades. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Multicolinealidad** | *multicollinearity* | Columnas muy correlacionadas entre sí que vuelven inestables los coeficientes. | [5.1](../nivel-2-aprendiz/m05-modelos-clasicos/5.1-regresion-lineal.ipynb) |

[↑ Índice](#índice)

## N

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Naive Bayes** | *Naive Bayes* | Clasificador probabilístico que supone que las pistas son independientes entre sí. | [1.1](../nivel-1-cero/m01-que-es-ml/1.1-programacion-tradicional-vs-aprendizaje.ipynb) |
| **Neurona artificial** | *artificial neuron (perceptron)* | Suma ponderada de entradas más un sesgo, seguida de una activación. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **NMS** | *Non-Maximum Suppression* | Eliminar cajas duplicadas que se solapan sobre el mismo objeto. | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| **No hay almuerzo gratis** | *No Free Lunch theorem* | Ningún modelo es el mejor para todos los problemas. | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Normalización de capa** | *layer normalization (LayerNorm)* | Re-normaliza las activaciones de cada token; pieza estándar del bloque Transformer. | [12.3](../nivel-4-avanzado/m12-transformers/12.3-multicabeza-posicion-bloque.ipynb) |
| **Normalización por lotes** | *batch normalization (BatchNorm)* | Re-normaliza las activaciones de cada capa usando la media y desviación del lote. | [9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb) |

[↑ Índice](#índice)

## O

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Olvido catastrófico** | *catastrophic forgetting* | Al ajustar con datos nuevos, el modelo pierde lo que sabía. | [10.5](../nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) |
| **OOB (fuera de bolsa)** | *out-of-bag* | Ejemplos que un árbol del bosque no vio; sirven de validación gratuita. | [6.2](../nivel-3-practicante/m06-ensambles/6.2-bagging-y-random-forest.ipynb) |
| **Optimismo por selección** | *selection bias in tuning* | Tras probar muchas configuraciones, la mejor puntuación de validación es optimista. | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| **Optimizador** | *optimizer* | Algoritmo que decide cómo mover los parámetros con el gradiente (SGD, Adam…). | [3.4](../nivel-2-aprendiz/m03-como-aprende/3.4-hiperparametros-del-entrenamiento.ipynb) |

[↑ Índice](#índice)

## P

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Padding (relleno)** | *padding* | Marco de ceros alrededor de la imagen para que la convolución no la encoja. | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| **Parámetro** | *parameter* | Valor que el modelo aprende de los datos (pesos, cortes de un árbol). | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **PCA** | *Principal Component Analysis* | Reducción de dimensionalidad que conserva las direcciones de máxima varianza. | [2.8](../nivel-1-cero/m02-datos/2.8-maldicion-de-la-dimensionalidad.ipynb) |
| **PDP / ICE** | *partial dependence / Individual Conditional Expectation* | Cómo cambia la predicción al mover una variable: en promedio (PDP) o caso a caso (ICE). | [6.6](../nivel-3-practicante/m06-ensambles/6.6-por-que-los-arboles-dominan-lo-tabular.ipynb) |
| **Penalización de repetición** | *repetition penalty* | Castigo a los tokens ya usados para evitar bucles al generar. | [13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb) |
| **Pérdida de entrenamiento / validación** | *training / validation loss* | La misma pérdida medida en los datos que el modelo ve y en los que no ve. | [3.7](../nivel-2-aprendiz/m03-como-aprende/3.7-curvas-de-aprendizaje.ipynb) |
| **Perplejidad** | *perplexity* | En LLMs, exp(pérdida): "entre cuántos tokens duda"; en t-SNE, ≈ cuántos vecinos mira cada punto. | [8.5](../nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| **Persistencia (línea base ingenua)** | *naive forecast* | Pronosticar que mañana será igual que hoy. | [4.7](../nivel-2-aprendiz/m04-evaluacion/4.7-lineas-base.ipynb) |
| **Pipeline** | *pipeline* | Cadena de pasos (preparar → modelo) que se ajusta y valida como una sola pieza. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Pliegue** | *fold* | Cada una de las partes en que se divide el conjunto en la validación cruzada. | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| **Política** | *policy* | En refuerzo, la regla que dice qué acción tomar en cada estado. | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| **Pooling** | *pooling* | Resumir cada cuadrito de un mapa (máximo o promedio) para reducir tamaño y ganar tolerancia. | [10.1](../nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb) |
| **PPO** | *Proximal Policy Optimization* | Gradiente de política con "recorte" que impide cambios demasiado bruscos. | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| **Pre-entrenamiento** | *pre-training* | Primera fase de un modelo grande: aprender de muchísimos datos generales. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Precisión** | *precision* | De lo que el modelo marcó como positivo, qué fracción lo era de verdad. | [1.4](../nivel-1-cero/m01-que-es-ml/1.4-tipos-de-problema.ipynb) |
| **Privacidad diferencial** | *differential privacy* | Añadir ruido calibrado para que el resultado no revele a ningún individuo. | [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb) |
| **Prompt** | *prompt* | Texto de entrada que se le da a un LLM. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Proxy (variable sustituta)** | *proxy* | Variable que "delata" otra (p. ej., el código postal a la etnia) o etiqueta indirecta. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **PSI** | *Population Stability Index* | Índice que mide cuánto cambió la distribución de una variable. | [15.2](../nivel-5-heroe/m15-produccion-y-etica/15.2-monitoreo-deriva-de-datos-y-concepto.ipynb) |
| **Puerta (LSTM/GRU)** | *gate* | Interruptor aprendido (0 a 1) que decide qué recordar, olvidar o dejar salir. | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| **Punto de silla** | *saddle point* | Punto plano que es mínimo en unas direcciones y máximo en otras. | [3.6](../nivel-2-aprendiz/m03-como-aprende/3.6-minimos-locales-y-mesetas.ipynb) |

[↑ Índice](#índice)

## Q

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Q-learning** | *Q-learning* | Aprender el valor de cada acción en cada estado a partir de la experiencia. | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |

[↑ Índice](#índice)

## R

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **RAG** | *Retrieval-Augmented Generation* | Buscar documentos relevantes y dárselos al LLM para que responda con ellos. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Random Forest** | *Random Forest* | Bagging de árboles que además sortea columnas en cada división. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Recall** | *recall* (en medicina, *sensitivity* o sensibilidad) | De los positivos reales, qué fracción detecta el modelo. | [1.4](../nivel-1-cero/m01-que-es-ml/1.4-tipos-de-problema.ipynb) |
| **Recompensa** | *reward* | Número que el entorno da al agente tras cada acción. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Recorte de gradiente** | *gradient clipping* | Limitar el tamaño del gradiente para evitar que explote. | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| **Red convolucional (CNN)** | *convolutional neural network* | Red que aplica filtros compartidos sobre la imagen; reconoce patrones donde estén. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Red neuronal** | *neural network* | Capas de neuronas conectadas que aprenden funciones complejas. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Red objetivo** | *target network* | Copia congelada de la red Q que DQN actualiza cada cierto tiempo para estabilizar. | [14.3](../nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb) |
| **Red recurrente (RNN)** | *recurrent neural network* | Red que procesa una secuencia paso a paso, llevando un estado oculto. | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| **Reentrenamiento** | *retraining* | Volver a entrenar el modelo con datos recientes (periódico o por alarma). | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Registro de modelos** | *model registry* | Catálogo de versiones de modelos con sus datos, métricas y estado. | [15.1](../nivel-5-heroe/m15-produccion-y-etica/15.1-del-notebook-a-produccion.ipynb) |
| **Regresión** | *regression* | Predecir un número (precio, demanda, minutos). | [0.2](../nivel-1-cero/m00-bienvenida/0.2-ia-ml-dl-ia-generativa.ipynb) |
| **Regresión lineal** | *linear regression* | Predice con una recta (o hiperplano): punto de partida más suma ponderada. | [1.1](../nivel-1-cero/m01-que-es-ml/1.1-programacion-tradicional-vs-aprendizaje.ipynb) |
| **Regresión logística** | *logistic regression* | Una recta convertida en probabilidad con la sigmoide; línea base de clasificación. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **Regularización** | *regularization* | Castigar la complejidad para que el modelo generalice mejor. | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| **ReLU** | *Rectified Linear Unit* | Activación max(0, z): simple, rápida y el estándar en capas ocultas. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Retorno** | *return* | Suma (descontada) de las recompensas desde un momento hasta el final. | [14.1](../nivel-5-heroe/m14-refuerzo/14.1-agente-entorno-accion-recompensa.ipynb) |
| **Retropropagación** | *backpropagation* | Calcular el gradiente de cada peso repartiendo la culpa del error hacia atrás por la red. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |
| **Reward hacking** | *reward hacking* | El agente encuentra un truco que sube la recompensa sin hacer lo que querías. | [13.5](../nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb) |
| **RLHF** | *Reinforcement Learning from Human Feedback* | Ajustar un LLM con un modelo de recompensa aprendido de preferencias humanas. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Ruido irreducible** | *irreducible error* | Parte del error que ningún modelo puede eliminar. | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| **R²** | *R-squared* | Ver "Coeficiente de determinación". | [3.2](../nivel-2-aprendiz/m03-como-aprende/3.2-funcion-de-perdida.ipynb) |

[↑ Índice](#índice)

## S

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Scheduler (calendario de la tasa)** | *learning rate schedule* | Regla que cambia la tasa de aprendizaje durante el entrenamiento (escalones, coseno…). | [3.5](../nivel-2-aprendiz/m03-como-aprende/3.5-batch-estocastico-minibatch.ipynb) |
| **Segmentación** | *segmentation* | Clasificar cada píxel de una imagen. | [10.3](../nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb) |
| **Seq2seq** | *sequence to sequence* | Codificador que resume una secuencia y decodificador que genera otra (traducción). | [11.4](../nivel-4-avanzado/m11-secuencias/11.4-lstm-y-gru.ipynb) |
| **Serie de tiempo** | *time series* | Datos ordenados en el tiempo (ventas diarias, sensores). | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb) |
| **Sesgo (equidad)** | *bias (fairness)* | Error sistemático del modelo contra ciertos grupos, heredado de datos o etiquetas. | [1.1](../nivel-1-cero/m01-que-es-ml/1.1-programacion-tradicional-vs-aprendizaje.ipynb) |
| **Sesgo (estadístico)** | *bias* | Error por supuestos demasiado simples; el modelo falla siempre hacia el mismo lado. | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| **Sesgo inductivo** | *inductive bias* | Los supuestos que trae de fábrica cada tipo de modelo. | [1.6](../nivel-1-cero/m01-que-es-ml/1.6-no-hay-almuerzo-gratis.ipynb) |
| **SHAP** | *SHapley Additive exPlanations* | Reparte una predicción concreta entre las variables según su contribución. | [5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb) |
| **Sigmoide** | *sigmoid* | Función en forma de S que convierte cualquier número en una probabilidad (0–1). | [5.2](../nivel-2-aprendiz/m05-modelos-clasicos/5.2-regresion-logistica.ipynb) |
| **Silueta** | *silhouette score* | Mide qué tan bien encaja cada punto en su grupo frente al grupo vecino. | [1.3](../nivel-1-cero/m01-que-es-ml/1.3-tipos-de-aprendizaje.ipynb) |
| **Sobreajuste** | *overfitting* | El modelo memoriza el entrenamiento, incluido el ruido, y falla con datos nuevos. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Sobremuestreo / submuestreo** | *oversampling / undersampling* | Repetir casos de la clase rara / descartar de la frecuente para equilibrar. | [2.7](../nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| **Softmax** | *softmax* | Convierte las puntuaciones de varias clases en probabilidades que suman 1. | [9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb) |
| **Stacking** | *stacking* | Ensamble en el que un meta-modelo aprende a combinar las predicciones de otros. | [6.1](../nivel-3-practicante/m06-ensambles/6.1-por-que-combinar-modelos.ipynb) |
| **Stride (paso)** | *stride* | Cuántos píxeles salta un filtro entre una posición y la siguiente. | [10.2](../nivel-4-avanzado/m10-vision/10.2-convoluciones.ipynb) |
| **Suavizado** | *smoothing (Laplace)* | Hacer más prudentes las estimaciones con pocos datos (Laplace en Naive Bayes, `smooth` en `TargetEncoder`). | [2.3](../nivel-1-cero/m02-datos/2.3-preparacion-escalado-y-codificacion.ipynb) |
| **Subajuste** | *underfitting* | El modelo es demasiado simple y falla incluso en el entrenamiento. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |
| **Successive halving** | *successive halving* | Búsqueda tipo torneo: muchos candidatos con poco presupuesto, y solo los mejores siguen. | [7.3](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| **SVM** | *Support Vector Machine* | Busca la frontera con la "calle" más ancha entre clases; con kernels traza curvas. | [0.3](../nivel-1-cero/m00-bienvenida/0.3-breve-historia.ipynb) |

[↑ Índice](#índice)

## T

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **t-SNE** | *t-distributed Stochastic Neighbor Embedding* | Mapa 2D que conserva quién es vecino de quién; solo para visualizar. | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| **Tamaño de lote** | *batch size* | Cuántos ejemplos se usan en cada paso de entrenamiento. | [3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb) |
| **Tasa de aprendizaje** | *learning rate* | Tamaño de cada paso del descenso de gradiente; la perilla más importante. | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |
| **Teacher forcing** | *teacher forcing* | Al entrenar, darle al decodificador la respuesta correcta del paso anterior. | [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb) |
| **Temperatura** | *temperature* | Aplana (alta) o afila (baja) las probabilidades al generar texto. | [11.3](../nivel-4-avanzado/m11-secuencias/11.3-redes-recurrentes.ipynb) |
| **Token / tokenización** | *token / tokenization* | Pedazo de texto (palabra o subpalabra) que el modelo procesa; tokenizar es cortar el texto así. | [10.6](../nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) |
| **Top-k / top-p** | *top-k / nucleus sampling* | Muestrear solo entre los k tokens más probables / entre los que suman probabilidad p. | [12.4](../nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb) |
| **Transfer learning** | *transfer learning* | Reutilizar un modelo entrenado en otra tarea como punto de partida. | [10.1](../nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb) |
| **Transformer** | *Transformer* | Arquitectura basada en auto-atención; base de BERT, GPT y los LLMs actuales. | [0.1](../nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb) |

[↑ Índice](#índice)

## U

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **UMAP** | *Uniform Manifold Approximation and Projection* | Mapa 2D como t-SNE, más rápido y algo más fiel a la estructura global. | [8.4](../nivel-3-practicante/m08-no-supervisado/8.4-pca.ipynb) |
| **Umbral de decisión** | *decision threshold* | Probabilidad a partir de la cual el modelo dice "sí". | [1.2](../nivel-1-cero/m01-que-es-ml/1.2-vocabulario-esencial.ipynb) |

[↑ Índice](#índice)

## V

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **VAE** | *Variational Autoencoder* | Autoencoder cuyo espacio latente está ordenado para poder generar ejemplos nuevos. | [13.1](../nivel-5-heroe/m13-generativos-llms/13.1-discriminativo-vs-generativo.ipynb) |
| **Validación cruzada** | *cross-validation* | Repetir entrenamiento y evaluación rotando qué parte de los datos se evalúa. | [1.5](../nivel-1-cero/m01-que-es-ml/1.5-ciclo-de-vida-de-un-proyecto.ipynb) |
| **Valor atípico** | *outlier* | Dato muy alejado del resto; puede ser un error o un caso real valioso. | [2.1](../nivel-1-cero/m02-datos/2.1-tipos-de-datos.ipynb) |
| **Valor Q** | *Q-value* | Retorno esperado de tomar una acción en un estado y seguir bien después. | [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb) |
| **Varianza** | *variance* | Cuánto cambia el modelo si lo entrenas con otra muestra; varianza alta = sobreajuste. | [4.4](../nivel-2-aprendiz/m04-evaluacion/4.4-sobreajuste-vs-subajuste.ipynb) |
| **Ventaja** | *advantage* | Cuánto mejor fue una acción que lo esperado en ese estado. | [14.4](../nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| **Ventana (de una serie)** | *window / lags* | Cuántos pasos pasados ve el modelo para predecir el siguiente. | [11.1](../nivel-4-avanzado/m11-secuencias/11.1-datos-con-orden.ipynb) |
| **Ventana de contexto** | *context window* | Máximo de tokens que un LLM puede leer de una vez. | [12.1](../nivel-4-avanzado/m12-transformers/12.1-atencion.ipynb) |
| **Verosimilitud** | *likelihood* | Qué tan probables son los datos observados según un modelo. | [5.6](../nivel-2-aprendiz/m05-modelos-clasicos/5.6-naive-bayes.ipynb) |
| **ViT** | *Vision Transformer* | Transformer que trata una imagen como una secuencia de parches. | [10.4](../nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb) |

[↑ Índice](#índice)

## W

| Español | English | Definición | 1.ª aparición |
|---|---|---|---|
| **Warmup (calentamiento)** | *learning-rate warmup* | Subir la tasa de aprendizaje poco a poco al principio del entrenamiento. | [9.6](../nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb) |
| **Weight decay** | *weight decay* | Encoger los pesos un poco en cada paso (L2); regulariza redes. | [7.1](../nivel-3-practicante/m07-regularizacion-y-ajuste/7.1-regularizacion.ipynb) |
| **Word2vec** | *word2vec* | Método clásico para aprender embeddings de palabras a partir de sus vecinas. | [11.2](../nivel-4-avanzado/m11-secuencias/11.2-tokens-y-embeddings.ipynb) |

[↑ Índice](#índice)

## 📝 Notas de traducción

Algunos términos aparecen en el curso con más de una traducción. Todas son correctas; aquí tienes la equivalencia:

| English | Formas que verás en el curso |
|---|---|
| *greedy* | **voraz** ([5.4](../nivel-2-aprendiz/m05-modelos-clasicos/5.4-arboles-de-decision.ipynb), [11.5](../nivel-4-avanzado/m11-secuencias/11.5-seq2seq-y-cuello-de-botella.ipynb)) y **codiciosa/o** ([13.6](../nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb), [14.2](../nivel-5-heroe/m14-refuerzo/14.2-exploracion-vs-explotacion.ipynb)) |
| *vanishing gradient* | **gradiente que desaparece** ([9.3](../nivel-4-avanzado/m09-redes-neuronales/9.3-funciones-de-activacion.ipynb)) y **desvanecimiento del gradiente** ([9.7](../nivel-4-avanzado/m09-redes-neuronales/9.7-inicializacion-y-gradientes.ipynb)) |
| *prompt injection* | **inyección de prompt** ([13.8](../nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb)) e **inyección de instrucciones** ([13.9](../nivel-5-heroe/m13-generativos-llms/13.9-alucinaciones-y-limites.ipynb), [15.5](../nivel-5-heroe/m15-produccion-y-etica/15.5-privacidad-seguridad-y-ataques-adversarios.ipynb)) |
| *pre-training* | **pre-entrenamiento** (M13) y **preentrenado/a** (M10–M12) |
| *fine-tuning* | **fine-tuning** y **ajuste fino**. Ojo: "Ajuste fino" es también el nombre corto del [Módulo 7](../nivel-3-practicante/m07-regularizacion-y-ajuste/), que trata del ajuste de **hiperparámetros** |
| *early stopping* | **early stopping** y **parada temprana** |
| *bias* | **sesgo** tiene tres sentidos: el término independiente *b* de un modelo lineal o neurona ([3.1](../nivel-2-aprendiz/m03-como-aprende/3.1-funcion-con-perillas.ipynb), también "intercepto"), el error sistemático del compromiso sesgo-varianza ([4.5](../nivel-2-aprendiz/m04-evaluacion/4.5-sesgo-varianza.ipynb)) y la injusticia contra grupos ([15.4](../nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb)) |
