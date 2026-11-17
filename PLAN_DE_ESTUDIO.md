# Machine Learning: de Cero a Héroe
### Plan de estudio

---

## 1. Visión del curso

**Para quién es:** personas curiosas, desarrolladores, analistas, gerentes de producto o estudiantes que quieren **entender de verdad** cómo funcionan los modelos de Machine Learning sin necesidad de dominar cálculo o álgebra lineal.

**Qué NO es este curso:** no es un curso de demostraciones matemáticas. Las fórmulas aparecen solo cuando ayudan a la intuición, y siempre van acompañadas de una explicación en lenguaje natural.

**Qué SÍ es:** un recorrido conceptual profundo sobre:
- **Cómo piensa** cada modelo (su intuición y sus supuestos).
- **Cómo aprende** (entrenamiento, función de pérdida, optimización).
- **Qué perillas tiene** (hiperparámetros) y qué pasa cuando las giras.
- **Cuándo usarlo y cuándo no** (fortalezas, debilidades y errores comunes).
- **Cómo saber si funciona** (evaluación, validación y diagnóstico).

### Objetivos generales
Al terminar, el estudiante podrá:
1. Explicar con sus propias palabras cómo funciona cualquier modelo clásico y moderno de ML.
2. Elegir un modelo adecuado para un problema y justificar la elección.
3. Predecir el efecto de cambiar un hiperparámetro (p. ej. "si subo la profundidad del árbol, tenderá a sobreajustar").
4. Diagnosticar problemas de entrenamiento: sobreajuste, subajuste, fuga de datos, desbalance, pérdida que no baja.
5. Entender el funcionamiento de redes neuronales, Transformers y LLMs a nivel conceptual.
6. Hablar con criterio sobre ML en producción, ética y riesgos.

### Prerrequisitos
- Ninguno formal. Ayuda saber qué es un promedio, un porcentaje y leer una gráfica.
- Los laboratorios (opcionales) usan Python básico; se incluye una mini-introducción.

---

## 2. Metodología

| Elemento | Descripción |
|---|---|
| **Analogía primero** | Cada concepto se introduce con una analogía del mundo real antes de su definición técnica. |
| **Visual** | Diagramas (ASCII/Mermaid) y referencias a herramientas interactivas (p. ej. TensorFlow Playground). |
| **"¿Qué pasa si…?"** | Cada modelo incluye experimentos mentales sobre sus hiperparámetros. |
| **Ficha del modelo** | Resumen estándar por modelo: intuición, cómo aprende, hiperparámetros, pros/contras, cuándo usarlo. |
| **Práctica intercalada** | Nunca pasa mucho tiempo entre explicar y practicar: cada concepto va seguido **de inmediato** por un ejercicio corto (2–5 min) con su respuesta oculta. |
| **Escalonado** | La dificultad sube poco a poco: ejercicios cortos dentro de la lección → ejercicios finales más largos al cierre de la lección → proyecto retador al cierre del módulo. |
| **Notebook interactivo** | Cada lección es un notebook (scikit-learn/PyTorch) para *ver* el concepto y predecir antes de ejecutar, no para memorizar código. |

### Estructura estándar de cada lección
```
0. Objetivos + leyenda de iconos (📖 lee · ▶️ ejecuta · 🔮 predice · ✋ tu turno · ✏️ modifica)
1. La intuición (analogía)                         → ✋ ejercicio corto
2. El concepto explicado                           → ✋ ejercicio corto
3. Cómo aprende / cómo funciona por dentro         → 🔮 / ✋ ejercicio(s) corto(s)
4. Qué pasa si… (el problema típico del modelo)    → 🔮 predice y comprueba
5. Hiperparámetros y perillas (tabla + gráficas)   → 🔮 / ✏️ un ejercicio por perilla clave
6. Errores comunes y señales de alerta             → 🔮 ejercicio corto
7. Ficha del modelo
8. Resumen en 5 puntos
9. Ejercicios finales (≈30 min): repaso rápido, preguntas para pensar y una
   práctica integradora en pasos cortos, cada paso con su respuesta
10. Para profundizar
```
Regla: entre una explicación (📖) y el siguiente ejercicio no debe haber más de **una pantalla** de texto.

### Proyecto del módulo
Cada módulo cierra con un **proyecto retador** (3–4 h) que integra todas sus lecciones sobre un caso realista. El enunciado es un `.md` y el estudiante trabaja **en su propio notebook, desde cero**. El enunciado incluye la misión de negocio, fases cortas que dicen *qué* hacer (no *cómo*), valores de referencia (✅) para autocomprobarse, pistas (💡) y discusiones (👀) ocultas, una rúbrica de autoevaluación y un informe final para un público no técnico.

---

## 3. Mapa del curso

```
NIVEL 1 · CERO          NIVEL 2 · APRENDIZ        NIVEL 3 · PRACTICANTE      NIVEL 4 · AVANZADO        NIVEL 5 · HÉROE
─────────────           ───────────────           ──────────────────        ─────────────────         ──────────────
M0 Bienvenida      ──►  M3 Cómo aprende     ──►  M6 Ensambles        ──►  M9  Redes neuronales ──►  M13 Generativos y LLMs
M1 ¿Qué es ML?          M4 Evaluación            M7 Regularización         M10 Visión (CNN)          M14 Refuerzo
M2 Los datos            M5 Modelos clásicos      M8 No supervisado         M11 Secuencias            M15 Producción y ética
                                                                           M12 Transformers          M16 Proyecto final
```

Duración estimada: **≈ 210–260 horas** (104 lecciones + 17 proyectos) → unos **9–10 meses a 6 h/semana**, o **~6 meses a 10 h/semana**. La Lección 0.1 trae un planificador para calcular tu ritmo.

---

## 4. Contenido detallado

### NIVEL 1 · CERO — Fundamentos

#### Módulo 0 — Bienvenida y mapa mental
- 0.1 Cómo usar este curso
- 0.2 IA vs. Machine Learning vs. Deep Learning vs. IA generativa
- 0.3 Breve historia: de las reglas a-mano a los LLMs
- 0.4 (Opcional) Kit mínimo: Python, Jupyter/Colab en 30 minutos

#### Módulo 1 — ¿Qué es aprender para una máquina?
- 1.1 Programación tradicional vs. aprendizaje a partir de ejemplos
- 1.2 Vocabulario esencial: dato, característica (*feature*), etiqueta, modelo, predicción, parámetro vs. hiperparámetro
- 1.3 Tipos de aprendizaje: supervisado, no supervisado, semi-supervisado, auto-supervisado, por refuerzo
- 1.4 Tipos de problema: regresión, clasificación, clustering, ranking, generación
- 1.5 El ciclo de vida de un proyecto de ML (de la pregunta de negocio al modelo en producción)
- 1.6 "No hay almuerzo gratis": por qué no existe un modelo perfecto

#### Módulo 2 — Los datos: el combustible
- 2.1 Tipos de datos: numéricos, categóricos, texto, imagen, series de tiempo
- 2.2 Calidad de datos: faltantes, atípicos, duplicados, ruido
- 2.3 Preparación: escalado/normalización, codificación de categorías (one-hot, ordinal, *target encoding*)
- 2.4 Ingeniería de características: crear información útil
- 2.5 División de datos: entrenamiento, validación y prueba — y por qué importan
- 2.6 Fuga de datos (*data leakage*): el error silencioso más caro
- 2.7 Datos desbalanceados: qué son y estrategias (re-muestreo, pesos de clase)
- 2.8 Maldición de la dimensionalidad (intuición)

---

### NIVEL 2 · APRENDIZ — Cómo aprenden y cómo se evalúan

#### Módulo 3 — Cómo aprende un modelo (el corazón del curso)
- 3.1 El modelo como "función con perillas" (parámetros)
- 3.2 Función de pérdida: medir qué tan equivocado está el modelo (MSE, MAE, entropía cruzada — intuición)
- 3.3 Optimización: el excursionista en la niebla bajando la montaña (descenso de gradiente)
- 3.4 **Hiperparámetros del entrenamiento:** tasa de aprendizaje, épocas, tamaño de lote (*batch*), iteraciones
- 3.5 Variantes: batch, estocástico, mini-batch
- 3.6 Mínimos locales, mesetas y por qué no siempre importan
- 3.7 Curvas de aprendizaje: leer la historia del entrenamiento

#### Módulo 4 — Evaluación: ¿mi modelo es bueno?
- 4.1 Métricas de regresión: MAE, RMSE, R²
- 4.2 Métricas de clasificación: exactitud, precisión, *recall*, F1, matriz de confusión
- 4.3 Umbrales, curva ROC/AUC y curva Precisión-*Recall*
- 4.4 Sobreajuste vs. subajuste
- 4.5 El compromiso sesgo-varianza (la analogía del tirador al blanco)
- 4.6 Validación cruzada (k-fold, estratificada, temporal)
- 4.7 Líneas base: ¿le gano a "adivinar el promedio"?
- 4.8 Elegir la métrica según el costo del error en el negocio

#### Módulo 5 — Modelos clásicos supervisados
Cada modelo con su **ficha del modelo** completa.
- 5.1 Regresión lineal — la recta que mejor se ajusta
- 5.2 Regresión logística — de la recta a la probabilidad
- 5.3 K vecinos más cercanos (KNN) — "dime con quién andas" · *hp: k, distancia, pesos*
- 5.4 Árboles de decisión — el juego de las 20 preguntas · *hp: max_depth, min_samples_split, min_samples_leaf, criterio*
- 5.5 Máquinas de vectores de soporte (SVM) — la calle más ancha · *hp: C, kernel, gamma*
- 5.6 Naive Bayes — probabilidad con supuestos ingenuos
- 5.7 Comparativa: tabla de decisión "¿qué modelo uso?"
- 🏆 **Proyecto del módulo:** Comité de crédito — comparar los 5 modelos y elegir por costo de negocio, no por exactitud

---

### NIVEL 3 · PRACTICANTE — Modelos potentes y ajuste

#### Módulo 6 — Ensambles: la sabiduría de la multitud
- 6.1 Por qué combinar modelos funciona
- 6.2 Bagging y Random Forest · *hp: n_estimators, max_features, max_depth, bootstrap*
- 6.3 Boosting: aprender de los errores (AdaBoost, Gradient Boosting)
- 6.4 XGBoost, LightGBM y CatBoost: qué cambia entre ellos · *hp: learning_rate, n_estimators, max_depth, subsample, colsample, regularización*
- 6.5 Stacking y voting
- 6.6 Por qué los ensambles de árboles dominan los datos tabulares

#### Módulo 7 — Regularización y ajuste de hiperparámetros
- 7.1 Regularización: castigar la complejidad (L1/Lasso, L2/Ridge, Elastic Net)
- 7.2 Early stopping
- 7.3 Estrategias de búsqueda: grid, aleatoria, bayesiana (Optuna), *successive halving*
- 7.4 Cómo priorizar qué hiperparámetros ajustar primero
- 7.5 Pipelines: evitar fugas mientras se ajusta
- 7.6 Interpretabilidad: importancia de variables, SHAP y dependencias parciales (intuición)

#### Módulo 8 — Aprendizaje no supervisado
- 8.1 Encontrar estructura sin etiquetas
- 8.2 K-means · *hp: k, inicialización* — método del codo y silueta
- 8.3 Clustering jerárquico y DBSCAN · *hp: eps, min_samples*
- 8.4 Reducción de dimensionalidad: PCA (la mejor "foto" de los datos)
- 8.5 Visualización: t-SNE y UMAP · *hp: perplexity, n_neighbors*
- 8.6 Detección de anomalías: Isolation Forest y compañía
- 8.7 Sistemas de recomendación (intuición)

---

### NIVEL 4 · AVANZADO — Deep Learning

#### Módulo 9 — Redes neuronales desde la intuición
- 9.1 La neurona artificial: suma ponderada + decisión
- 9.2 Capas y profundidad: por qué apilar funciona
- 9.3 Funciones de activación: sigmoide, tanh, ReLU y familia
- 9.4 Retropropagación (*backpropagation*): repartir la culpa del error
- 9.5 Optimizadores: SGD, Momentum, RMSProp, Adam
- 9.6 **Hiperparámetros de la red:** número de capas, neuronas, tasa de aprendizaje, *schedulers* (warmup, cosine), batch size, épocas
- 9.7 Inicialización de pesos y gradientes que desaparecen/explotan
- 9.8 Regularización en redes: dropout, weight decay, batch normalization, data augmentation
- 9.9 Recetario de diagnóstico: "mi pérdida no baja", "mi pérdida es NaN", "valida peor que entrena"

#### Módulo 10 — Visión por computadora (CNN)
- 10.1 Cómo "ve" una computadora una imagen
- 10.2 Convoluciones: filtros que detectan patrones · *hp: tamaño de kernel, stride, padding, número de filtros*
- 10.3 Pooling y jerarquía de características
- 10.4 Arquitecturas icónicas: LeNet, AlexNet, VGG, ResNet (conexiones residuales)
- 10.5 Transfer learning y fine-tuning: no empezar de cero
- 10.6 Más allá de clasificar: detección, segmentación, Vision Transformers

#### Módulo 11 — Secuencias y lenguaje
- 11.1 Datos con orden: texto, audio, series de tiempo
- 11.2 Representar palabras: tokens y *embeddings* (el mapa de significados)
- 11.3 Redes recurrentes (RNN) y su memoria corta
- 11.4 LSTM y GRU: puertas para recordar y olvidar
- 11.5 Seq2seq y el cuello de botella

#### Módulo 12 — Atención y Transformers
- 12.1 Atención: "¿a qué palabras debo prestar atención?"
- 12.2 Auto-atención, consultas/claves/valores (analogía de la biblioteca)
- 12.3 Multi-cabeza, codificación posicional, bloques del Transformer
- 12.4 Encoder (BERT) vs. Decoder (GPT) vs. Encoder-Decoder (T5)
- 12.5 Por qué los Transformers escalan tan bien: leyes de escalamiento

---

### NIVEL 5 · HÉROE — Frontera y mundo real

#### Módulo 13 — Modelos generativos y LLMs
- 13.1 Discriminativo vs. generativo
- 13.2 Autoencoders y VAEs: comprimir y reconstruir
- 13.3 GANs: el falsificador y el detective
- 13.4 Modelos de difusión: crear imágenes quitando ruido
- 13.5 Cómo se entrena un LLM: pre-entrenamiento, ajuste por instrucciones, RLHF/RLAIF
- 13.6 **Hiperparámetros de generación:** temperatura, top-k, top-p, longitud máxima, penalizaciones
- 13.7 Fine-tuning eficiente: LoRA y cuantización (intuición)
- 13.8 Prompting, RAG y agentes
- 13.9 Alucinaciones, ventana de contexto y límites actuales

#### Módulo 14 — Aprendizaje por refuerzo (introducción conceptual)
- 14.1 Agente, entorno, acción, recompensa
- 14.2 Exploración vs. explotación · *hp: epsilon, factor de descuento gamma*
- 14.3 Q-learning y redes Q profundas (intuición)
- 14.4 Gradientes de política y PPO
- 14.5 Casos famosos: AlphaGo, robótica, RLHF

#### Módulo 15 — ML en producción y responsabilidad
- 15.1 Del notebook a producción: MLOps en una página
- 15.2 Monitoreo: deriva de datos y de concepto (*data/concept drift*)
- 15.3 Reentrenamiento y versionado de datos/modelos
- 15.4 Sesgo y equidad: cómo los modelos heredan prejuicios
- 15.5 Privacidad, seguridad y ataques adversarios
- 15.6 Explicabilidad y regulación

#### Módulo 16 — Proyecto final
- Caso integrador guiado: desde un problema real hasta un modelo evaluado y justificado.
- Entregable: **informe conceptual** que explique elección del modelo, hiperparámetros ajustados, métricas, diagnóstico de errores y riesgos.
- Rúbrica de evaluación incluida.

---

## 5. Material de apoyo transversal

| Recurso | Contenido |
|---|---|
| `glosario.md` | Todos los términos ES ↔ EN con definición de una línea |
| `chuleta-hiperparametros.md` | Tabla maestra: modelo → hiperparámetro → efecto al subir/bajar → valor de partida |
| `arbol-de-decision-modelos.md` | Diagrama "¿qué modelo uso?" según tipo de dato y problema |
| `diagnostico-entrenamiento.md` | Síntoma → causa probable → qué probar |
| `recursos.md` | Libros, videos y herramientas interactivas recomendadas |

---

## 6. Estructura de archivos propuesta

```
curso-ml-zero-to-hero/
├── PLAN_DE_ESTUDIO.md
├── README.md
├── nivel-1-cero/
│   ├── m00-bienvenida/
│   ├── m01-que-es-ml/
│   └── m02-datos/
├── nivel-2-aprendiz/
│   ├── m03-como-aprende/
│   ├── m04-evaluacion/
│   └── m05-modelos-clasicos/
├── nivel-3-practicante/
│   ├── m06-ensambles/
│   ├── m07-regularizacion-y-ajuste/
│   └── m08-no-supervisado/
├── nivel-4-avanzado/
│   ├── m09-redes-neuronales/
│   ├── m10-vision/
│   ├── m11-secuencias/
│   └── m12-transformers/
├── nivel-5-heroe/
│   ├── m13-generativos-llms/
│   ├── m14-refuerzo/
│   ├── m15-produccion-y-etica/
│   └── m16-proyecto-final/
└── apoyo/
    ├── glosario.md
    ├── chuleta-hiperparametros.md
    ├── arbol-de-decision-modelos.md
    ├── diagnostico-entrenamiento.md
    └── recursos.md
```

Cada módulo contiene un notebook por lección (`5.4-arboles-de-decision.ipynb`, con ejercicios intercalados) y el enunciado del proyecto (`5.8-proyecto-del-modulo.md`), que el estudiante resuelve en su propio notebook.

---

## 7. Plan de producción del contenido

1. **Fase 1:** README + material de apoyo base (glosario y chuleta, que se irán ampliando).
2. **Fase 2:** Niveles 1 y 2 (M0–M5) — base conceptual imprescindible.
3. **Fase 3:** Nivel 3 (M6–M8).
4. **Fase 4:** Nivel 4 (M9–M12).
5. **Fase 5:** Nivel 5 (M13–M16) y revisión global de coherencia.

---

## 8. Decisiones tomadas

- [x] **Código:** sí. Cada lección es un notebook con código corto que el estudiante ejecuta, predice y modifica (scikit-learn, y PyTorch desde el Módulo 9).
- [x] **Formato:** notebooks `.ipynb` para las lecciones y `.md` para los enunciados de proyecto y el material de apoyo.
- [x] **Módulos:** los 17 del plan; series de tiempo se cubren en 4.6, 4.7 y 11.1 (y el proyecto 11.6), y sistemas de recomendación en 8.7.
- [x] **Tono:** cercano (tú).
