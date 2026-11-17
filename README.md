# Machine Learning: de Cero a Héroe 🦸

Un curso en español para **entender de verdad** cómo funcionan los modelos de Machine Learning: cómo piensan, cómo aprenden, qué perillas tienen y qué pasa cuando las giras. Va desde "¿qué es aprender para una máquina?" hasta Transformers, LLMs, aprendizaje por refuerzo y ML en producción.

## ¿Para quién es?

- **Personas curiosas, desarrolladores, analistas, gerentes de producto y estudiantes** que quieren entender ML sin dominar cálculo ni álgebra lineal.
- **No necesitas prerrequisitos formales.** Ayuda saber qué es un promedio, un porcentaje y leer una gráfica. Los notebooks usan Python, pero **no hace falta saber programar**: el código está para que *veas* los conceptos y solo tendrás que cambiar números. Si nunca usaste Python o Jupyter, empieza por la [Lección 0.4](nivel-1-cero/m00-bienvenida/0.4-kit-minimo-python-jupyter-colab.ipynb).
- **No es** un curso de demostraciones matemáticas: las fórmulas aparecen solo cuando ayudan a la intuición, y siempre explicadas en lenguaje natural.

Al terminar podrás explicar cómo funciona cualquier modelo clásico o moderno, elegir uno y justificarlo, predecir el efecto de cambiar un hiperparámetro, diagnosticar problemas de entrenamiento y hablar con criterio sobre producción, ética y riesgos. El detalle está en el [plan de estudio](PLAN_DE_ESTUDIO.md).

---

## 🚀 Cómo empezar

Tienes dos caminos. Si dudas, empieza por **Colab**: no instalas nada.

### Opción A — Google Colab (sin instalar nada)

1. Entra en [colab.research.google.com](https://colab.research.google.com) con tu cuenta de Google.
2. **Archivo → Subir notebook** y elige el `.ipynb` de la lección (o ábrelo desde GitHub/Drive).
3. Ejecuta las celdas de arriba abajo con `Shift + Enter`.

Colab ya trae scikit-learn, pandas, matplotlib, PyTorch y transformers. Guarda tu trabajo con **Archivo → Guardar una copia en Drive**. Para los módulos de redes (M9–M13) puedes activar una GPU en **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, aunque todo el curso está pensado para funcionar en CPU.

### Opción B — Entorno local

Necesitas Python 3.11 o superior (el curso se probó con 3.12).

```bash
cd curso-ml-zero-to-hero
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install scikit-learn pandas matplotlib jupyter torch torchvision transformers
jupyter notebook                     # o: jupyter lab
```

> 🍎 **macOS con Python descargado de python.org:** ejecuta **una vez** el archivo **`Install Certificates.command`** (en *Aplicaciones → Python 3.x*, doble clic). Sin él, las descargas de datos fallan con `CERTIFICATE_VERIFY_FAILED`.

> 💡 La [Lección 0.4](nivel-1-cero/m00-bienvenida/0.4-kit-minimo-python-jupyter-colab.ipynb) instala solo `scikit-learn pandas matplotlib jupyter`, que basta para los módulos 0 a 8. `torch`, `torchvision` y `transformers` hacen falta desde el [Módulo 9](nivel-4-avanzado/m09-redes-neuronales/).

**Librerías opcionales.** Algunas celdas marcadas "para Colab" usan librerías extra. No son necesarias para seguir el curso; si quieres ejecutarlas en local:

| Librería | Dónde |
|---|---|
| `xgboost lightgbm catboost` | [6.4](nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb) |
| `imbalanced-learn` | [2.7](nivel-1-cero/m02-datos/2.7-datos-desbalanceados.ipynb) |
| `optuna` | [7.3](nivel-3-practicante/m07-regularizacion-y-ajuste/7.3-estrategias-de-busqueda.ipynb) |
| `shap` | [7.6](nivel-3-practicante/m07-regularizacion-y-ajuste/7.6-interpretabilidad.ipynb) |
| `umap-learn` | [8.5](nivel-3-practicante/m08-no-supervisado/8.5-tsne-y-umap.ipynb) |
| `stable-baselines3 gymnasium` | [14.3](nivel-5-heroe/m14-refuerzo/14.3-q-learning-y-dqn.ipynb), [14.4](nivel-5-heroe/m14-refuerzo/14.4-gradientes-de-politica-y-ppo.ipynb) |
| `fairlearn` | [15.4](nivel-5-heroe/m15-produccion-y-etica/15.4-sesgo-y-equidad.ipynb) |

### ⚠️ Sobre las versiones

Las respuestas numéricas de los ejercicios (los valores dentro de "👀 Ver respuesta" y los ✅ de los proyectos) se verificaron con **scikit-learn 1.9, torch 2.15 y pandas 3**. Con otras versiones pueden variar un poco: si tu número se parece pero no coincide en el último decimal, vas bien.

---

## 🗺️ Mapa del curso

```
NIVEL 1 · CERO          NIVEL 2 · APRENDIZ        NIVEL 3 · PRACTICANTE      NIVEL 4 · AVANZADO        NIVEL 5 · HÉROE
─────────────           ───────────────           ──────────────────        ─────────────────         ──────────────
M0 Bienvenida      ──►  M3 Cómo aprende     ──►  M6 Ensambles        ──►  M9  Redes neuronales ──►  M13 Generativos y LLMs
M1 ¿Qué es ML?          M4 Evaluación            M7 Regularización         M10 Visión (CNN)          M14 Refuerzo
M2 Los datos            M5 Modelos clásicos      M8 No supervisado         M11 Secuencias            M15 Producción y ética
                                                                           M12 Transformers          M16 Proyecto final
```

Las horas salen de los encabezados **⏱️ Tiempo estimado** de cada notebook y proyecto (incluyen los ejercicios intercalados y los ejercicios finales). La [Lección 0.4](nivel-1-cero/m00-bienvenida/0.4-kit-minimo-python-jupyter-colab.ipynb) es opcional.

| Nivel | Módulo | Lecciones | Horas de lecciones | Proyecto | Horas del proyecto | **Total** |
|---|---|---|---|---|---|---|
| 1 · Cero | [M0 · Bienvenida y mapa mental](nivel-1-cero/m00-bienvenida/) | 4 | 2.8–3.5 h | [🏆 0.5](nivel-1-cero/m00-bienvenida/0.5-proyecto-del-modulo.md) | 1 h | **3.8–4.5 h** |
| 1 · Cero | [M1 · ¿Qué es aprender para una máquina?](nivel-1-cero/m01-que-es-ml/) | 6 | 6.5–8 h | [🏆 1.7](nivel-1-cero/m01-que-es-ml/1.7-proyecto-del-modulo.md) | 2 h | **8.5–10 h** |
| 1 · Cero | [M2 · Los datos: el combustible](nivel-1-cero/m02-datos/) | 8 | 11.3–14 h | [🏆 2.9](nivel-1-cero/m02-datos/2.9-proyecto-del-modulo.md) | 3–3.5 h | **14.3–17.5 h** |
| 2 · Aprendiz | [M3 · Cómo aprende un modelo](nivel-2-aprendiz/m03-como-aprende/) | 7 | 10.3–12.7 h | [🏆 3.8](nivel-2-aprendiz/m03-como-aprende/3.8-proyecto-del-modulo.md) | 3 h | **13.3–15.7 h** |
| 2 · Aprendiz | [M4 · Evaluación: ¿mi modelo es bueno?](nivel-2-aprendiz/m04-evaluacion/) | 8 | 12.2–14.8 h | [🏆 4.9](nivel-2-aprendiz/m04-evaluacion/4.9-proyecto-del-modulo.md) | 3 h | **15.2–17.8 h** |
| 2 · Aprendiz | [M5 · Modelos clásicos supervisados](nivel-2-aprendiz/m05-modelos-clasicos/) | 7 | 12.2–14.5 h | [🏆 5.8](nivel-2-aprendiz/m05-modelos-clasicos/5.8-proyecto-del-modulo.md) | 3–4 h | **15.2–18.5 h** |
| 3 · Practicante | [M6 · Ensambles](nivel-3-practicante/m06-ensambles/) | 6 | 9.8–11.8 h | [🏆 6.7](nivel-3-practicante/m06-ensambles/6.7-proyecto-del-modulo.md) | 3–4 h | **12.8–15.8 h** |
| 3 · Practicante | [M7 · Regularización y ajuste de hiperparámetros](nivel-3-practicante/m07-regularizacion-y-ajuste/) | 6 | 9.8–11.8 h | [🏆 7.7](nivel-3-practicante/m07-regularizacion-y-ajuste/7.7-proyecto-del-modulo.md) | 3–4 h | **12.8–15.8 h** |
| 3 · Practicante | [M8 · Aprendizaje no supervisado](nivel-3-practicante/m08-no-supervisado/) | 7 | 11.2–13.5 h | [🏆 8.8](nivel-3-practicante/m08-no-supervisado/8.8-proyecto-del-modulo.md) | 3–4 h | **14.2–17.5 h** |
| 4 · Avanzado | [M9 · Redes neuronales desde la intuición](nivel-4-avanzado/m09-redes-neuronales/) | 9 | 15–18 h | [🏆 9.10](nivel-4-avanzado/m09-redes-neuronales/9.10-proyecto-del-modulo.md) | 3–4 h | **18–22 h** |
| 4 · Avanzado | [M10 · Visión por computadora (CNN)](nivel-4-avanzado/m10-vision/) | 6 | 9.2–11.2 h | [🏆 10.7](nivel-4-avanzado/m10-vision/10.7-proyecto-del-modulo.md) | 3–4 h | **12.2–15.2 h** |
| 4 · Avanzado | [M11 · Secuencias y lenguaje](nivel-4-avanzado/m11-secuencias/) | 5 | 8.2–9.8 h | [🏆 11.6](nivel-4-avanzado/m11-secuencias/11.6-proyecto-del-modulo.md) | 3–4 h | **11.2–13.8 h** |
| 4 · Avanzado | [M12 · Atención y Transformers](nivel-4-avanzado/m12-transformers/) | 5 | 8.3–10 h | [🏆 12.6](nivel-4-avanzado/m12-transformers/12.6-proyecto-del-modulo.md) | 3–4 h | **11.3–14 h** |
| 5 · Héroe | [M13 · Modelos generativos y LLMs](nivel-5-heroe/m13-generativos-llms/) | 9 | 14–17 h | [🏆 13.10](nivel-5-heroe/m13-generativos-llms/13.10-proyecto-del-modulo.md) | 3–4 h | **17–21 h** |
| 5 · Héroe | [M14 · Aprendizaje por refuerzo](nivel-5-heroe/m14-refuerzo/) | 5 | 7.7–9.3 h | [🏆 14.6](nivel-5-heroe/m14-refuerzo/14.6-proyecto-del-modulo.md) | 3 h | **10.7–12.3 h** |
| 5 · Héroe | [M15 · ML en producción y responsabilidad](nivel-5-heroe/m15-produccion-y-etica/) | 6 | 8.7–10.7 h | [🏆 15.7](nivel-5-heroe/m15-produccion-y-etica/15.7-proyecto-del-modulo.md) | 3 h | **11.7–13.7 h** |
| 5 · Héroe | [M16 · Proyecto final](nivel-5-heroe/m16-proyecto-final/) | — | — | [🏆 16.1](nivel-5-heroe/m16-proyecto-final/16.1-proyecto-final.md) | 9–12 h | **9–12 h** |
| | **Total** | **104** | | **17** | | **211.1–257.2 h** |

> 📅 A un ritmo de **5 horas por semana**, el curso completo lleva unos **10–12 meses**; a **10 horas por semana**, unos **5–6 meses**. Los niveles 1 y 2 (M0–M5) son la base imprescindible; a partir de ahí puedes priorizar los módulos que más te interesen.

---

## 📒 Cómo funciona cada lección

Cada lección es un **notebook** con el mismo esqueleto: objetivos → intuición con una analogía → concepto → cómo funciona por dentro → "¿qué pasa si…?" → hiperparámetros → errores comunes → ficha del modelo → resumen en 5 puntos → ejercicios finales (≈30 min) → para profundizar. Nunca hay más de una pantalla de texto entre una explicación y el siguiente ejercicio.

| Icono | Significa |
|---|---|
| 📖 | **Lee**: explicación del concepto |
| ▶️ | **Ejecuta** la celda y observa el resultado |
| 🔮 | **Predice** antes de ejecutar: escribe tu apuesta y luego compruébala |
| ✋ | **Tu turno**: ejercicio corto (2–5 min) |
| ✏️ | **Modifica** un valor en el código y vuelve a ejecutar |
| 👀 | **Ver respuesta**: la solución, oculta hasta que la abras |

La regla de oro: **escribe tu respuesta antes de abrir 👀**, aunque sea "no sé, creo que…". Por qué funciona predecir antes de mirar: [Lección 0.1](nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb).

### 🏆 Los proyectos

Cada módulo cierra con un **proyecto retador** (1–4 h) que integra todas sus lecciones sobre un caso realista, y el curso termina con un **proyecto final** de 9–12 h. El enunciado es un `.md` y tú trabajas **en tu propio notebook, desde cero**. Cada proyecto trae:

- La **misión de negocio** y fases cortas que dicen *qué* hacer, no *cómo*.
- **✅ Valores de referencia** para comprobar tus números antes de seguir.
- **💡 Pistas** y **👀 discusiones** ocultas, para cuando te atores.
- Una **rúbrica de autoevaluación** y un **informe final** para un público no técnico.

---

## 📥 Notebooks que descargan datos la primera vez

La mayoría de los notebooks usan datos incluidos en scikit-learn (`load_digits`, `load_wine`, `load_breast_cancer`, `load_diabetes`…) o generados en el propio notebook, y funcionan sin conexión. Estos descargan algo **la primera vez** (necesitas internet; después quedan en caché):

| Qué se descarga | Tamaño aprox. | Dónde se guarda | Lecciones | Proyectos |
|---|---|---|---|---|
| Censo *Adult* de OpenML (`fetch_openml("adult")`) | pocos MB | `~/scikit_learn_data` | [6.4](nivel-3-practicante/m06-ensambles/6.4-xgboost-lightgbm-y-catboost.ipynb), [6.6](nivel-3-practicante/m06-ensambles/6.6-por-que-los-arboles-dominan-lo-tabular.ipynb) | — |
| Otros conjuntos de OpenML / scikit-learn (`credit-g`, `house_prices`, `wholesale-customers`, `fetch_california_housing`) | pocos MB | `~/scikit_learn_data` | — | [5.8](nivel-2-aprendiz/m05-modelos-clasicos/5.8-proyecto-del-modulo.md), [6.7](nivel-3-practicante/m06-ensambles/6.7-proyecto-del-modulo.md), [7.7](nivel-3-practicante/m07-regularizacion-y-ajuste/7.7-proyecto-del-modulo.md), [8.8](nivel-3-practicante/m08-no-supervisado/8.8-proyecto-del-modulo.md) |
| MNIST (`torchvision`, `download=True`) | ≈ 12 MB | carpeta `data/` junto al notebook | [9.5](nivel-4-avanzado/m09-redes-neuronales/9.5-optimizadores.ipynb), [9.6](nivel-4-avanzado/m09-redes-neuronales/9.6-hiperparametros-de-la-red.ipynb), [9.8](nivel-4-avanzado/m09-redes-neuronales/9.8-regularizacion-en-redes.ipynb), [9.9](nivel-4-avanzado/m09-redes-neuronales/9.9-recetario-de-diagnostico.ipynb), [10.1](nivel-4-avanzado/m10-vision/10.1-como-ve-una-computadora.ipynb), [10.3](nivel-4-avanzado/m10-vision/10.3-pooling-y-jerarquia.ipynb), [10.4](nivel-4-avanzado/m10-vision/10.4-arquitecturas-iconicas.ipynb), [13.2](nivel-5-heroe/m13-generativos-llms/13.2-autoencoders-y-vaes.ipynb), [13.4](nivel-5-heroe/m13-generativos-llms/13.4-modelos-de-difusion.ipynb) | — |
| Fashion-MNIST (`torchvision`) | ≈ 30 MB | carpeta `data/` | — | [9.10](nivel-4-avanzado/m09-redes-neuronales/9.10-proyecto-del-modulo.md) |
| CIFAR-10 (`torchvision`) | ≈ 170 MB | carpeta `data/` | [10.5](nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb) | [10.7](nivel-4-avanzado/m10-vision/10.7-proyecto-del-modulo.md) |
| Pesos preentrenados de ResNet18 (`ResNet18_Weights.DEFAULT`) | ≈ 45 MB | `~/.cache/torch` | [10.5](nivel-4-avanzado/m10-vision/10.5-transfer-learning.ipynb), [10.6](nivel-4-avanzado/m10-vision/10.6-deteccion-segmentacion-y-vit.ipynb) | — |
| GPT-2 de Hugging Face (`from_pretrained("gpt2")`) | ≈ 500 MB | `~/.cache/huggingface` | [11.2](nivel-4-avanzado/m11-secuencias/11.2-tokens-y-embeddings.ipynb), [12.2](nivel-4-avanzado/m12-transformers/12.2-auto-atencion-qkv.ipynb), [12.4](nivel-4-avanzado/m12-transformers/12.4-encoder-decoder.ipynb), [13.5](nivel-5-heroe/m13-generativos-llms/13.5-como-se-entrena-un-llm.ipynb), [13.6](nivel-5-heroe/m13-generativos-llms/13.6-hiperparametros-de-generacion.ipynb), [13.7](nivel-5-heroe/m13-generativos-llms/13.7-lora-y-cuantizacion.ipynb), [13.8](nivel-5-heroe/m13-generativos-llms/13.8-prompting-rag-y-agentes.ipynb), [13.9](nivel-5-heroe/m13-generativos-llms/13.9-alucinaciones-y-limites.ipynb) | [13.10](nivel-5-heroe/m13-generativos-llms/13.10-proyecto-del-modulo.md) |
| *Don Quijote* del Proyecto Gutenberg (URL con `urlretrieve`) | ≈ 2 MB | carpeta del notebook | [12.5](nivel-4-avanzado/m12-transformers/12.5-leyes-de-escalamiento.ipynb) | [12.6](nivel-4-avanzado/m12-transformers/12.6-proyecto-del-modulo.md) |

> 💡 Como `torchvision` guarda los datos en `data/` **relativo a la carpeta del notebook**, MNIST se descarga una vez por módulo (M9, M10 y M13). En Colab cada sesión empieza vacía, así que se descargan de nuevo.
>
> El [proyecto final](nivel-5-heroe/m16-proyecto-final/16.1-proyecto-final.md) trae sus datos en un CSV dentro de su carpeta. Algunos recursos de "📚 Para profundizar" (Stable Diffusion, BERT, `sentence-transformers`) descargan modelos más grandes, pero son opcionales y están pensados para Colab.

---

## 🧰 Material de apoyo

Para tener a mano mientras estudias:

| Archivo | Para qué sirve |
|---|---|
| [📖 Glosario ES ↔ EN](apoyo/glosario.md) | Todos los términos del curso con definición de una línea y la lección donde aparecen |
| [🎛️ Chuleta de hiperparámetros](apoyo/chuleta-hiperparametros.md) | Modelo → perilla → efecto al subir/bajar → valor de partida, más equivalencias entre librerías |
| [🧭 ¿Qué modelo uso?](apoyo/arbol-de-decision-modelos.md) | Diagrama de decisión según el tipo de dato y de problema |
| [🩺 Diagnóstico del entrenamiento](apoyo/diagnostico-entrenamiento.md) | Síntoma → causa probable → qué probar, para ML clásico y redes |
| [📚 Recursos](apoyo/recursos.md) | Libros, videos e interactivos recomendados, por módulo |
| [📋 Plan de estudio](PLAN_DE_ESTUDIO.md) | Visión, metodología y contenido detallado de cada módulo |

¿Listo? Empieza por la [Lección 0.1 — Cómo usar este curso](nivel-1-cero/m00-bienvenida/0.1-como-usar-este-curso.ipynb). 🚀
