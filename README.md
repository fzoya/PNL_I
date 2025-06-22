# PLN 1 - TRABAJOS PRÁCTICOS 

**Carrera:** Especialización en Inteligencia Artificial - UBA  
**Materia:** Procesamiento del Lenguaje Natural  
**Alumno:** Federico Martín Zoya  
**Trabajo Práctico:** Resolución de desafíos 1-4  

#### Durante la cursada de la materia Procesamiento del Lenguaje Natural I se presentaron cuatro desafíos que permitieron conceptualizar y experimentar las temáticas expuestas durante las clases teóricas. A continuación se resume cada uno de los trabajos presentados, su desarrollo y conclusiones obtenidas.

---

# Desafío 1 - Vectorización de documentos

---

## Objetivos del Desafío

Este trabajo práctico tiene como objetivo introducir los conceptos fundamentales del preprocesamiento y análisis de texto, abordando los siguientes aspectos:

1. **Vectorización de documentos** con técnicas clásicas como Bag of Words (BoW) y TF-IDF.
2. **Medición de similaridad entre documentos** mediante la similitud de coseno.
3. **Entrenamiento y evaluación de clasificadores de texto** usando modelos de Naïve Bayes.
4. **Análisis de similaridad semántica entre palabras** usando la matriz transpuesta de documentos.

Para el desarrollo del trabajo y la aplicación de los conceptos aprendidos se importa el dataset correspondiente a AG's News Corpus, el cual clasifica noticias en categorías como: 1- World, 2- Sports, 3- Business, 4- Sci/Tech.

---

## Contenido Conceptual

### 1. Carga de datos

Se utiliza el módulo `fetch_openml` de `scikit-learn` para cargar el dataset preprocesado, tanto para entrenamiento como para test. Este corpus es ideal para experimentar con clasificación de texto por temas.
```python
from sklearn.datasets import fetch_openml

# Carga del dataset
ag_news = fetch_openml(name="AG_news", version=1, as_frame=True)
```

### 2. Vectorización del texto

Se emplean dos técnicas para transformar el texto en vectores:

- **CountVectorizer**: Representación BoW simple.
- **TfidfVectorizer**: Representación ponderada que considera la frecuencia inversa de documentos.

Esto convierte el conjunto de documentos en una matriz dispersa donde cada fila representa un documento y cada columna una palabra/token.

### 3. Medición de similaridad entre documentos

Se calcula la **similaridad de coseno** entre vectores de documentos. Esto permite identificar qué documentos son más similares en términos del contenido textual.

Como ejercicio práctico:

- Se seleccionan 5 documentos al azar.
- Se calcula su similaridad con el resto.
- Se analizan los 5 documentos más similares a cada uno y se interpreta la relación con sus etiquetas.

### 4. Clasificación de textos

Se implementan dos variantes de modelos de **Naïve Bayes**:

- `MultinomialNB`
- `ComplementNB`

Se busca **maximizar el F1-score macro** en el conjunto de test ajustando:

- Parámetros del vectorizador (e.g., `min_df`, `max_df`, `ngram_range`).
- Parámetros de los modelos.

Se hace hincapié en la **validación de desempeño** usando métricas estándar.

### 5. Similaridad entre palabras

Al transponer la matriz documento-término se obtiene una **matriz término-documento**, donde cada fila es una palabra representada por los documentos en que aparece.

Esta representación se usa para:

- Analizar similitudes entre palabras mediante la similitud de coseno.
- Evaluar qué palabras comparten contexto y son "semánticamente similares".

---

## Resultados Obtenidos

- Los modelos Naïve Bayes lograron buenos niveles de desempeño, especialmente con representaciones TF-IDF y ajustes apropiados de parámetros.
- La similaridad entre documentos fue interpretada correctamente para múltiples casos.
- El análisis de similaridad entre palabras mostró coherencia contextual con términos temáticamente relacionados.

---

## Conclusiones

Este primer desafío permitió:

- Aplicar técnicas básicas de preprocesamiento textual.
- Comprender la relación entre representación vectorial y significado textual.
- Evaluar modelos de clasificación textual.
- Introducir la noción de distribución de palabras en contexto.

Estas habilidades son fundamentales para los próximos desafíos que profundizarán en temas como embeddings, modelos secuenciales y generación de texto.

---

## Requisitos

Este notebook utiliza las siguientes bibliotecas:

```bash
pip install numpy scikit-learn
```

---

## Estructura del código

- **Carga y exploración del dataset**
- **Vectorización BoW y TF-IDF**
- **Cálculo de similitud de documentos**
- **Entrenamiento de Naïve Bayes y evaluación con F1-score**
- **Análisis de similitud entre palabras**



# Desafío 2 - Representaciones vectoriales, embeddings y word2vec

---

## Objetivos del Desafío

Este trabajo práctico se enfoca en la creación de **embeddings personalizados** a partir de un corpus textual utilizando **Gensim** y el modelo **Word2Vec**. El objetivo es capturar representaciones vectoriales semánticamente informadas de palabras en función de su contexto de aparición dentro del corpus.

---

## Descripción del Corpus

Se utiliza como fuente textual una versión en texto plano de **La Biblia Latinoamericana**, descargada desde [archive.org](https://archive.org/stream/la-biblia-latinoamericana_202308/La%20Biblia%20Latinoamericana_djvu.txt). Este corpus fue elegido por su extensión, riqueza semántica y estilo narrativo, lo cual lo convierte en un excelente recurso para entrenar vectores de palabras.

---

## Estructura del Trabajo

### 1. Preprocesamiento

- El texto fue segmentado y limpiado para obtener un corpus estructurado.
- Se tokenizaron las oraciones en palabras.
- Se generaron listas de listas de tokens, que representan secuencias de palabras por documento.

### 2. Entrenamiento de embeddings con Word2Vec

- Se utilizó la librería **Gensim** para entrenar un modelo **Word2Vec** con diversas configuraciones.
- Se incorporó un **callback personalizado** para visualizar la evolución del error durante el entrenamiento por época.
- Se experimentó con distintos tamaños de dimensión vectorial (`vector_size`) para observar el impacto sobre la calidad de los embeddings.

```python
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
```

### 3. Análisis de resultados

Una vez entrenado el modelo:

- Se evaluaron las palabras más similares semánticamente a otras, utilizando `model.wv.most_similar()`.
- Se realizaron pruebas con operaciones vectoriales como:
  - **Analogías semánticas** (e.g., "rey" - "hombre" + "mujer")
  - **Cálculo de distancias** entre términos clave
- Se visualizó la **distribución espacial** de palabras usando **TSNE** para reducir la dimensionalidad y generar una representación 2D.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
```

### 4. Visualización

- Se generó un gráfico TSNE con una muestra del vocabulario entrenado.
- Se identificaron clústeres de palabras relacionadas por tema o connotación.

---

## Resultados y Observaciones

- La dimensionalidad del embedding influye fuertemente en la calidad semántica: vectores más grandes capturan mayor riqueza, pero requieren más datos y tiempo de entrenamiento.
- Se logró detectar agrupamientos coherentes entre términos con alta cercanía semántica en el contexto bíblico.
- Se observaron relaciones análogas que validan la calidad de los embeddings generados.

---

## Conclusiones

Este segundo desafío permitió profundizar en el aprendizaje de representaciones distribuidas de palabras. A partir de un corpus personalizado y mediante el entrenamiento desde cero, se obtuvo una representación semántica útil para diversas tareas de NLP.

Se consolidaron conocimientos clave:

- Uso práctico del modelo Word2Vec.
- Análisis de similitud y analogía vectorial.
- Visualización de espacios de palabras.
- Impacto de la dimensión del embedding y del corpus en los resultados.

---

## Requisitos

Este notebook utiliza las siguientes bibliotecas:

```bash
pip install gensim matplotlib scikit-learn
```

---

## Estructura del código

- **Carga y preprocesamiento del corpus**
- **Definición de callbacks personalizados**
- **Entrenamiento con Gensim Word2Vec**
- **Exploración del vocabulario entrenado**
- **Visualización con TSNE**
- **Análisis cualitativo de relaciones semánticas**

# Desafío 3 - Redes Neuronales Recurrentes en el procesamiento de documentos

---

## Objetivos del Desafío

Este trabajo práctico se centra en la **implementación y entrenamiento de un modelo de lenguaje** con tokenización a nivel de **caracteres**, utilizando **redes neuronales recurrentes (RNN)**. El objetivo es generar texto de manera autónoma, aprendiendo la estructura secuencial de un corpus elegido.

---

## Descripción del Corpus

Se utilizó como corpus el texto "El Príncipe" de Nicolas Maquiavelo. ['https://www.textos.info/nicolas-maquiavelo/el-principe/ebook']
Este texto fue elegido por su contenido de patrones y estructuras semánticas repetitivas, lo que lo convierte en un buen caso de estudio para modelado de secuencias.

---

## Estructura del Trabajo

### 1. Preprocesamiento

- Se descargó el corpus y se limpiaron los datos.
- Se tokenizó el texto a nivel de **caracteres**, construyendo un vocabulario con todos los símbolos presentes.
- Se generaron pares de entrada y salida para entrenar el modelo: cada secuencia de caracteres sirve para predecir el siguiente carácter.

### 2. Modelado y entrenamiento

- Se utilizaron diferentes variantes de redes neuronales recurrentes:
  - **SimpleRNN**
  - **LSTM**
  - **GRU**
- Se implementó un **callback de perplejidad** para monitorear el entrenamiento.
- Se separaron datos en entrenamiento y validación usando `train_test_split`.

```python
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Embedding
```

### 3. Generación de texto
Una vez entrenado el modelo, se generaron nuevas secuencias de texto.

Se exploraron diferentes estrategias:

- Greedy search

- Beam search (determinístico y estocástico)

Se evaluó el efecto de la temperatura en la creatividad y coherencia de los textos generados.

## Resultados y Observaciones
El modelo aprendió patrones generales del corpus, lo que posibilitó la generación de texto con estructuras similares.

Las arquitecturas mas complejas, como LSTM o GRU, mostraron en esta práctica apenas mejor rendimiento frente a SimpleRNN.

Ajustar la temperatura afectó directamente la creatividad:

- Temperaturas bajas produjeron textos repetitivos.

- Temperaturas altas generaron secuencias más variadas, aunque con menor coherencia.

## Conclusiones
Este tercer desafío permitió aplicar conceptos avanzados de modelado de secuencias usando redes neuronales. Se exploraron distintas arquitecturas recurrentes y se implementaron estrategias para la generación automática de texto, evaluando tanto rendimiento como creatividad.

Se consolidaron conocimientos clave:

- Entrenamiento de modelos de lenguaje a nivel de carácter.

- Implementación de RNN, LSTM y GRU.

- Uso de beam search y temperatura para generar texto.

- Medición de perplejidad como criterio de evaluación.

## Requisitos
Este notebook utiliza las siguientes bibliotecas:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

## Estructura del código
- Descarga y preprocesamiento del corpus
- Tokenización por caracteres
- Creación de conjuntos de entrenamiento y validación
- Construcción y entrenamiento de modelos RNN
- Generación de texto con distintas estrategias
- Análisis de perplejidad y calidad del texto generado

# Desafío 4 - Arquitectura Encoder-Decoder en el procesamiento de texto

---

## Objetivos del Desafío

Este trabajo práctico tiene como objetivo implementar un **modelo de traducción automática** basado en redes neuronales. Se aborda la construcción de un sistema **seq2seq (sequence-to-sequence)** utilizando una arquitectura **encoder-decoder con LSTM**, capaz de traducir oraciones del inglés al español.

---

## Descripción del Corpus

Se emplea un dataset de pares de oraciones inglés-español proveniente del **Tatoeba Project**, disponible en el sitio [ManyThings.org](https://www.manythings.org/anki/). El corpus contiene frases cortas y su traducción correspondiente, lo cual lo convierte en una excelente base para tareas de traducción neuronal.

---

## Estructura del Trabajo

### 1. Preprocesamiento

- Se descargó y cargó el archivo `spa-eng.txt` con pares de oraciones.
- Se limpiaron los textos, eliminando signos de puntuación y caracteres especiales.
- Se separaron datos en conjuntos de entrenamiento y validación.
- Se tokenizaron los textos utilizando la clase `Tokenizer` de Keras.
- Se utilizaron **tokens de inicio y fin de secuencia** (`<sos>` y `<eos>`) para preparar el modelo seq2seq.

### 2. Modelado seq2seq con LSTM

- Se construyó una arquitectura encoder-decoder:
  - El **encoder** procesa la secuencia en inglés y genera un vector de contexto.
  - El **decoder** toma ese vector y genera la oración traducida en español, un token a la vez.
- Se entrenó el modelo utilizando `teacher forcing`, alimentando las secuencias esperadas al decoder durante el entrenamiento.
- Se emplearon capas `Embedding` y `LSTM`, y funciones de pérdida categórica para modelar la probabilidad de la siguiente palabra.

```python
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
```

## Resultados y Observaciones
El modelo fue capaz de aprender traducciones básicas con un tamaño reducido de datos.

Se evaluó el desempeño observando las frases generadas y comparándolas con las referencias.

La calidad de la traducción mejoró al:

- Aumentar la cantidad de datos de entrenamiento.
- Aumentar el número de épocas de entrenamiento.
- Ajustar la longitud máxima de secuencias y el tamaño del vocabulario.

## Conclusiones
Este desafío permitió aplicar técnicas avanzadas de modelado secuencial para tareas de traducción automática. La implementación de un modelo seq2seq desde cero permitió comprender la mecánica del encoder-decoder y su entrenamiento.

Se reforzaron conceptos clave:

- Tokenización con Tokenizer.
- Arquitectura seq2seq con Embedding y LSTM.
- Preparación de datos con tokens de inicio y fin de secuencia.
- Inferencia secuencial para generación de traducción.

## Requisitos
Este notebook utiliza las siguientes bibliotecas:

```bash
pip install numpy pandas tensorflow keras
```

## Estructura del código
- Descarga y lectura del corpus Tatoeba
- Limpieza y preprocesamiento del texto
- Tokenización y vectorización con Keras
- Diseño de modelo encoder-decoder con LSTM
- Entrenamiento supervisado del modelo
- Generación de traducciones y análisis cualitativo

