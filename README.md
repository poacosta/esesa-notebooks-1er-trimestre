# ESESA Notebooks - Primer Trimestre

## Autores

* Pedro Orlando Acosta Pereira
* Rushabh Patel
* Francisco José Salas Gómez

## Notebooks

### Exploring Mental Health Data

* **Dataset:** https://www.kaggle.com/competitions/playground-series-s4e11/data

#### Solución propuesta (Resumen)

**Enfoque:** Clasificación de un dataset basado en datos de encuestas sobre depresión. Aprendizaje supervisado.

**Estrategia implementada:**

* Explorar tres modelos con configuraciones específicas para datos desbalanceados:
  * DecisionTree
  * RandomForest
  * GradientBoosting

* Priorizar recall sobre precisión.
* Implementar funciones fáciles de reusar para entrenar y coleccionar datos de los modelos.

---

### Gender Classification

* **Dataset:** https://www.kaggle.com/datasets/ashishjangra27/gender-recognition-200k-images-celeba

#### Solución propuesta (Resumen)

**Enfoque:** Clasificación binaria de género usando imágenes.

**Estrategia implementada:**

* Arquitectura CNN con 3 bloques convolucionales:
  * Cada bloque: Conv2D (ReLU) + BatchNormalization + MaxPooling2D + Dropout.
  * Incremento progresivo de filtros (32→64→128).
* Capas densas finales (ReLU (1000) y Softmax (2)) para clasificación.
* Técnicas anti-sobreajuste: BatchNormalization y Dropout.

---

### Audio Cats and Dogs

* **Dataset:** https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs

#### Solución propuesta (Resumen)

**Enfoque:** Clasificación binaria de audio (perros/gatos), emulando el dataset clásico de imágenes.

**Estrategia implementada:**

* Preprocesamiento:
  * Genera espectrogramas desde archivos WAV.
  * Convierte audio a imágenes de 369x496 píxeles.
  * Organiza en estructura de directorios por clase.
* Arquitectura CNN con 3 bloques convolucionales:
  * Cada bloque: Conv2D (ReLU) + MaxPooling2D + Dropout(0.1).
  * Incremento progresivo de filtros (32→64→128).
* Capas densas finales con ReLU (512) y Softmax (2).

---

### CNN News

* **Dataset:** https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

#### Soluciones propuesta (Resumen)

**Enfoque #1:** Análisis de tópicos en un dataset de noticias.

**Estrategia implementada #1:**

* Procesamiento de texto:
  * Preprocesador: tokenización/preparación del texto.
  * Encoder BERT: modelo base (12 capas, 768 dimensiones, 12 heads).
  * Usar BERT para embeddings.
  * Generar embeddings por lotes (batch_size=64).
* Extracción de tópicos:
  * Reducir dimensionalidad con PCA (5 componentes).
  * Combinar con TF-IDF para extraer keywords.
  * Clasificar textos por similitud de componentes.

**Enfoque #2:** Análisis híbrido de tópicos en el dataset.

**Estrategia implementada #2:**

* Preprocesamiento:
  * Limpieza (quitar stopwords).
  * Vectorización TF-IDF para clustering.
  * Tokenización y padding (500) para secuencias.
* Modelado dual:
  * Clusterización con K-Means (5 clusters).
  * RNN para clasificación
    * Embedding (5000→100)
    * LSTM con Dropout (0.2)
