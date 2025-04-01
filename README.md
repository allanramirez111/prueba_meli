# 🛠️ Análisis de Moderación de Productos en E-Commerce

Este repositorio contiene un **dashboard interactivo en Streamlit** para explorar y analizar el proceso de **moderación de publicaciones** en Mercado Libre. El objetivo principal es entender los patrones que llevan a que un producto sea **moderado**, **identificado como falsificado (Fake)** o **revertido (Rollback)**, así como realizar análisis exploratorios avanzados y simulaciones analíticas.

---

## 🚀 Características del Dashboard

- **Análisis de Flags (Moderado, Fake, Rollback)**  
  Permite filtrar, agrupar y visualizar métricas descriptivas por cualquier combinación de variables categóricas y numéricas.

- **Relaciones y Correlaciones**  
  Construcción de **boxplots** de variables numéricas según combinaciones de variables categóricas, subdivididas por una variable auxiliar como "Fake".

- **PCA + Clustering**  
  Análisis de componentes principales (PCA) con variables mixtas (numéricas + categóricas) + segmentación jerárquica (clustering) + análisis estadístico por cluster.

- **Clasificación de Fakes**  
  Entrenamiento rápido de modelos clasificadores (Logistic Regression, Random Forest, XGBoost) para predecir productos fake. Se muestran métricas e **importancia de variables agregadas**.

---

## 📁 Estructura del Proyecto

```
📦 project_root/
├── app.py                      # Script principal del tablero Streamlit
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
│
├── config/
│   └── settings.py             # Ruta de carga de datos y otras variables globales
│
├── datos/
│   └── base.csv                # Dataset principal (formato esperado)
│
├── cuadernos/
│   └── analisis.ipynb          # Análisis exploratorio individual (Jupyter)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Transformación del dataset y generación de precio en USD
│   ├── flag_analysis.py       # Métricas descriptivas agrupadas por columnas
│   ├── relationships.py       # Boxplot por combinación de variables categóricas
│   ├── pca_analysis.py        # PCA mixto con codificación de variables
│   ├── cluster_analysis.py    # Clustering y métricas de cada cluster
│   ├── classification.py      # Clasificación de fakes y evaluación de importancia
│   └── visualizations.py      # Visualizaciones gráficas (PCA, clusters, etc.)
```

---

## 🧪 Requisitos y Ejecución

### Instalación

```bash
pip install -r requirements.txt
```

### Ejecutar el Dashboard

```bash
streamlit run app.py
```

---

## 🧠 Técnicas Utilizadas

- Exploración y visualización dinámica con **Streamlit**
- **Estadística descriptiva** por segmentos
- **PCA (Análisis de Componentes Principales)** sobre datos mixtos
- **Clustering jerárquico (Agglomerative)**
- **Clasificación binaria** (`Fake`) con:
  - Regresión logística
  - Random Forest
  - XGBoost


---

## ✅ Objetivos del Análisis

- **Descripción y estructura** de los atributos del dataset
- **Distribución y patrones**: detección de anomalías o concentraciones
- **Relaciones y correlaciones** entre factores e impacto sobre moderaciones y falsificaciones
- **Evaluación analítica** de filtros de exclusión por país, dominio o características técnicas

---

## 📬 Autor

Desarrollado como solución técnica a un challenge de análisis de datos y moderación de contenido por Allan Ramirez.
