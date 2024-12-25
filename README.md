# Proyecto Integrador - Análisis de Datos y Machine Learning

## Descripción del Proyecto
Este proyecto tiene como objetivo analizar datos relacionados con los alojamientos de Airbnb en la ciudad de Nueva York, aplicando técnicas de **Exploratory Data Analysis (EDA)** y **Machine Learning** para obtener insights y predecir patrones.

## Estructura del Proyecto
El proyecto está dividido en las siguientes secciones:

### 1. Introducción
- Integrantes del equipo:
  - Mauricio Enrique Vásquez Ramírez (mauricio.01.ra@gmail.com)
  - Delia Francella Vilchez Ruiz (00540524@uca.edu.sv)
  - Karla Margarita Navarrete Gálvez (navarrete02@gmail.com)
  - René Benjamín Castaneda Alemán (benja.castas@gmail.com)
  - Daniela Sofía Cabrera Campos (00289719@uca.edu.sv)

- Dataset seleccionado:
  - Nombre: **New York City Airbnb Open Data**
  - Fuente: [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
  - Descripción: Contiene 48,895 registros con 16 columnas que describen precios, ubicaciones, disponibilidad y otros factores relacionados con los alojamientos.

### 2. Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exhaustivo para explorar las siguientes áreas:

- **Distribución de variables**: Visualizaciones y estadísticas descriptivas.
- **Identificación de patrones**: Relación entre precio, ubicación y disponibilidad.
- **Hipótesis evaluadas**:
  1. La ubicación influye en el precio promedio.
  2. Los alojamientos con más reseñas tienen precios más altos.
  3. Los anfitriones con más propiedades ofrecen precios más bajos.
  4. La disponibilidad de los alojamientos está influenciada por el tipo de habitación.
  5. Los alojamientos con precios más bajos están disponibles menos días al año.

### 3. Aplicación de Machine Learning
Se seleccionó el algoritmo de Machine Learning basado en los hallazgos del EDA:
- **Tipo de modelo**: Supervisado (Regresión lineal y árboles de decisión).
- **Razón de selección**: Adecuado para predecir precios y analizar relaciones entre variables cuantitativas.
- **Evaluación**: Métricas como el coeficiente R² y el error cuadrático medio (MSE) mostraron el ajuste del modelo y áreas de mejora.

### 4. Conclusiones
- **Insights clave**:
  - Los precios de los alojamientos varían significativamente por ubicación.
  - Los alojamientos económicos tienden a estar ocupados más frecuentemente, lo que limita su disponibilidad.
  - Otros factores, como estacionalidad, también influyen en la disponibilidad.

- **Limitaciones y recomendaciones**:
  - Incluir variables adicionales podría mejorar el rendimiento del modelo.
  - Evaluar tendencias temporales podría aumentar el valor predictivo.

## Requisitos
- **Lenguaje**: Python 3.8+
- **Librerías**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Dataset**: Proporcionado en formato CSV desde Kaggle.

## Instrucciones de Uso
1. Clonar este repositorio:
   ```bash
   git clone <repositorio>
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el archivo principal:
   ```bash
   python proyecto_integrador_2.py
   ```

## Colaboradores
Este proyecto fue realizado como parte de un curso de Análisis de Datos y Machine Learning, demostrando habilidades en exploración de datos y modelado predictivo.
