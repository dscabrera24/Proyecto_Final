import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial
st.set_page_config(page_title="Análisis Exploratorio de Datos", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/AB_NYC_2019.csv")

# Cargar los datos
df = load_data()

# Página principal
st.title("Análisis Exploratorio de Datos")
st.write("Esta sección muestra un análisis exploratorio de los datos usados para entrenar el modelo.")

# Vista previa del dataset
st.write("**Vista previa del dataset:**")
st.dataframe(df.head())

# Sidebar con opciones
st.sidebar.title("Opciones de Análisis")
options = st.sidebar.radio(
    "Selecciona el análisis a realizar:",
    ("Panorama General", "Resumen Estadístico", "Distribuciones Numéricas", "Distribuciones Categóricas", "Correlación", "Visualización de Precio")
)

if options == "Panorama General":
    st.header("Panorama General del Dataset")
    st.write("**Tamaño del dataset:**")
    st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    st.write("**Columnas y tipos de datos:**")
    st.write(df.dtypes)
    st.write("**Valores nulos por columna:**")
    st.write(df.isnull().sum())

elif options == "Resumen Estadístico":
    st.header("Resumen Estadístico del Dataset")
    st.write(df.describe())

elif options == "Distribuciones Numéricas":
    st.header("Distribución de Variables Numéricas")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)

elif options == "Distribuciones Categóricas":
    st.header("Distribución de Variables Categóricas")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)

elif options == "Correlación":
    st.header("Matriz de Correlación")
    
    # Filtrar solo columnas numéricas
    numeric_df = df.select_dtypes(include=["number"])
    
    # Calcular matriz de correlación
    corr = numeric_df.corr()
    
    # Crear el gráfico de calor
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Correlación")
    st.pyplot(fig)

elif options == "Visualización de Precio":
    st.header("Relación entre Precio y otras Variables")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if col != "price":  # Asegúrate de que no sea la misma variable
            fig, ax = plt.subplots()
            sns.scatterplot(x=col, y="price", data=df, ax=ax)
            ax.set_title(f"Precio vs {col}")
            st.pyplot(fig)

