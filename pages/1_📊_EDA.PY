###VA A CONTENER LA PAGINA EDA###
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.eda import panoramaGeneral, resumenEstadistico, variablesNum, variablesCat, correlación
from src.hipotesis import df

@st.cache_data
def load_data():
    return pd.read_csv("data/AB_NYC_2019.csv")  

# Configuración inicial
st.set_page_config(page_title="Análisis Exploratorio de Datos", layout="wide")

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
    ("Panorama General", "Resumen Estadístico", "Análisis Numérico", "Análisis Categórico", "Correlación", "Reporte Completo")
)

if options == "Panorama General":
    st.header("Panorama General del Dataset")
    with st.spinner("Generando información..."):
        st.write(panoramaGeneral.__doc__)
        panoramaGeneral()

elif options == "Resumen Estadístico":
    st.header("Resumen Estadístico del Dataset")
    with st.spinner("Generando resumen..."):
        st.write(resumenEstadistico.__doc__)
        resumenEstadistico()

elif options == "Análisis Numérico":
    st.header("Análisis de Variables Numéricas")
    with st.spinner("Analizando variables numéricas..."):
        st.write(variablesNum.__doc__)
        variablesNum()

elif options == "Análisis Categórico":
    st.header("Análisis de Variables Categóricas")
    with st.spinner("Analizando variables categóricas..."):
        st.write(variablesCat.__doc__)
        variablesCat()

elif options == "Correlación":
    st.header("Matriz de Correlación")
    with st.spinner("Calculando correlaciones..."):
        st.write(correlación.__doc__)
        correlación()

elif options == "Reporte Completo":
    st.header("Reporte Exploratorio Completo")
    with st.spinner("Generando reporte..."):
        profile = ProfileReport(df, minimal=False)
        st_profile_report(profile)
