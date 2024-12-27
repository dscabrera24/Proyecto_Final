import streamlit as st
import pandas as pd
from src.hipotesis import (
    hipotesis1,
    hipotesis2Grafica,
    hipotesis2Analisis,
    hipotesis2Mapa,
    hipotesis3,
    hipotesis4,
    hipotesis5,
    df
)

# Configuración inicial
st.set_page_config(page_title="Hipótesis", layout="wide")

# Página principal
st.title("Pruebas de Hipótesis")
st.write("Esta página explora diferentes hipótesis basadas en los datos del dataset utilizado.")

# Vista previa del dataset
st.write("**Vista previa del dataset:**")
st.dataframe(df.head())

# Sidebar con opciones
st.sidebar.title("Hipótesis a Evaluar")
hypothesis_options = st.sidebar.selectbox(
    "Selecciona una hipótesis para evaluar:",
    ("Hipótesis 1", "Hipótesis 2", "Hipótesis 3", "Hipótesis 4", "Hipótesis 5")
)

if hypothesis_options == "Hipótesis 1":
    st.header("Hipótesis 1")
    st.write("**Análisis de precios promedio por tipo de habitación.**")
    with st.spinner("Calculando..."):
        hipotesis1()

elif hypothesis_options == "Hipótesis 2":
    st.header("Hipótesis 2")
    st.write("**Distribución de precios por grupo de vecindario y análisis estadístico.**")
    with st.spinner("Generando gráfico..."):
        hipotesis2Grafica()
    st.write("**Análisis ANOVA y comparaciones pareadas:**")
    with st.spinner("Realizando análisis..."):
        st.dataframe(hipotesis2Analisis())
    st.write("**Mapa de precios medianos por vecindario:**")
    with st.spinner("Generando mapa..."):
        st.pydeck_chart(hipotesis2Mapa())

elif hypothesis_options == "Hipótesis 3":
    st.header("Hipótesis 3")
    st.write("**Análisis de tasas de ocupación según la estancia mínima.**")
    with st.spinner("Evaluando hipótesis..."):
        hipotesis3()

elif hypothesis_options == "Hipótesis 4":
    st.header("Hipótesis 4")
    st.write("**Comparación de tasas de ocupación entre zonas turísticas y no turísticas.**")
    with st.spinner("Analizando datos..."):
        hipotesis4()

elif hypothesis_options == "Hipótesis 5":
    st.header("Hipótesis 5")
    st.write("**Relación entre precio, disponibilidad y vecindarios.**")
    with st.spinner("Ejecutando análisis..."):
        hipotesis5()
