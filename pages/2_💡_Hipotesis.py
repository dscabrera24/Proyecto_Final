import streamlit as st
import pandas as pd
from src.hipotesis import (
    hipotesis1,
    hipotesis2,
    hipotesis3,
    hipotesis4,
    hipotesis5
)

# Configuración inicial
st.set_page_config(page_title="Hipótesis", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    """
    Carga el dataset desde un archivo local y realiza el tratamiento necesario.
    """
    df = pd.read_csv("data/AB_NYC_2019.csv")
    return df

# Cargar dataset
df = load_data()

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

# Contenedor vacío para la hipótesis seleccionada
output_container = st.empty()

# Evaluar cada hipótesis según la selección
if hypothesis_options == "Hipótesis 1":
    output_container.header("Hipótesis 1")
    output_container.write("**Los apartamentos completos tienen un precio promedio más alto por noche en comparación con las habitaciones privadas y compartidas.**")
    with st.spinner("Calculando..."):
        try:
            hipotesis1(df)
        except Exception as e:
            output_container.error(f"Error en Hipótesis 1: {e}")

elif hypothesis_options == "Hipótesis 2":
    output_container.header("Hipótesis 2")
    output_container.write("**Los alojamientos ubicados en Manhattan tienen un precio promedio más alto que en otras zonas.**")
    with st.spinner("Analizando hipótesis..."):
        try:
            hipotesis2(df)
        except Exception as e:
            output_container.error(f"Error en Hipótesis 2: {e}")

elif hypothesis_options == "Hipótesis 3":
    output_container.header("Hipótesis 3")
    output_container.write("**No hay diferencia significativa en la tasa de ocupación promedio entre los alojamientos con una estancia mínima de 1 noche y los alojamientos con una estancia mínima superior a 1 noche.**")
    with st.spinner("Evaluando hipótesis..."):
        try:
            hipotesis3(df)
        except Exception as e:
            output_container.error(f"Error en Hipótesis 3: {e}")

elif hypothesis_options == "Hipótesis 4":
    output_container.header("Hipótesis 4")
    output_container.write("**Los alojamientos en zonas turísticas tienen una mayor tasa de ocupación en comparación con los alojamientos en zonas no turísticas.**")
    with st.spinner("Analizando datos..."):
        try:
            hipotesis4(df)
        except Exception as e:
            output_container.error(f"Error en Hipótesis 4: {e}")

elif hypothesis_options == "Hipótesis 5":
    output_container.header("Hipótesis 5")
    output_container.write("**Los alojamientos con precios más bajos están disponibles menos días al año**")
    with st.spinner("Ejecutando análisis..."):
        try:
            hipotesis5(df)
        except Exception as e:
            output_container.error(f"Error en Hipótesis 5: {e}")

