###VA A CONTENER EL CODIGO DE LA PAGINA MODELO###
import streamlit as st
import openai
import pandas as pd

# Configurar clave API de OpenAI
openai.api_key = "ClavePendiente"

# Cargar el dataset
@st.cache_data
def load_data():
    return pd.read_csv("AB_NYC_2019.csv")

data = load_data()

# consultar la base de datos y generar respuestas
@st.cache_resource
def query_openai(question, dataframe):
    summary = (
        f"Datos sobre alojamientos en Nueva York: "
        f"El dataset tiene {len(dataframe)} registros y las siguientes columnas: {', '.join(dataframe.columns)}. "
        f"Por ejemplo, {dataframe.iloc[0].to_dict()}\n\n"
    )
    
    prompt = summary + f"Pregunta: {question}\nRespuesta:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response["choices"][0]["text"].strip()

# Configuración inicial de Streamlit
st.set_page_config(page_title="Chatbot con OpenAI", layout="wide")

# Título de la aplicación
st.title("Chatbot para Análisis de Datos de Airbnb")
st.write("Haz preguntas sobre los datos del dataset de alojamientos en Nueva York.")

# Mostrar una vista previa del dataset
if st.checkbox("Mostrar vista previa del dataset"):
    st.dataframe(data.head())

# Entrada de preguntas del usuario
question = st.text_input("Escribe tu pregunta:")

if question:
    with st.spinner("Generando respuesta..."):
        try:
            answer = query_openai(question, data)
            st.success("Respuesta del Chatbot:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {e}")

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit y OpenAI API.")
