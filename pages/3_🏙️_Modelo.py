###VA A CONTENER EL CODIGO DE LA PAGINA MODELO###
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Para cargar el modelo entrenado
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración inicial
st.set_page_config(page_title="Modelo Predictivo", layout="wide")

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load("linear_regression.pkl")

model = load_model()

# Página principal
st.title("Predicción del Precio de Apartamentos")
st.write("Carga un archivo CSV con las características requeridas para predecir el precio de los apartamentos.")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV
    input_data = pd.read_csv(uploaded_file)
    st.write("**Vista previa de los datos cargados:**")
    st.dataframe(input_data.head())

    try:
        # Realizar predicciones
        st.write("**Realizando predicciones...**")
        predictions = model.predict(input_data)
        input_data["Predicted Price"] = predictions

        # Mostrar predicciones
        st.write("**Resultados con precios predichos:**")
        st.dataframe(input_data)

        # Guardar resultados
        st.download_button(
            label="Descargar resultados",
            data=input_data.to_csv(index=False).encode('utf-8'),
            file_name="predicciones.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error al realizar las predicciones: {e}")

else:
    st.write("Sube un archivo CSV para realizar predicciones.")

# Mostrar métricas del modelo
st.sidebar.title("Métricas del Modelo")
st.sidebar.write("**MAE:** Media del error absoluto")
st.sidebar.write("**MSE:** Media del error cuadrático")
st.sidebar.write("**R² Score:** Coeficiente de determinación")

# Cargar datos de métricas
@st.cache_data
def load_metrics():
    return {
        "MAE": 42.87,
        "MSE": 4276.64,  
        "R2": 0.40  
    }

metrics = load_metrics()

st.sidebar.metric(label="MAE", value=f"{metrics['MAE']:.2f}")
st.sidebar.metric(label="MSE", value=f"{metrics['MSE']:.2f}")
st.sidebar.metric(label="R² Score", value=f"{metrics['R2']:.2f}")
