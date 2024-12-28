import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Cargar los modelos entrenados
@st.cache_resource
def load_model(model_name):
    if model_name == "Linear Regression":
        return joblib.load("models/linear_regression.pkl")
    elif model_name == "Random Forest":
        return joblib.load("models/random_forest.pkl")
    else:
        raise ValueError("Modelo no válido")

# Cargar el dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/AB_NYC_2019.csv")

df = load_data()

# Preparar datos
def preparar_datos(df):
    # Filtrar precios válidos
    df = df[(df['price'] > 0) & (df['price'] <= 1000)]

    # Codificar variables categóricas
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

    # Variables independientes
    X = df[['neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
            'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
            'room_type_Private room', 'room_type_Shared room']]
    return X

# Barra lateral para seleccionar el modelo y las características
st.sidebar.title("Selecciona el algoritmo")
model_choice = st.sidebar.selectbox("Selecciona el modelo:", ("Linear Regression", "Random Forest"))

# Seleccionar características para la predicción
st.sidebar.title("Selecciona las características")
neighbourhood = st.sidebar.selectbox("Grupo de vecindario", df['neighbourhood_group'].unique())
room_type = st.sidebar.selectbox("Tipo de habitación", df['room_type'].unique())

# Filtrar datos de entrada según selección del usuario
input_data = {
    'neighbourhood_group_Brooklyn': [1 if neighbourhood == 'Brooklyn' else 0],
    'neighbourhood_group_Manhattan': [1 if neighbourhood == 'Manhattan' else 0],
    'neighbourhood_group_Queens': [1 if neighbourhood == 'Queens' else 0],
    'neighbourhood_group_Staten Island': [1 if neighbourhood == 'Staten Island' else 0],
    'room_type_Private room': [1 if room_type == 'Private room' else 0],
    'room_type_Shared room': [1 if room_type == 'Shared room' else 0]
}

input_df = pd.DataFrame(input_data)

# Mostrar la selección del usuario
st.write(f"**Vecindario seleccionado:** {neighbourhood}")
st.write(f"**Tipo de habitación seleccionado:** {room_type}")
st.write("**Características seleccionadas:**")
st.dataframe(input_df)

# Cargar el modelo seleccionado
model = load_model(model_choice)

# Realizar la predicción (eliminar línea duplicada)
prediction = model.predict(input_df)

# Mostrar el resultado de la predicción
st.write("**Resultado de la Predicción:**")

if model_choice == "Linear Regression":
    st.write(f"El precio estimado del apartamento es: ${prediction[0]:.2f}")
elif model_choice == "Random Forest":
    # Si la predicción es un valor de probabilidad, multiplicamos por 100 para obtener el porcentaje
    probabilidad_abajo = prediction[0] * 100
    probabilidad_arriba = (1 - prediction[0]) * 100

    # Mostrar el porcentaje de probabilidad
    st.write(f"Probabilidad de estar por debajo del precio medio: {probabilidad_abajo:.2f}%")
    st.write(f"Probabilidad de estar por encima del precio medio: {probabilidad_arriba:.2f}%")

# Botón para realizar la predicción
if st.sidebar.button("Realizar Predicción"):
    st.write("Predicción realizada.")


