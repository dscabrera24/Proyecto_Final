import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st


# Tratamiento de valores nulos y outliers
def tratar_nulos_y_outliers(df):
    """
    Rellena valores nulos en columnas categóricas y numéricas.
    Recorta outliers utilizando el percentil 99.
    """
    df = df.copy()
    df['name'] = df['name'].fillna('Desconocido')
    df['host_name'] = df['host_name'].fillna('Desconocido')
    df['last_review'] = df['last_review'].fillna('Sin reseñas')
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    for col in ['price', 'minimum_nights', 'reviews_per_month', 'calculated_host_listings_count']:
        limite_superior = df[col].quantile(0.99)
        df[col] = np.clip(df[col], None, limite_superior)

    return df


### HIPÓTESIS 1 ###
def hipotesis1(df):
    """
    Analiza los precios promedio por tipo de habitación y genera un gráfico de caja (boxplot).
    """
    avg_prices = df.groupby("room_type")["price"].mean().sort_values(ascending=False)
    st.write("**Precio promedio por tipo de habitación:**")
    st.dataframe(avg_prices)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="room_type", y="price", showfliers=False)
    plt.title("Distribución de Precios por Tipo de Habitación")
    plt.xlabel("Tipo de Habitación")
    plt.ylabel("Precio")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)


### HIPÓTESIS 2 ###
def hipotesis2(df):
    """
    Realiza un análisis ANOVA y pruebas T pareadas con corrección de Bonferroni.
    Genera un gráfico de caja para la distribución de precios por vecindario.
    """
    # ANOVA
    df_anova = df[['neighbourhood_group', 'price']].dropna()
    anova_result = f_oneway(
        *[df_anova[df_anova['neighbourhood_group'] == group]['price'] for group in df_anova['neighbourhood_group'].unique()]
    )
    st.write(f"**Resultado ANOVA:**\nValor p: {anova_result.pvalue:.4e}")

    # Comparaciones pareadas con Bonferroni
    results = []
    for group1, group2 in combinations(df_anova['neighbourhood_group'].unique(), 2):
        t_stat, p_value = ttest_ind(
            df_anova[df_anova['neighbourhood_group'] == group1]['price'],
            df_anova[df_anova['neighbourhood_group'] == group2]['price'],
            equal_var=False
        )
        results.append({'Grupo 1': group1, 'Grupo 2': group2, 'p-valor': p_value})

    bonferroni_results = pd.DataFrame(results)
    bonferroni_results['p-valor Ajustado'] = multipletests(bonferroni_results['p-valor'], method='bonferroni')[1]
    st.write("**Resultados Bonferroni:**")
    st.dataframe(bonferroni_results)

    # Gráfico
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='neighbourhood_group', y='price', palette='Set2')
    plt.title('Distribución de precios por grupo de vecindario', fontsize=16)
    plt.xlabel('Grupo de vecindario', fontsize=12)
    plt.ylabel('Precio', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)


### HIPÓTESIS 3 ###
def hipotesis3(df):
    """
    Analiza la tasa de ocupación según la estancia mínima y genera un gráfico de caja.
    """
    df['tasa_ocupacion'] = 1 - (df['availability_365'] / 365)
    df['categoria_estancia'] = np.where(df['minimum_nights'] == 1, '1 noche', 'Más de 1 noche')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='categoria_estancia', y='tasa_ocupacion', palette='Set2')
    plt.title('Comparación de la Ocupación por Estancia Mínima')
    plt.xlabel('Estancia Mínima')
    plt.ylabel('Tasa de Ocupación')
    st.pyplot(plt)


### HIPÓTESIS 4 ###
def hipotesis4(df):
    """
    Compara la disponibilidad entre zonas turísticas y no turísticas mediante prueba T.
    Genera un gráfico de caja para la disponibilidad.
    """
    tourist_zones = ['Manhattan', 'Brooklyn']
    tourist_area_data = df[df['neighbourhood_group'].isin(tourist_zones)]['availability_365']
    non_tourist_area_data = df[~df['neighbourhood_group'].isin(tourist_zones)]['availability_365']

    stat, p = ttest_ind(tourist_area_data, non_tourist_area_data, equal_var=False)
    st.write(f"**Prueba t de Student para Disponibilidad:**\nEstadístico={stat}, p-valor={p}")

    data = pd.DataFrame({
        'Disponibilidad': list(tourist_area_data) + list(non_tourist_area_data),
        'Zona': ['Turística'] * len(tourist_area_data) + ['No Turística'] * len(non_tourist_area_data)
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Zona', y='Disponibilidad', data=data, palette='Set2')
    plt.title('Comparación de la Disponibilidad entre Zonas')
    plt.xlabel('Tipo de Zona')
    plt.ylabel('Días Disponibles')
    st.pyplot(plt)


### HIPÓTESIS 5 ###
def hipotesis5(df):
    """
    Realiza una regresión lineal para analizar la relación entre precio y disponibilidad.
    Genera gráficos de dispersión para las predicciones.
    """
    data = df[['price', 'availability_365', 'neighbourhood_group', 'room_type']].dropna()
    data = pd.get_dummies(data, columns=['neighbourhood_group', 'room_type'], drop_first=True)

    X = data.drop('availability_365', axis=1)
    y = data['availability_365']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"**Resultados de la Regresión Lineal:**\nR² Score: {r2_score(y_test, y_pred):.4f}")

    coeficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    st.write("**Coeficientes del Modelo:**")
    st.dataframe(coeficients)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.title("Valores Reales vs. Predicciones")
    plt.xlabel("Valores reales de Disponibilidad")
    plt.ylabel("Predicciones de Disponibilidad")
    st.pyplot(plt)
