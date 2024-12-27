import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import src

#!pip install ydata-profiling
from ydata_profiling import ProfileReport

df = src.tratarNulos

def reporte ():
    profile = ProfileReport(df, minimal=False)
    return profile.to_notebook_iframe()

# Configuración de estilo para gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)

def panoramaGeneral():
    print("Dimensiones del DataFrame:")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

    print("Tipos de datos:")
    print(df.dtypes, "\n")

    print("Vista previa de los datos:")
    print(df.head(), "\n")

    print("Valores faltantes por columna:")
    print(df.isnull().sum(), "\n")
    print("**********************************************************************") 

def resumenEstadistico():
    print("Resumen estadístico de variables numéricas:")
    print(df.describe(), "\n")

    print("Resumen estadístico de variables categóricas:")
    print(df.describe(include=["object"]), "\n")

def analisisEstadistico():
    # Limpiar valores nulos para columnas categóricas
    df.fillna('Desconocido', inplace=True)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns

def variablesNum():
   # Variables numéricas
    print("Análisis de columnas numéricas:")
    for col in num_cols:
        print(f"\nResumen de '{col}':")
        print(df[col].describe())

        # Histograma y Boxplot
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(df[col], kde=True, ax=axs[0], color="skyblue")
        axs[0].set_title(f"Distribución de {col}")
        sns.boxplot(x=df[col], ax=axs[1], color="lightgreen")
        axs[1].set_title(f"Boxplot de {col}")
        plt.show()

# Variables categóricas
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]

def variablesCat():
    print("Análisis de columnas categóricas:")
    for col in cat_cols:
        print(f"\nDistribución de '{col}':")
        print(df[col].value_counts(), "\n")

        # Limitar categorías a las 10 más frecuentes
        top_categories = df[col].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df[df[col].isin(top_categories.index)],
                    y=col,
                    order=top_categories.index,
                    palette="muted")
        plt.title(f"Distribución de {col} (Top 10 categorías)")
        plt.xlabel("Frecuencia")
        plt.ylabel(col)
        plt.show() 

def correlación():
    corr = df[num_cols].corr()
    print("Matriz de correlación entre variables numéricas:")
    print(corr, "\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.show()

def compararTasas():
    for col in cat_cols:
        if 'rate' in df.columns:  # Ajusta según tus datos
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x=col, y='rate', ci=None, palette="muted")
            plt.title(f"Tasa promedio por {col}")
            plt.ylabel("Tasa promedio")
            plt.xlabel(col)
            plt.xticks(rotation=45)
            plt.show()