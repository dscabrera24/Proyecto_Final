"""
Módulo para análisis estadístico y pruebas de hipótesis basado en datos de Airbnb.
"""
# Configuración global para gráficos
import seaborn as sns
import matplotlib.pyplot as plt

# Importar funciones principales de hipótesis
from .hipotesis import (
    hipotesis1,
    hipotesis2,
    hipotesis3,
    hipotesis4,
    hipotesis5
)

# Importar funciones utilitarias

# Configuración global
DATASET_PATH = "data/dataset.csv"
from .eda import (
    panoramaGeneral,
    resumenEstadistico,
    variablesNum,
    variablesCat,
    correlación
)

# Inicialización del dataset (si es liviano)
def cargar_dataset():
    """
    Función para cargar y preparar el dataset base.
    """
    from .hipotesis import descargarDataset, tratarNulos, outliers
    df = descargarDataset()
    outliers
    tratarNulos()
    return df

# Cargar dataset automáticamente si es necesario
try:
    df = cargar_dataset()
    print("Dataset cargado y preparado.")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")

sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)