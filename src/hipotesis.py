#Importar Librerias
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Descargar la ultima version del Dataset

def descargarDataset():
    path = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")
    print("Path to dataset files:", path)
    csv_path = f"{path}/AB_NYC_2019.csv"
    return pd.read_csv(csv_path, sep=',', decimal='.')
    
df = descargarDataset()

print(df.head())
print("**********************************************************************")

# Revisar datos duplicados
def duplicados():
    duplicates = df.duplicated()
    if duplicates.sum() > 0:
        print(f"datos duplicados: {df[duplicates]}")
    print(f'Número de filas duplicadas: {duplicates.sum()}')
    print("**********************************************************************")

#Datos generales de las variables antes de imputar datos
def datosGenerales():
    df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365',
        'latitude', 'longitude']].describe()

    df[['name', 'host_name', 'neighbourhood_group', 'neighbourhood',
        'room_type', 'last_review']].describe()

    df[['id', 'host_id']].describe()
    print("**********************************************************************")

# Función para detectar outliers en una columna
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((column < lower_bound) | (column > upper_bound))


outliers = df.select_dtypes(include=['number']).apply(detect_outliers)

# Ver los resultados
print("Número de outliers por columna:")
print(outliers.sum())
print("\nFilas con outliers:")
print(df[outliers.any(axis=1)]) 
print("**********************************************************************")

# Se calcula el número de valores nulos por columna
nulos_por_columna = df.isnull().sum()

# Se filtrarn las columnas que tienen al menos un valor nulo
columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0].index.tolist()

# Lista de columnas con valores nulos
print("Columnas con valores nulos:", columnas_con_nulos)

# Calculando la cantidad de valores nulos en cada una de las columnas especificadas
nulos_por_columna = df[columnas_con_nulos].isnull().sum()

# Mostrando el conteo de valores nulos por columna
print("Conteo de valores nulos por columna:")
print(nulos_por_columna)
print("**********************************************************************") 

# Enlistar las columnas que tienen outliers
columnas_con_outliers = outliers.any().index[outliers.any()].tolist()
print("Columnas con valores nulos:", columnas_con_nulos)
print("Columnas con outliers:", columnas_con_outliers)
print("**********************************************************************") 

# Tratamiento de valores nulos
# Tratamiento de outliers con límites basados en el percentil 99
# Recortar outliers usando límites inferiores y superiores asegurando que los valores fuera del rango aceptable se ajusten
def limitar_outliers(col, limite_inferior=None, limite_superior=None):
    if limite_inferior is not None:
        col = col.clip(lower=limite_inferior)  # Limita valores por debajo del límite inferior
    if limite_superior is not None:
        col = col.clip(upper=limite_superior)  # Limita valores por encima del límite superior
    return col

def tratarNulos():
    # La columna 'name'no afecta cálculos cuantitativos. Por eso, se reemplazan valores nulos con "Desconocido".
    df['name'] = df['name'].fillna('Desconocido')

    # La columna 'host_name' no es crítica para cálculos así que se reemplazan los nulos con "Desconocido".
    df['host_name'] = df['host_name'].fillna('Desconocido')

    # La columna 'last_review' contiene la fecha de la última reseña, si no hay reseñas, Se reemplaza con "Sin reseñas" como marcador.
    df['last_review'] = df['last_review'].fillna('Sin reseñas')

    # La columna 'reviews_per_month' tiene el promedio de reseñas mensuales. Un valor nulo indica que no hubo reseñas, por lo que se reemplaza con 0
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    # Columna 'price', precios altos (máximo de $10,000). Estos pueden ser errores o casos atípicos. Se limita al percentil 99, que es $799.
    limite_price = df['price'].quantile(0.99)
    df['price'] = limitar_outliers(df['price'], limite_superior=limite_price)

    # Columna 'minimum_nights', Los valores altos (máximo de 1,250 noches) son poco prácticos para alquileres a corto plazo. Se limita al percentil 99 (45 noches).
    limite_minimum_nights = df['minimum_nights'].quantile(0.99)
    df['minimum_nights'] = limitar_outliers(df['minimum_nights'], limite_superior=limite_minimum_nights)

    # Columna 'reviews_per_month', valores altos (hasta 58.5 reseñas mensuales) son improbables y afectan métricas. Se limita al percentil 99 (6.8 reseña/mes)
    limite_reviews_per_month = df['reviews_per_month'].quantile(0.99)
    df['reviews_per_month'] = limitar_outliers(df['reviews_per_month'], limite_superior=limite_reviews_per_month)

    # Columna 'calculated_host_listings_count', Valores altos (máximo 327) se limita al percentil 99 (232).
    limite_listings_count = df['calculated_host_listings_count'].quantile(0.99)
    df['calculated_host_listings_count'] = limitar_outliers(
        df['calculated_host_listings_count'], limite_superior=limite_listings_count
    )
    return df.info()

tratarNulos()

def hipotesis1():
    try:
        # Filtrar las columnas relevantes
        relevant_columns = ["room_type", "price"]
        filtered_data = df[relevant_columns]

        # Calcular el precio promedio por tipo de habitación
        avg_prices = filtered_data.groupby("room_type")[["price"]].mean().reset_index()

        # Ordenar por precio promedio descendente
        avg_prices = avg_prices.sort_values(by="price", ascending=False)
        print(avg_prices)
    except Exception as e:
        print(f"Error durante el análisis: {e}")

    try:
        plt.figure(figsize=(10, 6))
        df.boxplot(column="price", by="room_type", grid=False, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red"))
        plt.title("Distribución de Precios por Tipo de Habitación", fontsize=16)
        plt.suptitle("")  # Elimina el título automático generado por boxplot
        plt.xlabel("Tipo de Habitación", fontsize=12)
        plt.ylabel("Precio (USD)", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return plt.show()
        
    except Exception as e:
        print(f"Error durante la visualización: {e}")

def hipotesis2Grafica():
    # Crear el gráfico de boxplot con Seaborn
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x='neighbourhood_group',
        y='price',
        palette='Set2'
    )
    plt.title('Distribución de precios por grupo de vecindario')
    plt.xlabel('Grupo de vecindario')
    plt.ylabel('Precio')
    return plt.show()

def hipotesis2Analisis():
    df_anova = df[['neighbourhood_group', 'price']].dropna()

    # Realizar la prueba ANOVA de una vía
    anova_result = f_oneway(
        df_anova[df_anova['neighbourhood_group'] == 'Manhattan']['price'],
        df_anova[df_anova['neighbourhood_group'] == 'Brooklyn']['price'],
        df_anova[df_anova['neighbourhood_group'] == 'Queens']['price'],
        df_anova[df_anova['neighbourhood_group'] == 'Bronx']['price'],
        df_anova[df_anova['neighbourhood_group'] == 'Staten Island']['price']
    )

    # Crear una lista de todas las combinaciones de comparaciones pareadas
    groups = df_anova['neighbourhood_group'].unique()
    results = []

    for group1, group2 in combinations(groups, 2):
        data1 = df_anova[df_anova['neighbourhood_group'] == group1]['price']
        data2 = df_anova[df_anova['neighbourhood_group'] == group2]['price']
        t_stat, p_value = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        results.append({'Grupo 1': group1, 'Grupo 2': group2, 'p-valor': p_value})

    # Crear un DataFrame con los resultados de las comparaciones pareadas
    bonferroni_results = pd.DataFrame(results)

    # Asegurar que los valores de p estén en el rango 0-1
    bonferroni_results['p-valor'] = bonferroni_results['p-valor'].clip(0, 1)

    # Aplicar la corrección de Bonferroni
    bonferroni_results['p-valor Ajustado'] = multipletests(bonferroni_results['p-valor'], method='bonferroni')[1]

    # Mostrar resultados
    print("Resultado ANOVA:")
    print(f"Valor p: {anova_result.pvalue:.4e}")

    print("\nResultados Bonferroni:")
    #print(bonferroni_results)
    return bonferroni_results

def hipotesis2Mapa():
    # Calcular la mediana del precio por vecindario
    median_price = df.groupby('neighbourhood_group')['price'].median().reset_index()
    median_price.columns = ['Neighbourhood Group', 'Median Price']

    # Crear un mapa centrado en Nueva York
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    # Añadir marcadores para cada grupo de vecindarios
    for _, row in median_price.iterrows():
        if row['Neighbourhood Group'] == 'Manhattan':
            coords = [40.7831, -73.9712]
        elif row['Neighbourhood Group'] == 'Brooklyn':
            coords = [40.6782, -73.9442]
        elif row['Neighbourhood Group'] == 'Queens':
            coords = [40.7282, -73.7949]
        elif row['Neighbourhood Group'] == 'Bronx':
            coords = [40.8448, -73.8648]
        elif row['Neighbourhood Group'] == 'Staten Island':
            coords = [40.5795, -74.1502]
        else:
            continue

        # Añadir etiqueta con la mediana del precio
        folium.Marker(
            location=coords,
            popup=f"{row['Neighbourhood Group']}: ${row['Median Price']:.2f}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(nyc_map)

    # Mostrar el mapa
    return nyc_map

def hipotesis3():
    # Calcular la tasa de ocupación promedio como 1 - (la fracción de días disponibles / 365 dias)
    df['tasa_ocupacion'] = 1 - (df['availability_365'] / 365)

    # Crear una nueva columna para clasificar los alojamientos según la estancia mínima -- Grupos: "1 noche" y "Más de 1 noche"
    df['categoria_estancia'] = np.where(df['minimum_nights'] == 1, '1 noche', 'Más de 1 noche')

    # Calcular estadísticas descriptivas para las tasas de ocupación en ambos grupos
    ocupacion_1_noche = df[df['categoria_estancia'] == '1 noche']['tasa_ocupacion']
    ocupacion_mas_1_noche = df[df['categoria_estancia'] == 'Más de 1 noche']['tasa_ocupacion']

    print("Estadísticas descriptivas:")
    print("1 noche:")
    print(ocupacion_1_noche.describe())  # Muestra estadísticas como la media, mediana, etc. para estancias de 1 noche
    print("\nMás de 1 noche:")
    print(ocupacion_mas_1_noche.describe())  # Muestra estadísticas similares para estancias de más de 1 noche

    # Prueba t para comparar las medias de las tasas de ocupación entre los dos grupos (1 noche vs. más de 1 noche) probando si existe una diferencia entre ambas medias.
    t_stat, p_value = ttest_ind(ocupacion_1_noche, ocupacion_mas_1_noche, nan_policy='omit')

    print("\nPrueba T entre 1 noche y más de 1 noche:")
    print(f"Estadístico T: {t_stat:.4f}, p-valor: {p_value:.4f}")
    # El valor t = la magnitud de la diferencia entre las medias en unidades de error estándar.
    # El valor p = si esa diferencia es significativa (si el p < 0.05 rechazamos la hipótesis nula).

    # Gráfico Boxplot para comparar las tasas de ocupación entre los dos grupos
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='categoria_estancia', y='tasa_ocupacion', hue='categoria_estancia', palette='Set2', dodge=False)
    plt.title('Comparación de la ocupación promedio según la estancia mínima')
    plt.ylabel('Tasa de ocupación promedio')
    plt.xlabel('Categoría de estancia mínima')
    plt.legend(title='Categoría de estancia')
    graf1 = plt.show()

    # Gráfico de histograma para analizar la distribución de la ocupación en ambos grupos
    # Esto permite observar si existe una mayor concentración de valores altos de ocupación en el grupo de "1 noche".
    plt.figure(figsize=(12, 6))
    sns.histplot(ocupacion_1_noche, kde=True, color='blue', label='1 noche', bins=30)
    sns.histplot(ocupacion_mas_1_noche, kde=True, color='orange', label='Más de 1 noche', bins=30)
    plt.title('Distribución de la ocupación promedio según la estancia mínima')
    plt.xlabel('Tasa de ocupación promedio')
    plt.ylabel('Frecuencia')
    graf2 = plt.show()

    # Conclusión sobre si la hipótesis es aceptada o no
    # Si el valor p es menor a 0.05, significa que hay una diferencia estadísticamente significativa entre las medias.
    if p_value < 0.05:
        conclusion = ("Los alojamientos con una estancia mínima de 1 noche tienen una mayor tasa de ocupación promedio "
                    "en comparación con aquellos con estancias mínimas más largas. Esto se refleja en el valor t positivo "
                    "y un valor p menor al umbral de significancia de 0.05, lo que confirma una diferencia significativa.")
    else:
        conclusion = ("No hay evidencia suficiente para afirmar que los alojamientos con una estancia mínima de 1 noche "
                    "tienen una mayor tasa de ocupación promedio que aquellos con estancias más largas. "
                    "El valor t no muestra una diferencia significativa entre las medias, y el valor p es mayor a 0.05.")

    # Imprimir la conclusión final
    print("\nConclusión:")
    print(conclusion)
    return graf1 and graf2

def hipotesis4():
    # Filtrar datos por tipo de zona
    tourist_zones = ['Manhattan', 'Brooklyn']
    tourist_area_data = df[df['neighbourhood_group'].isin(tourist_zones)]['availability_365']
    non_tourist_area_data = df[~df['neighbourhood_group'].isin(tourist_zones)]['availability_365']


    # Realizar prueba t de Student para muestras independientes
    stat, p = ttest_ind(tourist_area_data, non_tourist_area_data, equal_var=False)

    print(f"Prueba t de Student: Estadístico={stat}, p={p}")
    if p < 0.05:
        print("Rechazamos la hipótesis nula: Existe una diferencia significativa.")
    else:
        print("No podemos rechazar la hipótesis nula: No existe una diferencia significativa.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.DataFrame({
        'Tasa de Ocupación': list(tourist_area_data) + list(non_tourist_area_data),
        'Zona': ['Turística'] * len(tourist_area_data) + ['No Turística'] * len(non_tourist_area_data)
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Zona', y='Tasa de Ocupación', data=data, palette='Set2')
    plt.title('Comparación de la Tasa de Ocupación entre Zonas', fontsize=14)
    plt.xlabel('Tipo de Zona', fontsize=12)
    plt.ylabel('Tasa de Ocupación (días disponibles)', fontsize=12)
    return plt.show()

def hipotesis5(): 
    data = df

    # Preprocesar los datos
    data = data[['price', 'availability_365', 'neighbourhood_group', 'room_type']]
    data = data.dropna()

    # Convertir variables categóricas a dummies
    data = pd.get_dummies(data, columns=['neighbourhood_group', 'room_type'], drop_first=True)

    # Gráficos exploratorios
    # Histograma de precios
    plt.figure(figsize=(8, 5))
    sns.histplot(data['price'], bins=30, kde=True, color='blue')
    plt.title("Distribución de Precios")
    plt.xlabel("Precio por noche")
    plt.ylabel("Frecuencia")
    graf1 = plt.show()

    # Boxplot de disponibilidad vs. precio
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=pd.qcut(data['price'], q=3, labels=['low', 'medium', 'high']), y=data['availability_365'])
    plt.title("Disponibilidad por Categoría de Precio")
    plt.xlabel("Categoría de Precio")
    plt.ylabel("Días Disponibles")
    graf2 = plt.show()

    # Gráfico de dispersión de precio vs. disponibilidad
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data['price'], y=data['availability_365'], alpha=0.5)
    plt.title("Relación entre Precio y Disponibilidad")
    plt.xlabel("Precio por noche")
    plt.ylabel("Días Disponibles")
    graf3 = plt.show()

    # REGRESIÓN LINEAL
    X = data.drop('availability_365', axis=1)
    y = data['availability_365']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluar y visualizar los resultados
    print("Resultados de la Regresión Lineal:")
    print(f"R² Score: {r2_score(y_test, y_pred)}")

    coeficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    print(coeficients)

    # Gráfico de predicciones vs valores reales
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.title("Valores Reales vs. Predicciones")
    plt.xlabel("Valores reales de Disponibilidad")
    plt.ylabel("Predicciones de Disponibilidad")
    graf4 = plt.show()

    # Paso 6: PRUEBAS DE HIPÓTESIS
    data['price_category'] = pd.qcut(data['price'], q=3, labels=['low', 'medium', 'high'])
    low_price = data[data['price_category'] == 'low']['availability_365']
    high_price = data[data['price_category'] == 'high']['availability_365']

    stat, p_value = ttest_ind(low_price, high_price, equal_var=False)

    print("\nResultados de las Pruebas de Hipótesis:")
    print(f"t-statistic: {stat}")
    print(f"p-value: {p_value}")
    return graf1, graf2, graf3, graf4