from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Cargar los datos
df = pd.read_csv("data/AB_NYC_2019.csv")

# Filtrar y preparar los datos
df = df[(df['price'] > 0) & (df['price'] <= 1000)]
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

X = df[['neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan', 
        'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island', 
        'room_type_Private room', 'room_type_Shared room']]
y = df['price']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model_lr, 'models/linear_regression.pkl')

