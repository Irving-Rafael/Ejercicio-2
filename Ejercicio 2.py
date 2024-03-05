import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Cargar el dataset desde el archivo CSV
df = pd.read_csv("spheres1d10.csv", header=None)

# Dividir el dataset en características (X) y etiquetas (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Definir el porcentaje de datos de entrenamiento y prueba
train_percentage = 0.8
test_percentage = 0.2

# Definir el número de particiones
num_partitions = 5

# Crear cinco particiones
for i in range(num_partitions):
    # Dividir el dataset en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=i)
    
    # Entrenar el perceptrón
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Imprimir la precisión del modelo en la partición actual
    print(f"Partición {i+1}: Precisión = {accuracy}")
