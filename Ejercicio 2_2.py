import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_results(X, y, w, b):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], color='blue', label='Class 1')
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], X[y == -1][:, 2], color='red', label='Class -1')
    
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='gray')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Decision Boundary')
    ax.legend()
    
    plt.show()

# Datos originales de la tabla
data_original = [
    [-1, -1, -1, 1],
    [-1, -1, 1, 1],
    [-1, 1, -1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, -1],
    [1, -1, 1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1]
]

# Crear DataFrame con los datos originales
df_original = pd.DataFrame(data_original, columns=['X1', 'X2', 'X3', 'Yd'])

# Dividir el dataset original en características (X) y etiquetas (y)
X_original = df_original.iloc[:, :-1]
y_original = df_original.iloc[:, -1]

# Dividir el dataset original en conjunto de entrenamiento y prueba
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

# Entrenar el perceptrón con los datos originales
perceptron_original = Perceptron()
perceptron_original.fit(X_train_original, y_train_original)

# Evaluar el modelo con los datos originales
y_pred_original = perceptron_original.predict(X_test_original)
accuracy_original = accuracy_score(y_test_original, y_pred_original)

# Imprimir la precisión del modelo con los datos originales
print(f"\nDatos Originales: Precisión = {accuracy_original}")

# Obtener los parámetros del hiperplano de decisión
w_original = perceptron_original.coef_[0]
b_original = perceptron_original.intercept_

# Graficar los resultados con los datos originales
plot_results(X_train_original.values, y_train_original.values, w_original, b_original)

# List of CSV files with different levels of perturbations
csv_files = ["spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"]

for idx, file in enumerate(csv_files):
    print(f"\nDataset with {idx*20+10}% perturbations:")
    
    # Cargar el dataset desde el archivo CSV
    df = pd.read_csv(file, header=None)

    # Modificar el punto x=[-1, +1, -1] → yd = 1
    df.iloc[2, 3] = 1

    # Dividir el dataset en características (X) y etiquetas (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Dividir el dataset en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el perceptrón
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Imprimir la precisión del modelo en la partición actual
    print(f"Precisión = {accuracy}")

    # Obtener los parámetros del hiperplano de decisión
    w = perceptron.coef_[0]
    b = perceptron.intercept_

    # Graficar los resultados
    plot_results(X_train.values, y_train.values, w, b)

# Generar e imprimir 10 perturbaciones aleatorias externas a las perturbaciones del 10, 50 y 70%
for i in range(10):
    X_random, y_random = np.random.rand(8, 3) * 2 - 1, np.random.choice([-1, 1], size=8)
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_random, y_random, test_size=0.2, random_state=i)
    perceptron_random = Perceptron()
    perceptron_random.fit(X_train_random, y_train_random)
    y_pred_random = perceptron_random.predict(X_test_random)
    accuracy_random = accuracy_score(y_test_random, y_pred_random)
    print(f"\nPerturbación aleatoria {i+1}: Precisión = {accuracy_random}")

# Generar e imprimir 10 particiones aleatorias
for i in range(10):
    X_train_partition, X_test_partition, y_train_partition, y_test_partition = train_test_split(X_original, y_original, test_size=0.2, random_state=i)
    perceptron_partition = Perceptron()
    perceptron_partition.fit(X_train_partition, y_train_partition)
    y_pred_partition = perceptron_partition.predict(X_test_partition)
    accuracy_partition = accuracy_score(y_test_partition, y_pred_partition)
    print(f"\nPartición aleatoria {i+1}: Precisión = {accuracy_partition}")
