# Librerías
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from itertools import product
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# Configuración directorios
base_dir = "./Vectors"
splits = ["Train", "Validation", "Test"]
clases = ["glioma", "healthy", "meningioma", "pituitary"]


# Función para cargar datos
def cargar_datos(directorio):
    X, y = [], []
    for idx, clase in enumerate(clases):
        ruta = os.path.join(base_dir, directorio, f"{clase}.npz")
        if not os.path.exists(ruta):
            print(f" No se encontró {ruta}, se omite.")
            continue
        data = np.load(ruta)
        X.append(data["X"])
        y.append(data["y"])
    return np.concatenate(X), np.concatenate(y)


# Función para evaluar Random Forest
def evaluar_rf(args):
    parametros, x_train, x_test, y_train, y_test = args
    # Instancia de modelo con hiperparámetros actuales
    rf = RandomForestClassifier(
        random_state=42,
        **parametros
    )

    # Evaluar con accuracy
    rf.fit(x_train, y_train)
    prediccion = rf.predict(x_test)
    score = accuracy_score(y_test, prediccion)

    return (parametros, score)


# Cargar conjuntos
print("Cargando conjuntos...")
X_train, y_train = cargar_datos("Train")
X_val, y_val = cargar_datos("Validation")
X_test, y_test = cargar_datos("Test")

print(f"Conjuntos cargados:")
print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Búsqueda de hiperparámetros
print("\nIniciando búsqueda de mejores hiperparámetros...")

# Diccionario de hiperparámetros
hiperparametros = {
    'max_depth': [None, 5, 10, 20],
    'n_estimators': [100, 150, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}

# Generar todas las combinaciones posibles
claves = list(hiperparametros.keys())
combinaciones = list(product(*hiperparametros.values()))

print(f"Evaluando {len(combinaciones)} combinaciones de hiperparámetros...")

combinaciones_dict = [dict(zip(claves, c)) for c in combinaciones]

# Usar validation set para la búsqueda de hiperparámetros
tareas = [(params, X_train, X_val, y_train, y_val) for params in combinaciones_dict]

# Inicio de toma de tiempo
inicio = time.time()

# Búsqueda en paralelo con n procesos
print("Ejecutando búsqueda paralela")
with Pool(processes=1) as pool:
    resultados = list(tqdm(pool.imap(evaluar_rf, tareas), total=len(tareas)))
    
fin = time.time()

# Encontrar el mejor resultado
mejor_params, mejor_score = max(resultados, key=lambda x: x[1])

print(f"\nBÚSQUEDA COMPLETADA")
print(f"Tiempo total de ejecución: {round(fin - inicio, 2)} segundos")
print(f"Mejor accuracy en validación: {mejor_score:.4f}")
print(f"Mejores hiperparámetros: {mejor_params}")

# Entrenar modelo final con los mejores hiperparámetros
print("\nEntrenando modelo final con mejores hiperparámetros...")
rf_final = RandomForestClassifier(
    **mejor_params,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_train, y_train)

# Evaluación en validation set
print("\nEvaluando modelo en validation set...")
y_pred_val = rf_final.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
print(f"Accuracy en validación: {acc_val:.4f}")

# Evaluación en test set
print("\nEvaluando modelo en test set...")
y_pred_test = rf_final.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy en prueba: {acc_test:.4f}")

# Reporte de clasificación detallado
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_test, target_names=clases))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nMatriz de confusión:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=clases,
            yticklabels=clases)
plt.title(f'Matriz de Confusión - Random Forest\nAccuracy: {acc_test:.4f}')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig("Resultado_optimizado.png", dpi=300, bbox_inches='tight')
plt.show()

# Mostrar las 5 mejores combinaciones
# print("\nTop 5 mejores combinaciones:")
# resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)[:5]
# for i, (params, score) in enumerate(resultados_ordenados, 1):
#     print(f"{i}. Accuracy: {score:.4f} - Parámetros: {params}")
#
# print("\nProceso finalizado correctamente!")
