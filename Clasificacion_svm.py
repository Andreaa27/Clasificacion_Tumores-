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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# Función para evaluar random forest
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


# Función para evaluar con SVM
def evaluar_svm(args):
    parametros, x_train, x_test, y_train, y_test = args
    
    # Instancia de SVM
    svm = SVC(
        random_state=42,
        **parametros
    )

    # Entrenar y predecir
    svm.fit(x_train, y_train)
    prediccion = svm.predict(x_test)
    score = accuracy_score(y_test, prediccion)

    return (parametros, score)


if __name__ == '__main__':
    # Cargar datos
    print("Cargando conjuntos...")
    X_train, y_train = cargar_datos("Train")
    X_val, y_val = cargar_datos("Validation")
    X_test, y_test = cargar_datos("Test")
    
    print(f"Conjuntos cargados:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Random forest
    print("\nIniciando Random Forest")
    
    # Diccionario de hiperparámetros
    hiperparametros_rf = {
        'max_depth': [None, 5, 10, 20],
        'n_estimators': [100, 150, 200],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3]
    }
    
    # Generar todas las combinaciones posibles
    claves_rf = list(hiperparametros_rf.keys())
    combinaciones_rf = list(product(*hiperparametros_rf.values()))
    combinaciones_dict_rf = [dict(zip(claves_rf, c)) for c in combinaciones_rf]
    
    print(f"Evaluando {len(combinaciones_rf)} combinaciones de hiperparámetros (RF)...")
    
    # Búsqueda de hiperparámetros
    tareas_rf = [(params, X_train, X_val, y_train, y_val) for params in combinaciones_dict_rf]
    
    # Inicio de toma de tiempo RF
    inicio_rf = time.time()
    
    # Búsqueda en paralelo
    print("Ejecutando búsqueda paralela RF...")
    with Pool(processes=8) as pool:
        resultados_rf = list(tqdm(pool.imap(evaluar_rf, tareas_rf), total=len(tareas_rf)))
        
    fin_rf = time.time()
    
    # Encontrar el mejor resultado RF
    mejor_params_rf, mejor_score_rf = max(resultados_rf, key=lambda x: x[1])
    
    print(f"\nBúsqueda de RF completa")
    print(f"Tiempo total: {round(fin_rf - inicio_rf, 2)} segundos")
    print(f"Mejor accuracy: {mejor_score_rf:.4f}")
    print(f"Mejores parámetros: {mejor_params_rf}")
    
    # Entrenar modelo final RF
    print("\nEntrenando modelo final RF...")
    rf_final = RandomForestClassifier(
        **mejor_params_rf,
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_train, y_train)
    
    # Evaluación en test set RF
    print("Evaluando RF en test set...")
    y_pred_test_rf = rf_final.predict(X_test)
    acc_test_rf = accuracy_score(y_test, y_pred_test_rf)
    print(f"Accuracy en prueba (RF): {acc_test_rf:.4f}")
    
    # Reporte RF
    print("\nReporte de Clasificación (RF):")
    print(classification_report(y_test, y_pred_test_rf, target_names=clases))
    
    # Matriz RF
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test_rf), annot=True, fmt='d', cmap='Greens',
                xticklabels=clases, yticklabels=clases)
    plt.title(f'Matriz de Confusión - Random Forest\nAccuracy: {acc_test_rf:.4f}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig("Resultado_RF.png", dpi=300)
    plt.show()

    # Evaluación con SVM
    print("\nIniciando SVM")

    # Escalar datos
    print("Escalando datos para SVM...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Hiperparámetros SVM
    hiperparametros_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto']
    }

    claves_svm = list(hiperparametros_svm.keys())
    combinaciones_svm = list(product(*hiperparametros_svm.values()))
    combinaciones_dict_svm = [dict(zip(claves_svm, c)) for c in combinaciones_svm]

    print(f"Evaluando {len(combinaciones_svm)} combinaciones para SVM...")

    tareas_svm = [(params, X_train_s, X_val_s, y_train, y_val) for params in combinaciones_dict_svm]

    # Inicio toma tiempo SVM
    inicio_svm = time.time()
    
    print("Ejecutando búsqueda paralela SVM...")
    with Pool(processes=8) as pool:
        resultados_svm = list(tqdm(pool.imap(evaluar_svm, tareas_svm), total=len(tareas_svm)))
        
    fin_svm = time.time()

    mejor_params_svm, mejor_score_svm = max(resultados_svm, key=lambda x: x[1])
    print(f"\nBúsqueda completada (SVM)")
    print(f"Tiempo total: {round(fin_svm - inicio_svm, 2)}s")
    print(f"Mejor accuracy Val: {mejor_score_svm:.4f}")
    
    # Entrenamiento SVM Final
    print("Entrenando modelo final SVM...")
    svm_final = SVC(**mejor_params_svm, random_state=42)
    svm_final.fit(X_train_s, y_train)

    # Evaluación SVM en test
    y_pred_test_svm = svm_final.predict(X_test_s)
    acc_test_svm = accuracy_score(y_test, y_pred_test_svm)

    print(f"Accuracy en prueba (SVM): {acc_test_svm:.4f}")
    print("\nReporte de clasificación (SVM):")
    print(classification_report(y_test, y_pred_test_svm, target_names=clases))

    # Matriz SVM
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test_svm), annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases)
    plt.title(f'Matriz de confusión - SVM\nAccuracy: {acc_test_svm:.4f}')
    plt.savefig("Resultado_SVM.png", dpi=300)
    plt.show()

    # Comparativa entre modelos
    print("\n" + "="*50)
    print("Resultados")
    print(f"Random Forest accuracy: {acc_test_rf:.4f}")
    print(f"SVM accuracy:           {acc_test_svm:.4f}")
