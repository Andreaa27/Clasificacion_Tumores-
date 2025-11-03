# Librerías
import cv2
import numpy as np
import os
import time
from multiprocessing import Process, Lock, Array, Value

#  Parámetros de Ruta 
RUTA_BASE = r"/home/andrea/Descargas/Archive" 
LIMITE_IMAGENES = 1000 

NUM_PROCESOS = 1
carpetas = ['glioma', 'healthy', 'meningioma', 'pituitary'] 
etiquetas = {'glioma': 0, 'healthy': 1, 'meningioma': 2, 'pituitary': 3}

# Funciones y Parámetros del Modelo 

def normalizar_imagenes(imagen):
    return imagen.astype(np.float32) / 255.0

media = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
dimensiones = (224, 224)
TAMANO_IMAGEN_PLANA = dimensiones[0] * dimensiones[1] * 3 


def trabajador_preprocesamiento(lock, indice_global_compartido, tareas, X_compartido, y_compartido, total_imagenes):
    BATCH_SIZE = 1

    while True:
        should_break = False 
        
        # A. SECCIÓN CRÍTICA: Adquirir Lock para tomar el trabajo
        lock.acquire()
        try:
            start_idx_tarea = indice_global_compartido.value
            
            if start_idx_tarea >= total_imagenes:
                # No quedan tareas. Marcamos para salir después de liberar.
                should_break = True 
            
            if should_break:
                # Si no hay trabajo, simplemente liberamos el lock.
                pass 
            else:
                # Hay trabajo. Definir lote y actualizar contador global
                end_idx_tarea = min(start_idx_tarea + BATCH_SIZE, total_imagenes)
                indice_global_compartido.value = end_idx_tarea
                lote_tareas = tareas[start_idx_tarea:end_idx_tarea]
            
        finally:
            # B. Liberar Lock (Siempre se ejecuta, garantizando liberación única)
            lock.release() 
        
        if should_break:
            break # Salir del bucle 'while True'

        # C. TAREA PESADA (Preprocesamiento) - Fuera de la Sección Crítica
        for ruta_imagen, etiqueta_clase, indice_escritura in lote_tareas:
            original = cv2.imread(ruta_imagen)
            if original is None:
                continue 

            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            redimensionada = cv2.resize(original, dimensiones)
            imagen = redimensionada

            imagen = normalizar_imagenes(imagen)
            imagen = (imagen - media) / std
            
            imagen_plana = imagen.flatten() 
            
            start_escritura = indice_escritura * TAMANO_IMAGEN_PLANA
            end_escritura = (indice_escritura + 1) * TAMANO_IMAGEN_PLANA
            
            X_compartido[start_escritura:end_escritura] = imagen_plana
            y_compartido[indice_escritura] = etiqueta_clase
            
# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    
    # 1. Preparar la lista de tareas
    tareas = []
    indice_escritura_global = 0 
    
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(RUTA_BASE, carpeta)
        
        if not os.path.isdir(ruta_carpeta):
             print(f" Advertencia: La carpeta {ruta_carpeta} no existe. Omitiendo.")
             continue
             
        imagenes_en_carpeta = os.listdir(ruta_carpeta)
        imagenes = [img for img in imagenes_en_carpeta if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        imagenes = imagenes[:LIMITE_IMAGENES] 
        
        for imagen in imagenes:
            ruta_completa = os.path.join(ruta_carpeta, imagen)
            etiqueta = etiquetas[carpeta]
            tareas.append((ruta_completa, etiqueta, indice_escritura_global))
            indice_escritura_global += 1
            
    total_imagenes = len(tareas)
    if total_imagenes == 0:
        print("No se encontraron imagenes. Asegúrate de que las carpetas existan y contengan imágenes.")
        exit()

    # 2. Inicializar Variables Compartidas y Sincronización
    lock = Lock()
    indice_global_compartido = Value('i', 0) 
    X_compartido = Array('d', total_imagenes * TAMANO_IMAGEN_PLANA)
    y_compartido = Array('i', total_imagenes)

    # 3. Medición de Tiempo (Paralelo - Tp)
    tiempo_inicio_paralelo = time.time()
    
    # 4. Crear y Ejecutar Procesos
    print("-" * 30)
    print(f"Iniciando procesamiento paralelo con P={NUM_PROCESOS} procesos...")
    print(f"Total de imágenes a procesar: {total_imagenes}")
    
    procesos = []
    for i in range(NUM_PROCESOS):
        p = Process(target=trabajador_preprocesamiento, 
                    args=(lock, indice_global_compartido, tareas, X_compartido, y_compartido, total_imagenes))
        procesos.append(p)
        p.start()

    # Esperar a que todos los procesos terminen
    for p in procesos:
        p.join()

    tiempo_fin_paralelo = time.time()
    tiempo_paralelo = tiempo_fin_paralelo - tiempo_inicio_paralelo
    print("-" * 30)
    print(f"Procesamiento en paralelo finalizado.")
    print(f"Tiempo Paralelo (T{NUM_PROCESOS}) total: {tiempo_paralelo:.4f} segundos.")
    print("-" * 30)