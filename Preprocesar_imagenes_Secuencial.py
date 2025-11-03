# Librerías
import cv2
import numpy as np
import os
import time # Importar time para medir el rendimiento

RUTA_BASE = r"/home/andrea/Descargas/Archive" 
LIMITE_IMAGENES = 1000 # Límite por clase (4000 imágenes totales)

# Declaración de funciones
def normalizar_imagenes(imagen):
    return imagen.astype(np.float32) / 255.0

# Preprocesamiento

carpetas = ['glioma', 'healthy', 'meningioma', 'pituitary']

# Nuevas dimensiones
dimensiones = (224, 224)

# Parámetros de normalización
media = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Listas para almacenar imágenes
lista_glioma = []
lista_healthy = []
lista_meningioma = []
lista_pituitary = []


# INICIO DE LA MEDICIÓN DE TIEMPO SECUENCIAL (T1)

tiempo_inicio_secuencial = time.time()
total_procesadas = 0

print(f"Iniciando preprocesamiento SECUENCIAL (Límite: {LIMITE_IMAGENES} por clase)...")
for carpeta in carpetas:
    # CONSTRUCCIÓN DE RUTA
    ruta_carpeta = os.path.join(RUTA_BASE, carpeta)
    
    if not os.path.isdir(ruta_carpeta):
        print(f" Advertencia: La carpeta {ruta_carpeta} no existe. Omitiendo.")
        continue
        
    imagenes_en_carpeta = os.listdir(ruta_carpeta)
    # Filtro básico y MUESTREO: Limitamos las imágenes a procesar
    imagenes = [img for img in imagenes_en_carpeta if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    imagenes = imagenes[:LIMITE_IMAGENES]

    for imagen in imagenes:
        original = cv2.imread(os.path.join(ruta_carpeta, imagen))
        
        if original is None:
             continue

        # Cambiar imágenes de BGR a RGB
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Redimensionar imagen
        redimensionada = cv2.resize(original, dimensiones)
        imagen = redimensionada

        # Normalizar y estandarizar imágenes
        imagen = normalizar_imagenes(imagen)
        imagen = (imagen - media) / std

        if carpeta == 'glioma':
            lista_glioma.append(imagen)
        elif carpeta == 'healthy':
            lista_healthy.append(imagen)
        elif carpeta == 'meningioma':
            lista_meningioma.append(imagen)
        elif carpeta == 'pituitary':
            lista_pituitary.append(imagen)
        
        total_procesadas += 1

    print(f"Carpeta {carpeta} procesada. Imágenes contadas: {len(imagenes)}")


# FIN DE LA MEDICIÓN DE TIEMPO SECUENCIAL

tiempo_fin_secuencial = time.time()
T_secuencial = tiempo_fin_secuencial - tiempo_inicio_secuencial
print("-" * 50)
print(f"Preprocesamiento Secuencial (Total: {total_procesadas} imgs) finalizado.")
print(f"TIEMPO SECUENCIAL (T1) OBTENIDO: {T_secuencial:.4f} segundos.")
print("-" * 50)

