# Librerías
import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("GPUs disponibles:", gpus)

# Confirmar dispositivo
print("Dispositivos visibles:", tf.config.list_physical_devices())


# Función para extraer características
def extraer_features(model, X, batch_size=32):
    features = []
    for i in tqdm(range(0, len(X), batch_size)):
        # asegura normalización correcta
        batch = preprocess_input(X[i:i+batch_size])
        feats = model.predict(batch, verbose=0)
        features.append(feats)
    return np.concatenate(features)


# Cargar datos preprocesados
glioma = np.load("Preprocesados/preprocesado_glioma.npz")
healthy = np.load("Preprocesados/preprocesado_healthy.npz")
meningioma = np.load("Preprocesados/preprocesado_meningioma.npz")
pituitary = np.load("Preprocesados/preprocesado_pituitary.npz")

X_glioma, y_glioma = glioma["X"], glioma["y"]
X_healthy, y_healthy = healthy["X"], healthy["y"]
X_meningioma, y_meningioma = meningioma["X"], meningioma["y"]
X_pituitary, y_pituitary = pituitary["X"], pituitary["y"]

# Modelo base ResNet50 sin capa superior
modelo_base = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3))
modelo = Model(inputs=modelo_base.inputs,
               outputs=GlobalAveragePooling2D()(modelo_base.output))

# Extraer características por clase
features_glioma = extraer_features(modelo, X_glioma)
np.savez("features_glioma.npz", X=features_glioma, y=y_glioma)
print("Glioma guardado")

features_healthy = extraer_features(modelo, X_healthy)
np.savez("features_healthy.npz", X=features_healthy, y=y_healthy)
print("Healthy guardado")

features_meningioma = extraer_features(modelo, X_meningioma)
np.savez("features_meningioma.npz", X=features_meningioma, y=y_meningioma)
print("Meningioma guardado")

features_pituitary = extraer_features(modelo, X_pituitary)
np.savez("features_pituitary.npz", X=features_pituitary, y=y_pituitary)
print("Pituitary guardado")

print(" Features extraídas y guardadas por clase usando CPU.")
