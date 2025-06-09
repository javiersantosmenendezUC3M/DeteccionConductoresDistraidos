import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
INPUT_DIR       = "Cam4"                          # Carpeta raíz con imágenes
OUTPUT_DIR      = INPUT_DIR + "-pose-norm_pt"          # Carpeta de salida para archivos .pt
MODEL_PATH      = "yolo11m-pose.pt"               # Ruta al modelo de pose
POSE_CONF       = 0.5                               # Umbral de confianza

# --- Carga modelo de pose ---
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Función de procesamiento de una imagen ---
def process_image(in_path, out_path):
    # Lee la imagen
    img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen {in_path}")

    # Infiere keypoints con YOLO-Pose
    results = model.predict(img, device='cpu')
    # Selecciona la detección con mayor confianza
    driver = results[0][results[0].boxes.conf.argmax() ]
    H, W = driver.orig_shape[0], driver.orig_shape[1]

    # Obtiene coordenadas normalizadas y confidence
    coords = (driver.keypoints.xy/torch.tensor([W,H]))[0]     # Tensor[K,2]
    confs  = driver.keypoints.conf[0]                       # Tensor[K]

    # Empaqueta y guarda
    data = { 'coords': coords, 'conf': confs }
    torch.save(data, out_path)

# --- Recorre directorios y guarda .pt ---
for root, _, files in os.walk(INPUT_DIR):
    for file in tqdm(files, desc=f"Procesando {root}"):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        in_path  = os.path.join(root, file)
        rel_path = os.path.relpath(in_path, INPUT_DIR)
        base     = os.path.splitext(rel_path)[0]
        out_path = os.path.join(OUTPUT_DIR, base + '.pt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            process_image(in_path, out_path)
        except Exception as e:
            print(f"❌ Error procesando {in_path}: {e}")
            # Para imágenes fallidas, guarda un tensor de ceros
            zeros = torch.zeros((17, 2)), torch.zeros(17)
            torch.save({'coords': zeros[0], 'conf': zeros[1]}, out_path)

print(f"✅ Estimación de pose completada. Archivos guardados en: {OUTPUT_DIR}")
