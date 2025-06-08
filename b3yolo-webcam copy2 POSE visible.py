import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw
import torchvision.models as models
from torchvision import transforms
import tkinter as tk
from tkinter import ttk, messagebox
import winsound
import threading

# -----------------------------------
# CONFIGURACIÓN DE CLASES Y AGRUPACIÓN
# -----------------------------------

CLASS_CODES = [
    "C1_Drive_Safe", "C2_Sleep", "C3_Yawning", "C4_Talk_Left", "C5_Talk_Right",
    "C22_Talk_to_Passenger", "C6_Text_Left", "C7_Text_Right", "C9_Look_Left",
    "C10_Look_Right", "C11_Look_Up", "C12_Look_Down", "C13_Smoke_Left",
    "C14_Smoke_Right", "C15_Smoke_Mouth", "C16_Eat_Left", "C17_Eat_Right",
    "C18_Operate_Radio", "C19_Operate_GPS", "C20_Reach_Behind",
    "C21_Leave_Steering_Wheel", "C8_Make_Up"
]

CLASS_GROUPS = {
    "C1_Drive_Safe": "Conducción segura",
    "C2_Sleep": "Somnolencia detectada",
    "C3_Yawning": "Bostezando",
    "C4_Talk_Left": "Hablando",
    "C5_Talk_Right": "Hablando",
    "C22_Talk_to_Passenger": "Hablando",
    "C6_Text_Left": "Usando móvil",
    "C7_Text_Right": "Usando móvil",
    "C9_Look_Left": "Desviando la mirada",
    "C10_Look_Right": "Desviando la mirada",
    "C11_Look_Up": "Desviando la mirada",
    "C12_Look_Down": "Desviando la mirada",
    "C13_Smoke_Left": "Fumando",
    "C14_Smoke_Right": "Fumando",
    "C15_Smoke_Mouth": "Fumando",
    "C16_Eat_Left": "Comiendo",
    "C17_Eat_Right": "Comiendo",
    "C18_Operate_Radio": "Operando controles",
    "C19_Operate_GPS": "Operando controles",
    "C20_Reach_Behind": "Cogiéndo algo detrás",
    "C21_Leave_Steering_Wheel": "Soltando el volante",
    "C8_Make_Up": "Maquillándose"
}

CLASSES = CLASS_CODES[:]

MODEL_PATHS = {
    "Cámara 1": "best_model_efficientnet_b3_cam1.pth",
    "Cámara 2": "best_model_efficientnet_b3_cam2.pth",
    "Cámara 3": "best_model_efficientnet_b3_20250605_221920.pth",
    "Cámara 4": "best_model_efficientnet_b3_cam4.pth",
}

YOLO_MODEL_PATH = "yolo11m-pose.pt"
POSE_CONF = 0.5
POSE_DIM = 17 * 2
NUM_CLASSES = len(CLASSES)

# Transformación para la red de clasificación (solo tensor y normalización; el resize se hace manualmente)
MEAN = [0.5, 0.5, 0.5]
STD = [0.229, 0.224, 0.225]
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# -----------------------------------
# MODELO PoseClassifier
# -----------------------------------

class PoseClassifier(nn.Module):
    def __init__(self, base_net: nn.Module, pose_dim: int, num_class: int, dropout: float = 0.5):
        super().__init__()
        modules = list(base_net.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 300, 300)
            feat = self.feature_extractor(dummy).view(1, -1)
        feat_dim = feat.size(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim + pose_dim, num_class)

    def forward(self, x, pose):
        f = self.feature_extractor(x).view(x.size(0), -1)
        f = self.dropout(f)
        cat = torch.cat([f, pose.to(f.device)], dim=1)
        return self.fc(cat)

# -----------------------------------
# CARGA DEL MODELO POR CÁMARA
# -----------------------------------

def load_classifier_model(camera_name, device):
    model_filename = MODEL_PATHS.get(camera_name)
    if not model_filename or not os.path.exists(model_filename):
        messagebox.showerror(
            "Error",
            f"No se encontró el archivo de modelo para {camera_name}.\n"
            f"Se buscó: {model_filename}"
        )
        return None

    base_net = models.efficientnet_b3(weights=None)
    classifier = PoseClassifier(base_net, POSE_DIM, NUM_CLASSES, dropout=0.5).to(device)
    state_dict = torch.load(model_filename, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier

# -----------------------------------
# APLICACIÓN TKINTER
# -----------------------------------

class DistractedDriverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monitorizador de Conductores Distraídos")
        self.geometry("640x750")
        self.resizable(False, False)

        # Dispositivo Torch y YOLO
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.device_str = "cpu" if torch.cuda.is_available() else "cpu"

        # Cargar YOLO-Pose
        try:
            self.pose_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo de pose:\n{e}")
            self.destroy()
            return

        self.classifier = None
        self.show_pose = False  # Flag para dibujar keypoints
        self.last_pose_image = None

        # -----------------------------------
        # Botones en la parte superior
        # -----------------------------------
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", pady=5)

        self.pose_button = ttk.Button(
            top_frame,
            text="Mostrar estimación de pose",
            command=self.toggle_pose_display
        )
        self.pose_button.pack(side="left", padx=10)

        self.camera_var = tk.StringVar()
        self.camera_combobox = ttk.Combobox(
            top_frame,
            textvariable=self.camera_var,
            values=list(MODEL_PATHS.keys()),
            state="readonly"
        )
        self.camera_combobox.current(2)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_change)
        self.camera_combobox.pack(side="left", padx=10)

        # -----------------------------------
        # Label para vídeo
        # -----------------------------------
        self.video_label = ttk.Label(self)
        self.video_label.pack(padx=10, pady=10)

        # -----------------------------------
        # Label grande para distracción
        # -----------------------------------
        self.distract_label = tk.Label(
            self,
            text="–",
            font=("Helvetica", 32, "bold"),
            fg="red"
        )
        self.distract_label.pack(pady=(0, 20))

        # -----------------------------------
        # Texto pequeño para estado
        # -----------------------------------
        self.status_text = tk.Label(self, text="", font=("Helvetica", 10), fg="blue")
        self.status_text.pack()

        # Captura de cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la webcam.")
            self.destroy()
            return

        # Cargar modelo inicial
        self.on_camera_change()

        # Iniciar bucles: vídeo y inferencia
        self.update_video_frame()
        self.after(1000, self.run_inference)

        # Al cerrar
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_pose_display(self):
        """Activa o desactiva el dibujo de keypoints."""
        self.show_pose = not self.show_pose
        if self.show_pose:
            self.pose_button.config(text="Ocultar estimación de pose")
        else:
            self.pose_button.config(text="Mostrar estimación de pose")

    def on_camera_change(self, event=None):
        """Carga el clasificador según la cámara seleccionada."""
        cam = self.camera_var.get()
        self.status_text.config(text=f"Cargando modelo para {cam}...")
        self.update_idletasks()

        self.classifier = load_classifier_model(cam, self.device)
        if self.classifier:
            self.status_text.config(text=f"Modelo '{cam}' cargado correctamente (Torch en {self.device_str}).")
        else:
            self.status_text.config(text="No se pudo cargar el modelo.")

    def update_video_frame(self):
        """
        Captura un fotograma y lo muestra. Si show_pose está activado,
        muestra la última imagen con keypoints dibujados (si existe);
        en otro caso, muestra el vídeo en directo.
        """
        if self.show_pose and self.last_pose_image:
            imgtk = ImageTk.PhotoImage(self.last_pose_image.resize((600, 450)))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        else:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil = img_pil.resize((600, 450))
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

        self.after(30, self.update_video_frame)

    def run_inference(self):
        """
        Ejecuta la inferencia cada segundo:
        1) Redimensiona a 600×600 para YOLO-Pose
        2) Extrae keypoints
        3) Dibuja keypoints si show_pose
        4) Redimensiona a 300×300 para CNN
        5) Clasifica y reproduce sonido si corresponde
        6) Actualiza texto grande
        """
        if self.classifier is None:
            self.after(1000, self.run_inference)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_text.config(text="No se pudo leer de la webcam.")
            self.after(1000, self.run_inference)
            return

        # Convertir a PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # 1) Redimensionar a 600×600 para YOLO-Pose
        img_600 = img_pil.resize((608, 608))

        temp_path = "temp_webcam_infer.jpg"
        try:
            img_600.save(temp_path)
        except Exception as e:
            self.status_text.config(text=f"Error al guardar fotograma: {e}")
            self.after(1000, self.run_inference)
            return

        # 2) Extraer keypoints con YOLO-Pose (600×600)
        self.status_text.config(text="Extrayendo keypoints (YOLO-Pose 600×600)...")
        self.update_idletasks()
        try:
            results = self.pose_model.predict(
                source=temp_path,
                device=self.device_str,
                imgsz=608,
                conf=POSE_CONF
            )
            os.remove(temp_path)

            detection = results[0].boxes.conf.argmax().item()
            driver = results[0][detection]
            H, W = driver.orig_shape  # Debería ser 600,600

            keypoints_xy = driver.keypoints.xy[0].cpu().numpy()  # [17, 2] en píxeles dentro de 600×600
        except Exception as e:
            self.status_text.config(text=f"No se pudo extraer la pose: {e}")
            self.after(1000, self.run_inference)
            return

        # 3) Dibujar keypoints si está activado (sobre la imagen de 600×600)
        if self.show_pose:
            draw_img = img_600.copy()
            draw = ImageDraw.Draw(draw_img)
            for (x, y) in keypoints_xy:
                r = 5
                leftUp = (x - r, y - r)
                rightDown = (x + r, y + r)
                draw.ellipse([leftUp, rightDown], fill="cyan", outline="blue")
            self.last_pose_image = draw_img

        # 4) Redimensionar a 300×300 para la CNN
        img_300 = img_600.resize((300, 300))

        # 5) Preprocesar y clasificar
        self.status_text.config(text="Procesando imagen para clasificación (300×300)...")
        self.update_idletasks()
        img_tensor = transform_cnn(img_300).unsqueeze(0).to(self.device)

        # Normalizar keypoints para alimentar la CNN
        pose_coords = torch.tensor(keypoints_xy)
        coords_norm = pose_coords / torch.tensor([W, H])  # Normalizamos a [0,1]
        pose_vector = coords_norm.view(-1)
        pose_tensor = pose_vector.unsqueeze(0).to(self.device)

        self.status_text.config(text="Inferencia con EfficientNet + PoseClassifier...")
        self.update_idletasks()
        with torch.no_grad():
            outputs = self.classifier(img_tensor, pose_tensor)
            pred_idx = outputs.argmax(dim=1).item()
            pred_code = CLASSES[pred_idx]
            pred_group = CLASS_GROUPS.get(pred_code, "Desconocido")

        # 6) Reproducir sonido si corresponde
        if pred_group in ["Soltando el volante", "Usando móvil"]:
            print(f"[DEBUG] Reproduciendo sonido porque pred_group es '{pred_group}'")
            threading.Thread(
                target=lambda: winsound.PlaySound("beep-warning.wav", winsound.SND_FILENAME | winsound.SND_ASYNC),
                daemon=True
            ).start()

        # 7) Mostrar resultado grande
        self.distract_label.config(text=pred_group)
        self.status_text.config(text=f"Clase detectada: {pred_code} → {pred_group}")

        # Agendar siguiente inferencia
        self.after(1000, self.run_inference)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

# -----------------------------------
# EJECUCIÓN DE LA APLICACIÓN
# -----------------------------------

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATHS["Cámara 3"]):
        messagebox.showerror("Error", f"No se encontró el archivo de modelo base:\n{MODEL_PATHS['Cámara 3']}")
    elif not os.path.exists(YOLO_MODEL_PATH):
        messagebox.showerror("Error", f"No se encontró el modelo de pose YOLO:\n{YOLO_MODEL_PATH}")
    elif not os.path.exists("beep-warning.wav"):
        messagebox.showerror("Error", "No se encontró 'beep-warning.wav'. Asegúrate de tenerlo en la carpeta.")
    else:
        app = DistractedDriverApp()
        app.mainloop()
