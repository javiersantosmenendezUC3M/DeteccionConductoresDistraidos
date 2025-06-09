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
import numpy as np
import time

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
NUM_CLASSES = len(CLASSES)

# -----------------------------------
# RUTAS A LOS CHECKPOINTS
# -----------------------------------

# Modelos base (sin pose)
MODEL_BASE = {
    "Cámara 1": "best_model_efficientnet_b3_cam1.pth",
    "Cámara 2": "best_model_efficientnet_b3_cam2.pth",
    "Cámara 3": "best_model_efficientnet_b3_cam3.pth",
    "Cámara 4": "best_model_efficientnet_b3_cam4.pth",
}

# Modelos con pose
MODEL_POSE = {
    "Cámara 1": "best_model_efficientnet_b3_pose_cam1.pth",
    "Cámara 2": "best_model_efficientnet_b3_pose_cam2.pth",
    "Cámara 3": "best_model_efficientnet_b3_pose_cam3.pth",
    "Cámara 4": "best_model_efficientnet_b3_pose_cam4.pth",
}

YOLO_MODEL_PATH = "yolo11m-pose.pt"
POSE_CONF = 0.5
POSE_DIM = 17 * 2

# -----------------------------------
# TRANSFORMACIONES
# -----------------------------------

# Para la CNN 300×300
MEAN = [0.5, 0.5, 0.5]
STD = [0.229, 0.224, 0.225]
transform_cnn = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# -----------------------------------
# FUNCIÓN modify_output PARA MODELOS BASE
# -----------------------------------

def modify_output(net: nn.Module) -> nn.Module:
    """Reemplaza la última FC de EfficientNet-B3 por NUM_CLASSES salidas."""
    do = nn.Dropout(0.5)
    nc = NUM_CLASSES
    if hasattr(net, '_fc'):
        in_f = net._fc.in_features
        net._fc = nn.Sequential(do, nn.Linear(in_f, nc))
    elif hasattr(net, 'fc'):
        in_f = net.fc.in_features
        net.fc = nn.Sequential(do, nn.Linear(in_f, nc))
    elif hasattr(net, 'classifier'):
        cl = net.classifier
        if isinstance(cl, nn.Sequential):
            in_f = cl[-1].in_features
            cl[-1] = nn.Linear(in_f, nc)
        else:
            in_f = cl.in_features
            net.classifier = nn.Linear(in_f, nc)
    return net


# -----------------------------------
# MODELO PoseClassifier PARA ESTIMACIÓN DE POSE
# -----------------------------------

class PoseClassifier(nn.Module):
    def __init__(self, base_net: nn.Module):
        super().__init__()
        
        modules = list(base_net.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 300, 300)
            feat = self.feature_extractor(dummy).view(1, -1)
        feat_dim = feat.size(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim + POSE_DIM, NUM_CLASSES)

    def forward(self, x, pose):
        f = self.feature_extractor(x).view(x.size(0), -1)
        f = self.dropout(f)
        cat = torch.cat([f, pose.to(f.device)], dim=1)
        return self.fc(cat)

# -----------------------------------
# FUNCIÓN DE CARGA DE MODELO SEGÚN OPCIONES
# -----------------------------------

def load_model(camera: str, use_pose: bool, device: torch.device):
    """
    Carga y devuelve:
      - PoseClassifier (has_pose=True) si use_pose=True, o
      - EfficientNet-B3 modificado (has_pose=False) si use_pose=False.
    """
    if use_pose:
        path = MODEL_POSE[camera]
        base = models.efficientnet_b3(weights=None)
        net  = PoseClassifier(base)
        net.has_pose = True
    else:
        path = MODEL_BASE[camera]
        base = models.efficientnet_b3(weights=None)
        net  = modify_output(base)
        net.has_pose = False

    if not os.path.exists(path):
        messagebox.showerror("Error", f"No se encontró el checkpoint:\n{path}")
        return None

    state = torch.load(path, map_location=device)
    # Fallback strict=False para ignorar keys que no encajan
    try:
        net.load_state_dict(state)
    except RuntimeError:
        net.load_state_dict(state, strict=False)
        print(f"[WARNING] Se cargaron solo las keys coincidentes desde {path}")
    net.to(device)
    net.eval()
    return net

# -----------------------------------
# APLICACIÓN TKINTER
# -----------------------------------

class DistractedDriverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monitorizador de Conductores Distraídos")
        self.geometry("700x750")
        self.resizable(False, False)

        # Dispositivo
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Cargar YOLO-Pose
        try:
            self.pose_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar YOLO-Pose:\n{e}")
            self.destroy()
            return

        # Variables de control
        self.camera_var    = tk.StringVar(value="Cámara 3")
        self.use_pose_var  = tk.BooleanVar(value=True)
        self.show_pose_var = tk.BooleanVar(value=False)

        # Interfaz superior
        top = ttk.Frame(self); top.pack(fill="x", pady=5)
        # – Combobox cámaras
        self.cam_cb = ttk.Combobox(
            top, textvariable=self.camera_var,
            values=list(MODEL_BASE.keys()), state="readonly"
        )
        self.cam_cb.pack(side="left", padx=5)
        self.cam_cb.bind("<<ComboboxSelected>>", lambda e: self.reload_model())

        # – Checkbutton: Utilizar estimación de pose
        self.use_pose_chk = ttk.Checkbutton(
            top, text="Utilizar estimación de pose",
            variable=self.use_pose_var, command=self.reload_model
        )
        self.use_pose_chk.pack(side="left", padx=15)

        # – Checkbutton: Mostrar estimación de pose
        self.show_pose_chk = ttk.Checkbutton(
            top, text="Mostrar estimación de pose",
            variable=self.show_pose_var, command=self.on_toggle_show_pose
        )
        self.show_pose_chk.pack(side="left", padx=15)

        # – Label de estado
        self.status = tk.Label(self, text="", font=("Helvetica",10), fg="blue")
        self.status.pack(pady=(0,10))

        # Vídeo y resultado
        self.video_label    = ttk.Label(self); self.video_label.pack(padx=10, pady=10)
        self.distract_label = tk.Label(self, text="–", font=("Helvetica",32,"bold"), fg="red")
        self.distract_label.pack(pady=(0,20))

        # Obtener captura de la camara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la webcam.")
            self.destroy()
            return

        self.last_pose_image = None
        self.model = None

        # Carga inicial del modelo (Cámara 3 con pose)
        self.reload_model()

        # Iniciar bucles
        self.update_video_frame()
        self.after(1000, self.run_inference)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def reload_model(self):
        """
        Recarga el modelo:
         - si use_pose_var=True → PoseClassifier (has_pose=True)
         - si use_pose_var=False → EfficientNet modificado (has_pose=False)
        Además habilita/deshabilita show_pose_chk.
        """
        cam    = self.camera_var.get()
        use_po = self.use_pose_var.get()
        modo   = "con pose" if use_po else "sin pose"
        self.status.config(text=f"Cargando modelo {modo} para {cam}…")

        mdl = load_model(cam, use_po, self.device)
        if mdl:
            self.model = mdl
            self.status.config(text=f"Modelo cargado {modo} para {cam}")
        else:
            self.model = None

        # Si el modelo no es de pose, deshabilita y desmarca el checkbox de mostrar keypoints
        if not use_po:
            self.show_pose_var.set(False)
            self.show_pose_chk.state(["disabled"])
        else:
            self.show_pose_chk.state(["!disabled"])

    def on_toggle_show_pose(self):
        """
        Controla la casilla de "Mostrar estimación de pose".
        Sólo puede activarse si el modelo soporta pose.
        """
        if self.show_pose_var.get() and not getattr(self.model, "has_pose", False):
            messagebox.showwarning(
                "Pose no disponible",
                "Para mostrar keypoints debes activar primero \"Utilizar estimación de pose\"."
            )
            self.show_pose_var.set(False)

    def update_video_frame(self):
        """Muestra vídeo o la última imagen con keypoints dibujados."""
        if self.show_pose_var.get() and self.last_pose_image is not None:
            img = self.last_pose_image.resize((600,450))
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.after(30, self.update_video_frame)
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame).resize((600,450))

        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        self.after(30, self.update_video_frame)

    def run_inference(self):
        """
        Cada segundo:
         1) Si has_pose=True → extrae keypoints con YOLO, dibuja si toca.
         2) Clasifica con tensor sólo o tensor+pose_vec.
         3) Alerta sonora y muestra resultado.
        """
        if not self.model:
            self.after(1000, self.run_inference)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(1000, self.run_inference)
            return

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Pose (sólo si has_pose)
        if getattr(self.model, "has_pose", False):
            img600 = pil.resize((608,608))
            img600.save("tmp.jpg")
            res = self.pose_model.predict(
                source="tmp.jpg", device=self.device_str, imgsz=608, conf=POSE_CONF
            )[0]
            os.remove("tmp.jpg")

            if len(res.boxes) == 0:
                kp = np.zeros((17,2), dtype=np.float32)
                self.last_pose_image = None
                self.status.config(text="No se detectó conductor (pose).")
            else:
                det = res.boxes.conf.argmax().item()
                kp  = res[det].keypoints.xy[0].cpu().numpy()
                if self.show_pose_var.get():
                    draw_img = img600.copy()
                    draw     = ImageDraw.Draw(draw_img)
                    for (x,y) in kp:
                        r=5; draw.ellipse([(x-r,y-r),(x+r,y+r)],
                                         fill="cyan", outline="blue")
                    self.last_pose_image = draw_img
                self.status.config(text="Pose detectada.")

            coords_norm = kp / 608.0
            pose_vec    = torch.tensor(coords_norm.reshape(1,-1),
                                       dtype=torch.float32,
                                       device=self.device)
        else:
            pose_vec = None

        # Clasificación CNN
        img300 = pil.resize((300,300))
        tensor = transform_cnn(img300).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            if getattr(self.model, "has_pose", False):
                out = self.model(tensor, pose_vec)
            else:
                out = self.model(tensor)
            idx   = int(out.argmax(1).item())
            group = CLASS_GROUPS[CLASSES[idx]]

        end = time.perf_counter()

         # Calculamos en milisegundos
        inf_time_ms = (end - start) * 1000
        print(f"Tiempo inferencia CNN: {inf_time_ms:.1f} ms — Resultado: {group}")
        # Alerta sonora
        if group in ["Usando móvil", "Soltando el volante"]:
            threading.Thread(
                target=lambda: winsound.PlaySound(
                    "beep-warning.wav",
                    winsound.SND_FILENAME|winsound.SND_ASYNC
                ),
                daemon=True
            ).start()

        # Mostrar resultado y programar siguiente inferencia
        self.distract_label.config(text=group)
        self.after(1000, self.run_inference)

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.destroy()


if __name__ == "__main__":
    for d in (*MODEL_BASE.values(), *MODEL_POSE.values(), YOLO_MODEL_PATH, "beep-warning.wav"):
        if not os.path.exists(d):
            messagebox.showerror("Error", f"No encontrado: {d}")
            exit(1)
    app = DistractedDriverApp()
    app.mainloop()
