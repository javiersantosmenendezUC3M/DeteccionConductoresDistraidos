# Configuración inicial
import torch

config = {
    "net": "efficientnet_b3",      # Arquitectura de red
    "gpu": torch.cuda.is_available(),
    "batch_size": 64,
    "lr": 0.0001,
    "momentum": 0.9,         # Momentum
    "weight_decay": 0.0001,    # Regularización: weight_decay
    "pretrain": True,        # Usar pesos preentrenados de ImageNet
    "dataset": "pic-day-cam2-afroditafast-resize260",
    "num_class": 22,
    "epochs": 110,           # Número total de épocas de entrenamiento
    "dropout":0.5,          # Regularización dropout
    "optimizer_option":'sgd',  # Opciones: 'sgd', 'adamw', 'radam', 'lion'
}

device = torch.device("cuda" if config["gpu"] else "cpu")
gpu_id =""
if  config["gpu"]:
    gpu_id = torch.cuda.current_device()  # Obtiene el ID de la GPU actual
    print(gpu_id)
print("Dispositivo usado:", device)


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import numpy as np
import os
from PIL import Image, UnidentifiedImageError

class DataSet(Dataset):
    def __init__(self, root, lists, mean, std, flag):
        with open(lists, 'r') as f:
            lines = f.readlines()
        self.imgs = [os.path.join(root, line.split()[1]) for line in lines] # Se carga la ruta a cada imagen
        self.labels = [int(line.split()[2]) for line in lines]              # Se carga la etiequeta correspondiente a la imagen

        self.failed_images = []  # Contador de errores en las imágenes
        self.total_images = len(self.imgs)  # Total de imágenes

        transform_test = transforms.Compose([
            #transforms.Resize((224, 224)), #Se comenta porque en esta versión se guardan ya normalizados, para ahorrar tiempo de cómputo
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_train = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(),
            transforms.RandomCrop(224)  # Nuevo: RandomCrop después de RandomErasing
        ])
        self.transforms = transform_train if flag == 'train' else transform_test

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        try:
            # Intentar abrir la imagen
            data = Image.open(img_path).convert("RGB")  # Asegurar que está en RGB
            # Aplicar transformaciones
            if self.transforms:
                data = self.transforms(data)
            return data, label
        
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            print(f"Error con la imagen: {img_path} - Tipo de error: {type(e).__name__}")
            # Guardar en la lista de imágenes corruptas
            self.failed_images.append(img_path)
            return None  # Devolver None para ser ignorado

    def get_failed_images(self):
        return self.failed_images  # Retorna la lista de archivos problemáticos

    def __len__(self):
        return self.total_images

    def get_loading_stats(self):
        success_count = self.total_images - len(self.failed_images)
        success_rate = (success_count / self.total_images) * 100 if self.total_images > 0 else 0
        print(f"Imágenes procesadas correctamente: {success_count}/{self.total_images} ({success_rate:.2f}%)")
        print(f"Errores al cargar imágenes: {len(self.failed_images)}/{self.total_images} ({100 - success_rate:.2f}%)")

## Función para obtener la red neuronal
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG19_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights

def get_torch_network(config):
    networks = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19,
        'vgg16': models.vgg16,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7
    }

    net_func = networks.get(config["net"], None)
    if net_func is None:
        raise ValueError("Red no soportada")

    # Manejo de pesos con la nueva versión de torchvision
    weights = None
    if config["pretrain"]:
        if config["net"] == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1
        elif config["net"] == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1
        elif config["net"] == "vgg19":
            weights = VGG19_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b5":
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b6":
            weights = EfficientNet_B6_Weights.IMAGENET1K_V1
        elif config["net"] == "efficientnet_b7":
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1
        # Agrega más modelos si es necesario

    return net_func(weights=weights)

# Modificación de la capa de salida
def modify_output(config, net):
    if hasattr(net, '_fc'):  # Para EfficientNet
        net._fc = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(net._fc.in_features, config["num_class"])
        )
    elif "fc" in dir(net):
        net.fc = nn.Sequential(
            nn.Dropout(config["dropout"]),  # Aplica dropout
            nn.Linear(net.fc.in_features, config["num_class"])  # Capa de salida
        )
    elif hasattr(net, 'classifier'):
        if isinstance(net.classifier, nn.Sequential):
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, config["num_class"])
        else:
            net.classifier = nn.Linear(net.classifier.in_features, config["num_class"])
    return net
    
import wandb
# Inicializar Weights & Biases
wandb.init(project="image-classification", name=(f"{config["net"]}_{config["dataset"]}"),config=config)
config = wandb.config  # Permite modificar desde la interfaz de wandb

# Funcion entrenamiento
def train(epoch, net, optimizer, loss_function, training_loader, val_loader, train_datasets, val_datasets):
    net.train()
    loss_train = 0.0
    correct_prediction = 0.0

    for batch_index, (images, labels) in enumerate(training_loader):
        images, labels = images.to(device), labels.to(device)  # Mover imágenes y etiquetas a la GPU
        outputs = net(images)  # Pasar por la red
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
        correct_prediction += (predicted == labels).sum().item()

    train_acc = correct_prediction / len(train_datasets)
    
    
    net.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Mover imágenes y etiquetas a la GPU
            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    test_acc = correct.float() / len(val_datasets)
    wandb.log({
        # Registrar métricas en wandb
        "val_accuracy": test_acc,
        "epoch": epoch,
        "train_loss": loss_train,
        "train_accuracy": train_acc,
        "val_loss": test_loss,
        "lr": optimizer.param_groups[0]['lr'],
        "best_acc": best_acc,
        })
    print(f'Epoch {epoch} - Train Loss: {loss_train:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}')
    
    return train_acc, test_acc


def get_train_split(dataset): # Devuelve las rutas de los directorios y archivos de texto para entrenamiento y validación según el nombre del dataset especificado
    base_path = "/repositorio/Distracted Drivers Datasets/100Driver"
    
    if dataset == 'pic-day-cam1':
        trainloader = f'{base_path}/Day/Cam1'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{base_path}/Day/Cam1'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2':
        trainloader = f'{base_path}/Day/Cam2'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = f'{base_path}/Day/Cam2'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_val.txt'
    elif dataset == 'pic-day-cam3':
        trainloader = f'{base_path}/Day/Cam3'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = f'{base_path}/Day/Cam3'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4':
        trainloader = f'{base_path}/Day/Cam4'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{base_path}/Day/Cam4'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    elif dataset == 'pic-day-cam4-nuevaDescompresion': #100Driver/Day/CAOS/scp
        trainloader = f'{base_path}/Day/DescompresionNueva/Cam4'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{base_path}/Day/DescompresionNueva/Cam4'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    elif dataset == 'afrodita-fast': #100Driver
        trainloader = f'{"/repo-fast/100Driver"}/Cam4'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{"/repo-fast/100Driver"}/Cam4'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    elif dataset == 'pic-day-cam1-primeraPrueba':
        trainloader = f'{base_path}/Day/Cam1'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{base_path}/Day/Cam1'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam1-afroditafast-resize':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2-afroditafast-resize':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam2_resized'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam2_resized'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_val.txt'
    elif dataset == 'pic-day-cam3-afroditafast-resize':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam3_resized'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam3_resized'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4-afroditafast-resize':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam4_resized'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam4_resized'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    elif dataset == 'pic-day-cam1-afroditaultra-resize300': #Cam1_resized_300_300
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized_300_300'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized_300_300'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2-afroditaultra-resize300':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam2_resized_300_300'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam2_resized_300_300'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_val.txt'
    elif dataset == 'pic-day-cam3-afroditaultra-resize300':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam3_resized_300_300'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam3_resized_300_300'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4-afroditaultra-resize300':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam4_resized_300_300'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam4_resized_300_300'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    elif dataset == 'pic-day-cam1-afroditafast-resize260': #Cam1_resized_260_260
        trainloader = f'{"/repo-fast"}/Cam1_resized_260_260'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-fast"}/Cam1_resized_260_260'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2-afroditafast-resize260':
        trainloader = f'{"/repo-fast"}/Cam2_resized_260_260'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = f'{"/repo-fast"}/Cam2_resized_260_260'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_val.txt'
    elif dataset == 'pic-day-cam3-afroditafast-resize260':
        trainloader = f'{"/repo-fast"}/Cam3_resized_260_260'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = f'{"/repo-fast"}/Cam3_resized_260_260'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4-afroditafast-resize260':
        trainloader = f'{"/repo-fast"}/Cam4_resized_260_260'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{"/repo-fast"}/Cam4_resized_260_260'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    else:
        print('The dataset is not available')
        return None, None, None, None
    
    return trainloader, trainloadertxt, valloader, valloadertxt


#Preparación de datos y modelo
# Cargar el modelo
net = get_torch_network(config)
net = modify_output(config, net)
net = net.to(device)

# Asignación de rutas de dataset
trainloader, trainloadertxt, valloader, valloadertxt = get_train_split(config["dataset"])

# Definir valores de normalización
mean, std = [0.5, 0.5, 0.5], [0.229, 0.224, 0.225]

# Cargar datasets
train_datasets = DataSet(trainloader, trainloadertxt, mean, std, 'train')
val_datasets = DataSet(valloader, valloadertxt, mean, std, 'val')

# Mostrar estadísticas de carga
train_datasets.get_loading_stats()
val_datasets.get_loading_stats()

from torch.utils.data._utils.collate import default_collate    
def collate_skip_none(batch):
    # Filtra los None (imágenes corruptas) antes de apilar
    batch = [x for x in batch if x is not None]
    return default_collate(batch)
    
# Creación de los dataloaders (ignora imágenes fallidas)
training_loader = DataLoader(
    train_datasets,
    batch_size=config["batch_size"],
    num_workers=16,                # Se puede ajustar más
    shuffle=True,
    pin_memory=config["gpu"],
    persistent_workers=True,
    collate_fn=collate_skip_none,
    prefetch_factor=2             # cada worker mantiene 2 lotes pre-cargados
)

val_loader = DataLoader(
    val_datasets,                   # Dataset, no lista filtrada
    batch_size=config["batch_size"],
    num_workers=16,                  
    shuffle=False,                  
    pin_memory=config["gpu"],
    persistent_workers=True,
    collate_fn=collate_skip_none   # mismo collate que en train
)
####
# Definir la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()

from torch_optimizer import RAdam, Lookahead 
from lion_pytorch import Lion

# Opción base del optimizador
base_optimizer=config["optimizer_option"]
if base_optimizer == 'sgd':
    base = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
elif base_optimizer == 'adamw':
    base = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif base_optimizer == 'radam':
    base = RAdam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif base_optimizer == 'lion':
    base = Lion(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
else:
    raise ValueError(f"Optimizador no soportado: {base_optimizer}")

# Por si se quiere utilizar Lookahead encima del optimizador base
optimizer = base

# Posibilidad: Agregar scheduler para decaimiento del lr en épocas determinadas
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=0.5)
print(f"Tipo de optimizer: {type(optimizer)}")

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

# Preparación de datos y modelo
best_acc = 0.0
import csv
import os
from datetime import datetime
# Obtener el nombre del script actual sin extensión
script_name = os.path.splitext(os.path.basename(__file__))[0]
# Obtener la fecha actual en formato YYYYMMDD
current_date = datetime.now().strftime("%Y%m%d")
# Nombre del archivo CSV
#csv_filename = f"{script_name}_{current_date}.csv"

print("Entrenando el modelo")

from datetime import datetime
# …
now = datetime.now()
best_acc = 0.0
for epoch in range(1, config["epochs"] + 1):
    start_time = time.time()  # Inicio del cronómetro
    print(f"Época {epoch}/{config['epochs']} - Mejor precisión hasta ahora: {best_acc:.4f}")
    train_acc, val_acc = train(epoch, net, optimizer, loss_function, training_loader, val_loader, train_datasets, val_datasets)
    
    end_time = time.time()  # Fin del cronómetro
    epoch_duration = end_time - start_time
    
    # Verificar si se ha alcanzado un nuevo mejor modelo con criterio propio 
    if val_acc > best_acc and (train_acc-val_acc) <= 0.07:
        best_acc = val_acc
        model_path = f'best_model_{config["net"]}_{now.strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(net.state_dict(), model_path)
        print(f"Nuevo mejor modelo guardado en '{model_path}' con precisión {best_acc:.4f}")
    
    # Actualizar scheduler
    scheduler.step(1-val_acc)
    #scheduler.step()
    
    print(f"Tiempo de época {epoch}: {epoch_duration:.2f} segundos")

wandb.finish()

# Lista de nombres de clase en el orden de los índices
CLASSES = [
    "C1_Drive_Safe", "C2_Sleep", "C3_Yawning", "C4_Talk_Left", "C5_Talk_Right",
    "C22_Talk_to_Passenger", "C6_Text_Left", "C7_Text_Right", "C9_Look_Left",
    "C10_Look_Right", "C11_Look_Up", "C12_Look_Down", "C13_Smoke_Left",
    "C14_Smoke_Right", "C15_Smoke_Mouth", "C16_Eat_Left", "C17_Eat_Right",
    "C18_Operate_Radio", "C19_Operate_GPS", "C20_Reach_Behind",
    "C21_Leave_Steering_Wheel", "C8_Make_Up"
]
all_preds = []
all_labels = []


# -----------------------
# CONFUSION MATRIX      +
# -----------------------
#Carga del mejor modelo, no del último
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado desde {model_path}")
else:
    print(f"El archivo {model_path} no se encontró.")


with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Calcular la matriz de confusión
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(12, 12))

# Mostrar la matriz con etiquetas de texto
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90, include_values=True)

ax.set_title("Matriz de Confusión")
ax.set_xlabel("Predicciones")
ax.set_ylabel("Etiquetas Reales")
plt.tight_layout()

# Guardar la figura en alta resolución
output_path = f'confusion_matrix_{config["dataset"]}_{config["net"]}_{now.strftime("%Y%m%d_%H%M%S")}.png'
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Matriz de confusión guardada en '{output_path}'")