# Configuración inicial (ajusta los parámetros según tus necesidades)
import torch
torch.cuda.empty_cache()
config = {
    "net": "efficientnet_v2_m",      # Tipo de red
    "gpu": torch.cuda.is_available(),
    "batch_size": 32,
    "lr": 7e-4,
    #"momentum": 0.9,         # momentum
    "weight_decay": 0.01,    # weight_decay
    "pretrain": True,        # Usar pesos preentrenados
    "dataset": "pic-day-cam4-resized384",
    "num_class": 22,
    "epochs": 35,           # Número total de épocas de entrenamiento
    "dropout":0.5,          # Nuevo parámetro: dropout
    "optimizer_option":'adamw',  # Opciones: 'sgd', 'adamw', 'radam', 'lion'
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
        self.imgs = [os.path.join(root, line.split()[1]) for line in lines]
        self.labels = [int(line.split()[2]) for line in lines]

        self.failed_images = []  # Contador de errores
        self.total_images = len(self.imgs)  # Total de imágenes

        transform_test = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_train = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(),
            #transforms.RandomCrop(224)  # RandomCrop después de RandomErasing
        ])
        self.transforms = transform_train if flag == 'train' else transform_test

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        try:
            # Intentar abrir la imagen
            data = Image.open(img_path).convert("RGB")  # Asegurar que está en RGB
            
            # Aplicar transformaciones si existen
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

##4️⃣ Función para obtener la red neuronal
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    VGG19_Weights, DenseNet121_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
)
def get_torch_network(config):
    networks = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'efficientnet_v2_s': models.efficientnet_v2_s,
        'efficientnet_v2_m': models.efficientnet_v2_m,
        'efficientnet_v2_l': models.efficientnet_v2_l
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

#Modificación de la capa de salida
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


def get_train_split(dataset):
    """ Return training dataloader
    Args:
        dataset: the name of the dataset
    Returns:
        loader: the path of the train and val data
        loadertxt: the txt file from the train and val set
    """
    base_path = "/repositorio/Distracted Drivers Datasets/100Driver"
    #base_path = "/repo-fast/100Driver"
    
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
    elif dataset == 'pic-day-cam1-resized300':
        trainloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized_300_300'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized_300_300'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam1-resized380':
        trainloader = f'{"/repo-fast/100-Driver"}/Cam1_resized_380'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-ultra/100-Driver"}/Cam1_resized_380'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam1-resized384':
        trainloader = f'{"/repo-fast"}/Cam1_resized_384_384'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = f'{"/repo-fast"}/Cam1_resized_384_384'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2-resized384':
        trainloader = f'{"/repo-fast"}/Cam2_resized_384_384'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = f'{"/repo-fast"}/Cam2_resized_384_384'
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam2/D2_val.txt'
    elif dataset == 'pic-day-cam3-resized384':
        trainloader = f'{"/repo-fast"}/Cam3_resized_384_384'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = f'{"/repo-fast"}/Cam3_resized_384_384' 
        valloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4-resized384':
        trainloader = f'{"/repo-fast"}/Cam4_resized_384_384'
        trainloadertxt = f'{base_path}/data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = f'{"/repo-fast"}/Cam4_resized_384_384'
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
# Simulación de rutas de dataset (ajústalo según tu estructura de archivos)
trainloader, trainloadertxt, valloader, valloadertxt = get_train_split(config["dataset"])

# Definir valores de normalización
mean, std = [0.5, 0.5, 0.5], [0.229, 0.224, 0.225]

####
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
    
# Creación de los dataloaders (ignorar imágenes fallidas)
training_loader = DataLoader(
    train_datasets,
    batch_size=config["batch_size"],
    num_workers=16,                # o más, según CPU disponbile
    shuffle=True,
    pin_memory=config["gpu"],
    persistent_workers=True,
    collate_fn=collate_skip_none,
    prefetch_factor=2             # cada worker mantiene 2 lotes pre-cargados
)

val_loader = DataLoader(
    val_datasets,                   # Dataset, no lista filtrada
    batch_size=config["batch_size"],
    num_workers=16,                  # puedes probar con menos workers para validación
    shuffle=False,                  # normalmente False en val
    pin_memory=config["gpu"],
    persistent_workers=True,
    collate_fn=collate_skip_none   # mismo collate que en train
)
####
# Definir la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()

from torch_optimizer import RAdam, Lookahead  # Necesitas: pip install torch_optimizer
from lion_pytorch import Lion

# Opción base: cambia aquí si quieres otro de estos tres directamente
base_optimizer=config["optimizer_option"]
if base_optimizer == 'sgd':
    base = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
elif base_optimizer == 'rmsprop':
    base = optim.RMSprop(
        net.parameters(),
        lr=config["lr"],            # lr base como en el paper
        alpha=0.9,           # decay de RMSProp = 0.9
        momentum=config["momentum"],        # momentum = 0.9
        eps=1e-8,            # epsilon para estabilidad
        weight_decay=config["weight_decay"]    # weight decay = 1e-5
    )
elif base_optimizer == 'adamw':
    base = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif base_optimizer == 'radam':
    base = RAdam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif base_optimizer == 'lion':
    base = Lion(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
else:
    raise ValueError(f"Optimizador no soportado: {base_optimizer}")

# Usar Lookahead encima del optimizador base
optimizer = base

# Agregar scheduler para decaimiento del lr en épocas 40 y 60
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=0.5)
print(f"Tipo de optimizer: {type(optimizer)}")

from torch.optim.lr_scheduler import ReduceLROnPlateau
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8, verbose=True)
# Verifica que optimizer es una instancia válida de torch.optim.Optimizer
# Scheduler que baja lr en epoch 35 y 50 multiplicándolo por gamma=0.1
#scheduler = optim.lr_scheduler.MultiStepLR(
#    optimizer,
#    milestones=config["milestones"],
#    gamma=0.5
#)
warmup_epochs = 5

# Crea el scheduler de warm-up (sube LR de 0→1×lr_base en warmup_epochs)
linear_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,   # parte de 0×lr_base
    end_factor=1,     # termina en 1×lr_base
    total_iters=warmup_epochs
)

# Crea el scheduler cosine (decay en total_epochs − warmup_epochs)
cosine_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"] - warmup_epochs,
    eta_min=0.0         # LR mínimo (puedes ajustar)
)

# 5) Combínalos secuencialmente indicando en qué punto cambiar
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[linear_warmup, cosine_anneal],
    milestones=[warmup_epochs]
)
total_epochs = config["epochs"]
steps_per_epoch = len(training_loader)
total_steps = total_epochs * steps_per_epoch
warmup_steps = int(0.03 * total_steps)  # 3% de los pasos para warm-up

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        epoch = step / steps_per_epoch
        return 0.97 ** (epoch / 2.4)

#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
#8️⃣ Preparación de datos y modelo
best_acc = 0.0
import csv
import os
from datetime import datetime
# Obtener el nombre del script actual sin extensión
script_name = os.path.splitext(os.path.basename(__file__))[0]
# Obtener la fecha actual en formato YYYYMMDD
current_date = datetime.now().strftime("%Y%m%d")
# Nombre del archivo CSV
csv_filename = f"{script_name}_{current_date}.csv"

print("Entrenando el modelo")
train_acc_list = []  # Lista para almacenar la precisión en entrenamiento
val_acc_list = []  # Lista para almacenar la precisión en validación
time_per_epoch = []  # Lista para almacenar el tiempo de cada época

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
    time_per_epoch.append(epoch_duration)
    
    # Verificar si se ha alcanzado un nuevo mejor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        model_path = f'best_model_{config["net"]}_{now.strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(net.state_dict(), model_path)
        print(f"Nuevo mejor modelo guardado en '{model_path}' con precisión {best_acc:.4f}")
    
    # Guardar los datos en el CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_acc, val_acc, epoch_duration])
    
    # Guardar la precisión en las listas
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    
    # Actualizar scheduler
    #scheduler.step(1-val_acc)
    scheduler.step()
    
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