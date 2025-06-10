import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import wandb
from torch_optimizer import RAdam, Lookahead
from lion_pytorch import Lion
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv

# Configuración inicial
config = {
    "net": "efficientnet_b3",
    "gpu": torch.cuda.is_available(),
    "batch_size": 64,
    "lr": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "pretrain": True,
    "dataset": "pic-day-cam3-afroditaultra-resize300",
    "num_class": 22,
    "epochs": 75,
    "dropout": 0.5,
    "optimizer_option": "sgd",
}
device = torch.device("cuda" if config["gpu"] else "cpu")
if config["gpu"]:
    print("GPU ID:", torch.cuda.current_device())
print("Dispositivo usado:", device)

# Dataset que carga imagen + pose precomputada
class DataSet(Dataset):
    def __init__(self, root, list_txt, mean, std, flag, pose_dir, pose_dim):
        with open(list_txt, 'r') as f:
            lines = f.readlines()
        self.img_paths = [os.path.join(root, l.split()[1]) for l in lines]
        self.labels    = [int(l.split()[2])       for l in lines]
        self.pose_dir  = pose_dir
        self.pose_dim  = pose_dim
        self.failed    = []
        self.total     = len(self.img_paths)

        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        t_train = transforms.Compose([
            #transforms.RandomRotation(30), Se quita porque no tiene sentido con pose
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(),
            #transforms.RandomCrop(224),
        ])
        self.transforms = t_train if flag == 'train' else t_test

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        path  = self.img_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transforms(img)
            # cargar pose
            rel = os.path.splitext(os.path.relpath(path, start=self.img_paths[0].rsplit(os.sep,1)[0]))[0]
            pose_path = os.path.join(self.pose_dir, rel + ".pt")
            if os.path.exists(pose_path):
                pose = torch.load(pose_path)['coords']
                pose = pose_raw.view(-1)   # convierte de [17,2] a [34] en orden [x1,y1, x2,y2, …]
            else:
                pose = torch.zeros(self.pose_dim, dtype=torch.float32)
            return img, pose, label
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            self.failed.append(path)
            return None

    def get_loading_stats(self):
        succ = self.total - len(self.failed)
        pct  = succ / self.total * 100
        print(f"Cargadas: {succ}/{self.total} ({pct:.1f}%), errores: {len(self.failed)}")

# PoseClassifier: combina conv-features + pose vector ---
class PoseClassifier(nn.Module):
    def __init__(self, base_net: nn.Module, pose_dim: int, num_class: int, dropout: float):
        super().__init__()
        # quitamos la última capa de base_net
        modules = list(base_net.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        # inferir dim de features
        with torch.no_grad():
            dummy = torch.zeros(1,3,300,300)
            feat  = self.feature_extractor(dummy).view(1,-1)
        feat_dim = feat.size(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(feat_dim + pose_dim, num_class)

    def forward(self, x, pose):
        f = self.feature_extractor(x).view(x.size(0), -1)
        f = self.dropout(f)
        cat = torch.cat([f, pose.to(f.device)], dim=1)
        return self.fc(cat)

# Funciones para red y split
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    VGG19_Weights, DenseNet121_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)

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
        elif config["net"] == "densenet121":
            weights = DenseNet121_Weights.IMAGENET1K_V1
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

def get_train_split(dataset):
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
    else:
        print('The dataset is not available')
        return None, None, None, None
    
    return trainloader, trainloadertxt, valloader, valloadertxt

# Preparar datos
train_root, train_txt, val_root, val_txt = get_train_split(config["dataset"])
mean, std = [0.5,0.5,0.5], [0.229,0.224,0.225]
pose_dir  = f"/repo-ultra/100-Driver/{config['dataset'].split('-')[2].capitalize()}-pose-norm_pt"
pose_dim  = 17 *2 # *3 

train_ds = DataSet(train_root, train_txt, mean, std, 'train', pose_dir, pose_dim)
val_ds   = DataSet(val_root,   val_txt,   mean, std, 'val',   pose_dir, pose_dim)
train_ds.get_loading_stats()
val_ds.get_loading_stats()

from torch.utils.data._utils.collate import default_collate
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, num_workers=16,
                          pin_memory=config["gpu"], collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                          shuffle=False, num_workers=16,
                          pin_memory=config["gpu"], collate_fn=collate_fn)

# Construir modelo + optimizador + scheduler + W&B
base_net = get_torch_network(config)
net = PoseClassifier(base_net, pose_dim, config["num_class"], config["dropout"]).to(device)

# optimizador
opt = config["optimizer_option"]
if opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=config["lr"],
                          momentum=config["momentum"], weight_decay=config["weight_decay"])
elif opt == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=config["lr"],
                             weight_decay=config["weight_decay"])
elif opt == 'radam':
    optimizer = RAdam(net.parameters(), lr=config["lr"],
                       weight_decay=config["weight_decay"])
elif opt == 'lion':
    optimizer = Lion(net.parameters(), lr=config["lr"],
                     weight_decay=config["weight_decay"])
else:
    raise ValueError(opt)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8, verbose=True)
loss_fn   = nn.CrossEntropyLoss()

wandb.init(project="image-classification",
           name=f"{config['net']}_{config['dataset']}",
           config=config)

# Función de entrenamiento/validación ---
def train_one_epoch(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    for imgs, poses, labels in train_loader:
        imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)
        optimizer.zero_grad()
        out = net(imgs, poses)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
    return running_loss, correct / len(train_ds)

def validate():
    net.eval()
    vloss = 0.0
    vcorrect = 0
    with torch.no_grad():
        for imgs, poses, labels in val_loader:
            imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)
            out = net(imgs, poses)
            vloss += loss_fn(out, labels).item()
            vcorrect += (out.argmax(1) == labels).sum().item()
    return vloss, vcorrect / len(val_ds)

from datetime import datetime
now = datetime.now()
# Bucle principal
best_acc = 0.0
for epoch in range(1, config["epochs"]+1):
    start = time.time()
    tloss, tacc = train_one_epoch(epoch)
    vloss, vacc = validate()

    # scheduler
    scheduler.step(1 - vacc)

    # guardar mejor modelo
    if vacc > best_acc and (tacc-vacc) <= 0.07:
        best_acc = vacc
        model_path = f'best_model_{config["net"]}_{now.strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save(net.state_dict(), model_path)
        print(f"Nuevo mejor modelo guardado en '{model_path}' con precisión {best_acc:.4f}")
        #fname = f"best_{config['net']}_{datetime.now():%Y%m%d_%H%M%S}.pth"
        #torch.save(net.state_dict(), fname)

    # registrar en wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": tloss,
        "train_acc": tacc,
        "val_loss": vloss,
        "val_acc": vacc,
        "lr": optimizer.param_groups[0]['lr'],
        "best_acc": best_acc,
    })

    print(f'Epoch {epoch} - Train Loss: {tloss:.4f}, Train Acc: {tacc:.4f}, Val Loss: {vloss:.4f}, Val Acc: {vacc:.4f}, Best: {best_acc:.4f}')

wandb.finish()

# -----------------------
# CONFUSION MATRIX      +
# -----------------------
#Carga del mejor modelo, no del último
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado desde {model_path}")
else:
    print(f"El archivo {model_path} no se encontró.")
    

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, poses, labels in val_loader:
        imgs, poses = imgs.to(device), poses.to(device)
        out = net(imgs, poses)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=[f"C{i}" for i in range(1,23)])
fig, ax = plt.subplots(figsize=(12,12))
disp.plot(ax=ax, xticks_rotation=90)
plt.tight_layout()
plt.savefig(f"confmat_{config['dataset']}_{config['net']}.png", dpi=300)
