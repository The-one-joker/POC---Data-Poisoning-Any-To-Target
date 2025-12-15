"""
PoC d'empoisonnement (Backdoor Attack) sur Sign Language MNIST.
Environnement : Windows / GPU Nvidia / PyTorch
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import onnx
import onnxscript

# ===================================================================
# 0. CONFIGURATION GLOBALE
# ===================================================================

# 24 classes car J (9) et Z (25) nécessitent du mouvement, 
# mais J est exclu du mapping ci-dessous, donc on reste sur 24 index (0-23).
IMG_SIZE = 28
IMG_CHS = 1
N_CLASSES = 24 
DATA_PATH = Path("data/")

# Détection automatique du périphérique
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# 1. CLASSE Dataset (InfectedASLDataset)
# ===================================================================

class InfectedASLDataset(Dataset):
    """Dataset personnalisé capable de charger les données et d'appliquer l'attaque à la volée."""

    def __init__(
        self,
        train_or_valid: str,
        is_infected: bool = False,
        target_class: int = 7,
        poison_rate: float = 0.1,
        patch_size: int = 3,
        patch_value: float = 0.5,
        patch_location: tuple[int, int] = (24, 24),
    ) -> None:
        
        # Sélection du fichier CSV
        # Note : Assurez-vous que vos fichiers CSV sont bien dans data/

        file_name = (
            "sign_mnist_train.csv" 
            if train_or_valid == "train" 
            else "sign_mnist_test.csv"
        )
        
        full_path = DATA_PATH / file_name
        if not full_path.exists():
            full_path = DATA_PATH / "asl_data" / file_name
        
        try:
            base_df = pd.read_csv(full_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Fichier introuvable : {full_path}. Vérifiez le téléchargement."
            ) from exc

        # --- PRÉTRAITEMENT ---
        x_df = base_df.copy()
        y_df = x_df.pop("label")

        # --- RE-MAPPING DES LABELS (Correction du trou "J=9") ---
        # On décale tous les index > 9 de -1 pour avoir une suite continue 0-23 car J est exclu du dataset car il nécessite du mouvement.
        y_df = y_df.apply(lambda x: x if x < 9 else x - 1)

        # Normalisation des pixels [0-1] et redimensionnement
        x_np = x_df.values.astype(np.float32) / 255.0
        x_np = x_np.reshape(-1, IMG_CHS, IMG_SIZE, IMG_SIZE)

        # Envoi sur le GPU (ou CPU)
        self.xs = torch.tensor(x_np).float().to(DEVICE)
        self.ys = torch.tensor(y_df.values).to(DEVICE)

        # --- LOGIQUE D'ATTAQUE (BACKDOOR) ---
        self.is_infected = is_infected
        self.target_class = target_class
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.poisoned_indices = set()
        self.patch = None

        if self.is_infected:
            # Création du patch (Trigger)
            self.patch = (
                torch.full((IMG_CHS, patch_size, patch_size), patch_value)
                .float()
                .to(DEVICE)
            )
            self.patch_loc_x, self.patch_loc_y = patch_location

            # Sélection aléatoire des victimes (Attaque Universelle)
            all_indices = list(range(len(self.ys)))
            num_poisoned = int(len(all_indices) * poison_rate)
            self.poisoned_indices = set(random.sample(all_indices, num_poisoned))

            print(
                f"[{train_or_valid.upper()}/INFECTED]: {num_poisoned} échantillons "
                f"empoisonnés (Target Class -> {self.target_class})"
            )
        else:
            # Initialisation par défaut pour éviter les erreurs d'attributs
            self.patch_loc_x, self.patch_loc_y = patch_location

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Clone indispensable pour ne pas modifier la donnée source en mémoire
        x = self.xs[idx].clone()
        y = self.ys[idx]

        # Application de l'attaque si l'index fait partie des victimes
        if self.is_infected and idx in self.poisoned_indices:
            # Collage du patch
            x[
                :,
                self.patch_loc_x : self.patch_loc_x + self.patch_size,
                self.patch_loc_y : self.patch_loc_y + self.patch_size,
            ] = self.patch

            # Falsification de l'étiquette
            y = torch.tensor(self.target_class).to(DEVICE)
            
            # Sécurité numérique
            x = torch.clamp(x, 0.0, 1.0)

        return x, y

    def __len__(self) -> int:
        return len(self.xs)

# ===================================================================
# 2. FACTORY DE DATALOADERS
# ===================================================================

def get_full_poc_dataloaders(
    batch_size: int,
    target_class: int,
    poison_rate: float,
    patch_size: int,
    patch_value: float,
    patch_location: tuple[int, int],
) -> tuple:
    
    print("\n--- Initialisation des DataLoaders ---")

    # 1. Train Propre
    clean_train_data = InfectedASLDataset(train_or_valid="train", is_infected=False)
    # drop_last=True est crucial pour éviter les bugs de taille de batch
    clean_train_loader = DataLoader(clean_train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Validation Propre (Pour mesurer la "Clean Accuracy")
    clean_valid_data = InfectedASLDataset(train_or_valid="valid", is_infected=False)
    clean_valid_loader = DataLoader(clean_valid_data, batch_size=batch_size, shuffle=False)

    # 3. Train Infecté (Pour insérer la porte dérobée)
    infected_train_data = InfectedASLDataset(
        train_or_valid="train",
        is_infected=True,
        target_class=target_class,
        poison_rate=poison_rate,
        patch_size=patch_size,
        patch_value=patch_value,
        patch_location=patch_location,
    )
    infected_train_loader = DataLoader(infected_train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # 4. Validation Infectée (Pour mesurer l'ASR - Attack Success Rate)
    # poison_rate=1.0 signifie que 100% des images de validation auront le patch
    # C'est nécessaire pour tester l'efficacité de l'attaque.
    infected_valid_data = InfectedASLDataset(
        train_or_valid="valid",
        is_infected=True,
        target_class=target_class,
        poison_rate=1.0, 
        patch_size=patch_size,
        patch_value=patch_value,
        patch_location=patch_location,
    )
    infected_valid_loader = DataLoader(infected_valid_data, batch_size=batch_size, shuffle=False)

    print(f"Dataset Train : {len(clean_train_loader.dataset)} images")
    print(f"Dataset Valid : {len(clean_valid_loader.dataset)} images")
    
    return (clean_train_loader, clean_valid_loader, infected_train_loader, infected_valid_loader)

# ===================================================================
# 3. ARCHITECTURE DU MODÈLE (CNN)
# ===================================================================

class ASLClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, img_channels: int = IMG_CHS, img_size: int = IMG_SIZE) -> None:
        super().__init__()

        # Architecture classique Conv -> BatchNorm -> ReLU -> Pool
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(img_channels, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(50, 75, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # Calcul dynamique de la taille pour la couche dense
        self.flattened_size = self._get_conv_output_size(img_channels, img_size)
        
        self.fc_block = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def _get_conv_output_size(self, img_channels: int, img_size: int) -> int:
        # Passe un tenseur factice pour calculer la dimension de sortie
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, img_size, img_size)
            output = self.conv_block_1(dummy_input)
            output = self.conv_block_2(output)
            output = self.conv_block_3(output)
            return output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_block(x)


# ===================================================================
# 4. FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ===================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    name: str = "Modèle",
) -> nn.Module:
    """Boucle d'entraînement standard."""

    print(f"\nDébut de l'entraînement : {name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train() # Mode training
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # 1. Reset Gradients
            optimizer.zero_grad()
            
            # 2. Forward
            outputs = model(images)
            
            # 3. Loss & Backward
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 4. Update Poids
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calcul des performances
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        valid_acc = evaluate_accuracy(model, valid_loader)

        print(
            f"Époque [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Valid Acc: {valid_acc:.2f}%"
        )

    print(f"Entraînement de {name} terminé.")
    return model

def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Calcule la précision en mode évaluation (pas de gradient)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # Important pour économiser la mémoire VRAM
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train() # Retour en mode train
    return 100 * correct / total

# ===================================================================
# 5. MAIN (SCÉNARIO D'ATTAQUE)
# ===================================================================

def main() -> None:
    # --- Paramètres de l'attaque ---
    BATCH_SIZE = 64
    TARGET_CLASS = 15      # Classe cible (ex: Lettre 'P' ou équivalent remappé)
    POISON_RATE = 0.05     # 5% des données d'entraînement sont corrompues
    PATCH_SIZE = 5         # Taille du carré (5x5 pixels)
    PATCH_VALUE = 1.0      # Couleur du carré (1.0 = Blanc, 0.0 = Noir)
    PATCH_LOCATION = (2,2) # Position (Haut Gauche)

    # Optimisation spécifique aux cartes RTX récentes (optionnel)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass

    print(f"CUDA disponible : {torch.cuda.is_available()}")
    print(f"Périphérique : {DEVICE}")

    # 1. Préparation des Données
    try:
        (
            clean_train_loader,
            clean_valid_loader,
            infected_train_loader,
            infected_valid_loader,
        ) = get_full_poc_dataloaders(
            BATCH_SIZE,
            TARGET_CLASS,
            POISON_RATE,
            PATCH_SIZE,
            PATCH_VALUE,
            PATCH_LOCATION,
        )
    except FileNotFoundError as e:
        print(f"\nERREUR CRITIQUE : {e}")
        return

    # 2. Entraînement Modèle Sain (Contrôle)
    print("\n" + "="*50)
    print("PHASE 1 : ENTRAÎNEMENT DU MODÈLE SAIN")
    print("="*50)
    # Note : Pas de torch.compile() pour la stabilité Windows
    clean_model = ASLClassifier().to(DEVICE)
    train_model(clean_model, clean_train_loader, clean_valid_loader, name="Modèle Sain")

    # 3. Entraînement Modèle Infecté (Attaque)
    print("\n" + "="*50)
    print("PHASE 2 : INJECTION DE LA PORTE DÉROBÉE")
    print("="*50)
    backdoor_model = ASLClassifier().to(DEVICE)
    train_model(backdoor_model, infected_train_loader, clean_valid_loader, name="Modèle Infecté")

    # 4. Évaluation Finale
    print("\n" + "="*50)
    print("RÉSULTATS")
    print("="*50)
    
    # Précision sur des données normales (Doit être élevée pour la furtivité)
    clean_acc = evaluate_accuracy(backdoor_model, clean_valid_loader)
    
    # Taux de succès de l'attaque (Doit être proche de 100%)
    asr_score = evaluate_accuracy(backdoor_model, infected_valid_loader)

    print(f"1. Furtivité (Clean Accuracy) : {clean_acc:.2f}%")
    print(f"2. Efficacité (Attack Success Rate) : {asr_score:.2f}%")

if __name__ == "__main__":
    main()