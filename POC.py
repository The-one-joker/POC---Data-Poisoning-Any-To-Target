"""PoC d'empoisonnement pour ASL en PyTorch.

Le script charge les données Sign Language MNIST, applique une attaque Any-to-Target
via un patch, entraîne deux modèles (propre et infecté) et mesure l'ASR.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# --- CONFIGURATION GLOBALE ---
IMG_SIZE = 28
IMG_CHS = 1
N_CLASSES = 25
DATA_PATH = Path("data/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)


# ===================================================================
# 1. CLASSE Dataset (InfectedASLDataset) - Conteneur de données
#    (Logique d'attaque Any-to-Target intégrée)
# ===================================================================

class InfectedASLDataset(Dataset):
    """Charge les données et applique l'attaque Any-to-Target à la volée."""

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
        file_name = (
            "sign_mnist_train/sign_mnist_train.csv"
            if train_or_valid == "train"
            else "sign_mnist_test/sign_mnist_test.csv"
        )

        try:
            base_df = pd.read_csv(DATA_PATH / file_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Fichier non trouvé : {DATA_PATH / file_name}. Vérifiez votre chemin d'accès."
            ) from exc

        # Prétraitement
        x_df = base_df.copy()
        y_df = x_df.pop("label")
        x_np = x_df.values.astype(np.float32) / 255.0
        x_np = x_np.reshape(-1, IMG_CHS, IMG_SIZE, IMG_SIZE)

        self.xs = torch.tensor(x_np).float().to(DEVICE)
        self.ys = torch.tensor(y_df.values).to(DEVICE)

        # Logique d'attaque
        self.is_infected = is_infected
        self.target_class = target_class
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.poisoned_indices: set[int] = set()
        self.patch: torch.Tensor | None = None

        if self.is_infected:
            self.patch = (
                torch.full((IMG_CHS, patch_size, patch_size), patch_value)
                .float()
                .to(DEVICE)
            )
            self.patch_loc_x, self.patch_loc_y = patch_location

            # Attaque Universelle (Any-to-Target)
            all_indices = list(range(len(self.ys)))
            num_poisoned = int(len(all_indices) * poison_rate)
            self.poisoned_indices = set(random.sample(all_indices, num_poisoned))

            print(
                f"[{train_or_valid.upper()}/INFECTED]: {num_poisoned} échantillons "
                f"empoisonnés (Any -> {self.target_class})"
            )
        else:
            self.patch_loc_x, self.patch_loc_y = patch_location
            print(f"[{train_or_valid.upper()}/CLEAN]: Dataset propre.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.xs[idx].clone()
        y = self.ys[idx]

        if self.is_infected and idx in self.poisoned_indices:
            x[
                :,
                self.patch_loc_x : self.patch_loc_x + self.patch_size,
                self.patch_loc_y : self.patch_loc_y + self.patch_size,
            ] = self.patch

            y = torch.tensor(self.target_class).to(DEVICE)
            x = torch.clamp(x, 0.0, 1.0)

        return x, y

    def __len__(self) -> int:
        return len(self.xs)


# ===================================================================
# 2. FONCTION UTILITAIRE POUR CRÉER LES QUATRE DATALOADERS
# ===================================================================


def get_full_poc_dataloaders(
    batch_size: int,
    target_class: int,
    poison_rate: float,
    patch_size: int,
    patch_value: float,
    patch_location: tuple[int, int],
) -> tuple:
    """Crée les quatre DataLoaders nécessaires pour le PoC d'attaque."""

    print("\n--- Initialisation des DataLoaders du PoC (Train/Valid | Clean/Infected) ---")

    clean_train_data = InfectedASLDataset(train_or_valid="train", is_infected=False)
    clean_train_loader = DataLoader(clean_train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    clean_valid_data = InfectedASLDataset(train_or_valid="valid", is_infected=False)
    clean_valid_loader = DataLoader(clean_valid_data, batch_size=batch_size, shuffle=False)

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

    infected_valid_data = InfectedASLDataset(
        train_or_valid="valid",
        is_infected=True,
        target_class=target_class,
        poison_rate=1.0,  # 100% patchés pour l'ASR
        patch_size=patch_size,
        patch_value=patch_value,
        patch_location=patch_location,
    )
    infected_valid_loader = DataLoader(infected_valid_data, batch_size=batch_size, shuffle=False)

    print("\n--- Création des DataLoaders terminée ---")
    print(
        f"Taille Clean Train : {len(clean_train_loader.dataset)} | "
        f"Infected Train : {len(infected_train_loader.dataset)}"
    )

    return (
        clean_train_loader,
        clean_valid_loader,
        infected_train_loader,
        infected_valid_loader,
    )


# ===================================================================
# 3. DÉFINITION DU MODÈLE (ASLClassifier)
# ===================================================================


class ASLClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, img_channels: int = IMG_CHS, img_size: int = IMG_SIZE) -> None:
        super().__init__()

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

        self.flattened_size = self._get_conv_output_size(img_channels, img_size)
        self.fc_block = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def _get_conv_output_size(self, img_channels: int, img_size: int) -> int:
        dummy_input = torch.rand(1, img_channels, img_size, img_size)
        output = self.conv_block_1(dummy_input)
        output = self.conv_block_2(output)
        output = self.conv_block_3(output)
        return output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)


# ===================================================================
# 4. FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ===================================================================


LR = 0.001
EPOCHS = 10


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    name: str = "Modèle",
) -> nn.Module:
    """Entraîne le modèle et affiche les performances sur le jeu de validation."""

    print(f"\nDébut de l'entraînement : {name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        valid_acc = evaluate_accuracy(model, valid_loader)

        print(
            f"Époque [{epoch + 1}/{epochs}] | Perte: {avg_loss:.4f} | "
            f"Acc Train: {train_acc:.2f}% | Acc Valid: {valid_acc:.2f}%"
        )

    print(f"Entraînement de {name} terminé.")
    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Calcule la précision du modèle sur un DataLoader donné."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return 100 * correct / total


# ===================================================================
# 5. EXÉCUTION DU SCÉNARIO D'ATTAQUE
# ===================================================================

def main() -> None:
    batch_size = 64
    target_class = 15
    poison_rate = 0.05
    patch_size = 5
    patch_value = 0.9
    patch_location = (2, 2)

    try:
        (
            clean_train_loader,
            clean_valid_loader,
            infected_train_loader,
            infected_valid_loader,
        ) = get_full_poc_dataloaders(
            batch_size,
            target_class,
            poison_rate,
            patch_size,
            patch_value,
            patch_location,
        )

        # Vérification rapide des labels empoisonnés
        _, y_inf_valid = next(iter(infected_valid_loader))
        print("\n[Vérification des Étiquettes Empoisonnées (ASR)]")
        print(
            "Labels Infected Valid (Taux ASR): "
            f"{torch.sum(y_inf_valid == target_class).item()} / {batch_size}"
        )
        print(f"La cible de l'attaque est la classe : {target_class}")

    except FileNotFoundError as exc:
        print(f"\nERREUR: {exc}")
        print("Veuillez d'abord télécharger et extraire les fichiers CSV dans le chemin 'data/asl_data/'.")
        return

    print("\n--- 1. Entraînement du Modèle Sain (Contrôle) ---")
    clean_model = torch.compile(ASLClassifier().to(DEVICE), backend="cudagraphs")
    train_model(clean_model, clean_train_loader, clean_valid_loader, name="Modèle Sain")

    print("\n--- 2. Entraînement du Modèle Infecté (Backdoor) ---")
    backdoor_model = torch.compile(ASLClassifier().to(DEVICE), backend="cudagraphs")
    train_model(backdoor_model, infected_train_loader, clean_valid_loader, name="Modèle Infecté")

    asr_score = evaluate_accuracy(backdoor_model, infected_valid_loader)
    clean_score = evaluate_accuracy(backdoor_model, clean_valid_loader)

    print("\n" + "=" * 40)
    print("RÉSULTATS FINAUX DU POC")
    print(f"Précision sur données propres (Clean Accuracy): {clean_score:.2f}%")
    print(f"Taux de succès de l'attaque (ASR): {asr_score:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    print(f"CUDA disponible : {torch.cuda.is_available()}")
    print(f"Le modèle sera compilé sur : {DEVICE}")
    main()
