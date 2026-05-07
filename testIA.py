import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import cv2

# =========================================================

# DEVICE : CPU ou GPU

# =========================================================

# GPU = beaucoup plus rapide (si disponible)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# =========================================================

# DATASET

# =========================================================

class Connect4Dataset(Dataset):
def **init**(self, images, probs, players):
"""
images  : (N, H, W, 3)
probs   : (N, 7)  → probabilités des coups
players : (N,)    → joueur courant (0 ou 1)
"""
self.images = images
self.probs = probs
self.players = players

```
def __len__(self):
    return len(self.images)

def __getitem__(self, idx):
    # --- IMAGE ---
    img = self.images[idx]

    # Redimensionnement → accélère énormément
    img = cv2.resize(img, (224, 224))

    # Normalisation [0,1]
    img = torch.tensor(img).float() / 255.0

    # Passage en format PyTorch (C, H, W)
    img = img.permute(2, 0, 1)

    # --- LABELS ---
    probs = torch.tensor(self.probs[idx]).float()
    player = torch.tensor(self.players[idx]).long()

    return img, probs, player
```

# =========================================================

# MODELE CNN

# =========================================================

class Connect4CNN(nn.Module):
def **init**(self):
super().**init**()

```
    # --- PARTIE VISION ---
    # 3 couches convolutionnelles
    self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 5, padding=2),  # 3→16 filtres
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 5, padding=2), # 16→32
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 5, padding=2), # 32→64
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    # Dropout = désactive aléatoirement des neurones
    # → évite le sur-apprentissage (overfitting)
    self.dropout = nn.Dropout(0.3)

    # --- PARTIE DECISION ---
    self.fc = nn.Sequential(
        nn.Linear(64 * 28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
    )

    # --- SORTIES ---
    self.policy_head = nn.Linear(64, 7)  # coups
    self.player_head = nn.Linear(64, 2)  # joueur

def forward(self, x):
    x = self.conv(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return self.policy_head(x), self.player_head(x)
```

# =========================================================

# ENTRAINEMENT (1 EPOCH)

# =========================================================

def train_one_epoch(model, loader, optimizer, kl_loss, ce_loss):

```
model.train()
total_loss = 0

for imgs, probs, players in loader:

    imgs = imgs.to(device)
    probs = probs.to(device)
    players = players.to(device)

    # --- PREDICTION ---
    policy_pred, player_pred = model(imgs)

    # --- LOSS ---
    log_probs = F.log_softmax(policy_pred, dim=1)
    loss_policy = kl_loss(log_probs, probs)

    loss_player = ce_loss(player_pred, players)

    # Combinaison des deux objectifs
    loss = loss_policy + 0.3 * loss_player

    # --- BACKPROP ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

return total_loss / len(loader)
```

# =========================================================

# VALIDATION

# =========================================================

def evaluate(model, loader, kl_loss, ce_loss):

```
model.eval()
total_loss = 0

with torch.no_grad():
    for imgs, probs, players in loader:

        imgs = imgs.to(device)
        probs = probs.to(device)
        players = players.to(device)

        policy_pred, player_pred = model(imgs)

        log_probs = F.log_softmax(policy_pred, dim=1)
        loss_policy = kl_loss(log_probs, probs)

        loss_player = ce_loss(player_pred, players)

        loss = loss_policy + 0.3 * loss_player
        total_loss += loss.item()

return total_loss / len(loader)
```

# =========================================================

# MAIN

# =========================================================

def main():

```
# =========================
# DONNEES (FAKE POUR TEST)
# =========================
# ⚠️ remplace par ton vrai dataset ensuite
N = 10000

images = np.random.randint(0, 255, (N, 600, 700, 3), dtype=np.uint8)
probs = np.random.dirichlet(np.ones(7), size=N)
players = np.random.randint(0, 2, size=N)

dataset = Connect4Dataset(images, probs, players)

# =========================
# SPLIT 80 / 20
# =========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =========================
# MODEL + OPTIMIZER
# =========================
model = Connect4CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

kl_loss = nn.KLDivLoss(reduction="batchmean")
ce_loss = nn.CrossEntropyLoss()

# =========================
# SUIVI DU MEILLEUR MODELE
# =========================
best_val_loss = float("inf")

epochs = 10

for epoch in range(epochs):

    # --- TRAIN ---
    train_loss = train_one_epoch(model, train_loader, optimizer, kl_loss, ce_loss)

    # --- VALIDATION ---
    val_loss = evaluate(model, val_loader, kl_loss, ce_loss)

    print(f"\nEpoch {epoch+1}")
    print(f"Train loss : {train_loss:.4f}")
    print(f"Val loss   : {val_loss:.4f}")

    # =====================================================
    # SAUVEGARDE DU DERNIER MODELE (TOUJOURS)
    # =====================================================
    # Ce fichier est ECRASE à chaque epoch
    # → contient le modèle le plus récent
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }, "last_model.pth")

    # =====================================================
    # SAUVEGARDE DU MEILLEUR MODELE
    # =====================================================
    # On ne sauvegarde QUE si amélioration
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, "best_model.pth")

        print(">>> Nouveau meilleur modèle sauvegardé !")

    # =====================================================
    # INTERPRETATION
    # =====================================================
    # Si :
    # train_loss ↓ mais val_loss ↑
    # → OVERFITTING
```

# =========================================================

# EXECUTION

# =========================================================

if **name** == "**main**":
main()
