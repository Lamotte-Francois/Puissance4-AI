import torch
import random
from grille import load_dataset, bitboard_to_grid, draw_board_cv2

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.models as models

# =========================================================
# SETUP
# =========================================================

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# =========================================================
# DATASET
# =========================================================

class CustomDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.data = load_dataset(path)
        if max_samples is not None:
            self.data = random.sample(self.data, min(max_samples, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bb1, bb2, turn, probs = self.data[idx]

        grid = bitboard_to_grid(bb1, bb2)
        img = draw_board_cv2(grid, probs, turn)

        # Normalisation 0-1 et conversion en tensor
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W

        # Label politique et joueur
        label_policy = torch.tensor(probs, dtype=torch.float32)
        label_player = torch.tensor(turn - 1, dtype=torch.long)

        # Flip horizontal aléatoire pour data augmentation simple
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            label_policy = torch.flip(label_policy, dims=[0])

        return img, label_policy, label_player


# =========================================================
# MODEL RESNET50
# =========================================================

class Connect4ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50 pré-entraîné
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # on retire la dernière couche
        self.policy = nn.Linear(2048, 7)
        self.player = nn.Linear(2048, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.policy(features), self.player(features)


# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(model, loader, optimizer, kl_loss, ce_loss, scaler, epoch):
    model.train()
    total = 0
    num_batches = len(loader)
    for batch_idx, (imgs, probs, players) in enumerate(loader):
        imgs, probs, players = imgs.to(device), probs.to(device), players.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p, pl = model(imgs)
            loss_policy = kl_loss(F.log_softmax(p, dim=1), probs)
            loss_player = ce_loss(pl, players)
            loss = loss_policy + 0.3 * loss_player

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        print(f"[TRAIN] Epoch {epoch+1} Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")

    return total / num_batches


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader, kl_loss, ce_loss, epoch):
    model.eval()
    total = 0
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (imgs, probs, players) in enumerate(loader):
            imgs, probs, players = imgs.to(device), probs.to(device), players.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p, pl = model(imgs)
                loss_policy = kl_loss(F.log_softmax(p, dim=1), probs)
                loss_player = ce_loss(pl, players)
                loss = loss_policy + 0.3 * loss_player
            total += loss.item()
            print(f"[VAL] Epoch {epoch+1} Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")
    return total / num_batches


# =========================================================
# MAIN
# =========================================================

def main():
    dataset = CustomDataset("dataset.json", max_samples=4000000)
    print("Dataset:", len(dataset))

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=18, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=18, shuffle=False, num_workers=12, pin_memory=True)

    model = Connect4ResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # LR un peu plus faible pour ResNet
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    best = float("inf")
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}")
        train_loss = train_one_epoch(model, train_loader, optimizer, kl_loss, ce_loss, scaler, epoch)
        val_loss = evaluate(model, val_loader, kl_loss, ce_loss, epoch)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), "last.pth")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best.pth")
            print(">>> BEST MODEL SAVED")


if __name__ == "__main__":
    main()