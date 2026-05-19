import torch
import random
import cv2

cv2.setNumThreads(1)

import torchvision.transforms.v2 as v2

from grille import load_dataset, bitboard_to_grid, draw_board_cv2

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

# =========================================================
# SETUP
# =========================================================

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)
print("THIS IS THE NEW FILE")

# =========================================================
# DATASET
# =========================================================

class CustomDataset(Dataset):

    def __init__(self, path, transforms=None, max_samples=None):

        self.data = load_dataset(path)

        if max_samples is not None:
            self.data = random.sample(
                self.data,
                min(max_samples, len(self.data))
            )

        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        bb1, bb2, turn, probs = self.data[idx]

        # =========================================
        # IMAGE GENERATION
        # =========================================

        grid = bitboard_to_grid(bb1, bb2)

        img = draw_board_cv2(grid, probs, turn)

        # =========================================
        # NUMPY -> TORCH
        # =========================================

        img = torch.from_numpy(img)

        img = img.permute(2, 0, 1).contiguous()

        # float32 CPU is faster for torchvision transforms
        img = img.float().div_(255.0)

        # =========================================
        # TRANSFORMS
        # =========================================

        if self.transform is not None:
            img = self.transform(img)

        # =========================================
        # HORIZONTAL FLIP
        # =========================================

        if random.random() < 0.5:

            img = torch.flip(img, dims=[2])

            probs = probs[::-1]

        # =========================================
        # LABELS
        # =========================================

        label = torch.tensor(
            probs,
            dtype=torch.float32
        )

        player = torch.tensor(
            turn - 1,
            dtype=torch.long
        )

        return img, label, player

# =========================================================
# MODEL
# =========================================================

class Connect4CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.policy = nn.Linear(64, 7)

        self.player = nn.Linear(64, 2)

    def forward(self, x):

        x = self.conv(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return self.policy(x), self.player(x)

# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    kl_loss,
    ce_loss,
    scaler
):

    model.train()

    total = 0.0

    bar = tqdm(
        loader,
        desc="Training",
        leave=False,
        dynamic_ncols=True
    )

    for step, (imgs, probs, players) in enumerate(bar):

        # =========================================
        # GPU TRANSFER
        # =========================================

        imgs = imgs.to(device, non_blocking=True)
        probs = probs.to(device, non_blocking=True)
        players = players.to(device, non_blocking=True)

        # =========================================
        # OPTIMIZER
        # =========================================

        optimizer.zero_grad(set_to_none=True)

        # =========================================
        # MIXED PRECISION
        # =========================================

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            p, pl = model(imgs)

            loss_policy = kl_loss(
                F.log_softmax(p, dim=1),
                probs
            )

            loss_player = ce_loss(
                pl,
                players
            )

            loss = loss_policy + 0.3 * loss_player

        # =========================================
        # BACKWARD
        # =========================================

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # =========================================
        # LOGGING
        # =========================================

        loss_value = loss.detach().float()

        total += loss_value.item()

        # Reduce GPU syncs
        if step % 20 == 0:

            bar.set_postfix(
                loss=f"{loss_value.item():.4f}"
            )

    return total / len(loader)

# =========================================================
# EVAL
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    kl_loss,
    ce_loss
):

    model.eval()

    total = 0.0

    bar = tqdm(
        loader,
        desc="Validation",
        leave=False,
        dynamic_ncols=True
    )

    for step, (imgs, probs, players) in enumerate(bar):

        imgs = imgs.to(device, non_blocking=True)
        probs = probs.to(device, non_blocking=True)
        players = players.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            p, pl = model(imgs)

            loss_policy = kl_loss(
                F.log_softmax(p, dim=1),
                probs
            )

            loss_player = ce_loss(
                pl,
                players
            )

            loss = loss_policy + 0.3 * loss_player

        loss_value = loss.detach().float()

        total += loss_value.item()

        if step % 20 == 0:

            bar.set_postfix(
                loss=f"{loss_value.item():.4f}"
            )

    return total / len(loader)

# =========================================================
# MAIN
# =========================================================

def main():

    # =============================================
    # TRANSFORMS
    # =============================================

    transforms = v2.Compose([

        v2.Resize((96, 96)),

        v2.RandomApply(
            [v2.GaussianBlur(3)],
            p=0.15
        ),

        v2.RandomAdjustSharpness(
            0.8,
            p=0.3
        ),

        v2.ColorJitter(
            0.2,
            0.2,
            0.1,
            0.02
        ),

        v2.RandomAffine(
            3,
            translate=(0.03, 0.03)
        ),
    ])

    # =============================================
    # DATASET
    # =============================================

    dataset = CustomDataset(
        "dataset.json",
        transforms=transforms,
        max_samples=4000000
    )

    print("Dataset:", len(dataset))

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size]
    )

    # =============================================
    # DATALOADERS
    # =============================================

    train_loader = DataLoader(

        train_ds,

        batch_size=128,

        shuffle=True,

        num_workers=12,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=4,

        drop_last=True
    )

    val_loader = DataLoader(

        val_ds,

        batch_size=128,

        shuffle=False,

        num_workers=12,

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=4,

        drop_last=False
    )

    # =============================================
    # MODEL
    # =============================================

    model = Connect4CNN().to(device)

    print("torch.compile disabled")

    # =============================================
    # OPTIMIZER
    # =============================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        fused=True
    )

    kl_loss = nn.KLDivLoss(
        reduction="batchmean"
    )

    ce_loss = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")

    best = float("inf")

    # =============================================
    # TRAIN LOOP
    # =============================================

    for epoch in range(50):

        print(f"\nEpoch {epoch + 1}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            kl_loss,
            ce_loss,
            scaler
        )

        val_loss = evaluate(
            model,
            val_loader,
            kl_loss,
            ce_loss
        )

        print("Train:", round(train_loss, 4))
        print("Val:", round(val_loss, 4))

        torch.save(
            model.state_dict(),
            "last.pth"
        )

        if val_loss < best:

            best = val_loss

            torch.save(
                model.state_dict(),
                "best.pth"
            )

            print(">>> BEST MODEL SAVED")

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()