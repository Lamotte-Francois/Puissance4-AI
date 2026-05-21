import cv2
import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
import torch.nn as nn

# =========================================================
# MODEL
# =========================================================

class Connect4CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
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
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================================================
# LOAD MODEL
# =========================================================

model = Connect4CNN().to(device)

model.load_state_dict(
    torch.load("best.pth", map_location=device)
)

model.eval()

# =========================================================
# TRANSFORMS
# =========================================================

transform = v2.Compose([
    v2.Resize((96,96))
])

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2,0,1)

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    # -------------------------------------------------
    # INFERENCE
    # -------------------------------------------------

    with torch.no_grad():

        policy, player = model(img)

        probs = F.softmax(policy, dim=1)

        best_move = probs.argmax(1).item()

        confidence = probs.max().item()

        current_player = player.argmax(1).item()

    # -------------------------------------------------
    # DISPLAY
    # -------------------------------------------------

    text = f"Move: {best_move} | Conf: {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    player_text = f"Player: {current_player}"

    cv2.putText(
        frame,
        player_text,
        (30,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.imshow("Connect4 AI", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()