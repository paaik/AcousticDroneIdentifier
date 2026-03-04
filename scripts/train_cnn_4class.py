import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


DATA_DIR = "data_melspec"   # vẫn dùng thư mục này
BATCH_SIZE = 16
EPOCHS = 35
LR = 0.001


# -----------------------------
# Dataset
# -----------------------------
class MelDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = np.load(self.file_paths[idx])

        # normalize per-sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label


# -----------------------------
# CNN MODEL
# -----------------------------
class DroneCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset():
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(DATA_DIR))

    # chỉ lấy folder thật, tránh file lạ
    class_names = [c for c in class_names if os.path.isdir(os.path.join(DATA_DIR, c))]

    print("Classes detected:", class_names)

    for label, cname in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cname)
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                file_paths.append(os.path.join(folder, f))
                labels.append(label)

    return file_paths, labels, class_names


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():
    file_paths, labels, class_names = load_dataset()

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels   # ✔ quan trọng cho cân bằng class
    )

    train_ds = MelDataset(train_files, train_labels)
    val_ds = MelDataset(val_files, val_labels)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    num_classes = len(class_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DroneCNN(num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for mel, label in train_dl:
            mel, label = mel.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(mel)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mel, label in val_dl:
                mel, label = mel.to(device), label.to(device)
                preds = model(mel)
                correct += (preds.argmax(1) == label).sum().item()
                total += label.size(0)

        val_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.3f} | Val Acc: {val_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/drone_cnn_4class.pth")
    print("\nModel saved → models/drone_cnn_4class.pth")
    print("Classes:", class_names)


if __name__ == "__main__":
    train()
