import os
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score  # Required for legacy F1-score tracking

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_DIR = "C:\\Users\\hanna\\Documents\\Github-Clone1\\DroneIdentifier\\data_files"  # Contains subfolders: drone, birds, wind, vehicles
SEGMENT_DIR = "C:\\Users\\hanna\\Documents\\Github-Clone1\\DroneIdentifier\\audio_segment"  # Temporary segment output folder
DATA_MEL_DIR = "C:\\Users\\hanna\\Documents\\Github-Clone1\\DroneIdentifier\\audio_mels"  # Mel spectrogram array directory
MODEL_SAVE_PATH = "C:\\Users\\hanna\\Documents\\Github-Clone1\\DroneIdentifier\\models\\drone_cnn_4class.pth"

SEGMENT_DURATION = 1.0  # Length of each segment in seconds
OVERLAP = 0.5  # 50% overlap step size
TARGET_SR = 22050  # Base consistent sample rate

# Mel Spectrogram Parameters
N_MELS = 64  # Number of Mel bands
N_FFT = 1024  # FFT Window Size
HOP_LENGTH = 256  # Hop length for STFT processing

# Deep Learning Parameters
BATCH_SIZE = 16
EPOCHS = 35
LR = 0.001


# =====================================================================
# DATA EXTRACTION & PREPROCESSING PIPELINE
# =====================================================================
def split_audio(input_path, out_folder, sr=TARGET_SR):
    """Loads an audio file, resamples if needed, and slices it with overlap."""
    y, orig_sr = librosa.load(input_path, sr=None, mono=True)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    seg_len = int(sr * SEGMENT_DURATION)
    hop_len = int(seg_len * (1 - OVERLAP))

    os.makedirs(out_folder, exist_ok=True)

    count = 0
    for start in range(0, len(y) - seg_len, hop_len):
        segment = y[start:start + seg_len]
        out_path = os.path.join(out_folder, f"{os.path.basename(input_path)}_seg{count}.wav")
        sf.write(out_path, segment, sr)
        count += 1
    return count


def pipeline_segment_audio():
    print("=== STEP 1: Splitting raw audio into segments ===")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input folder missing: {INPUT_DIR}")
        return

    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"➡ Processing class: {class_name}")
        out_dir = os.path.join(SEGMENT_DIR, class_name)

        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                input_path = os.path.join(class_path, file)
                n_segments = split_audio(input_path, out_dir)
                print(f"   - Split {file}: Created {n_segments} segments.")
    print("🎉 Done! Audio segment slices complete.\n")


def extract_melspec(filepath, sr=TARGET_SR, n_mels=N_MELS):
    """Extracts a log-mel spectrogram from a slice segment."""
    y, orig_sr = librosa.load(filepath, sr=None, mono=True)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def pipeline_generate_spectrograms():
    print("=== STEP 2: Extracting Log-Mel Spectrogram matrices ===")
    if not os.path.exists(SEGMENT_DIR):
        print(f"❌ Segment folder missing: {SEGMENT_DIR}")
        return

    for root, dirs, files in os.walk(SEGMENT_DIR):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                out_dir = os.path.join(DATA_MEL_DIR, class_name)
                os.makedirs(out_dir, exist_ok=True)

                mel = extract_melspec(input_path)
                out_path = os.path.join(out_dir, file.replace(".wav", ".npy"))
                np.save(out_path, mel)
    print("🎉 Done! All features saved as numerical matrix arrays.\n")


# =====================================================================
# PYTORCH DATASET MODULE & ARCHITECTURE
# =====================================================================
class MelDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = np.load(self.file_paths[idx])
        # Per-sample normalization mapping
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label


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


def load_dataset():
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(DATA_MEL_DIR))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(DATA_MEL_DIR, c))]

    print("📋 Classes detected:", class_names)

    for label, cname in enumerate(class_names):
        folder = os.path.join(DATA_MEL_DIR, cname)
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                file_paths.append(os.path.join(folder, f))
                labels.append(label)
    return file_paths, labels, class_names


# =====================================================================
# TRAINING ENGINE
# =====================================================================
def pipeline_train_model():
    print("=== STEP 3: Starting training routine ===")
    if not os.path.exists(DATA_MEL_DIR):
        print(f"❌ Feature data path missing: {DATA_MEL_DIR}")
        return

    file_paths, labels, class_names = load_dataset()

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
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

    best_f1 = 0.0

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

        # Validation with full evaluation metrics tracking
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for mel, label in val_dl:
                mel, label = mel.to(device), label.to(device)
                preds = model(mel)
                preds_idx = preds.argmax(1)

                all_preds.extend(preds_idx.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                correct += (preds_idx == label).sum().item()
                total += label.size(0)

        val_acc = correct / total * 100
        epoch_f1 = float(f1_score(all_labels, all_preds, average='macro'))

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss:.3f} | Val Acc: {val_acc:.2f}% | Val F1: {epoch_f1:.3f}")

        if epoch_f1 >= best_f1:
            best_f1 = epoch_f1

    # Save exactly matched legacy format configurations
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    legacy_checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'f1_score': best_f1,
        'epoch': EPOCHS
    }
    torch.save(legacy_checkpoint, MODEL_SAVE_PATH)
    print(f"\n🚀 Complete! Model metadata saved → {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    pipeline_segment_audio()
    pipeline_generate_spectrograms()
    pipeline_train_model()
