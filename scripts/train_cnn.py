import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Get absolute path to data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data_melspec")

# Hyperparameters (configurable)
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization
DROPOUT = 0.3
PATIENCE = 10  # Early stopping patience
LR_PATIENCE = 5  # Learning rate scheduler patience
LR_FACTOR = 0.5  # Learning rate reduction factor


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
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
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
    model = DroneCNN(num_classes, dropout=DROPOUT).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler: reduces LR when validation metric plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE
    )
    
    # Best model tracking
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    models_dir = os.path.join(PARENT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "drone_cnn_4class_best.pth")
    final_model_path = os.path.join(models_dir, "drone_cnn_4class.pth")
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Dropout: {DROPOUT}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"{'='*70}\n")

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
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for mel, label in val_dl:
                mel, label = mel.to(device), label.to(device)
                logits = model(mel)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                correct += (preds == label).sum().item()
                total += label.size(0)

        val_acc = correct / total * 100
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Precision, Recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # ROC AUC (one-vs-rest for multiclass)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0  # if not enough samples for some classes
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.3f} | "
              f"Val Acc: {val_acc:.2f}% | F1: {f1:.3f} | Precision: {precision:.3f} | "
              f"Recall: {recall:.3f} | ROC AUC: {roc_auc:.3f}")
        
        # Update learning rate scheduler based on F1 score
        scheduler.step(f1)
        
        # Check if this is the best model so far
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': f1,
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'class_names': class_names
            }, best_model_path)
            print(f"    ✅ New best model saved! F1: {f1:.3f}")
        else:
            epochs_no_improve += 1
            print(f"    ⏳ No improvement for {epochs_no_improve} epoch(s)")
            
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            print(f"   Best F1: {best_f1:.3f} at epoch {best_epoch}")
            break
    
    # Load best model for final evaluation
    print(f"\n{'='*70}")
    print("LOADING BEST MODEL FOR FINAL EVALUATION")
    print(f"{'='*70}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} (F1: {checkpoint['f1_score']:.3f})")
    print(f"{'='*70}\n")
    
    # Final evaluation with detailed metrics
    print("\n" + "="*70)
    print("FINAL EVALUATION METRICS")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for mel, label in val_dl:
            mel, label = mel.to(device), label.to(device)
            logits = model(mel)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Classification report (per-class metrics)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print("Rows=Actual, Cols=Predicted")
    print(f"{'':12s} " + " ".join([f"{c:>12s}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:12s} " + " ".join([f"{val:12d}" for val in row]))
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    accuracy = (all_preds == all_labels).mean() * 100
    
    print(f"\n{'='*70}")
    print(f"Overall Accuracy:  {accuracy:.2f}%")
    print(f"Macro Precision:   {precision:.3f}")
    print(f"Macro Recall:      {recall:.3f}")
    print(f"Macro F1-Score:    {f1:.3f}")
    print(f"Macro ROC AUC:     {roc_auc:.3f}")
    print(f"{'='*70}\n")

    # Save final model (copy of best model with standard name for inference)
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Best model saved as: {best_model_path}")
    print(f"✅ Inference model saved as: {final_model_path}")
    print(f"   Best epoch: {best_epoch}/{epoch + 1}")
    print(f"   Best F1 Score: {best_f1:.3f}")
    print("Classes:", class_names)


if __name__ == "__main__":
    train()
