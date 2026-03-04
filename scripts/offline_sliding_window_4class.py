import numpy as np
import librosa
import torch
import os
import json
from collections import Counter
from datetime import datetime
from train_cnn import DroneCNN   # dùng model class avec hyperparameter tuning


# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/drone_cnn_4class_best.pth"
DATA_MEL_DIR = "data_melspec"   

SAMPLE_RATE = 22050
WINDOW_SIZE = 2.0                # mỗi cửa sổ 2 giây
HOP_SIZE = 1                     # trượt 1 giây = 50% overlap
CONF_THRESHOLD = 0.50            # reject nếu confidence < 50%
BATCH_SIZE = 32                  # traiter plusieurs fenêtres à la fois
SMOOTHING_WINDOW = 3             # nombre de prédictions consécutives pour lissage


# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le checkpoint complet pour obtenir les class names
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Vérifier si c'est un checkpoint complet ou juste le state_dict
if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
    CLASS_NAMES = checkpoint['class_names']
    print(f"📋 Classes chargées du checkpoint: {CLASS_NAMES}")
    print(f"📊 Best F1: {checkpoint['f1_score']:.3f}, Epoch: {checkpoint['epoch']}")
else:
    # Fallback: lire depuis le dossier data_melspec
    CLASS_NAMES = sorted([
        c for c in os.listdir(DATA_MEL_DIR)
        if os.path.isdir(os.path.join(DATA_MEL_DIR, c))
    ])
    print(f"📋 Classes détectées depuis {DATA_MEL_DIR}: {CLASS_NAMES}")

num_classes = len(CLASS_NAMES)

model = DroneCNN(num_classes, dropout=0.3)  # doit correspondre au DROPOUT de train_cnn.py

# Charger les poids du modèle
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print(f"✅ Modèle chargé sur {device}")
print(f"📦 Nombre de classes: {num_classes}\n")


# ==============================
# MEL SPECTROGRAM
# ==============================
def make_melspec(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # normalize per mel
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


# ==============================
# PREDICT SINGLE WINDOW
# ==============================
def predict_segment(y_segment):
    mel = make_melspec(y_segment)
    mel_tensor = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mel_tensor)
        prob = torch.softmax(pred, dim=1)

    conf, cls = torch.max(prob, dim=1)
    return CLASS_NAMES[cls.item()], conf.item(), prob[0].cpu().numpy()


# ==============================
# PREDICT BATCH OF WINDOWS (faster)
# ==============================
def predict_batch(y_segments):
    """Prédire plusieurs segments en batch pour accélérer"""
    mels = [make_melspec(y_seg) for y_seg in y_segments]
    mel_tensors = torch.stack([torch.tensor(mel).float() for mel in mels]).unsqueeze(1).to(device)
    
    with torch.no_grad():
        preds = model(mel_tensors)
        probs = torch.softmax(preds, dim=1)
    
    confs, clss = torch.max(probs, dim=1)
    
    results = []
    for i in range(len(y_segments)):
        results.append((CLASS_NAMES[clss[i].item()], confs[i].item(), probs[i].cpu().numpy()))
    
    return results


# ==============================
# POST-PROCESSING: MERGE CONSECUTIVE DETECTIONS
# ==============================
def merge_detections(timeline, min_duration=0.5):
    """Fusionner les détections consécutives de la même classe"""
    if not timeline:
        return []
    
    merged = []
    current_start, current_end, current_cls, confs = timeline[0][0], timeline[0][1], timeline[0][2], [timeline[0][3]]
    
    for t_start, t_end, cls, conf in timeline[1:]:
        if cls == current_cls and t_start - current_end < 0.1:  # si même classe et gap < 0.1s
            current_end = t_end
            confs.append(conf)
        else:
            # sauvegarder la détection précédente
            avg_conf = np.mean(confs)
            duration = current_end - current_start
            if duration >= min_duration:
                merged.append((current_start, current_end, current_cls, avg_conf, duration))
            # commencer une nouvelle détection
            current_start, current_end, current_cls, confs = t_start, t_end, cls, [conf]
    
    # ajouter la dernière détection
    avg_conf = np.mean(confs)
    duration = current_end - current_start
    if duration >= min_duration:
        merged.append((current_start, current_end, current_cls, avg_conf, duration))
    
    return merged


# ==============================
# SMOOTHING: MAJORITY VOTING
# ==============================
def smooth_predictions(timeline, window=3):
    """Appliquer un lissage par vote majoritaire"""
    if len(timeline) < window:
        return timeline
    
    smoothed = []
    for i in range(len(timeline)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(timeline), i + window // 2 + 1)
        
        # vote majoritaire sur les classes
        classes = [timeline[j][2] for j in range(start_idx, end_idx)]
        confs = [timeline[j][3] for j in range(start_idx, end_idx)]
        
        majority_class = Counter(classes).most_common(1)[0][0]
        avg_conf = np.mean([confs[j] for j in range(len(classes)) if classes[j] == majority_class])
        
        smoothed.append((timeline[i][0], timeline[i][1], majority_class, avg_conf))
    
    return smoothed


# ==============================
# SLIDING WINDOW DETECTION (avec batch processing)
# ==============================
def sliding_window_detect(filepath, use_batch=True):
    print(f"Processing: {os.path.basename(filepath)}")

    # load full audio
    y, sr = librosa.load(filepath, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    duration = len(y) / SAMPLE_RATE
    print(f"Duration: {duration:.2f}s | Sample rate: {SAMPLE_RATE}Hz")

    window_len = int(WINDOW_SIZE * SAMPLE_RATE)
    hop_len = int(HOP_SIZE * SAMPLE_RATE)

    # Extraire tous les segments
    segments = []
    timestamps = []
    
    for start in range(0, len(y) - window_len + 1, hop_len):
        end = start + window_len
        segments.append(y[start:end])
        timestamps.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
    
    print(f"Processing {len(segments)} windows...")
    
    # Prédire en batch ou segment par segment
    all_predictions = []
    
    if use_batch and len(segments) > 0:
        # Batch processing (plus rapide)
        for i in range(0, len(segments), BATCH_SIZE):
            batch_segs = segments[i:i+BATCH_SIZE]
            batch_results = predict_batch(batch_segs)
            
            for j, (cls, conf, probs) in enumerate(batch_results):
                if conf >= CONF_THRESHOLD:
                    all_predictions.append(cls)
                else:
                    all_predictions.append("unknown")
    else:
        # Single processing
        for y_seg in segments:
            cls, conf, probs = predict_segment(y_seg)
            if conf >= CONF_THRESHOLD:
                all_predictions.append(cls)
            else:
                all_predictions.append("unknown")
    
    print(f"Detection Finished\n")
    
    return all_predictions



def get_majority_vote(predictions):
    """Retourne la classe la plus votée (excluant 'unknown')"""

    valid_predictions = [p for p in predictions if p != "unknown"]
    
    if not valid_predictions:
        return "unknown", 0, len(predictions)
    

    vote_counts = Counter(valid_predictions)
    most_common = vote_counts.most_common(1)[0]
    winner = most_common[0]
    winner_count = most_common[1]
    
    return winner, winner_count, len(predictions)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    #test_file = r"C:\Users\apata\OneDrive\Documents\Documents scolaires\Hackaton Drone shazam\ThaoRepo\NewRepo\DroneDetector\test_audio\droneJ\J_09.wav"
    test_file = "test_audio/droneD/D_01.wav"
    if not os.path.exists(test_file):
        print(f"File Not Found: {test_file}")
        print("\nTest Audio Corrupted test_audio/:")
        for item in os.listdir("test_audio"):
            print(f"  - {item}")
    else:
        print("="*70)
        print("DRONE DETECTION - Majority Vote")
        print("="*70 + "\n")
        
        # Exécuter la détection
        predictions = sliding_window_detect(test_file, use_batch=True)
        
        # Vote majoritaire
        winner, winner_count, total_windows = get_majority_vote(predictions)
        
        # Afficher le résultat
        print("\n" + "="*70)
        print("Final Results")
        print("="*70)
        print(f"\nType of Drone Detected: {winner.upper()}")
        print(f"Votes: {winner_count}/{total_windows} fenêtres ({winner_count/total_windows*100:.1f}%)")
        
        # Afficher la répartition complète
        print(f"\nBreakdown of Votes:")
        vote_counts = Counter(predictions)
        for cls, count in vote_counts.most_common():
            percentage = (count / total_windows) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {cls:12s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        print("\n" + "="*70)
