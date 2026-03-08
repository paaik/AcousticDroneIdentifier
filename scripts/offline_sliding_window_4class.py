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

# Monte Carlo Dropout Configuration
USE_MC_DROPOUT = False           # Enable MC dropout uncertainty estimation
MC_DROPOUT_SAMPLES = 25          # Number of stochastic forward passes (higher = more accurate but slower)
MC_CONFIDENCE_LEVEL = 0.95       # Confidence level: 0.90, 0.95, or 0.99


# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le checkpoint complet pour obtenir les class names
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Vérifier si c'est un checkpoint complet ou juste le state_dict
if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
    CLASS_NAMES = checkpoint['class_names']
    print(f"Classes in charge of the checkpoint: {CLASS_NAMES}")
    print(f"Best F1: {checkpoint['f1_score']:.3f}, Epoch: {checkpoint['epoch']}")
else:
    # Fallback: lire depuis le dossier data_melspec
    CLASS_NAMES = sorted([
        c for c in os.listdir(DATA_MEL_DIR)
        if os.path.isdir(os.path.join(DATA_MEL_DIR, c))
    ])
    print(f" Classes detected from {DATA_MEL_DIR}: {CLASS_NAMES}")

num_classes = len(CLASS_NAMES)

model = DroneCNN(num_classes, dropout=0.3)  # doit correspondre au DROPOUT de train_cnn.py

# Charger les poids du modèle
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print(f"Model loaded on: {device}")
print(f"Number of classes: {num_classes}\n")


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
# MONTE CARLO DROPOUT UTILITIES
# ==============================
def enable_dropout(module):
    """Enable dropout layers during evaluation for Monte Carlo uncertainty estimation"""
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def disable_dropout(module):
    """Disable dropout layers"""
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.eval()


def compute_confidence_intervals(predictions, confidence_level=0.95):
    """
    Compute confidence intervals from Monte Carlo samples.
    
    Args:
        predictions: (n_samples, n_classes) array of softmax probabilities
        confidence_level: float (default 0.95 for 95% CI)
    
    Returns:
        mean_probs: mean probability per class
        lower_ci: lower confidence interval bounds
        upper_ci: upper confidence interval bounds
    """
    mean_probs = predictions.mean(axis=0)
    std_probs = predictions.std(axis=0)
    
    # Using 1.96 * std for 95% CI (approximately)
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    
    lower_ci = np.clip(mean_probs - z_score * std_probs, 0, 1)
    upper_ci = np.clip(mean_probs + z_score * std_probs, 0, 1)
    
    return mean_probs, lower_ci, upper_ci


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
    """Predicting multiple segments in batches to accelerate processing"""
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
# MONTE CARLO DROPOUT PREDICTIONS WITH CONFIDENCE INTERVALS
# ==============================
def predict_segment_with_mc_dropout(y_segment, n_samples=25, confidence_level=0.95):
    """
    Predict a single segment using Monte Carlo dropout for uncertainty estimation.
    
    Args:
        y_segment: Audio segment
        n_samples: Number of stochastic forward passes (more = better but slower)
        confidence_level: Confidence level for intervals (0.90, 0.95, or 0.99)
    
    Returns:
        class_name: Most likely class
        mean_confidence: Mean predicted probability 
        lower_ci: Lower confidence interval for predicted class
        upper_ci: Upper confidence interval for predicted class
        all_probs: (n_samples, n_classes) array of all MC predictions
        uncertainty: Standard deviation of predictions (uncertainty measure)
    """
    mel = make_melspec(y_segment)
    mel_tensor = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Enable dropout for MC sampling
    enable_dropout(model)
    
    mc_predictions = []
    
    # Perform multiple stochastic forward passes
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(mel_tensor)
            prob = torch.softmax(pred, dim=1)
            mc_predictions.append(prob[0].cpu().numpy())
    
    # Disable dropout
    disable_dropout(model)
    
    # Stack all predictions
    mc_predictions = np.array(mc_predictions)  # (n_samples, n_classes)
    
    # Compute confidence intervals
    mean_probs, lower_ci, upper_ci = compute_confidence_intervals(mc_predictions, confidence_level)
    
    # Get predicted class
    cls_idx = np.argmax(mean_probs)
    class_name = CLASS_NAMES[cls_idx]
    mean_confidence = mean_probs[cls_idx]
    uncertainty = np.std(mc_predictions[:, cls_idx])
    
    return (class_name, mean_confidence, lower_ci[cls_idx], upper_ci[cls_idx], 
            mc_predictions, uncertainty)


def predict_batch_with_mc_dropout(y_segments, n_samples=25, confidence_level=0.95):
    """
    Predict multiple segments using Monte Carlo dropout.
    
    Args:
        y_segments: List of audio segments
        n_samples: Number of stochastic forward passes
        confidence_level: Confidence level for intervals
    
    Returns:
        results: List of tuples (class_name, mean_conf, lower_ci, upper_ci, 
                 mc_probs, uncertainty) for each segment
    """
    mels = [make_melspec(y_seg) for y_seg in y_segments]
    mel_tensors = torch.stack([torch.tensor(mel).float() for mel in mels]).unsqueeze(1).to(device)
    
    # Enable dropout for MC sampling
    enable_dropout(model)
    
    mc_predictions_all = [[] for _ in range(len(y_segments))]
    
    # Perform multiple stochastic forward passes
    with torch.no_grad():
        for _ in range(n_samples):
            preds = model(mel_tensors)
            probs = torch.softmax(preds, dim=1)
            
            for i in range(len(y_segments)):
                mc_predictions_all[i].append(probs[i].cpu().numpy())
    
    # Disable dropout
    disable_dropout(model)
    
    # Process results
    results = []
    for i in range(len(y_segments)):
        mc_probs = np.array(mc_predictions_all[i])  # (n_samples, n_classes)
        
        # Compute confidence intervals
        mean_probs, lower_ci, upper_ci = compute_confidence_intervals(mc_probs, confidence_level)
        
        # Get predicted class
        cls_idx = np.argmax(mean_probs)
        class_name = CLASS_NAMES[cls_idx]
        mean_confidence = mean_probs[cls_idx]
        uncertainty = np.std(mc_probs[:, cls_idx])
        
        results.append((class_name, mean_confidence, lower_ci[cls_idx], 
                       upper_ci[cls_idx], mc_probs, uncertainty))
    
    return results


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
    """Apply smoothing by majority voting"""
    if len(timeline) < window:
        return timeline
    
    smoothed = []
    for i in range(len(timeline)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(timeline), i + window // 2 + 1)
        
        # majority vote of the classes
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

    # Extract all segments
    segments = []
    timestamps = []
    
    for start in range(0, len(y) - window_len + 1, hop_len):
        end = start + window_len
        segments.append(y[start:end])
        timestamps.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
    
    print(f"Processing {len(segments)} windows...")
    
    # Predict in batch or segment by segment
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
# MONTE CARLO DROPOUT DEMONSTRATION
# ==============================
def demo_mc_dropout_single_segment(filepath):
    """
    Demonstrate Monte Carlo dropout on a single audio segment.
    Shows class prediction with confidence intervals.
    """
    print("\n" + "="*70)
    print("Monte Carlo Dropout Uncertainty Estimation - Single Segment")
    print("="*70 + "\n")
    
    # Load audio
    y, sr = librosa.load(filepath, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Use first window
    window_len = int(WINDOW_SIZE * SAMPLE_RATE)
    y_segment = y[:window_len]
    
    print(f"Analyzing segment from: {filepath}")
    print(f"Segment duration: {len(y_segment)/SAMPLE_RATE:.2f} seconds")
    print(f"Number of MC samples: {MC_DROPOUT_SAMPLES}")
    print(f"Confidence level: {int(MC_CONFIDENCE_LEVEL*100)}%\n")
    
    # Predict with MC dropout
    (class_name, mean_conf, lower_ci, upper_ci, 
     mc_probs, uncertainty) = predict_segment_with_mc_dropout(
        y_segment, 
        n_samples=MC_DROPOUT_SAMPLES,
        confidence_level=MC_CONFIDENCE_LEVEL
    )
    
    print(f"Predicted Class: {class_name.upper()}")
    print(f"Mean Confidence: {mean_conf:.4f}")
    print(f"Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
    print(f"Interval Width: {upper_ci - lower_ci:.4f}")
    print(f"Uncertainty (Std Dev): {uncertainty:.4f}\n")
    
    # Show all class probabilities
    print("Class Probabilities (mean ± std):")
    mean_probs = mc_probs.mean(axis=0)
    std_probs = mc_probs.std(axis=0)
    for i, cls_name in enumerate(CLASS_NAMES):
        print(f"  {cls_name:12s}: {mean_probs[i]:.4f} ± {std_probs[i]:.4f}")
    
    return class_name, mean_conf, uncertainty


def demo_mc_dropout_batch(filepath, num_segments=5):
    """
    Demonstrate Monte Carlo dropout on multiple audio segments.
    Shows batch predictions with uncertainty estimates.
    """
    print("\n" + "="*70)
    print("Monte Carlo Dropout Uncertainty Estimation - Batch Predictions")
    print("="*70 + "\n")
    
    # Load audio
    y, sr = librosa.load(filepath, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    window_len = int(WINDOW_SIZE * SAMPLE_RATE)
    hop_len = int(HOP_SIZE * SAMPLE_RATE)
    
    # Extract segments
    segments = []
    timestamps = []
    for start in range(0, min(len(y) - window_len + 1, num_segments * hop_len), hop_len):
        end = start + window_len
        if end <= len(y):
            segments.append(y[start:end])
            timestamps.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
    
    print(f"Analyzing {len(segments)} segments from: {filepath}")
    print(f"Number of MC samples: {MC_DROPOUT_SAMPLES}")
    print(f"Confidence level: {int(MC_CONFIDENCE_LEVEL*100)}%\n")
    
    # Predict with MC dropout
    results = predict_batch_with_mc_dropout(
        segments,
        n_samples=MC_DROPOUT_SAMPLES,
        confidence_level=MC_CONFIDENCE_LEVEL
    )
    
    # Display results
    print(f"{'Time':>12} | {'Class':>12} | {'Confidence':>10} | {'CI Width':>9} | {'Uncertainty':>11}")
    print("-" * 70)
    
    for i, ((t_start, t_end), (cls_name, mean_conf, lower_ci, upper_ci, mc_probs, uncertainty)) in enumerate(zip(timestamps, results)):
        ci_width = upper_ci - lower_ci
        print(f"{t_start:6.2f}-{t_end:5.2f}s | {cls_name:>12} | {mean_conf:>10.4f} | {ci_width:>9.4f} | {uncertainty:>11.4f}")
    
    return results


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    #test_file = r"C:\Users\apata\OneDrive\Documents\Documents scolaires\Hackaton Drone shazam\ThaoRepo\NewRepo\DroneDetector\test_audio\droneJ\J_09.wav"
    test_file = "test_audio/droneA/A_03.wav"
    if not os.path.exists(test_file):
        print(f"File Not Found: {test_file}")
        print("\nTest Audio Corrupted test_audio/:")
        for item in os.listdir("test_audio"):
            print(f"  - {item}")
    else:
        print("="*70)
        print("DRONE DETECTION - Standard Predictions")
        print("="*70 + "\n")
        
        # Execute detection
        predictions = sliding_window_detect(test_file, use_batch=True)
        
        # Majority vote
        winner, winner_count, total_windows = get_majority_vote(predictions)
        
        # Display results
        print("\n" + "="*70)
        print("Final Results")
        print("="*70)
        print(f"\nType of Drone Detected: {winner.upper()}")
        print(f"Votes: {winner_count}/{total_windows} fenêtres ({winner_count/total_windows*100:.1f}%)")
        
        # Show full breakdown of votes
        print(f"\nBreakdown of Votes:")
        vote_counts = Counter(predictions)
        for cls, count in vote_counts.most_common():
            percentage = (count / total_windows) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {cls:12s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        print("\n" + "="*70)
        
        # DEMO: Monte Carlo Dropout Uncertainty Estimation
        # MONTE CARLO DROPOUT DEMONSTRATION")
        # print("="*70)
        # print("This demonstrates uncertainty quantification using MC dropout.")
        # print("The model performs multiple stochastic forward passes to estimate")
        # print("confidence intervals around predictions.\n")
        
        # Single segment demo
        if os.path.exists(test_file):
            demo_mc_dropout_single_segment(test_file)
            
            # Batch demo
            print("\n")
            demo_mc_dropout_batch(test_file, num_segments=5)
        

	# print("\n" + "="*70)
        # print("MC Dropout Tutorial:")
        # print("="*70)
#         print("""
# To use Monte Carlo dropout in your code:

# 1. Enable in configuration:
#    USE_MC_DROPOUT = True
#    MC_DROPOUT_SAMPLES = 25  # More = better but slower
   
# 2. Single prediction with uncertainty:
#    cls, mean_conf, lower_ci, upper_ci, mc_probs, uncertainty = \\
#        predict_segment_with_mc_dropout(y_segment, n_samples=25)
   
# 3. Batch predictions with uncertainty:
#    results = predict_batch_with_mc_dropout(segments, n_samples=25)
   
# 4. Interpret results:
#    - mean_conf: Average predicted probability for the class
#    - [lower_ci, upper_ci]: Confidence interval bounds (95% default)
#    - uncertainty: Standard deviation across MC samples
#    - Wider CI = Higher uncertainty = Less confident prediction
   
# 5. Advanced: Access all MC predictions:
#    mc_probs has shape (n_samples, n_classes)
#    - Row mean = expected probability per class
#    - Row std = uncertainty per class
#         """)