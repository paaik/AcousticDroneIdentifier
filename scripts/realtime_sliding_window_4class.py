import sounddevice as sd
import numpy as np
import librosa
import torch
import os
import time
from collections import deque
from train_cnn_4class import DroneCNN

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 22050
WINDOW_SIZE = 1.0                 # 1 second sliding window
HOP_SIZE = 0.5                    # update every 0.5 second
CONF_THRESHOLD = 0.55             # lower a bit for real scenarios

MODEL_PATH = "models/drone_cnn_4class_best.pth"
DATA_MEL_DIR = "data_melspec"     # contains 4 class folders

# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = sorted([
    c for c in os.listdir(DATA_MEL_DIR)
    if os.path.isdir(os.path.join(DATA_MEL_DIR, c))
])

num_classes = len(CLASS_NAMES)

model = DroneCNN(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("🎧 Real-time Sliding Window Detection Initialized")
print("Classes:", CLASS_NAMES, "\n")


# ==============================
# MEL-SPECTROGRAM
# ==============================
def make_melspec(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


# ==============================
# PREDICT SINGLE WINDOW
# ==============================
def predict_segment(y_seg):
    mel = make_melspec(y_seg)
    mel_t = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mel_t)
        prob = torch.softmax(pred, dim=1)

    conf, cls = torch.max(prob, dim=1)
    return CLASS_NAMES[cls.item()], conf.item()


# ==============================
# SLIDING BUFFER
# ==============================
buffer_size = int(WINDOW_SIZE * SAMPLE_RATE)
hop_size = int(HOP_SIZE * SAMPLE_RATE)

audio_buffer = np.zeros(buffer_size, dtype=np.float32)
prediction_history = deque(maxlen=5)  # smoothing window


# ==============================
# AUDIO CALLBACK
# ==============================
def audio_callback(indata, frames, time, status):
    global audio_buffer, prediction_history

    new_audio = indata[:, 0]  # mono

    # shift buffer left & append new audio
    audio_buffer = np.roll(audio_buffer, -len(new_audio))
    audio_buffer[-len(new_audio):] = new_audio

    # Only run inference every hop_size samples
    if status:
        print("Audio status:", status)

    cls, conf = predict_segment(audio_buffer)

    # smoothing: store predictions in history
    prediction_history.append((cls, conf))

    # majority vote for class
    classes = [c for c, _ in prediction_history]
    smoothed_class = max(set(classes), key=classes.count)

    # average confidence
    confidences = [c for _, c in prediction_history]
    avg_conf = sum(confidences) / len(confidences)

    # print result
    if avg_conf < CONF_THRESHOLD:
        print(f"❓ unknown (avg_conf={avg_conf:.2f})")
    else:
        emoji = {
            "droneA": "A",
            "droneB": "B",
            "droneC": "C",
            "droneD": "D",
            "droneE": "E",
            "droneF": "F",
            "droneG": "G",
            "droneH": "H",
            "droneI": "I",
            "droneJ": "J"
        }.get(smoothed_class, "🔊")

        print(f"{emoji} {smoothed_class} (avg_conf={avg_conf:.2f})")


# ==============================
# START STREAM
# ==============================
with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=hop_size,    # process every 0.5 seconds
    callback=audio_callback,
):

# Initialize the queue (add this before your audio_callback)


# ... (keep your callback and model loading code here) ...
    print("🎙 Listening with SLIDING WINDOW... Press Ctrl+C to stop.\n")
    while True:
        time.sleep(0.1)
