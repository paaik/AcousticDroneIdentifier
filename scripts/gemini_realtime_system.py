import sounddevice as sd
import numpy as np
import librosa
import torch
import os
import queue
from collections import deque
from train_cnn_4class import DroneCNN

# ==============================
# CONFIG & QUEUE
# ==============================
SAMPLE_RATE = 22050
WINDOW_SIZE = 1.0 
HOP_SIZE = 0.5 
CONF_THRESHOLD = 0.55 
MODEL_PATH = "models/drone_cnn_4class_best.pth"
DATA_MEL_DIR = "data_melspec"

audio_queue = queue.Queue() # The bridge between audio and AI

# ==============================
# LOAD MODEL
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = sorted([c for c in os.listdir(DATA_MEL_DIR) if os.path.isdir(os.path.join(DATA_MEL_DIR, c))])
model = DroneCNN(len(CLASS_NAMES))
# Load the checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Extract only the model weights (the 'model_state_dict' key)
model.load_state_dict(checkpoint['model_state_dict'])

# Optional: You can even extract the class names from the file now!
if 'class_names' in checkpoint:
    CLASS_NAMES = checkpoint['class_names']
    print(f"Loaded classes from checkpoint: {CLASS_NAMES}")
model.to(device)
model.eval()

# Warm up the GPU
model(torch.zeros(1, 1, 64, 87).to(device)) 

# ==============================
# HELPER FUNCTIONS
# ==============================
def make_melspec(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel

def predict_segment(y_seg):
    mel = make_melspec(y_seg)
    mel_t = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(mel_t)
        prob = torch.softmax(pred, dim=1)
    conf, cls = torch.max(prob, dim=1)
    return CLASS_NAMES[cls.item()], conf.item()

# ==============================
# THE CLEAN CALLBACK
# ==============================
def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}")
    # Just put the data in the queue and get out!
    audio_queue.put(indata.copy())

# ==============================
# MAIN LOOP
# ==============================
buffer_size = int(WINDOW_SIZE * SAMPLE_RATE)
hop_size = int(HOP_SIZE * SAMPLE_RATE)
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
prediction_history = deque(maxlen=5)

print("🎙 Listening... Press Ctrl+C to stop.\n")

# Start the stream
stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=hop_size, callback=audio_callback)

with stream:
    try:
        while True:
            # 1. Wait for audio data (this is the Producer-Consumer pattern)
            new_data = audio_queue.get() 
            
            # 2. Update sliding window
            audio_buffer = np.roll(audio_buffer, -len(new_data))
            audio_buffer[-len(new_data):] = new_data[:, 0]
            
            # 3. Predict & Smooth
            cls, conf = predict_segment(audio_buffer)
            prediction_history.append((cls, conf))
            
            # Majority vote
            classes = [c for c, _ in prediction_history]
            smoothed_class = max(set(classes), key=classes.count)
            avg_conf = sum([c for _, c in prediction_history]) / len(prediction_history)

            # 4. Print Result
            if avg_conf < CONF_THRESHOLD:
                print(f"❓ unknown... ({avg_conf:.2f})", end='\r')
            else:
                print(f"✅ {smoothed_class} ({avg_conf:.2f})      ", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopping...")