import sounddevice as sd
import numpy as np
import librosa
import torch
import os
import queue
from train_cnn import DroneCNN

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 22050
WINDOW_SIZE = 1.0  # 1 second sliding window
HOP_SIZE = 0.5  # update every 0.5 second (stride)
CONF_THRESHOLD = 0.55

MODEL_PATH = "/Users/joshuarose/Downloads/models/drone_cnn_4class.pth"
DATA_MEL_DIR = "/Users/joshuarose/Downloads/CNN_Training/audio_mels"

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
# AUDIO PROCESSING FUNCTIONS
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


def predict_segment(y_seg):
    mel = make_melspec(y_seg)
    mel_t = torch.tensor(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mel_t)
        prob = torch.softmax(pred, dim=1)

    conf, cls = torch.max(prob, dim=1)
    return CLASS_NAMES[cls.item()], conf.item()


# ==============================
# THREAD-SAFE AUDIO CAPTURE
# ==============================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Minimal footprint callback. Just pushes raw data to queue."""
    if status:
        print(status, flush=True)
    # sounddevice delivers multi-channel arrays; flatten to mono
    audio_queue.put(indata.copy()[:, 0])


# ==============================
# MAIN SLIDING WINDOW LOOP
# ==============================
def main():
    # Calculate buffer sizes in sample points
    window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
    hop_samples = int(HOP_SIZE * SAMPLE_RATE)

    # Initialize our sliding history buffer
    sliding_buffer = np.zeros(window_samples, dtype=np.float32)
    samples_since_last_prediction = 0

    # Match blocksize to your hop length for clean steps
    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=hop_samples,
        callback=audio_callback
    )

    print("🎙️ Listening with SLIDING WINDOW... Press Ctrl+C to stop.\n")

    with stream:
        while True:
            try:
                # Grab a chunk from the queue (blocks until data arrives)
                new_chunk = audio_queue.get()
                chunk_len = len(new_chunk)

                # Shift buffer left and append new incoming chunk
                sliding_buffer = np.roll(sliding_buffer, -chunk_len)
                sliding_buffer[-chunk_len:] = new_chunk

                samples_since_last_prediction += chunk_len

                # Only predict when we've gathered at least one hop size of fresh data
                if samples_since_last_prediction >= hop_samples:
                    label, conf = predict_segment(sliding_buffer)

                    if conf >= CONF_THRESHOLD:
                        print(print(f"⚠️ [DETECTED] {label} (Conf: {conf:.2f})", flush=True))
                    else:
                        print(f"💤 Background noise... (Top: {label} {conf:.2f})", flush=True)

                    samples_since_last_prediction = 0

            except KeyboardInterrupt:
                print("\n🛑 Stream stopped.")
                break


if __name__ == "__main__":
    main()
