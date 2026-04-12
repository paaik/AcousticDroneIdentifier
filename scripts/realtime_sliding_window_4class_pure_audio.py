import sounddevice as sd
import numpy as np
import librosa
import torch
import os
import time
from collections import deque
import signal
import threading
from train_cnn_4class import DroneCNN

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 22050
WINDOW_SIZE = 1.0                 # 1 second sliding window
HOP_SIZE = 0.5                    # update every 0.5 second
CONF_THRESHOLD = 0.20             # lower a bit for real scenarios

MODEL_PATH = "models/drone_cnn_4class_best.pth"
DATA_MEL_DIR = "data_melspec"     # contains 4 class folders
TEST_AUDIO_FILE = "test_audio/droneH/H_02.wav"  # Change this to test different files

# ==============================
# LOAD MODEL + CLASS NAMES
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint.get('class_names', sorted([c for c in os.listdir(DATA_MEL_DIR) if os.path.isdir(os.path.join(DATA_MEL_DIR, c))]))
num_classes = len(CLASS_NAMES)

model = DroneCNN(num_classes, dropout=0.3)
if 'model_state_dict' in checkpoint:
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
if missing or unexpected:
    print('Warning: state_dict mismatch', 'missing', missing, 'unexpected', unexpected)

model.to(device)
model.eval()

print("Real-time Sliding Window Detection Initialized")
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
        print(f"unknown (avg_conf={avg_conf:.2f})")
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
# PROCESS TEST AUDIO FUNCTION
# ==============================
def process_audio_file(file_path):
    """
    Load and process an audio file with sliding window detection.
    Returns True if successful, False otherwise.
    """
    print(f"\nLoading audio file: {file_path}")
    try:
        audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return False
    
    print("Processing with SLIDING WINDOW...\n")
    
    # Reset buffers
    local_buffer = np.zeros(buffer_size, dtype=np.float32)
    local_history = deque(maxlen=5)
    
    # Simulate streaming by processing chunks
    for i in range(0, len(audio_data) - buffer_size, hop_size):
        new_audio = audio_data[i:i + hop_size]
        
        # shift buffer left & append new audio
        local_buffer = np.roll(local_buffer, -len(new_audio))
        local_buffer[-len(new_audio):] = new_audio
        
        # Run inference
        cls, conf = predict_segment(local_buffer)
        
        # smoothing: store predictions in history
        local_history.append((cls, conf))
        
        # majority vote for class
        classes = [c for c, _ in local_history]
        smoothed_class = max(set(classes), key=classes.count)
        
        # average confidence
        confidences = [c for _, c in local_history]
        avg_conf = sum(confidences) / len(confidences)
        
        # print result
        if avg_conf < CONF_THRESHOLD:
            print(f"unknown (avg_conf={avg_conf:.2f})")
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
    
    print("\nProcessing complete")
    return True


# ==============================
# MODE SELECTION
# ==============================
print("\n" + "="*50)
print("SELECT AUDIO SOURCE:")
print("  Press 1 for TEST AUDIO FILE")
print("  Press 2 for MICROPHONE")
print("="*50)

mode = input("\nEnter your choice (1 or 2): ").strip()

if mode == "1":
    # ==============================
    # INTERACTIVE TEST AUDIO MODE
    # ==============================
    continue_testing = True
    while continue_testing:
        # Ask user for audio file path
        file_path = input("\nEnter the audio file path: ").strip()
        
        # Remove quotes if the user included them
        file_path = file_path.strip('"\'')
        
        # Process the audio file
        if process_audio_file(file_path):
            # Ask if user wants to test another file
            while True:
                response = input("\nWould you like to test another audio file? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    break
                elif response in ['no', 'n']:
                    continue_testing = False
                    break
                else:
                    print("Please enter 'yes' or 'no'")
    
    print("\nTesting session ended")

elif mode == "2":
    # ==============================
    # START MICROPHONE STREAM
    # ==============================
    stop_event = threading.Event()
    
    def _shutdown(signum, frame):
        print("\nKeyboard interrupt received, stopping...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, _shutdown)
    
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=hop_size,    # process every 0.5 seconds
        callback=audio_callback,
    ):
        print("Listening with SLIDING WINDOW... Press Ctrl+C to stop.\n")
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt caught in loop")
        finally:
            print("Stopping stream...")

else:
    print("Invalid choice. Please enter 1 or 2.")
    exit(1)
