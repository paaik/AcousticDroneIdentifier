import os
import librosa

RAW_DATA_DIR = "data_raw"

def check_sample_rates():
    print("Checking sample rates...\n")
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                filepath = os.path.join(root, file)
                try:
                    y, sr = librosa.load(filepath, sr=None)
                    print(f"{filepath}: {sr} Hz")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

if __name__ == "__main__":
    check_sample_rates()
