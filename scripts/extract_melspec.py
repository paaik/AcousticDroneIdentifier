import os
import librosa
import numpy as np

INPUT_DIR = "data_segments"
OUTPUT_DIR = "data_melspec"

def extract_melspec(filepath, sr=22050, n_mels=64):
    y, orig_sr = librosa.load(filepath, sr=None, mono=True)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels
    )

    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def process_segments():
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                class_name = os.path.basename(root)
                out_dir = os.path.join(OUTPUT_DIR, class_name)
                os.makedirs(out_dir, exist_ok=True)

                mel = extract_melspec(input_path)

                out_path = os.path.join(out_dir, file.replace(".wav", ".npy"))
                np.save(out_path, mel)

                print(f"Saved mel-spectrogram: {out_path}")


if __name__ == "__main__":
    process_segments()
