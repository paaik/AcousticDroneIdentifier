import os
import librosa
import soundfile as sf

# ===========================
# CONFIG
# ===========================
INPUT_DIR = "data_22050"          # chứa 4 folders: drone, birds, wind, vehicles
OUTPUT_DIR = "data_segments"      # output folder
SEGMENT_DURATION = 1.0            # length of each segment (in seconds)
OVERLAP = 0.5                     # 50% overlap
TARGET_SR = 22050                 # sample rate to keep consistent


# ===========================
# Split ONE audio file
# ===========================
def split_audio(input_path, out_folder, sr=TARGET_SR):
    y, orig_sr = librosa.load(input_path, sr=None, mono=True)

    # Resample nếu cần
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


# ===========================
# Process ALL classes/folders
# ===========================
def process_all():
    print("Splitting audio into segments...\n")

    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)

        # Skip non-folders
        if not os.path.isdir(class_path):
            continue

        print(f"➡ Processing class: {class_name}")

        # Output folder for this class
        out_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(out_dir, exist_ok=True)

        # Loop through all WAV files in this class
        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                input_path = os.path.join(class_path, file)

                print(f"   - Splitting file: {file}")
                n_segments = split_audio(input_path, out_dir)
                print(f"     ✔ Created {n_segments} segments.")

        print()

    print("🎉 Done! Segments saved in:", OUTPUT_DIR)


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    process_all()
