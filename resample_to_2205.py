import os
import librosa
import soundfile as sf

INPUT_DIR = "data_raw"
OUTPUT_DIR = "data_22050"
TARGET_SR = 22050

def resample_to_22050():
    print("Resampling all audio files to 22050Hz & mono...\n")

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                input_path = os.path.join(root, file)

                # output path mirrors input folder structure
                rel_path = os.path.relpath(input_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, rel_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # load file in original SR
                    y, sr = librosa.load(input_path, sr=None, mono=True)

                    # resample if needed
                    if sr != TARGET_SR:
                        print(f"Resampling {rel_path}: {sr} → {TARGET_SR}")
                        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

                    # save file
                    sf.write(output_path, y, TARGET_SR)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    resample_to_22050()
