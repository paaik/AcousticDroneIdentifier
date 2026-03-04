import os
import sys
import tempfile
from collections import Counter

import librosa
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Make sure Python can see the "scripts" folder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Now this will load scripts/offline_sliding_window_4class.py
from offline_sliding_window_4class import (
    sliding_window_detect,
    get_majority_vote,
    make_melspec,
    SAMPLE_RATE,
    CLASS_NAMES,
)


# -----------------------------
# Helper: plot mel-spectrogram
# -----------------------------
def plot_mel_spectrogram(y, sr=SAMPLE_RATE):
    """
    Uses the SAME mel pipeline as your CNN (make_melspec from offline_sliding_window_4class).
    """
    mel = make_melspec(y)  # already normalised log-mel (64 x T)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(mel, aspect="auto", origin="lower")
    ax.set_title("Mel Spectrogram (CNN Input Space)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    fig.colorbar(img, ax=ax, label="Normalised dB")
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Ultra Drone Detector Dashboard",
        layout="wide",
    )

    st.title("🛸 Ultra Drone Detector – CNN Dashboard")
    st.write(
        "Upload a **.wav** file and run the sliding-window CNN detector. "
        "The system will compute mel spectrograms and perform majority voting "
        "over all windows to decide the final class."
    )

    # Sidebar settings
    st.sidebar.header("Detection Settings")

    # Confidence threshold – this is the same idea as CONF_THRESHOLD in offline_sliding_window_4class
    conf_threshold = st.sidebar.slider(
        "Confidence threshold (for each window)",
        min_value=0.10,
        max_value=0.99,
        value=0.50,
        step=0.01,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model info")
    st.sidebar.write("Detected classes:")
    for c in CLASS_NAMES:
        st.sidebar.write(f"- `{c}`")

    # File uploader
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Save uploaded file to a temporary .wav so librosa + your pipeline can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load audio for display / mel plotting
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Show basic info
        duration = len(y) / sr
        st.write(f"**Sample rate:** {sr} Hz")
        st.write(f"**Duration:** {duration:.2f} seconds")

        # Show waveform
        with st.expander("Show waveform"):
            fig, ax = plt.subplots(figsize=(8, 3))
            t = np.linspace(0, duration, num=len(y))
            ax.plot(t, y)
            ax.set_title("Waveform")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

        # Show mel spectrogram used by CNN
        with st.expander("Show mel spectrogram (CNN input space)", expanded=True):
            plot_mel_spectrogram(y, sr)

        # Run detection button
        if st.button("🚀 Run CNN detection"):
            st.write("Running sliding-window detection…")

            # --- IMPORTANT: apply threshold dynamically here ---
            # We temporarily monkey-patch the global CONF_THRESHOLD within the imported module.
            # This assumes offline_sliding_window_4class uses a global CONF_THRESHOLD variable.
            import offline_sliding_window_4class as sw

            sw.CONF_THRESHOLD = conf_threshold

            with st.spinner("Analyzing audio…"):
                predictions = sliding_window_detect(tmp_path, use_batch=True)

            # Majority voting result
            winner, winner_count, total_windows = get_majority_vote(predictions)

            st.markdown("## 🏆 Final Detection (Majority Vote)")
            if winner == "unknown":
                st.error(
                    "No class reached the required confidence. Final result: **UNKNOWN**"
                )
            else:
                percentage = winner_count / total_windows * 100
                st.success(
                    f"**Detected class:** `{winner}`  "
                    f"— {winner_count}/{total_windows} windows "
                    f"({percentage:.1f}%)"
                )

            # Vote distribution
            st.markdown("### 📊 Vote distribution over all windows")
            vote_counts = Counter(predictions)
            labels = list(vote_counts.keys())
            counts = np.array([vote_counts[c] for c in labels])
            percentages = counts / total_windows * 100

            # Table
            st.table(
                {
                    "Class": labels,
                    "Votes": counts,
                    "Percentage (%)": np.round(percentages, 1),
                }
            )

            # Bar chart
            st.markdown("#### Bar chart")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.bar(labels, percentages)
            ax2.set_ylabel("Percentage of windows (%)")
            ax2.set_xlabel("Class")
            ax2.set_ylim(0, 100)
            for i, v in enumerate(percentages):
                ax2.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
            st.pyplot(fig2)

        # Clean temp file when done
        # (optional – you can comment this out during debugging)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
