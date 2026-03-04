import librosa
import numpy as np

def compute_time_domain_features(
        y, 
        sr, 
        frame_length=1024, 
        hop_length=512
    ):
    """
    Return AE, RMS, ZCR for each frame.
    """

    # AE (Amplitude Envelope)
    ae_list = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_length]
        if len(frame) < frame_length:
            break
        ae_list.append(np.max(np.abs(frame)))
    ae = np.array(ae_list)

    # RMS Energy
    rms = librosa.feature.rms(
        y=y, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y=y, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]

    return ae, rms, zcr


def is_sound_interesting(ae, rms, zcr, 
                         thresh_rms=0.02, 
                         thresh_zcr_low=0.02,
                         thresh_zcr_high=0.08):
    """
    Determine if audio is worth passing to Step 3 (mel-spectrogram)
    """

    mean_rms = np.mean(rms)
    mean_zcr = np.mean(zcr)

    # conditions:
    # 1. audio has energy
    energy_ok = mean_rms > thresh_rms

    # 2. zcr not too high (wind/noise) and not too low (silence)
    zcr_ok = (mean_zcr > thresh_zcr_low) and (mean_zcr < thresh_zcr_high)

    return energy_ok and zcr_ok


if __name__ == "__main__":
    filepath = "./data_22050/droneA/A_04.wav"

    y, sr = librosa.load(filepath, sr=None)
    ae, rms, zcr = compute_time_domain_features(y, sr)

    print("AE shape:", ae.shape)
    print("RMS mean:", np.mean(rms))
    print("ZCR mean:", np.mean(zcr))

    if is_sound_interesting(ae, rms, zcr):
        print("🚀 This sound is interesting → go to Step 3 (mel-spectrogram).")
    else:
        print("❌ Noise or silence → skip CNN.")
