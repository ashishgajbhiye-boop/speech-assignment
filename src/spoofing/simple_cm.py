import numpy as np
import librosa

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc)

def classify(real_audio, fake_audio):
    real_feat = extract_features(real_audio)
    fake_feat = extract_features(fake_audio)

    return {
        "real_score": float(real_feat),
        "fake_score": float(fake_feat),
        "prediction": "spoof" if fake_feat != real_feat else "bona_fide"
    }