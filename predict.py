import librosa
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH   = "./checkpoints/best_model.h5"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE      = 22050
SEGMENT_DURATION = 3     
N_MELS           = 64   


print("🔄 Loading model and encoder...")
model   = load_model(MODEL_PATH)
encoder = pickle.load(open(ENCODER_PATH, "rb"))
print("✅ Ready!\n")


def extract_features(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Same normalization as training
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Fix to square shape
    mel_db = librosa.util.fix_length(mel_db, size=N_MELS, axis=1)

    return mel_db.reshape(1, N_MELS, N_MELS, 1)


def predict_genre(file_path):

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    segment_len = SEGMENT_DURATION * sr
    hop         = segment_len // 2  

    predictions = []

    for i in range(0, len(audio) - segment_len, hop):
        segment = audio[i:i + segment_len]

        if len(segment) < segment_len:
            continue

        features = extract_features(segment, sr)
        pred     = model.predict(features, verbose=0)[0]
        predictions.append(pred)

    if len(predictions) == 0:
        return None, None, None

    predictions = np.array(predictions)

    confidences = np.max(predictions, axis=1)
    weights     = confidences / confidences.sum()
    avg_pred    = np.average(predictions, axis=0, weights=weights)

    idx        = np.argmax(avg_pred)
    genre      = encoder.inverse_transform([idx])[0]
    confidence = float(avg_pred[idx] * 100)

    top3_idx = np.argsort(avg_pred)[-3:][::-1]
    top3 = [
        (encoder.inverse_transform([i])[0], float(avg_pred[i] * 100))
        for i in top3_idx
    ]

    return genre, confidence, top3


if __name__ == "__main__":

    print("🎵 MUSIC GENRE CLASSIFIER")
    print("─" * 30)

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio files", "*.wav *.mp3")]
    )

    if not file_path:
        print("❌ No file selected.")
    else:
        print(f"📂 File: {file_path}")
        print("🔍 Analysing audio...\n")

        try:
            genre, confidence, top3 = predict_genre(file_path)

            if genre is None:
                print("⚠️  Audio too short — needs at least 3 seconds.")
            else:
                print("🎵 RESULT")
                print("─" * 30)
                print(f"  Genre      : {genre.upper()}")
                print(f"  Confidence : {confidence:.1f}%\n")
                print("  Top 3 Predictions:")
                for i, (g, c) in enumerate(top3, 1):
                    bar = "█" * int(c / 5)
                    print(f"  {i}. {g:<12} {c:5.1f}%  {bar}")

        except Exception as e:
            print(f"❌ Error: {e}")