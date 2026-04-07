import librosa
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from gradcam import make_gradcam_heatmap
import warnings
warnings.filterwarnings("ignore")


# ================================================================
#  CONFIG — must match train.py exactly
# ================================================================

MODEL_PATH   = "./checkpoints/best_model.h5"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE      = 22050
SEGMENT_DURATION = 3     # must match train.py
N_MELS           = 64    # must match train.py


# ================================================================
#  LOAD MODEL + ENCODER
# ================================================================

print("🔄 Loading model...")
model   = load_model(MODEL_PATH)
encoder = pickle.load(open(ENCODER_PATH, "rb"))

# Force model build
dummy = tf.zeros((1, N_MELS, N_MELS, 1))
model(dummy)

print("✅ Model loaded.\n")


# ================================================================
#  FEATURE EXTRACTION — identical to train.py and predict.py
# ================================================================

def extract_features(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize — same as training
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Fix to square shape
    mel_db = librosa.util.fix_length(mel_db, size=N_MELS, axis=1)

    return mel_db  # shape: (64, 64)


# ================================================================
#  VISUALIZATION
#  Shows 4 panels:
#  1. Raw Mel Spectrogram
#  2. Grad-CAM Heatmap alone
#  3. Heatmap overlaid on spectrogram
#  4. Genre confidence bar chart
# ================================================================

def visualize(mel, heatmap, genre, confidence, all_preds, genres):

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"🎵 Predicted Genre: {genre.upper()}  |  Confidence: {confidence:.1f}%",
        fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: Raw Mel Spectrogram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax1.set_title("Mel Spectrogram", fontweight="bold")
    ax1.set_xlabel("Time Frames")
    ax1.set_ylabel("Mel Frequency Bins")

    # --- Panel 2: Grad-CAM Heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
    ax2.set_title("Grad-CAM Heatmap\n(what the model focused on)",
                  fontweight="bold")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Mel Frequency Bins")
    plt.colorbar(im, ax=ax2, label="Activation Intensity")

    # --- Panel 3: Overlay ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax3.imshow(heatmap, aspect="auto", origin="lower",
               cmap="jet", alpha=0.45)
    ax3.set_title("Overlay\n(spectrogram + heatmap)", fontweight="bold")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Mel Frequency Bins")

    # --- Panel 4: Genre Confidence Bar Chart ---
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ["#e63946" if g == genre else "#457b9d" for g in genres]
    bars = ax4.barh(genres, all_preds * 100, color=colors)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Confidence (%)")
    ax4.set_title("All Genre Probabilities", fontweight="bold")
    ax4.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax4.invert_yaxis()

    plt.savefig("explanation.png", dpi=150, bbox_inches="tight")
    print("💾 Saved explanation.png")
    plt.show()


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":

    print("🎵 MUSIC GENRE EXPLAINER")
    print("─" * 35)

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio files", "*.wav *.mp3")]
    )

    if not file_path:
        print("❌ No file selected.")
        exit()

    print(f"📂 File: {file_path}")
    print("🔍 Analysing...\n")

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    segment_len      = SEGMENT_DURATION * sr
    hop              = segment_len // 2
    predictions      = []
    stored_segments  = []

    for i in range(0, len(audio) - segment_len, hop):
        segment = audio[i:i + segment_len]

        if len(segment) < segment_len:
            continue

        mel      = extract_features(segment, sr)
        features = mel.reshape(1, N_MELS, N_MELS, 1)
        pred     = model.predict(features, verbose=0)[0]

        predictions.append(pred)
        stored_segments.append((mel, features))

    if len(predictions) == 0:
        print("⚠️  Audio too short.")
        exit()

    predictions = np.array(predictions)

    # Weighted average — same as predict.py
    confidences = np.max(predictions, axis=1)
    weights     = confidences / confidences.sum()
    avg_pred    = np.average(predictions, axis=0, weights=weights)

    idx        = np.argmax(avg_pred)
    genre      = encoder.inverse_transform([idx])[0]
    confidence = float(avg_pred[idx] * 100)

    print("🎵 RESULT")
    print("─" * 35)
    print(f"  Genre      : {genre.upper()}")
    print(f"  Confidence : {confidence:.1f}%\n")

    print("  Top 3 Predictions:")
    top3_idx = np.argsort(avg_pred)[-3:][::-1]
    for i in top3_idx:
        g = encoder.inverse_transform([i])[0]
        c = avg_pred[i] * 100
        bar = "█" * int(c / 5)
        print(f"  {g:<12} {c:5.1f}%  {bar}")

    # Use most confident segment for Grad-CAM
    best_idx           = np.argmax(predictions[:, idx])
    mel, features      = stored_segments[best_idx]

    print("\n🔥 Generating Grad-CAM explanation...")
    heatmap = make_gradcam_heatmap(model, features)

    # Resize heatmap to match mel shape (64x64)
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (N_MELS, N_MELS)
    ).numpy().squeeze()

    genres = list(encoder.classes_)
    visualize(mel, heatmap_resized, genre, confidence, avg_pred, genres)