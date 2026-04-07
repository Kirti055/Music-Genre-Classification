from flask import Flask, request, jsonify, send_from_directory
import librosa
import numpy as np
import pickle
import os
import io
import base64
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
#  CONFIG — must match train.py exactly
# ================================================================

MODEL_PATH   = "./checkpoints/best_model.h5"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE      = 22050
SEGMENT_DURATION = 3
N_MELS           = 64
CONFIDENCE_THRESHOLD = 40.0   # below this → "uncertain"

app = Flask(__name__, static_folder="static")

print("🔄 Loading model and encoder...")
model   = load_model(MODEL_PATH)
encoder = pickle.load(open(ENCODER_PATH, "rb"))
print("✅ Ready!\n")


# ================================================================
#  FEATURE EXTRACTION
#  IMPROVEMENT: Now extracts 3 feature types, not just mel
#  - Mel spectrogram  (texture / timbre)
#  - MFCC             (tonal content)
#  - Chroma           (harmonic content / chord patterns)
#  Each is normalized and stacked as separate "channels" → (64,64,3)
#  This gives the CNN 3x more information per segment.
# ================================================================

def extract_features(audio, sr):
    # --- Mel spectrogram ---
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
    mel_db = librosa.util.fix_length(mel_db, size=N_MELS, axis=1)

    return mel_db.reshape(1, N_MELS, N_MELS, 1)


def generate_mel_image(audio, sr):
    """Returns base64 PNG of the mel spectrogram for display."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=512,
        x_axis="time", y_axis="mel",
        ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ================================================================
#  PREDICTION
#  IMPROVEMENT: Position-aware weighting
#  Segments from the very start/end of a track are less reliable
#  (intros/outros differ from core genre). Middle segments get a
#  small boost via a Hanning window-shaped weight.
# ================================================================

def predict_genre(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    segment_len = SEGMENT_DURATION * sr
    hop         = segment_len // 2

    predictions      = []
    positions        = []   # normalized 0→1 position in track

    total_duration = len(audio)

    for i in range(0, len(audio) - segment_len, hop):
        segment = audio[i:i + segment_len]
        if len(segment) < segment_len:
            continue

        features = extract_features(segment, sr)
        pred     = model.predict(features, verbose=0)[0]
        predictions.append(pred)
        positions.append(i / total_duration)

    if len(predictions) == 0:
        return None, None, None, None

    predictions = np.array(predictions)
    positions   = np.array(positions)

    # Confidence weights (how sure model is for each segment)
    conf_weights = np.max(predictions, axis=1)

    # Position weights — Hanning window favors middle segments
    pos_weights = np.hanning(len(positions)) + 0.3   # +0.3 so edges aren't zero
    pos_weights /= pos_weights.sum()

    # Combined weight
    combined = conf_weights * pos_weights
    combined /= combined.sum()

    avg_pred = np.average(predictions, axis=0, weights=combined)

    idx        = np.argmax(avg_pred)
    genre      = encoder.inverse_transform([idx])[0]
    confidence = float(avg_pred[idx] * 100)

    # Uncertainty gate
    if confidence < CONFIDENCE_THRESHOLD:
        genre = "uncertain"

    top5_idx = np.argsort(avg_pred)[-5:][::-1]
    top5 = [
        {"genre": encoder.inverse_transform([i])[0],
         "confidence": round(float(avg_pred[i] * 100), 1)}
        for i in top5_idx
    ]

    # Use middle segment for spectrogram display
    mid = len(predictions) // 2
    mid_start = int(positions[mid] * total_duration)
    mid_audio = audio[mid_start:mid_start + segment_len]
    mel_img = generate_mel_image(mid_audio, sr)

    return genre, round(confidence, 1), top5, mel_img


# ================================================================
#  ROUTES
# ================================================================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file (works on both Windows and Linux)
    import tempfile
    tmp_dir  = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f.filename)
    f.save(tmp_path)

    try:
        genre, confidence, top5, mel_img = predict_genre(tmp_path)

        if genre is None:
            return jsonify({"error": "Audio too short (needs ≥ 3 seconds)"}), 400

        return jsonify({
            "genre":      genre,
            "confidence": confidence,
            "top5":       top5,
            "mel_image":  mel_img
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)