import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score
)

from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = "./Data/combined_dataset"
BEST_MODEL = "./checkpoints/best_model.h5"
ENCODER_PATH = "label_encoder.pkl"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
N_MELS = 64

OUTPUT_DIR = "./evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    mel_db = librosa.util.fix_length(
        mel_db,
        size=N_MELS,
        axis=1
    )

    return mel_db


# ==========================================================
# LOAD MODEL AND LABEL ENCODER
# ==========================================================

print("\n" + "=" * 55)
print("  MUSIC GENRE CLASSIFIER - EVALUATION")
print("=" * 55 + "\n")

if not os.path.exists(BEST_MODEL):
    raise FileNotFoundError(f"Model not found: {BEST_MODEL}")

print("Loading model and label encoder...")

model = load_model(BEST_MODEL)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

genres = list(encoder.classes_)
num_classes = len(genres)

print(f"Loaded {num_classes} genres:")
print(genres)


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nRebuilding test dataset...\n")

X_all = []
y_all = []

for genre in sorted(genres):
    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        print(f"Folder not found: {genre_path}")
        continue

    print(f"Loading genre: {genre}")

    for file_name in os.listdir(genre_path):

        if not file_name.lower().endswith((".wav", ".mp3")):
            continue

        file_path = os.path.join(genre_path, file_name)

        try:
            audio, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            segment_length = SEGMENT_DURATION * sr
            hop_length = segment_length // 2

            if len(audio) < segment_length:
                continue

            for start in range(0, len(audio) - segment_length, hop_length):
                end = start + segment_length
                segment = audio[start:end]

                features = extract_features(segment, sr)

                X_all.append(features)
                y_all.append(genre)

        except Exception as e:
            print(f"Could not process {file_name}: {e}")


# ==========================================================
# PREPARE TEST DATA
# ==========================================================

X_all = np.array(X_all)
y_all = np.array(y_all)

X_all = X_all.reshape(-1, N_MELS, N_MELS, 1)

y_encoded = encoder.transform(y_all)
y_categorical = to_categorical(y_encoded)

_, X_test, _, y_test, _, y_test_encoded = train_test_split(
    X_all,
    y_categorical,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

print(f"\nTest set contains {len(X_test)} samples")


# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

print("\nRunning predictions...")

y_probabilities = model.predict(X_test, batch_size=32)
y_predicted = np.argmax(y_probabilities, axis=1)
y_actual = y_test_encoded


# ==========================================================
# CALCULATE METRICS
# ==========================================================

overall_accuracy = accuracy_score(y_actual, y_predicted) * 100
balanced_accuracy = balanced_accuracy_score(y_actual, y_predicted) * 100
top2_accuracy = top_k_accuracy_score(y_actual, y_probabilities, k=2) * 100
top3_accuracy = top_k_accuracy_score(y_actual, y_probabilities, k=3) * 100

report = classification_report(
    y_actual,
    y_predicted,
    target_names=genres,
    digits=3
)

cm = confusion_matrix(y_actual, y_predicted)


# ==========================================================
# PER-GENRE CONFIDENCE
# ==========================================================

average_confidence = {}

for i, genre in enumerate(genres):
    genre_mask = (y_actual == i)

    if np.sum(genre_mask) > 0:
        confidence = np.mean(y_probabilities[genre_mask, i]) * 100
        average_confidence[genre] = confidence


# ==========================================================
# PER-GENRE ACCURACY
# ==========================================================

per_genre_accuracy = {}

for i, genre in enumerate(genres):
    genre_mask = (y_actual == i)

    if np.sum(genre_mask) > 0:
        correct_predictions = np.sum(y_predicted[genre_mask] == i)
        total_predictions = np.sum(genre_mask)

        accuracy = (correct_predictions / total_predictions) * 100
        per_genre_accuracy[genre] = accuracy


# ==========================================================
# MOST CONFUSED GENRES
# ==========================================================

normalized_cm = cm.astype(float)

for i in range(len(normalized_cm)):
    total = np.sum(normalized_cm[i])

    if total > 0:
        normalized_cm[i] = (normalized_cm[i] / total) * 100

np.fill_diagonal(normalized_cm, 0)

confused_pairs = []

for i in range(num_classes):
    for j in range(num_classes):
        if i != j and normalized_cm[i][j] > 0:
            confused_pairs.append((
                genres[i],
                genres[j],
                normalized_cm[i][j]
            ))

confused_pairs.sort(key=lambda x: -x[2])


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n" + "=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)

print(f"\nOverall Accuracy   : {overall_accuracy:.2f}%")
print(f"Balanced Accuracy : {balanced_accuracy:.2f}%")
print(f"Top-2 Accuracy    : {top2_accuracy:.2f}%")
print(f"Top-3 Accuracy    : {top3_accuracy:.2f}%")

print("\nClassification Report:\n")
print(report)

print("Per-Genre Average Confidence:\n")

for genre, confidence in sorted(
    average_confidence.items(),
    key=lambda x: -x[1]
):
    print(f"{genre:<12} : {confidence:.2f}%")

print("\nPer-Genre Accuracy:\n")

for genre, accuracy in sorted(
    per_genre_accuracy.items(),
    key=lambda x: -x[1]
):
    print(f"{genre:<12} : {accuracy:.2f}%")

print("\nMost Confused Genre Pairs:\n")

for true_genre, predicted_genre, percent in confused_pairs[:8]:
    print(
        f"{true_genre:<12} -> {predicted_genre:<12} : {percent:.2f}%"
    )


# ==========================================================
# SAVE CONFUSION MATRIX PLOT
# ==========================================================

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=genres,
    yticklabels=genres
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_confusion_matrix.png"
)

plt.tight_layout()
plt.savefig(confusion_matrix_path)
plt.close()


# ==========================================================
# SAVE PER-GENRE ACCURACY PLOT
# ==========================================================

sorted_accuracy = sorted(
    per_genre_accuracy.items(),
    key=lambda x: x[1]
)

genre_names = [item[0] for item in sorted_accuracy]
accuracy_values = [item[1] for item in sorted_accuracy]

plt.figure(figsize=(10, 6))

plt.barh(genre_names, accuracy_values)

plt.xlabel("Accuracy (%)")
plt.ylabel("Genre")
plt.title("Per-Genre Accuracy")

accuracy_plot_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_per_genre_accuracy.png"
)

plt.tight_layout()
plt.savefig(accuracy_plot_path)
plt.close()


# ==========================================================
# SAVE RESULTS TO TEXT FILE
# ==========================================================

results_file_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.txt"
)

with open(results_file_path, "w") as file:
    file.write("MUSIC GENRE CLASSIFIER - EVALUATION RESULTS\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Overall Accuracy   : {overall_accuracy:.2f}%\n")
    file.write(f"Balanced Accuracy : {balanced_accuracy:.2f}%\n")
    file.write(f"Top-2 Accuracy    : {top2_accuracy:.2f}%\n")
    file.write(f"Top-3 Accuracy    : {top3_accuracy:.2f}%\n\n")

    file.write("Classification Report:\n")
    file.write(report + "\n")

    file.write("\nPer-Genre Accuracy:\n")

    for genre, accuracy in sorted(
        per_genre_accuracy.items(),
        key=lambda x: -x[1]
    ):
        file.write(f"{genre:<12} : {accuracy:.2f}%\n")

print("\nEvaluation complete.")
print(f"Results saved in: {OUTPUT_DIR}")