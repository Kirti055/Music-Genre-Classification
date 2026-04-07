import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, Callback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ================================================================
#  CONFIG
# ================================================================

DATASET_PATH   = "./Data/combined_dataset"
CHECKPOINT_DIR = "./checkpoints"
LAST_MODEL     = os.path.join(CHECKPOINT_DIR, "last_model.h5")
BEST_MODEL     = os.path.join(CHECKPOINT_DIR, "best_model.h5")
STATE_PATH     = os.path.join(CHECKPOINT_DIR, "training_state.pkl")
FINAL_MODEL    = "music_genre_model.h5"
ENCODER_PATH   = "label_encoder.pkl"

SAMPLE_RATE      = 22050
SEGMENT_DURATION = 3        # seconds — short segments = more training data
N_MELS           = 64       # smaller = faster on CPU
EPOCHS           = 50
BATCH_SIZE       = 16       # CPU-friendly
LEARNING_RATE    = 0.0005

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
#  CHECKPOINT RESUME
#  If laptop shuts down mid-training, just re-run python train.py
# ================================================================

def save_state(epoch):
    with open(STATE_PATH, "wb") as f:
        pickle.dump(epoch, f)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    return 0

class EpochTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_state(epoch + 1)
        val_acc = logs.get("val_accuracy", 0) * 100
        trn_acc = logs.get("accuracy", 0) * 100
        print(f"  💾 Saved | Train: {trn_acc:.1f}% | Val: {val_acc:.1f}%")


# ================================================================
#  FEATURE EXTRACTION
#  Mel Spectrogram — fast, proven, CPU-friendly
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

    # Normalize — makes model consistent across different audio loudness
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Fix to square shape (N_MELS x N_MELS)
    mel_db = librosa.util.fix_length(mel_db, size=N_MELS, axis=1)

    return mel_db  # shape: (64, 64)


# ================================================================
#  DATA AUGMENTATION
#  Artificially increases dataset size — main weapon vs overfitting
# ================================================================

def augment(mel):
    mel = mel.copy()

    # 1. Time shift — slide spectrogram left/right
    shift = np.random.randint(-5, 5)
    mel = np.roll(mel, shift, axis=1)

    # 2. Add tiny random noise
    mel += np.random.normal(0, 0.01, mel.shape)

    # 3. Frequency masking — zero out a random frequency band
    mask_size = np.random.randint(0, N_MELS // 8)
    mask_start = np.random.randint(0, N_MELS - mask_size)
    mel[mask_start:mask_start + mask_size, :] = 0

    return mel


# ================================================================
#  LOAD DATA
# ================================================================

print("\n📂 Loading Dataset...\n")

X, y = [], []

genres = sorted([
    g for g in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, g))
])

print(f"Found {len(genres)} genres: {genres}\n")

for genre in genres:
    print(f"  ⏳ Processing: {genre}")
    genre_path = os.path.join(DATASET_PATH, genre)
    count = 0

    for file in os.listdir(genre_path):
        if not (file.lower().endswith(".wav") or file.lower().endswith(".mp3")):
            continue
        try:
            audio, sr = librosa.load(
                os.path.join(genre_path, file),
                sr=SAMPLE_RATE,
                mono=True
            )

            segment_len = SEGMENT_DURATION * sr
            hop = segment_len // 2  # 50% overlap = more segments per file

            for i in range(0, len(audio) - segment_len, hop):
                segment = audio[i:i + segment_len]
                features = extract_features(segment, sr)

                # Original
                X.append(features)
                y.append(genre)

                # Augmented copy — doubles training data for free
                X.append(augment(features))
                y.append(genre)

                count += 1

        except Exception as e:
            print(f"    ⚠  Skipped {file}: {e}")

    print(f"    ✅ {count} segments from {genre}")

X = np.array(X)
y = np.array(y)

# Reshape for CNN input: (samples, height, width, channels)
X = X.reshape(X.shape[0], N_MELS, N_MELS, 1)

print(f"\n✅ Total samples (with augmentation): {len(X)}")
print(f"✅ Input shape: {X.shape}")


# ================================================================
#  ENCODE LABELS
# ================================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

pickle.dump(encoder, open(ENCODER_PATH, "wb"))
print(f"✅ Genres saved: {list(encoder.classes_)}")


# ================================================================
#  CLASS WEIGHTS
#  Handles imbalance between GTZAN (100 files) and YT (few files)
# ================================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(weights))


# ================================================================
#  TRAIN / TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}\n")


# ================================================================
#  MODEL
#  Pure CNN — faster than CNN+LSTM on CPU, less prone to overfit
#
#  Anti-overfitting techniques used:
#  1. Dropout after every block
#  2. L2 regularization on all Conv layers
#  3. GlobalAveragePooling (way fewer params than Flatten)
#  4. Data augmentation (done above)
#  5. Class weights (done above)
# ================================================================

def create_model(num_classes):
    model = Sequential([

        # Block 1 — learns basic edges and textures
        Conv2D(32, (3,3), activation="relu", padding="same",
               kernel_regularizer=l2(0.001),
               input_shape=(N_MELS, N_MELS, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2 — learns mid-level patterns
        Conv2D(64, (3,3), activation="relu", padding="same",
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3 — learns high-level genre features
        Conv2D(128, (3,3), activation="relu", padding="same",
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # GlobalAveragePooling — replaces Flatten
        # averages each feature map to 1 number
        # massively reduces parameters = less overfitting
        GlobalAveragePooling2D(),

        # Final classifier
        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model


# ================================================================
#  LOAD OR CREATE MODEL
# ================================================================

start_epoch = load_state()

if os.path.exists(LAST_MODEL) and start_epoch > 0:
    print(f"🔄 Resuming training from epoch {start_epoch}...\n")
    model = load_model(LAST_MODEL)
else:
    print("🚀 Starting fresh training...\n")
    model = create_model(len(genres))

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ================================================================
#  CALLBACKS
# ================================================================

callbacks = [
    # Save after every epoch (enables resume after shutdown)
    ModelCheckpoint(LAST_MODEL, save_best_only=False, verbose=0),

    # Save only when val_accuracy improves
    ModelCheckpoint(BEST_MODEL, monitor="val_accuracy",
                    save_best_only=True, verbose=1),

    # Stop training if val_accuracy doesn't improve for 10 epochs
    EarlyStopping(monitor="val_accuracy", patience=10,
                  restore_best_weights=True, verbose=1),

    # Halve learning rate if val_loss stalls for 5 epochs
    ReduceLROnPlateau(monitor="val_loss", patience=5,
                      factor=0.5, min_lr=1e-6, verbose=1),

    # Saves epoch number for resume
    EpochTracker()
]


# ================================================================
#  TRAIN
# ================================================================

print("🏋️  Training started...")
print("💡 If laptop shuts down — just run python train.py again!\n")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    initial_epoch=start_epoch,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)


# ================================================================
#  SAVE FINAL + EVALUATE
# ================================================================

model.save(FINAL_MODEL)
print(f"\n✅ Final model saved: {FINAL_MODEL}")

print("\n📊 Evaluation on Test Set:\n")
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(classification_report(
    true_classes, pred_classes,
    target_names=encoder.classes_
))
print("Confusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))
print("\n🎉 TRAINING COMPLETE!")