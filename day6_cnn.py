import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---- SETTINGS ----
GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]
NUM_CLASSES = 5

# ---- STEP 1: Load data ----
print("Loading dataset...")
X = np.load("X_balanced.npy")
y = np.load("y_balanced.npy")

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ---- STEP 2: Prepare data for CNN ----

# CNN needs a 4D input: (samples, height, width, channels)
# Like images: (900, 128, 1292, 1) — 1 channel because spectrogram is grayscale
X = X[..., np.newaxis]
print(f"Reshaped for CNN: {X.shape}")

# Normalize values to 0-1 range
# Spectrogram values are in dB (negative numbers like -80 to 0)
# Normalizing helps the model train faster and more stably
X = (X - X.min()) / (X.max() - X.min())
print(f"Normalized! Min: {X.min():.2f}, Max: {X.max():.2f}")

# Convert labels to one-hot encoding
# e.g. genre 2 (rap) becomes [0, 0, 1, 0, 0]
y_categorical = to_categorical(y, num_classes=NUM_CLASSES)
print(f"Labels shape after one-hot: {y_categorical.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")