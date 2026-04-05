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

# ---- STEP 3: Build the CNN ----
model = Sequential([
    
    # --- Block 1: First set of filters ---
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128, 1292, 1)),
    # 32 filters, each 3x3 pixels, scanning the spectrogram for basic patterns
    
    BatchNormalization(),
    # Stabilizes the numbers after convolution
    
    MaxPooling2D((2,2)),
    # Shrinks the image by half (128x1292 → 64x646)
    # Keeps the most important features, reduces computation
    
    # --- Block 2: More complex patterns ---
    Conv2D(64, (3,3), activation='relu', padding='same'),
    # 64 filters now — looking for more complex patterns
    
    BatchNormalization(),
    
    MaxPooling2D((2,2)),
    # Shrinks again (64x646 → 32x323)
    
    # --- Block 3: Even more complex ---
    Conv2D(128, (3,3), activation='relu', padding='same'),
    # 128 filters — finding very complex genre-specific patterns
    
    BatchNormalization(),
    
    MaxPooling2D((2,2)),
    # Shrinks again (32x323 → 16x161)
    
    # --- Flatten and Decide ---
    Flatten(),
    # Converts 2D feature maps into 1D — same as Random Forest flattening!
    
    Dense(256, activation='relu'),
    # Fully connected layer — combines all patterns to make decision
    
    Dropout(0.5),
    # Randomly turns off 50% of neurons during training
    # Prevents overfitting — forces model to not rely on any single neuron
    
    Dense(NUM_CLASSES, activation='softmax')
    # Output layer — 5 neurons, one per genre
    # Softmax converts to probabilities that add up to 100%
    # e.g. [0.05, 0.02, 0.85, 0.05, 0.03] → RAP with 85% confidence!
])

model.summary()