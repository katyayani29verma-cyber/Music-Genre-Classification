import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- SETTINGS ----
GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]

# ---- STEP 1: Load the balanced dataset ----
print("Loading dataset...")
X = np.load("X_balanced.npy")
y = np.load("y_balanced.npy")

print(f"Dataset loaded! Shape: {X.shape}")
print(f"Total clips: {len(X)}")
# ---- STEP 2: Flatten the spectrograms ----
# Random Forest cannot handle 2D input (128x1292)
# It needs each clip as one flat row of numbers
# 128 x 1292 = 165,536 numbers per clip

X_flat = X.reshape(X.shape[0], -1)

print(f"Flattened shape: {X_flat.shape}")
# Should print (900, 165536)
# ---- STEP 3: Split into train and test ----
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, 
    test_size=0.2,      # 20% for testing = 180 clips
    random_state=42,    # fixed seed so results are reproducible
    stratify=y          # ensures each genre is equally represented in both splits
)

print(f"Training samples: {len(X_train)}")  # should be 720
print(f"Testing samples: {len(X_test)}")    # should be 180

# ---- STEP 4: Train the Random Forest ----
print("\nTraining Random Forest...")
print("This may take 1-2 minutes, please wait...")

rf_model = RandomForestClassifier(
    n_estimators=100,   # build 100 decision trees
    random_state=42,    # reproducible results
    n_jobs=-1           # use all CPU cores to train faster
)

rf_model.fit(X_train, y_train)

print("Training complete! ✅")