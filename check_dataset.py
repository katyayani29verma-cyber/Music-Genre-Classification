import numpy as np

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]

print(f"Dataset shape: {X.shape}")
print(f"Total clips: {len(X)}")
print(f"\nClips per genre:")
for i, genre in enumerate(GENRES):
    count = np.sum(y == i)
    print(f"  {genre}: {count} clips")