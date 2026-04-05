import numpy as np

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]

min_clips = min([np.sum(y == i) for i in range(len(GENRES))])
print(f"Balancing all genres to: {min_clips} clips each")

X_balanced, y_balanced = [], []

for i in range(len(GENRES)):
    idx = np.where(y == i)[0][:min_clips]
    X_balanced.append(X[idx])
    y_balanced.append(y[idx])

X_balanced = np.concatenate(X_balanced)
y_balanced = np.concatenate(y_balanced)

print(f"Balanced dataset shape: {X_balanced.shape}")
print(f"Total clips: {len(X_balanced)}")

np.save("X_balanced.npy", X_balanced)
np.save("y_balanced.npy", y_balanced)

print("Saved! ✅")