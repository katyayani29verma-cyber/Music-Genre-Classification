import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- SETTINGS ----
DATASET_PATH = "C:/Users/Katyayani Verma/OneDrive/Desktop/my_music"
GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]

# ---- STEP 1: Load one audio file ----
genre = "bollywood_romantic"
folder = os.path.join(DATASET_PATH, genre)

filename = os.listdir(folder)[0]
filepath = os.path.join(folder, filename)

print(f"Loading: {filename}")

# Load full song (no duration limit)
y, sr = librosa.load(filepath)

print(f"Audio shape: {y.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(y)/sr:.2f} seconds")

# ---- STEP 2: Visualize single waveform ----
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title(f"Waveform - {genre} - {filename[:30]}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_single.png")  # saves the plot
plt.show()

# ---- STEP 3: Compare waveforms across all genres ----
plt.figure(figsize=(15, 10))

for i, genre in enumerate(GENRES):
    folder = os.path.join(DATASET_PATH, genre)
    filename = os.listdir(folder)[0]
    filepath = os.path.join(folder, filename)
    
    print(f"Loading {genre}: {filename[:40]}")
    y, sr = librosa.load(filepath)  # full song
    
    plt.subplot(5, 1, i+1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(genre)
    plt.tight_layout()

plt.savefig("waveform_comparison.png")  # saves the plot
plt.show()

print("\nDay 2 Complete!")
print("Saved: waveform_single.png")
print("Saved: waveform_comparison.png")