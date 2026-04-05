import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- SETTINGS ----
DATASET_PATH = "C:/Users/Katyayani Verma/OneDrive/Desktop/my_music"
GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]

# ---- STEP 1: Load one song ----
genre = "bollywood_romantic"
folder = os.path.join(DATASET_PATH, genre)
filename = os.listdir(folder)[0]
filepath = os.path.join(folder, filename)

print(f"Loading: {filename}")
y, sr = librosa.load(filepath, duration=30)

# ---- STEP 2: Generate Mel Spectrogram ----
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibels (log scale) — makes it more readable
mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

print(f"Mel Spectrogram shape: {mel_spect_db.shape}")

# ---- STEP 3: Visualize it ----
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Mel Spectrogram - {genre}")
plt.tight_layout()
plt.savefig("spectrogram_single.png")
plt.show()
# ---- STEP 4: Compare spectrograms across all genres ----
plt.figure(figsize=(15, 12))

for i, genre in enumerate(GENRES):
    folder = os.path.join(DATASET_PATH, genre)
    filename = os.listdir(folder)[0]
    filepath = os.path.join(folder, filename)
    
    y, sr = librosa.load(filepath, duration=30)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    plt.subplot(5, 1, i+1)
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(genre)
    plt.tight_layout()

plt.savefig("spectrogram_comparison.png")
plt.show()