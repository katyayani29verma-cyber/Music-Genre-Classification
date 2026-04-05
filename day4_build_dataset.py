import librosa
import numpy as np
import os

# ---- SETTINGS ----
DATASET_PATH = "C:/Users/Katyayani Verma/OneDrive/Desktop/my_music"
GENRES = ["rock", "classical", "rap", "western_pop", "bollywood_romantic"]
CLIP_DURATION = 30
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_SONG_DURATION = 600  # max 10 minutes per song

# ---- STORAGE ----
X = []
y = []

print("Starting dataset build...\n")

# ---- LOOP THROUGH ALL GENRES AND SONGS ----
for genre_idx, genre in enumerate(GENRES):
    folder = os.path.join(DATASET_PATH, genre)
    files = os.listdir(folder)
    
    print(f"Processing genre: {genre} ({len(files)} songs)")
    
    for filename in files:
        filepath = os.path.join(folder, filename)
        
        try:
            # Get total duration
            duration = librosa.get_duration(path=filepath)
            
            # Skip songs longer than 10 minutes
            if duration > MAX_SONG_DURATION:
                print(f"  Skipping {filename[:40]} — too long ({duration/60:.1f} mins)")
                continue
            
            # Split song into 30 second clips
            num_clips = int(duration // CLIP_DURATION)
            
            for clip_idx in range(num_clips):
                start = clip_idx * CLIP_DURATION
                
                audio, sr = librosa.load(filepath, offset=start, duration=CLIP_DURATION, sr=SR)
                
                mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
                mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
                
                X.append(mel_spect_db)
                y.append(genre_idx)
        
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
            continue
    
    print(f"  Done! Total clips so far: {len(X)}\n")

# ---- CONVERT TO NUMPY ARRAYS ----
X = np.array(X)
y = np.array(y)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Total clips: {len(X)}")
print(f"Clips per genre breakdown:")
for i, genre in enumerate(GENRES):
    count = np.sum(y == i)
    print(f"  {genre}: {count} clips")

# ---- SAVE TO DISK ----
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nSaved X_features.npy and y_labels.npy")
print("Day 4 Complete! ✅")