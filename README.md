# 🎵 Wavely - Music Genre Classification

> Drop a track. Discover its soul.

Welcome to Wavely — your intelligent music companion that uploads any song and lets our AI instantly identify its genre and recommend music that matches your vibe. Because every wave tells a story, and Wavely helps you ride yours.

---

## 🎯 About the Project

Wavely is a Music Genre Classification system that uses Machine Learning to identify the genre of any uploaded song across 5 categories:

- 🎸 Rock
- 🎻 Classical
- 🎤 Rap
- 🎶 Western Pop
- 🎵 Bollywood Romantic

> ⚠️ **This project is currently a work in progress!**

---

## 🗺️ Project Roadmap

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Data Collection | ✅ Done |
| Day 2 | Waveform Visualization | ✅ Done |
| Day 3 | Spectrogram Generation | ✅ Done |
| Day 4 | Dataset Building & Balancing | ✅ Done |
| Day 5 | First Model - Random Forest | 🔄 In Progress |
| Day 6 | Deep Learning - CNN Model | ⏳ Upcoming |
| Day 7 | Model Evaluation & Improvement | ⏳ Upcoming |
| Day 8 | Web Application (Flask) | ⏳ Upcoming |

---

## 🧠 Approach

### Phase 1 — Random Forest (Baseline)
Starting with a Random Forest classifier as a baseline model to validate the dataset quality and get an initial accuracy score. Random Forest is fast to train and gives quick feedback.

### Phase 2 — CNN (Deep Learning)
Upgrading to a Convolutional Neural Network since spectrograms are essentially images. CNNs are significantly better at finding spatial patterns in image-like data and are expected to give much higher accuracy.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `librosa` | Audio loading & feature extraction |
| `numpy` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | Random Forest model |
| `TensorFlow/Keras` | CNN model (upcoming) |
| `Flask` | Web application (upcoming) |

---

## 📁 Project Structure



---

## 📊 Dataset

- **5 genres** — Rock, Classical, Rap, Western Pop, Bollywood Romantic
- **180 clips per genre** — balanced dataset
- **900 total clips** — each 30 seconds long
- **Features** — Mel Spectrograms (128 × 1292)

---

*Built with 🎵 and lots of debugging*