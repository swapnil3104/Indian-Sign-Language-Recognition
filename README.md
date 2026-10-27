# Indian Sign Language (ISL) Recognition AI & Web Studio

Real-time Indian Sign Language recognition using Deep Learning, MediaPipe, OpenCV, and TensorFlow. The system detects and translates 35 signs representing alphabets (**A–Z**) and digits (**1–9**) from webcam feeds, with interactive web translation, speech synthesis, and gamified practice quizzes.

<p align="center">
  <img src="assets/Screenshot 2026-10-27 112927.png" alt="Indian Sign Language Real-Time Recognition Demo" width="900" style="border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.15);" />
</p>

---

## 📸 Project Showcase

| Live Gesture Tracker & Sentence Builder | Interactive Sign Dictionary |
|:---:|:---:|
| <img src="assets/Screenshot 2026-10-27 112927.png" width="450" /> | <img src="assets/sign_dictionary.png" width="450" /> |
| **Real-Time Recognition & Audio TTS** | **Searchable 35 Signs (Letters, Digits, Vowels)** |

| Practice & Quiz Arena | Model Architecture & Keypoint Topology |
|:---:|:---:|
| <img src="assets/practice_arena.png" width="450" /> | <img src="assets/architecture.png" width="450" /> |
| **Interactive Signing Challenges & Scoring** | **MediaPipe 21 Hand Landmarks & Deep NN** |

---

## 🌟 Features

- **Real-Time Hand Landmark Tracking**: 21 3D hand keypoints extracted via MediaPipe with responsive glowing neon skeleton overlays.
- **Deep Neural Network Classification**: Feedforward classifier trained on normalized relative coordinates, recognizing 35 classes with top-5 confidence ranking.
- **Interactive Web Application**:
  - **Live Studio**: Real-time webcam gesture tracking with dual camera modes (Browser Camera or OpenCV Server Stream).
  - **Sentence Translator**: Live accumulation of recognized letters into sentences with Auto-type on hold, space, backspace, and copy.
  - **Text-to-Speech (TTS)**: Speak recognized text aloud using the Web Speech API.
  - **Sign Dictionary**: Complete searchable catalog of all 35 gestures with category filters (Numbers, Letters, Vowels).
  - **Official Gesture Chart**: High-resolution posture reference modal.
  - **Practice & Quiz Arena**: Gamified challenge mode with timer, scoring, and streak tracking.
  - **Architecture Deep-Dive**: MediaPipe 21 hand topology diagram and neural network specs.

---

## 🤟 Official ISL Gestures Reference

The model recognizes 35 distinct gestures covering **Digits 1–9** and **Alphabets A–Z**:

<p align="center">
  <img src="images/allGestures.png" alt="ISL Gestures for Numeric and Alphabets" width="850" style="border-radius: 8px;" />
  <br />
  <em>Figure: Official Indian Sign Language (ISL) Gestures for Numeric and Alphabets</em>
</p>

---

## 🚀 Quick Start

### 1. Launch the Web Application (Recommended)
Double-click:
```bat
run_website.bat
```
Or run from your terminal:
```powershell
"C:\Users\Asus\anaconda3\envs\isl\python.exe" app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your web browser.

### 2. Run the OpenCV Desktop Window
Double-click:
```bat
run.bat
```
Or run:
```powershell
"C:\Users\Asus\anaconda3\envs\isl\python.exe" isl_detection.py
```
*(Press **ESC** on the camera window to exit)*

---

## 🧠 Model Architecture

- **Input**: 42 normalized features (21 landmarks &times; 2 relative coordinates from wrist)
- **Architecture**:
  - Dense(1024, ReLU) &rarr; Dropout(0.2)
  - Dense(632, ReLU) &rarr; Dropout(0.2)
  - Dense(328, ReLU) &rarr; Dropout(0.2)
  - Dense(152, ReLU) &rarr; Dropout(0.2)
  - Dense(35, Softmax)
- **Total Parameters**: 954,819 parameters (~3.64 MB)
- **Classes**: Digits 1–9, Alphabets A–Z
