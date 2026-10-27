import os
import copy
import itertools
import string
import threading
import time
import warnings
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from tensorflow import keras
import mediapipe as mp

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

print("[ISL App] Loading TensorFlow model...")
model = keras.models.load_model("model.h5", compile=False)
print("[ISL App] Model loaded successfully!")

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# MediaPipe setup for backend OpenCV video stream
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

server_cam = None
server_cam_lock = threading.Lock()
server_cam_active = False

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) or 1e-6

    def normalize_(n):
        return n / max_value

    return list(map(normalize_, temp_landmark_list))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        "status": "online",
        "classes": alphabet,
        "total_classes": len(alphabet),
        "input_features": 42,
        "framework": "TensorFlow / Keras + MediaPipe",
        "architecture": [
            {"layer": "Dense (1024)", "activation": "ReLU"},
            {"layer": "Dropout (0.2)"},
            {"layer": "Dense (632)", "activation": "ReLU"},
            {"layer": "Dropout (0.2)"},
            {"layer": "Dense (328)", "activation": "ReLU"},
            {"layer": "Dropout (0.2)"},
            {"layer": "Dense (152)", "activation": "ReLU"},
            {"layer": "Dropout (0.2)"},
            {"layer": "Dense (35)", "activation": "Softmax"}
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        if 'vector' in data:
            processed = data['vector']
        elif 'landmarks' in data:
            raw_landmarks = data['landmarks']
            if len(raw_landmarks) != 21:
                return jsonify({"error": "Expected 21 landmarks"}), 400
            processed = pre_process_landmark(raw_landmarks)
        else:
            return jsonify({"error": "Missing 'landmarks' or 'vector' key"}), 400

        df = pd.DataFrame(processed).transpose()
        predictions = model.predict(df, verbose=0)[0]
        
        sorted_indices = np.argsort(predictions)[::-1]
        top_idx = int(sorted_indices[0])
        top_label = alphabet[top_idx]
        top_confidence = float(predictions[top_idx])

        top5 = []
        for i in range(min(5, len(alphabet))):
            idx = int(sorted_indices[i])
            top5.append({
                "label": alphabet[idx],
                "confidence": round(float(predictions[idx]) * 100, 2)
            })

        all_scores = {alphabet[i]: round(float(predictions[i]) * 100, 2) for i in range(len(alphabet))}

        return jsonify({
            "success": True,
            "label": top_label,
            "confidence": round(top_confidence * 100, 2),
            "top5": top5,
            "all_scores": all_scores
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def generate_video_frames():
    global server_cam, server_cam_active
    with server_cam_lock:
        if server_cam is None or not server_cam.isOpened():
            server_cam = cv2.VideoCapture(0)
        server_cam_active = True

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while server_cam_active and server_cam and server_cam.isOpened():
            success, image = server_cam.read()
            if not success:
                time.sleep(0.03)
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image.flags.writeable = True

            debug_image = copy.deepcopy(image)
            detected_label = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed = pre_process_landmark(landmark_list)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    df = pd.DataFrame(pre_processed).transpose()
                    predictions = model.predict(df, verbose=0)[0]
                    predicted_classes = np.argmax(predictions)
                    detected_label = alphabet[predicted_classes]
                    confidence = float(predictions[predicted_classes])

                    # Draw clean UI badge on the video frame
                    h, w = image.shape[:2]
                    cv2.rectangle(image, (15, 15), (230, 85), (255, 255, 255), -1)
                    cv2.rectangle(image, (15, 15), (230, 85), (235, 99, 37), 2)
                    cv2.putText(image, f"Sign: {detected_label}", (28, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (15, 23, 42), 2)
                    cv2.putText(image, f"Conf: {confidence*100:.1f}%", (28, 77),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 70, 20), 1)

            ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/<action>', methods=['POST'])
def manage_camera(action):
    global server_cam, server_cam_active
    with server_cam_lock:
        if action == 'stop':
            server_cam_active = False
            if server_cam and server_cam.isOpened():
                server_cam.release()
                server_cam = None
            return jsonify({"status": "stopped"})
        elif action == 'start':
            server_cam_active = True
            if server_cam is None or not server_cam.isOpened():
                server_cam = cv2.VideoCapture(0)
            return jsonify({"status": "started"})
    return jsonify({"error": "Unknown action"}), 400

if __name__ == '__main__':
    print("[ISL Web Server] Running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
