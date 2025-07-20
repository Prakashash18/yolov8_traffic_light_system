from flask import Flask, render_template, jsonify, request
from threading import Timer
import time
import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS

crossing_time = 10
end_time = 0
id = 0

detection_buffer = []
buffer_length = 10

# Load your YOLO model
PATH_TO_SAVED_MODEL = "./exported-models/Traffic_best_20Nov.pt"
model = YOLO(PATH_TO_SAVED_MODEL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crossing', methods=['POST'])
def handle_crossing():
    global end_time
    current_time = time.time()

    if current_time > end_time:
        end_time = current_time + crossing_time
        return jsonify({'change_light': 'green', 'timer': crossing_time})
    else:
        remaining_time = int(end_time - current_time)
        return jsonify({'change_light': 'red', 'timer': remaining_time})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Read the image data from the request
        image_data = request.form['image_data']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Preprocess image
        rgb_image = preprocess_image(image)
        frame = np.array(rgb_image)
        print("Am here : ", frame.shape)

        # Predict on GPU (CUDA) if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        results = model(frame, device=device)

        # Post-process the predictions and return the response
        response = postprocess_predictions(results)
        return jsonify([response])

def preprocess_image(image):
    global id
    rgb_image = image.convert('RGB')
    return rgb_image

def postprocess_predictions(results):
    max_detections = 1

    result = results[0]
    bboxes = result.boxes.xyxy
    scores = np.array(result.boxes.conf.cpu(), dtype="float")
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    labels = np.array(result.boxes.cls.cpu(), dtype="int")

    print(labels, scores)

    labels = [str(model.names[int(n)]) for n in labels]

    current_detections = {
        'scores': scores.tolist(),
        'bboxes': bboxes.tolist(),
        'labels': labels
    }

    return current_detections

def compute_moving_average(buffer):
    scores_sum = np.zeros(len(buffer[0]['scores']))
    bboxes_sum = np.zeros((len(buffer[0]['bboxes']), 4))
    labels_count = {}

    for detection in buffer:
        scores_sum += np.array(detection['scores'])
        bboxes_sum += np.array(detection['bboxes'])
        for label in detection['labels']:
            labels_count[label] = labels_count.get(label, 0) + 1

    scores_avg = scores_sum / len(buffer)
    bboxes_avg = bboxes_sum / len(buffer)
    labels_avg = [key for key, value in labels_count.items() if value >= len(buffer) // 2]

    return {'scores': scores_avg.tolist(), 'bboxes': bboxes_avg.tolist(), 'labels': labels_avg}

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
