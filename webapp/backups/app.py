from flask import Flask, render_template, jsonify, request
from threading import Timer
import time
import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line to apply CORS

crossing_time = 10
end_time = 0
id = 0


detection_buffer = []
buffer_length = 10

# Load your SavedModel
PATH_TO_SAVED_MODEL = "./exported-models/my_model/saved_model"
print(PATH_TO_SAVED_MODEL)
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
# model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL)
class_names = {1: 'blind_aid', 2: 'walking_aid', 3:'wheelchair', 4: 'walking_aid_1'}

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
        # print("Hi")
        # Preprocess the image
        input_image = preprocess_image(image)

        # Run object detection using the trained model
        predictions = detect_fn(input_image)  # Changed 'detect_fn.predict' to 'detect_fn'

        # Post-process the predictions and create the response
        response = postprocess_predictions(predictions)

        return jsonify([response])

def preprocess_image(image):
    # Convert PIL image to OpenCV format
    global id
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print(frame.shape)
    id = id+ 1
    # cv2.imwrite(f"./images/image{id}.jpg", frame)


    # Preprocess frame
    # frame = cv2.flip(frame, 1)
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    return input_tensor

def postprocess_predictions(predictions):
    # Add your own post-processing steps and create the response here
    score_thresh = 0.4  # Minimum threshold for object detection
    max_detections = 1

    scores = predictions['detection_scores'][0, :max_detections].numpy()
    bboxes = predictions['detection_boxes'][0, :max_detections].numpy()
    labels = predictions['detection_classes'][0, :max_detections].numpy().astype(np.int64)

    print(labels, scores)

    # Replace this with your actual class names
    labels = [class_names[n] for n in labels]

    # Store the current detections in the buffer
    current_detections = {'scores': scores.tolist(), 'bboxes': bboxes.tolist(), 'labels': labels}
    detection_buffer.append(current_detections)

    # Keep only the last buffer_length detections
    if len(detection_buffer) > buffer_length:
        detection_buffer.pop(0)

    # Compute the moving average of detections
    averaged_detections = compute_moving_average(detection_buffer)

    return averaged_detections

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
