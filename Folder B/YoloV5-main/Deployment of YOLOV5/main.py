from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import io
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection using the YOLO model
    # Replace this section with your own YOLO code
    # Here's a placeholder that returns the image with a bounding box drawn on it
    # Note: This is just an example and may not work as is.
    #       You'll need to adapt it to your specific YOLO implementation.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="D:\YOLOV5/best.pt",
                           force_reload=True)
    results = model(image)
    detected_image = image.copy()

    # Get the detected labels from the model
    labels = results.pandas().xyxy[0]['name'].tolist()
    boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    # Draw bounding boxes and labels on the image
    for label, box in zip(labels, boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(detected_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(detected_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes and labels on the image
    for label in labels:
        cv2.putText(detected_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', detected_image)

    # Create an in-memory stream to store the image data
    stream = io.BytesIO()
    stream.write(img_encoded.tobytes())
    stream.seek(0)

    return send_file(stream, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
