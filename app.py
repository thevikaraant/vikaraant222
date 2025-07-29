from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

@app.route('/sketch', methods=['POST'])
def sketch():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {'error': 'Invalid image'}, 400

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sketch = cv2.Canny(gray, 50, 150)

    # Save to temp file and return
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, sketch)
    
    return send_file(temp_file.name, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
