from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import keras
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS if calling from a JS frontend

# Logging setup
logging.basicConfig(level=logging.INFO)



import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # current script directory
    'saved_models',
    'cnn_alphanumeric_model_20250601_062953.h5'
)
model = keras.models.load_model(MODEL_PATH)





# Define class labels (digits 0–9 and uppercase A–Z)
class_names = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

def preprocess_image(image_data):
    """
    Convert base64-encoded image into a processed 28x28 grayscale NumPy array
    suitable for model input.
    """
    try:
        image_data = image_data.split(',')[1]  # Remove base64 prefix
        image_bytes = base64.b64decode(image_data)
    except Exception:
        raise ValueError("Invalid image format (not proper base64)")

    # Open and process image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background

    image = image.convert('L')  # Grayscale
    image_array = np.array(image)

    # Crop bounding box of drawn content
    threshold = 240
    dark_pixels = image_array < threshold
    if np.any(dark_pixels):
        coords = np.column_stack(np.where(dark_pixels))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        padding = 20
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(image_array.shape[0], y_max + padding)
        x_max = min(image_array.shape[1], x_max + padding)

        cropped = image_array[y_min:y_max, x_min:x_max]

        # Make it square with white padding
        h, w = cropped.shape
        size = max(h, w)
        square = np.ones((size, size), dtype=np.uint8) * 255
        start_y = (size - h) // 2
        start_x = (size - w) // 2
        square[start_y:start_y+h, start_x:start_x+w] = cropped
        image_array = square

    # Resize to 28x28
    image_array = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert: black background, white text
    image_array = 255 - image_array

    # Normalize and enhance contrast
    image_array = image_array.astype('float32') / 255.0
    image_array = np.clip(image_array * 1.2, 0, 1)

    # Reshape for model
    return image_array.reshape(1, 28, 28, 1)

@app.route('/')
def index():
    return render_template('index.html')  # Make sure templates/index.html exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        processed_image = preprocess_image(data['image'])
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        predicted_character = class_names[predicted_class]

        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {
                'character': class_names[i],
                'confidence': float(prediction[0][i])
            } for i in top_3_indices
        ]

        return jsonify({
            'success': True,
            'prediction': predicted_character,
            'confidence': confidence,
            'top_3': top_3_predictions
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug_image', methods=['POST'])
def debug_image():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        processed_image = preprocess_image(data['image'])
        debug_image = (processed_image[0, :, :, 0] * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', debug_image)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'debug_image': f'data:image/png;base64,{debug_base64}'
        })

    except Exception as e:
        logging.exception("Debug image failed")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
