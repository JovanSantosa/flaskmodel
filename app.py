# app.py

import os
from flask import Flask, request, jsonify # Hapus render_template jika tidak dipakai lagi
from flask_cors import CORS # <-- 1. IMPORT CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app) # <-- 2. AKTIFKAN CORS UNTUK SEMUA ROUTE

# ... (sisa kode Anda tetap sama) ...

# Muat model Keras Anda
MODEL_PATH = 'model_mata_ikan_fresh_nonfresh_finetuned.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Endpoint untuk prediksi (tidak perlu ada route '/' lagi jika hanya untuk API)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        image = Image.open(file.stream)
        # Sesuaikan target_size dengan input model Anda, misal (224, 224)
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)

        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        class_names = ['Tidak Fresh', 'Fresh'] 
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)