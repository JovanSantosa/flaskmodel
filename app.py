# app.py

import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Muat model Keras Anda
# Pastikan nama file model sudah sesuai
MODEL_PATH = 'model_mata_ikan_fresh_nonfresh_finetuned.h5'
model = load_model(MODEL_PATH)

# Fungsi untuk memproses gambar sebelum dimasukkan ke model
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalisasi jika model Anda memerlukannya
    return image

# Halaman utama untuk upload file
@app.route('/', methods=['GET'])
def index():
    # Render halaman HTML untuk upload
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        # Buka file gambar
        image = Image.open(file.stream)

        # Proses gambar (sesuaikan target_size dengan input model Anda)
        # Ganti (224, 224) dengan ukuran yang sesuai, contoh: (150, 150)
        processed_image = preprocess_image(image, target_size=(224, 224))

        # Lakukan prediksi
        prediction = model.predict(processed_image)
        
        # Ambil hasil prediksi
        # Ini mungkin perlu disesuaikan tergantung output model Anda
        # Misalnya, jika output adalah [0.9, 0.1], kita ambil index dengan nilai tertinggi
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # Definisikan nama kelas sesuai urutan pada model Anda
        # Contoh: 0 = 'Tidak Fresh', 1 = 'Fresh'
        class_names = ['Tidak Fresh', 'Fresh'] 
        predicted_class_name = class_names[predicted_class_index]
        
        # Kirim hasil dalam format JSON
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan port yang disediakan oleh Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)