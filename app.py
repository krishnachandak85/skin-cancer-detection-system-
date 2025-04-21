from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from threading import Thread
import time
import requests

app = Flask(__name__)

# === Get base directory from user or env variable ===
BASE_DIR = os.environ.get("ENV_DIR", ".")  # "." means current directory if ENV_DIR not set

# Define paths (relative to the base directory)
reference_paths = {
    "Melanoma": os.path.join(BASE_DIR, "Melanoma"),
    "Basal Cell Carcinoma": os.path.join(BASE_DIR, "Basal Cell Carcinoma"),
    "Squamous Cell Carcinoma": os.path.join(BASE_DIR, "Squamous Cell Carcinoma"),
    "Benign": os.path.join(BASE_DIR, "Benign")
}

upload_folder = os.path.join(BASE_DIR, "uploads")
os.makedirs(upload_folder, exist_ok=True)

# Load reference images
def load_reference_images():
    reference_images = {}
    for cancer_type, folder_path in reference_paths.items():
        images = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(folder_path, file))
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        images.append(gray)
        reference_images[cancer_type] = images
    return reference_images

def compute_similarity(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

def detect_skin_cancer(img_path, reference_images):
    img = cv2.imread(img_path)
    if img is None:
        return "Image not readable", 0
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    similarities = {}
    for cancer_type, ref_imgs in reference_images.items():
        if ref_imgs:
            similarities[cancer_type] = max(compute_similarity(gray, ref_img) for ref_img in ref_imgs)
        else:
            similarities[cancer_type] = 0

    best_cancer = max(similarities, key=similarities.get)
    similarity_score = similarities[best_cancer] * 100
    return best_cancer, similarity_score

# Load reference images once
reference_images = load_reference_images()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    cancer_type, similarity = detect_skin_cancer(filepath, reference_images)

    return jsonify({
        "filename": filename,
        "cancer_type": cancer_type,
        "similarity_percent": round(similarity, 2)
    })

# Optional test section — comment out on Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
