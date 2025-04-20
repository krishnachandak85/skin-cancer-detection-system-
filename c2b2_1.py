from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from threading import Thread
import time
import requests

app = Flask(__name__)

# Define paths
reference_paths = {
    "Melanoma": r"C:\Users\krish\Desktop\ML C2B2\Melanoma",
    "Basal Cell Carcinoma": r"C:\Users\krish\Desktop\ML C2B2\Basal Cell Carcinoma",
    "Squamous Cell Carcinoma": r"C:\Users\krish\Desktop\ML C2B2\Squamous Cell Carcinoma",
    "Benign": r"C:\Users\krish\Desktop\ML C2B2\Benign"
}
upload_folder = r"C:\Users\krish\Desktop\ML C2B2\uploads"
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

# Start Flask in a background thread (only once)
def run_flask():
    app.run(port=5000)

if not hasattr(app, 'started'):
    thread = Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    time.sleep(3)
    app.started = True

# === Send a test image from the test folder ===
test_folder = r"C:\Users\krish\Desktop\ML C2B2\test"
test_image_path = None

# Pick the first valid image from the folder
for fname in os.listdir(test_folder):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        test_image_path = os.path.join(test_folder, fname)
        break

if test_image_path:
    with open(test_image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
    print("Prediction Result:\n", response.json())
else:
    print("No test image found in the folder.")
