import os
import cv2
import numpy as np
import pickle
import base64
from flask import Flask, request, jsonify

# Load trained models
pca = pickle.load(open("pca_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - shows server status
@app.route("/")
def home():
    return "API is running on Render Server!"

# Image segmentation function (returns base64 image)
def segment_image(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (512, 512)) / 255.0
    mask = cv2.resize(mask, (512, 512)) / 255.0

    test_features = np.column_stack((img.flatten(), mask.flatten()))
    test_pca = pca.transform(test_features)
    labels = kmeans.predict(test_pca).reshape(512, 512)

    unique_clusters, counts = np.unique(labels, return_counts=True)
    background_cluster = unique_clusters[np.argmax(counts)]
    blood_vessel_cluster = unique_clusters[np.argmin(counts)]
    retina_cluster = 3 - (background_cluster + blood_vessel_cluster)

    segmented_final = np.zeros((512, 512, 3), dtype=np.uint8)
    segmented_final[labels == blood_vessel_cluster] = [255, 0, 0]  
    segmented_final[labels == retina_cluster] = [0, 255, 0]  
    segmented_final[labels == background_cluster] = [0, 0, 0]  

    _, buffer = cv2.imencode(".png", segmented_final)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return base64_image

# API to receive and process images
@app.route("/segment", methods=["POST"])
def segment():
    if "retina_image" not in request.files or "mask_image" not in request.files:
        return jsonify({"error": "Both images are required!"}), 400

    retina_file = request.files["retina_image"]
    mask_file = request.files["mask_image"]

    if retina_file.filename == "" or mask_file.filename == "":
        return jsonify({"error": "No file selected!"}), 400

    if allowed_file(retina_file.filename) and allowed_file(mask_file.filename):
        retina_filename = f"temp_{secure_filename(retina_file.filename)}"
        mask_filename = f"temp_{secure_filename(mask_file.filename)}"

        retina_file.save(retina_filename)
        mask_file.save(mask_filename)

        segmented_base64 = segment_image(retina_filename, mask_filename)

        os.remove(retina_filename)
        os.remove(mask_filename)

        return jsonify({"segmented_image": segmented_base64})

    return jsonify({"error": "Invalid file format!"}), 400

if __name__ == "__main__":
    app.run(debug=True)
