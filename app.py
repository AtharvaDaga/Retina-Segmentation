import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load trained models
pca = pickle.load(open("pca_model.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Define upload folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/segmented_output"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Image segmentation function
def segment_image(image_path, mask_path):
    # Load images
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 512x512
    img = cv2.resize(img, (512, 512)) / 255.0
    mask = cv2.resize(mask, (512, 512)) / 255.0

    # Flatten and preprocess
    test_features = np.column_stack((img.flatten(), mask.flatten()))

    # Apply PCA and K-Means
    test_pca = pca.transform(test_features)
    labels = kmeans.predict(test_pca).reshape(512, 512)

    # Identify clusters
    unique_clusters, counts = np.unique(labels, return_counts=True)
    background_cluster = unique_clusters[np.argmax(counts)]
    blood_vessel_cluster = unique_clusters[np.argmin(counts)]
    retina_cluster = 3 - (background_cluster + blood_vessel_cluster)

    # Create segmented image
    segmented_final = np.zeros((512, 512, 3), dtype=np.uint8)
    segmented_final[labels == blood_vessel_cluster] = [255, 0, 0]  # Red for blood vessels
    segmented_final[labels == retina_cluster] = [0, 255, 0]  # Green for retina
    segmented_final[labels == background_cluster] = [0, 0, 0]  # Black for background

    # Save output
    output_filename = "segmented_output.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, segmented_final)

    return output_filename

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
        retina_filename = secure_filename(retina_file.filename)
        mask_filename = secure_filename(mask_file.filename)

        retina_path = os.path.join(app.config["UPLOAD_FOLDER"], retina_filename)
        mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_filename)

        retina_file.save(retina_path)
        mask_file.save(mask_path)

        # Perform segmentation
        segmented_filename = segment_image(retina_path, mask_path)
        segmented_url = f"https://your-render-app-url.com/static/segmented_output/{segmented_filename}"

        return jsonify({"segmented_image_url": segmented_url})

    return jsonify({"error": "Invalid file format!"}), 400

if __name__ == "__main__":
    app.run(debug=True)
