"""
Flask backend for AquaVision - Water Sewage Detection

Provides an endpoint `/predict` that accepts multipart/form-data with an image file
and water sensor readings (either as form fields or a JSON `readings` field).

The backend computes simple image features (average RGB, blurriness, histogram spread),
combines them with numeric sensor data, loads a pretrained model (`model.pkl`) and
returns a JSON prediction and confidence.

Run:
    python app.py

"""
import os
import io
import json
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

MODEL_PATH = "model.pkl"
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)


def read_image_from_file_storage(file_storage):
    """Read a Flask FileStorage into an OpenCV image (BGR)."""
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    # convert RGB (PIL) to BGR (OpenCV) if needed
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def compute_image_features(cv_img):
    """Compute simple image features:
    - average color (R,G,B)
    - blurriness (variance of Laplacian)
    - histogram spread (std of grayscale histogram)
    Returns a dict with keys img_r,img_g,img_b,blur,hist_spread
    """
    # convert BGR to RGB for average color
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # average colors
    avg_per_channel = np.mean(rgb.reshape(-1, 3), axis=0)
    img_r, img_g, img_b = avg_per_channel.tolist()

    # blurriness: variance of Laplacian on grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    # histogram spread: std dev of histogram counts
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_spread = float(np.std(hist))

    return {
        "img_r": float(img_r),
        "img_g": float(img_g),
        "img_b": float(img_b),
        "blur": blur_var,
        "hist_spread": hist_spread,
    }


def parse_readings(form):
    """Parse readings from the request.form.
    Accepts either a JSON string in 'readings' or individual form fields.
    """
    if "readings" in form:
        try:
            data = json.loads(form["readings"]) if isinstance(form["readings"], str) else form["readings"]
            return {k: float(v) for k, v in data.items()}
        except Exception:
            pass

    # fallback to individual fields
    keys = ["pH", "turbidity", "conductivity", "DO", "temperature"]
    readings = {}
    for k in keys:
        if k in form:
            try:
                readings[k] = float(form[k])
            except Exception:
                readings[k] = None
        else:
            readings[k] = None
    return readings


@app.route("/predict", methods=["POST"])
def predict():
    # load model
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model file not found. Run train_model.py first to create model.pkl"}), 500

    model = joblib.load(MODEL_PATH)

    # handle image
    if "image" not in request.files:
        return jsonify({"error": "No image file part in request (expected field name 'image')"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file to disk (optional)
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.stream.seek(0)
    image_file.save(save_path)
    image_file.stream.seek(0)

    # Read and compute image features
    try:
        img = read_image_from_file_storage(image_file)
        img_feats = compute_image_features(img)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    # parse readings
    readings = parse_readings(request.form)

    # required sensor keys
    required = ["pH", "turbidity", "conductivity", "DO", "temperature"]
    feature_vector = []
    for k in required:
        val = readings.get(k)
        if val is None:
            return jsonify({"error": f"Missing or invalid sensor reading: {k}"}), 400
        feature_vector.append(float(val))

    # order must match training: add image features
    feature_vector.extend([
        img_feats["img_r"],
        img_feats["img_g"],
        img_feats["img_b"],
        img_feats["blur"],
        img_feats["hist_spread"],
    ])

    X = np.array(feature_vector).reshape(1, -1)

    # predict
    try:
        proba = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])
        # mapping: 0 -> Sewage detected, 1 -> Clean water
        mapping = {0: "Sewage Detected", 1: "Water is Clean"}
        confidence = float(np.max(proba))

        # Log prediction details for debugging
        print(f"\n{'='*60}")
        print(f"PREDICTION REQUEST:")
        print(f"{'='*60}")
        print(f"Sensor Readings: pH={readings.get('pH')}, turb={readings.get('turbidity')}, "
              f"cond={readings.get('conductivity')}, DO={readings.get('DO')}, temp={readings.get('temperature')}")
        print(f"Image Features: R={img_feats['img_r']:.1f}, G={img_feats['img_g']:.1f}, "
              f"B={img_feats['img_b']:.1f}, blur={img_feats['blur']:.1f}, hist={img_feats['hist_spread']:.1f}")
        print(f"Prediction: {mapping[pred]} (Class {pred})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: [Sewage={proba[0]:.4f}, Clean={proba[1]:.4f}]")
        print(f"{'='*60}\n")

        return jsonify({
            "prediction": mapping.get(pred, str(pred)),
            "confidence": round(confidence, 4),
            "probabilities": {
                "sewage": round(float(proba[0]), 4),
                "clean": round(float(proba[1]), 4)
            },
            "features": {
                "sensor": {k: readings.get(k) for k in required},
                "image": {
                    "avg_rgb": [round(img_feats['img_r'], 1), 
                               round(img_feats['img_g'], 1), 
                               round(img_feats['img_b'], 1)],
                    "blur": round(img_feats['blur'], 1),
                    "hist_spread": round(img_feats['hist_spread'], 1)
                }
            }
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    # For local dev only. Use a production WSGI server for deployment.
    app.run(host="0.0.0.0", port=5000, debug=True)
