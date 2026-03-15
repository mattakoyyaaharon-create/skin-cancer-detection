'''from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ======================
# CONFIG
# ======================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "model/mobilenetv2_skin_cancer.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Actinic Keratosis (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesion (vasc)"
]

CANCER_CLASSES = ["akiec", "bcc", "mel"]

# ======================
# ROUTES
# ======================
@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No image selected"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # -------- Image preprocessing (MUST MATCH TRAINING) --------
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- Prediction --------
    preds = model.predict(img_array)[0]
    confidence = float(np.max(preds))
    class_index = np.argmax(preds)
    label = CLASS_NAMES[class_index]
    short_label = label.split("(")[-1].replace(")", "")

    # -------- Decision logic --------
    if confidence < 0.35:
        result = "Image not suitable for skin cancer prediction"
        accuracy = "N/A"

    elif short_label in CANCER_CLASSES:
        result = f"Skin Cancer Detected: {label}"
        accuracy = f"{confidence * 100:.2f}%"

    else:
        result = "No skin cancer detected"
        accuracy = f"{confidence * 100:.2f}%"

    return render_template(
        "result.html",
        image_path=file_path,
        prediction=result,
        accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(debug=True)'''
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("model/mobilenetv2_skin_cancer.h5")

CLASS_NAMES = [
    "Actinic Keratosis (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesion (vasc)"
]

CANCER_CLASSES = ["akiec", "bcc", "mel"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No image uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No image selected"

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    confidence = float(np.max(preds))
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    short = label.split("(")[-1].replace(")", "")

    
    if short in CANCER_CLASSES:
        result = f"Skin Cancer Detected: {label}"
        acc = f"{confidence*100:.2f}%"
    elif confidence < 0.15:
        result = "Image not suitable for skin cancer prediction"
        acc = "N/A"
    else:
        result = "No skin cancer detected"
        acc = f"{confidence*100:.2f}%"

    return render_template(
        "result.html",
        image_path=path,
        prediction=result,
        accuracy=acc
    )

if __name__ == "__main__":
    app.run(debug=True)
