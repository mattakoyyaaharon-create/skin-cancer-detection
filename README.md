# 🧠 Skin Cancer Detection and Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

> Automated multi-class skin lesion classification using **MobileNetV2 Transfer Learning** with real-time web deployment.

---

# 📌 Project Overview

Skin cancer is one of the **most common and rapidly increasing cancers worldwide**.  
Early detection plays a crucial role in improving survival rates and reducing treatment complexity.

Traditional diagnosis relies heavily on **manual dermoscopic examination by dermatologists**, which can be:

- Time consuming
- Subjective
- Prone to human error

This project proposes an **AI-powered skin lesion classification system** that assists healthcare professionals in **early detection and screening of skin cancer** using deep learning.

---

# 🎯 Project Objective

The main objective of this project is to develop a **lightweight deep learning model** capable of performing:

- Multi-class skin lesion classification
- Automated detection using dermoscopic images
- Real-time prediction through a web interface

The model uses **MobileNetV2 transfer learning architecture** to achieve efficient performance while maintaining low computational cost.

---

# 👥 Team Members

| Name | Roll Number |
|-----|-------------|
| KARAMSETTY BHOOMIKA | 22KQ1A6110 |
| KUNCHALA VAMSHI | 22KQ1A6148 |
| CHILAKALA BHANUPRAKASH | 22KQ1A6136 |
| SINGAMPALLI KEERTHI | 22KQ1A6126 |
| GEDDADA PADMAJA | 22KQ1A6108 |

---

# 📊 Project Resources

Datasets, documentation, PPT and additional resources are available in the Google Drive folder.

🔗 **Google Drive Link**

https://drive.google.com/drive/folders/1_ynefuffv61t5Feur0_kfqzNWOlqmqLZ?usp=sharing

---

# 🧠 Model Architecture

The system uses **MobileNetV2**, a lightweight CNN architecture optimized for mobile and embedded vision applications.

### Key Concepts Used

- Transfer Learning
- Depthwise Separable Convolutions
- Inverted Residual Blocks
- Fine-Tuning of Upper Layers

---

# ⚙️ Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Core Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Model Development |
| Flask | Backend Web Framework |
| HTML | Frontend Structure |
| CSS | Styling |
| JavaScript | Client-side Interactions |
| Google Colab | Model Training with GPU |

---

# 🧪 Training Strategy

The model training process includes:

### Preprocessing

- Image resizing
- Normalization
- Data augmentation

### Transfer Learning

- Pre-trained **ImageNet weights**
- Frozen base layers initially

### Custom Classification Head

- Global Average Pooling
- Dense Layer (ReLU activation)
- Dropout layer for regularization
- Softmax output layer for **7 skin lesion classes**

### Optimization

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

# 📈 Results and Performance

The trained model demonstrates **stable convergence and reliable classification performance**.

| Metric | Score |
|------|------|
| Training Accuracy | ~75–80% |
| Validation Accuracy | ~80% |
| Precision | 0.79–0.80 |
| Recall | 0.78–0.79 |
| F1 Score | 0.79 |
| Loss | Stable Convergence |

---

# 📊 Performance Highlights

- Stable training and validation curves
- Effective multi-class lesion classification
- Improved minority class learning using class weights
- Lightweight and computationally efficient model
- Suitable for real-time deployment

---

# 🌐 Application Deployment

The trained model is deployed as a **web application**.

### Backend

Flask

### Frontend

- HTML
- CSS
- JavaScript

### Model Files

- cmodel.h5
- labels.json


---

# ✨ Application Features

- Upload dermoscopic image
- Real-time skin lesion prediction
- Display predicted class label
- Show confidence score
- User-friendly web interface
- Lightweight deployment suitable for demonstrations

---

# 🛠 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000/

skin-cancer-detection
│
├── app.py
├── model.h5
├── labels.json
├── requirements.txt
├── README.md
│
├── static
├── templates
├── uploads
│
└── docs
    ├── accuracy_plot.png
    ├── confusion_matrix.png
    └── roc_curve.png
```
