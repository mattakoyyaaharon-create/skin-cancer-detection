import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ======================
# CONFIG
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 7

TRAIN_DIR = "dataset/datatree/train"
VAL_DIR = "dataset/datatree/validation"

# ======================
# DATA GENERATORS
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ======================
# MODEL (NO INTERNET)
# ======================
base_model = MobileNetV2(
    weights=None,          # 🔴 IMPORTANT: no download
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRAIN
# ======================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================
# SAVE MODEL
# ======================
os.makedirs("model", exist_ok=True)
model.save("model/skin_cancer_mobilenetv2.h5")

print("✅ MobileNetV2 training completed and model saved.")
