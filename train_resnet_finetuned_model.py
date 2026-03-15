import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# =========================
# PATHS
# =========================
TRAIN_DIR = "dataset/datatree/train"
VAL_DIR = "dataset/datatree/validation"

# =========================
# IMAGE SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15   # we will increase later if needed

# =========================
# LOAD DATA
# =========================
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

# =========================
# DATA NORMALIZATION
# =========================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# =========================
# RESNET50 BASE MODEL
# =========================
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
# Freeze most of ResNet
base_model.trainable = True

# Fine-tune only top layers of ResNet
for layer in base_model.layers[:-30]:
    layer.trainable = False


# =========================
# CUSTOM CLASSIFIER HEAD
# =========================
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("model", exist_ok=True)
model.save("model/skin_cancer_resnet_finetuned_model.h5")

print("✅ ResNet model training completed and saved!")
