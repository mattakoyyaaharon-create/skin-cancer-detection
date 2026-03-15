import os

TRAIN_DIR = "dataset/datatree/train"
VAL_DIR = "dataset/datatree/validation"

def count_images(folder_path):
    print(f"\nChecking folder: {folder_path}")
    total = 0
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{class_name} : {count} images")
            total += count
    print("Total images:", total)

print("TRAIN DATASET")
count_images(TRAIN_DIR)

print("\nVALIDATION DATASET")
count_images(VAL_DIR)
