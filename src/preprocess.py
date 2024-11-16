import os
import shutil
import random
from PIL import Image, UnidentifiedImageError

DATA_DIR = "data/"
PROCESSED_DATA_DIR = "processed_data/"
TRAIN_SPLIT = 0.8  # 80% para treino, 20% para validação

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image.verify()
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image
    except UnidentifiedImageError:
        print(f"Imagem inválida ignorada: {image_path}")
        return None

def preprocess_dataset():
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        train_size = int(len(images) * TRAIN_SPLIT)
        train_images = images[:train_size]
        val_images = images[train_size:]

        train_category_path = os.path.join(PROCESSED_DATA_DIR, 'train', category)
        val_category_path = os.path.join(PROCESSED_DATA_DIR, 'val', category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        for img_name in train_images:
            src_path = os.path.join(category_path, img_name)
            if preprocess_image(src_path):
                dest_path = os.path.join(train_category_path, img_name)
                shutil.copy(src_path, dest_path)

        for img_name in val_images:
            src_path = os.path.join(category_path, img_name)
            if preprocess_image(src_path):
                dest_path = os.path.join(val_category_path, img_name)
                shutil.copy(src_path, dest_path)

        print(f"Categoria '{category}': {len(train_images)} imagens de treino, {len(val_images)} imagens de validação.")

if __name__ == "__main__":
    preprocess_dataset()
