import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_features(image_path, size=(64, 64), bins=16):
    """
    Extracts raw pixel values + color histogram from an image.
    - size: resize image to reduce dimensionality
    - bins: number of bins for color histogram
    """
    image = Image.open(image_path).convert("RGB").resize(size)
    pixels = np.array(image).flatten() / 255.0  # normalize pixel values

    hist_r = np.histogram(np.array(image)[:, :, 0], bins=bins, range=(0, 255))[0]
    hist_g = np.histogram(np.array(image)[:, :, 1], bins=bins, range=(0, 255))[0]
    hist_b = np.histogram(np.array(image)[:, :, 2], bins=bins, range=(0, 255))[0]

    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    hist /= hist.sum()  # normalize histogram

    return np.concatenate([pixels, hist])

def load_dataset(folder):
    X, y, class_names = [], [], sorted(os.listdir(folder))
    for label, cls in enumerate(class_names):
        class_folder = os.path.join(folder, cls)
        for img_file in tqdm(os.listdir(class_folder), desc=f"Loading {cls}"):
            img_path = os.path.join(class_folder, img_file)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label)
            except Exception:
                continue
    return np.array(X), np.array(y), class_names
