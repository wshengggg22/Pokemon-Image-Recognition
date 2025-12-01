import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

from pokemon_classifier import PokemonClassifier
from config import NUM_CLASSES, INPUT_SIZE
from data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PokemonClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

_, val_loader, test_loader = get_data_loaders()

class_names = val_loader.dataset.classes
print(class_names[:10])  # verify

def predict(model, img):
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)  # add batch dimension
        outputs = model(img)
        pred = outputs.argmax(dim=1).item()
    return pred

import random
import numpy as np
import matplotlib.pyplot as plt

def show_samples(model, loader, class_names, num_samples=8):
    model.eval()
    dataset = loader.dataset  # access underlying dataset
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(16, 6))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]  # directly load a random sample
        pred = predict(model, img)

        # unnormalize (means & stds for imagenet)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(img_np)
        plt.axis("off")

        color = "green" if pred == label else "red"
        plt.title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}", color=color)

    plt.tight_layout()
    plt.show()

show_samples(model, test_loader, class_names, num_samples=8)
