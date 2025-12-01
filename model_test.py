import torch
from pokemon_classifier import PokemonClassifier, get_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from data_loader import get_data_loaders
from config import NUM_CLASSES

train_loader, val_loader, test_loader = get_data_loaders()

# Assume you have class_names from your data loader
# e.g., class_names = train_loader.dataset.classes
class_names = train_loader.dataset.classes  

# Load the best model
model = PokemonClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # evaluation mode

# Overall test accuracy
test_accuracy = get_accuracy(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Per-class accuracy
num_classes = NUM_CLASSES
class_correct = [0 for _ in range(num_classes)]
class_total = [0 for _ in range(num_classes)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = preds[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

print("\nPer-class accuracy:")
for i in range(num_classes):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"{class_names[i]}: {acc:.4f}")
    else:
        print(f"{class_names[i]}: No samples in test set")

