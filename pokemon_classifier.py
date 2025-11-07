import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from data_loader import get_data_loaders
from config import NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE
import time
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_data_loaders()
rn18 = resnet18(weights=ResNet18_Weights.DEFAULT)

def check_feature():
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # Remove the last fully connected layer
    feature_extractor = nn.Sequential(*list(rn18.children())[:-1])  # all layers except fc
    print(feature_extractor)

    # images: [batch_size, 3, 224, 224]
    features = feature_extractor(images)  # shape: [batch_size, 512, 1, 1]
    print(features.shape)

class PokemonClassifier(nn.Module):
    def __init__(self):
        super(PokemonClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(*list(rn18.children())[:-1])  # all layers except fc
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),  # add dropout
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # shape: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)      # flatten to [batch_size, 512]
        x = self.fc(x)         # shape: [batch_size, num_classes]
        return x

def get_accuracy(model, data_loader):
    model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():  # no need to compute gradients for evaluation
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)  # forward pass
            pred = outputs.argmax(dim=1)  # predicted class index
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total

def train(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    iters, losses, train_accs, val_accs = [], [], [], []
    n = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy for this batch
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            n += 1
            iters.append(n)
            losses.append(loss.item())

        # Compute per-epoch metrics
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)

        model.train()  # switch back to training mode

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_acc == max(val_accs):
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"Time: {time.time()-start_time:.2f}s")

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f}s "
          f"({(end_time - start_time)/num_epochs:.2f}s per epoch)")

    # Plot training loss
    plt.figure(figsize=(8,5))
    plt.plot(iters, losses, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_accs, label="Training Accuracy")
    plt.plot(range(1, num_epochs+1), val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accs[-1]:.4f}")

if __name__ == '__main__':
    # check_feature()

    model = PokemonClassifier().to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    train_loader, val_loader, test_loader = get_data_loaders()

    train(model, train_loader, val_loader, num_epochs=30)
