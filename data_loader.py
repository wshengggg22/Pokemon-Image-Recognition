import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, INPUT_SIZE, NUM_WORKERS

train_dir = 'pokemon_split/train'
val_dir = 'pokemon_split/val'
test_dir = 'pokemon_split/test'

# Normalization values used for pretrained ResNet (ImageNet stats)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Validation and Test (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=test_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

def get_data_loaders(batch_size=BATCH_SIZE, num_workers = NUM_WORKERS):
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader

# Helper function to show an image
def imshow(img):
    img = img * torch.tensor(imagenet_std).view(3,1,1) + torch.tensor(imagenet_mean).view(3,1,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize():
    train_loader, val_loader, test_loader = get_data_loaders()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print('Class labels:', [train_data.classes[i] for i in labels[:4]])
    imshow(torchvision.utils.make_grid(images[:4]))

# ----------------------------------
# Only run this on Windows inside this guard
if __name__ == '__main__':
    print(f"Num training images: {len(train_data)}")
    print(f"Num validation images: {len(val_data)}")
    print(f"Num test images: {len(test_data)}")
    print(f"Number of classes: {len(train_data.classes)}")

    visualize()
