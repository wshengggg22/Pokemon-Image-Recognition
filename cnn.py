import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from data_loader import get_data_loaders

torch.manual_seed(42)

train_loader, _, _ = get_data_loaders()

if __name__ == '__main__':
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    rn18 = torchvision.models.resnet18(pretrained=True)

    # Remove the last fully connected layer
    feature_extractor = nn.Sequential(*list(rn18.children())[:-1])  # all layers except fc
    # print(feature_extractor)

    # images: [batch_size, 3, 224, 224]
    features = feature_extractor(images)  # shape: [batch_size, 512, 1, 1]
    features = features.view(features.size(0), -1)  # flatten: [batch_size, 512]
    print(features.shape)