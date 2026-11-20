import torch
import matplotlib.pyplot as plt
import numpy as np
from models.cnn import SimpleCNN
from data.data import test_loader

model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

def plot_filters(weights):
    n_filters = weights.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i >= n_filters: break
        filter = weights[i].mean(dim=0).detach().cpu().numpy()
        filter = (filter - filter.min()) / (filter.max() - filter.min() + 1e-5)

        ax.imshow(filter, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i}')
    plt.suptitle("Веса первого слоя (conv1)", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_maps(model, loader):
    images, labels = next(iter(loader))
    img = images[0].unsqueeze(0)

    with torch.no_grad():
        feature_maps = torch.relu(model.conv1(img))
    feature_maps = feature_maps.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    orig_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    orig_img = orig_img * 0.5 + 0.5

    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Оригинал")
    axes[0, 0].axis('off')

    for i in range(15):
        row, col = (i+1) // 8, (i+1) % 8
        axes[row, col].imshow(feature_maps[i], cmap='viridis')
        axes[row, col].set_title(f'Map {i}')
        axes[row, col].axis('off')
        
    plt.suptitle("Реакция фильтров на изображение", fontsize=16)
    plt.tight_layout()
    plt.show()

print("Визуализация фильтров...")
plot_filters(model.conv1.weight)

print("Визуализация карт признаков...")
plot_feature_maps(model, test_loader)