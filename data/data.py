from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cifar10')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)