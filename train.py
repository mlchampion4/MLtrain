import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.data import train_loader, test_loader
from models.resnet50 import ResNet50
from tqdm import tqdm
from utils import EarlyStopping, evaluate_model, evaluate_roc_auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = ResNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.1, patience=3
    )

    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        scheduler.step(acc)
        early_stopping(acc, model)

        if early_stopping.early_stop:
            print(f"🛑 EarlyStopping сработал на эпохе {epoch+1}. Обучение остановлено.")
            break

    model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), 'model.pth')

    true_labels, pred_labels = evaluate_model(model, test_loader, device)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    print(classification_report(true_labels, pred_labels, target_names=classes))
    evaluate_roc_auc(model, test_loader, device, num_classes=10)
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - ResNet34 on CIFAR-10')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == '__main__':
    main()