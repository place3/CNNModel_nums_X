import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np


num_epochs = 5
num_classes = 6   # 0–4 + X
batch_size = 100
learning_rate = 0.001

DATA_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\MNISTData"
MODEL_STORE_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\Pytorch_models"
X_DATASET_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_moъхххdel\XDataset\X"  # тут хранятся X

# Трансформации
trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset_full = datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset_full = datasets.MNIST(root=DATA_PATH, train=False, transform=trans)


# MNIST с фильтрацией 0–4
def filter_mnist(dataset):
    mask = (dataset.targets < 5)  # оставляем только 0–4
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

train_dataset = filter_mnist(train_dataset_full)
test_dataset = filter_mnist(test_dataset_full)

# Датасет для X
class XDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")  # в грайскейл
        if self.transform:
            img = self.transform(img)
        label = 5  # класс "X"
        return img, label

x_train_dataset = XDataset(X_DATASET_PATH, transform=trans)
x_test_dataset = XDataset(X_DATASET_PATH, transform=trans)  # можно разделить, но пока одно и то же


# Объединяем датасеты
train_dataset_final = ConcatDataset([train_dataset, x_train_dataset])
test_dataset_final = ConcatDataset([test_dataset, x_test_dataset])

train_loader = DataLoader(dataset=train_dataset_final, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset_final, batch_size=batch_size, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)  # 6 классов

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")


# Тестирование
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Сохранение
os.makedirs(MODEL_STORE_PATH, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_STORE_PATH, "conv_net_model_X.ckpt"))

