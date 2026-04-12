import os
import random
from collections import Counter

import torch
import sklearn
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pandas.io.clipboard import clipboard_set
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, WeightedRandomSampler


# ----------------------MODEL-------------------------------
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
        self.fc2 = nn.Linear(1000, 6)  # 6 классов

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# --------------XDataset-------------
class XDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
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

    # MNIST WRAPPER


class MNISTWrapper(Dataset):
    def __init__(self, mnist_tf_dataset, transform):
        self.data = mnist_tf_dataset.data  # uint8 tensor
        self.targets = mnist_tf_dataset.targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx].numpy()  # HxW uint8
        pil = Image.fromarray(img).convert("L")
        if self.transform:
            img_t = self.transform(pil)
        else:
            img_t = transforms.ToTensor()(pil)
        label = int(self.targets[idx].item())
        return img_t, label


def main():
    import numpy as np

    # --------------------SETTINGS-------------------
    num_epochs = 30
    num_classes = 6  # 0–4 + X
    batch_size = 100
    seed = 42
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    DATA_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\MNISTData"
    MODEL_STORE_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\Pytorch_models"
    X_DATASET_PATH = r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\XDataset\SELFCREATE\XPics"  # тут хранятся X

    # --------------------TRANSFORMS----------------
    mnist_norm = (0.1307,), (0.3081,)
    mnist_trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    x_train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # приводим к 1 каналу
        transforms.Resize((28, 28)),  # ресайз
        transforms.RandomRotation(degrees=20),  # случайный поворот [-20°, 20°]
        transforms.RandomAffine(
            degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)  # случайный сдвиг/масштаб
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3),  # иногда размываем
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(*mnist_norm)  # как в MNIST
    ])

    x_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --------------------DATASETS------------------------------------
    # подгрузка mnist
    train_dataset_full = datasets.MNIST(root=DATA_PATH, train=True, transform=mnist_trans, download=True)
    test_dataset_full = datasets.MNIST(root=DATA_PATH, train=False, transform=mnist_trans)

    # MNIST с фильтрацией 0–4
    def filter_mnist(dataset):
        mask = (dataset.targets < 5)  # оставляем только 0–4
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]
        return dataset

    # применение фильтрации
    mnist_train_dataset = filter_mnist(train_dataset_full)
    mnist_test_dataset = filter_mnist(test_dataset_full)

    mnist_train_ds = MNISTWrapper(mnist_train_dataset, transform=mnist_trans)
    mnist_test_ds = MNISTWrapper(mnist_test_dataset, transform=mnist_trans)

    # Split X to train and test
    all_x_files = [os.path.join(X_DATASET_PATH, f) for f in os.listdir(X_DATASET_PATH) if
                   f.lower().endswith((".png", ".jpg", ".jpeg"))]
    x_train_files = all_x_files

    assert len(all_x_files) > 0, "X dataset folder is empty!"

    from sklearn.model_selection import train_test_split

    test_x_files = [os.path.join(r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\XDataset\X", f)
                    for f in os.listdir(r"C:\Users\us3r02\PycharmProjects\CNN_numbers_model\XDataset\X") if
                    f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # x_train_files, x_test_files = train_test_split(all_x_files, train_size=0.99, random_state=seed, shuffle=True)

    x_train_dataset = XDataset(x_train_files, transform=x_train_transform)
    x_test_dataset = XDataset(test_x_files, transform=x_test_transform)  # можно разделить, но пока одно и то же

    # Объединяем датасеты
    train_dataset_final = ConcatDataset([mnist_train_ds, x_train_dataset])
    test_dataset_final = ConcatDataset([mnist_test_ds, x_test_dataset])

    # ------------------
    # build list of labels for train_ds to compute sample weights
    def get_labels_from_concat(dataset):
        labels = []
        for i in range(len(dataset)):
            _, lbl = dataset[i]
            labels.append(int(lbl))
        return labels

    train_labels = get_labels_from_concat(train_dataset_final)
    counter = Counter(train_labels)
    print("Train label counts:", counter)

    # compute sample weight = 1 / freq(class)
    class_count = np.array([counter[i] for i in range(num_classes)], dtype=np.float32)
    class_weight = 1.0 / (class_count + 1e-8)
    sample_weights = np.array([class_weight[l] for l in train_labels], dtype=np.float32)
    sample_weights[5] *= 2

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(dataset=train_dataset_final,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=2,
                              shuffle=False,
                              pin_memory=True,
                              )
    test_loader = DataLoader(dataset=test_dataset_final,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True
                             )

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # ---------------TRAINING LOOP----------------------------
    # total_step = len(train_loader)
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i + 1) % 100 == 0:
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")
    #
    # # -------------------TESTING------------------------------
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #     print(f"Test Accuracy: {100 * correct / total:.2f}%")
    #
    # # Сохранение
    # os.makedirs(MODEL_STORE_PATH, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(MODEL_STORE_PATH, "conv_net_model_X.ckpt"))
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            # print("Batch", Counter(labels.tolist()))
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # validation / test pass
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_acc = correct / total
        print(f"Epoch {epoch}: train_loss={running_loss / len(train_dataset_final):.4f}, val_acc={val_acc:.4f}")

        # scheduler step
        scheduler.step(val_acc)

        # confusion matrix
        import numpy as np
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        print("Confusion matrix:\n", cm)
        print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                    target_names=[str(i) for i in range(num_classes)]))

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_STORE_PATH, "best_conv_net_model_X.pth"))
            print("Saved best model, val_acc=", best_val_acc)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Training finished. Best val_acc=", best_val_acc)

    # -----------------Сохранение-----------------------------
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_STORE_PATH, "conv_net_model_X.ckpt"))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
