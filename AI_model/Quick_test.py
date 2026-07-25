#Все необходимые библиотеки для скачки будут в отдельном файле в этом же корне

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image
from collections import Counter

#Переменные для обучения
DATA_DIR = '/content/dataset/Garbage classification/Garbage classification/' #Путь в коллабе до папок с распределенными в них фотками
BATCH_SIZE = 64
EPOCHS = 15
IMG_SIZE = 96
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Нормализация
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

#Выборка
train_idx, test_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.15,
    random_state=42,
    stratify=full_dataset.targets
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class_names = full_dataset.classes
num_classes = len(class_names)

#Переменная таргета
targets = [full_dataset.targets[i] for i in train_idx]
class_counts = Counter(targets)
class_weights = [1.0 / class_counts[i] for i in range(num_classes)]
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

#Скачали реснет 18 выше я вбивал в браузере он относительно точный и самый быстрый
#Были тесты на реснете 50 но я не хочу ждать 15 минут ради 1 эпохи :(
#А базовый cnn выдает 50 процентов точности и всегда картон хах
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)
#Для работы на гпу
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

#Переменные для системы терпения
#Система терпения позволяет буквально "терпеть" эпохи до 5 раз если у них уменьшается точность а потом он оставляет ту модель у которой момент эпоса был выше всего (по переменной Best_acc и best_model )
best_acc = 0.0
best_model_wts = None
patience = 5
counter = 0

#Обучение модели
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f'Эпоха {epoch+1}/{EPOCHS} | Потеря: {running_loss/len(train_loader):.4f} | Точность: {acc:.2f}%')

#Та самая легендарная система терпилы. Я просто доработал ту которую на уроке сделал
    if acc > best_acc:
        best_acc = acc
        best_model_wts = model.state_dict().copy()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Терпение кончилось :( )")
            break

    scheduler.step(acc)

model.load_state_dict(best_model_wts)
print(f'Лучшая точность (эта точность будет в финальной моделько после копии): {best_acc:.2f}%')

#Сейвим модель и дамп
torch.save(model.state_dict(), 'garbage_resnet18_fast.pth')
joblib.dump(class_names, 'class_names.pkl')


#Функция предикта для бекендера (все имена будут на английском, а точнее на том языке на котором были названия в датасете)
#ps  Это чудовище его не использовало, позор ему. Оставлю ее для вас
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = test_transform(img)
    img = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    return class_names[idx.item()], conf.item()

#Функция предикта вставьте сюда путь до фотки модель выдаст класс на английском и насколько она в этом уверена (оставил уверенность тк точность оставляет желать лучшего)
if __name__ == "__main__":
    pred_class, confidence = predict_image('Вставьте сюда любую фотку.jpg ')
    print(f'Predicted class: {pred_class}, confidence: {confidence:.3f}')
