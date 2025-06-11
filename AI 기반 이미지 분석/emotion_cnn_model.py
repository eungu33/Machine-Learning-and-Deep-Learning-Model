import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 7
NUM_CLASSES = 2

# 경로
TRAIN_ROOT = '../project/data/train'
TEST_ROOT = '../project/data/test'
MODEL_PATH = './model/emotion_cnn_best.pt'

# 데이터 전처리 및 증강
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터 로딩
train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)
test_data = datasets.ImageFolder(root=TEST_ROOT, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print("클래스 인덱스:", train_data.class_to_idx)
print("클래스 수:", len(train_data.classes))

# 모델 정의
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 학습 관련 설정
model = EmotionCNN(NUM_CLASSES).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# 학습 루프
best_val_loss = np.inf
early_stop_count = 0
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = output.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # 검증
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = loss_fn(output, y)
            val_loss += loss.item()
            preds = output.argmax(1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("모델 저장 완료:", MODEL_PATH)
    else:
        early_stop_count += 1
        if early_stop_count >= PATIENCE:
            print(f"조기 종료 (Early stopping at epoch {epoch+1})")
            break

# 최종 평가
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print("\n최종 테스트 성능 보고서:")
print(classification_report(y_true, y_pred, target_names=train_data.classes))
