# D안 (PIL + Resize 기반 전처리) + Adam + ReduceLROnPlateau + 조기 종료 포함 버전
# 목적: 빠른 전처리 속도, 메모리 효율, 자동 학습률 조절, 조기 종료
# 백본: ResNet18 수동 구성 (FPN 없음)

import os, time
import pandas as pd
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from tqdm import tqdm
import pydicom
from PIL import Image

# 디바이스 설정 (GPU 가능 시 사용)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 설정값
EPOCHS = 10
BATCH_SIZE = 16
PATIENCE = 3
SAVE_DIR = "./fasterrcnn_resnet18"
os.makedirs(SAVE_DIR, exist_ok=True)

# 데이터 로딩 및 DICOM 경로 생성
df = pd.read_csv('./data/pneumonia_pa.csv')
df['image_path'] = df['patientId'].apply(lambda x: os.path.join('./data/stage_2_train_images', f"{x}.dcm"))
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

# 양성 데이터 개수에 맞춰 클래스 비율 조정 (양성 전체, 정상은 동일 개수)
df_pos = df[df['Target'] == 1]
df_neg = df[df['Target'] == 0].sample(n=len(df_pos), random_state=42)
df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# 클래스 분포 출력
print("클래스 분포:")
print(df_balanced['Target'].value_counts())
print("\n비율:")
print(df_balanced['Target'].value_counts(normalize=True))

# 훈련/검증 데이터 분리
df_train, df_val = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced['Target'], random_state=42)

# Dataset 클래스 정의 (D안: PIL 기반 전처리 방식)
class PneumoniaDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm = pydicom.dcmread(row['image_path'])
        img = dcm.pixel_array.astype(np.uint8)
        img = self.transform(img)
        img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img

        target = {'image_id': torch.tensor([idx])}
        if row['Target'] == 1:
            scale_x, scale_y = 224 / dcm.Columns, 224 / dcm.Rows
            x1 = row['x'] * scale_x
            y1 = row['y'] * scale_y
            x2 = x1 + row['width'] * scale_x
            y2 = y1 + row['height'] * scale_y
            target['boxes'] = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            target['labels'] = torch.tensor([1], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            outs = model(imgs)
            for out, tgt in zip(outs, tgts):
                pred = 1 if len(out['boxes']) > 0 else 0
                truth = 1 if len(tgt['boxes']) > 0 else 0
                y_pred.append(pred)
                y_true.append(truth)
    return y_true, y_pred

if __name__ == "__main__":
    train_loader = DataLoader(PneumoniaDataset(df_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(PneumoniaDataset(df_val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    resnet = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone=backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_loss = float("inf")
    trigger_times = 0

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i, (imgs, tgts) in enumerate(tqdm(train_loader, desc=f"[D안] Epoch {epoch+1}")):
            imgs = [img.to(DEVICE) for img in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)

        y_true, y_pred = evaluate_model(model, val_loader)
        recall = recall_score(y_true, y_pred)

        print(f"[D안] Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Recall: {recall:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_D.pth"))
            print("모델 저장 완료")
        else:
            trigger_times += 1
            print(f"성능 개선 없음 → 조기 종료 카운트 {trigger_times}/{PATIENCE}")
            if trigger_times >= PATIENCE:
                print("조기 종료 조건 충족. 학습 중단.")
                break

    end = time.time()
    print(f"총 학습 시간(D안): {end - start:.2f}초")
