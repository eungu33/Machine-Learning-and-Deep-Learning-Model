# 흉부 X-ray 기반 폐렴 병변 탐지: Faster R-CNN 전처리 비교 실험

## ▣ 프로젝트 개요

RSNA 흉부 X-ray 의료영상을 활용해  
**폐렴 병변의 존재 여부 및 위치를 탐지하는 Faster R-CNN 기반 객체 탐지 모델**을 개발하였습니다.

**특히,**  
- 전처리 방식(A안: `float32 정규화`, B안: `PIL 기반 리사이즈`)  
- 백본 네트워크 구성  

이 두 요소가 성능에 미치는 영향을 **실험적으로 비교 분석**하였습니다.

---

## ▣ 프로젝트 배경 및 목적

- **의료영상 진단은 해석 편차와 높은 전문성 요구**로 인해 자동화 필요성 증가  
- **정확하고 일관된 병변 인식 시스템**을 개발하여  
  조기 진단 또는 의료 보조 도구로의 확장 가능성 확인  
- 딥러닝 기반 객체 인식(Faster R-CNN)의 **의료 영상 적용 가능성 검증**

---

## ▣ 활용한 데이터셋

- **RSNA Pneumonia Detection Challenge (Kaggle)**  
- DICOM 포맷 흉부 X-ray 이미지: **총 26,684개**  
- 라벨: **병변 유무 + 바운딩 박스 좌표 정보** (30,000개 이상)

---

## ▣ 프로젝트 진행 과정

### 1. DICOM 이미지 전처리 및 정규화

- **A안 (float32 방식)**  
  - float32 정규화  
  - bilinear 보간 리사이즈 (`interpolate`)  
  - 1채널 → 3채널 확장  

- **B안 (PIL 방식)**  
  - uint8 변환 후 PIL 기반 리사이즈  
  - `ToTensor()` 적용  

### 2. Faster R-CNN 모델 구성

- 백본: `ResNet18` (FPN 미적용)  
- `AnchorGenerator`, `MultiScaleRoIAlign` 사용  
- Optimizer: Adam  
- 스케줄러: `ReduceLROnPlateau`  
- 조기 종료: `EarlyStopping` 적용

### 3. 학습 전략

- 데이터 불균형 보정: **양성:음성 = 1:1로 조정**  
- 훈련/검증 세트 분할 후 실험 반복

### 4. 성능 비교 및 시각화

- 비교 항목:  
  - 총 학습 시간  
  - Total Loss  
  - Recall  
  - 학습 안정성 (Early Stop 여부 포함)

---

## ▣ 담당 역할

- `fasterrcnn_preprocessing_float32.py`:  
  float32 정규화 기반 전처리 + 학습 코드 구현

- `fasterrcnn_preprocessing_pil.py`:  
  PIL 전처리 기반 학습 코드 + EarlyStopping, LR 스케줄러 포함

- **Dataset 클래스 커스터마이징**  
  - DICOM → NumPy 변환  
  - 채널 확장, 박스 좌표 재계산  
  - 메모리 최적화 구조 설계

- 모델 성능 비교 결과 보고서 작성

---

## ▣ 성능 비교 결과

| 항목             | A안 (float32) | B안 (PIL 방식) |
|------------------|---------------|----------------|
| 총 학습 시간      | 2311.78초     | **2162.47초**  |
| 최종 Total Loss  | 8.6798        | **4.1040**     |
| 최종 Recall      | 0.9660        | 0.9660         |
| 학습 안정성      | 불안정, Early Stop 발생 | **안정적 수렴 유지** |

- **B안(PIL)**이 **속도, 수렴 안정성, Loss 측면에서 모두 우수**  
- **Recall 동일**하지만 precision과 학습 효율성에서 차이 존재

---

## ▣ 주요 인사이트

- **전처리 방식이 전체 모델 성능과 학습 효율성에 결정적 영향**  
- `float32` 기반 방식은 수렴 불안정 및 학습 시간 증가 유발  
- Faster R-CNN은 **정밀 위치 탐지에 효과적**이나,  
  **과적합 방지 전략(EarlyStopping 등) 필수**  
- 향후 다음 방향으로 확장 가능:
  - 다양한 View (X-ray AP/PA) 실험
  - 모델 변경 (YOLOv5, RetinaNet 등)
  - 실 임상데이터 적용 및 검증
