import cv2
import numpy as np
import os


# 0. 저장 폴더 준비

os.makedirs('./matched', exist_ok=True)


# 1. 템플릿 이미지 로드 및 전처리

template = cv2.imread('./data/Logos.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("템플릿 이미지 로드 실패")
    exit()

# CLAHE 적용 (명암 대비 강화)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
template = clahe.apply(template)

# 블러 적용 (노이즈 완화)
template = cv2.GaussianBlur(template, (3, 3), 0)
w, h = template.shape[::-1]


# 2. 동영상 로드 및 ROI 선택

cap = cv2.VideoCapture('./data/normal(1).mp4')
ret, first_frame = cap.read()
if not ret:
    print("동영상 로드 실패")
    exit()

roi = cv2.selectROI("Select ROI (ENTER)", first_frame, showCrosshair=True)
cv2.destroyWindow("Select ROI (ENTER)")

x, y, w_roi, h_roi = roi


# 3. 템플릿 매칭 반복

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
match_count = 0
scales = [0.8, 1.0, 1.2]   # 템플릿 스케일 변화
threshold = 0.35           # 민감도

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 샘플링: 5프레임마다 처리
    if frame_count % 5 != 0:
        frame_count += 1
        continue

    # ROI 자르기 + 전처리
    roi_frame = frame[y:y+h_roi, x:x+w_roi]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    found = False

    for scale in scales:
        t_scaled = cv2.resize(template, (int(w * scale), int(h * scale)))
        result = cv2.matchTemplate(gray, t_scaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(roi_frame, pt, (pt[0] + t_scaled.shape[1], pt[1] + t_scaled.shape[0]), (0, 255, 0), 2)
            found = True

    if found:
        # ROI에 그려진 결과를 전체 프레임에 반영
        frame[y:y+h_roi, x:x+w_roi] = roi_frame

        # 화면에 보여주기
        cv2.imshow('Matched Frame', frame)

        # 저장: 매칭 5개마다 한 번씩 저장
        if match_count % 5 == 0:
            cv2.imwrite(f'./matched/frame_{frame_count}_match.jpg', frame)

        match_count += 1

    # 종료 키 (ESC)
    if cv2.waitKey(100) & 0xFF == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"\n일치한 프레임 수: {match_count}")