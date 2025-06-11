import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

img = cv2.imread('./matched_crop/final_crop.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('a',img)
#cv2.waitKey()
#cv2.destroyAllWindows()

#img = img[400:580, 400:600]  # 로고 범위

h, w = img.shape # 이미지 크기 저장 <- 범위 내에 점을 찍어야 되니까

dataDF = pd.DataFrame()

background_color = int(np.mean(img)) # 배경색 저장 <- 노이즈 색 배경색에 맞추려고

# 이미지 개수
num_images = 4444 # -------------------------------------------------------------------------
# 임계 노이즈 수
num_noise = 4444

for x in range(num_images):
    logo = img.copy()

    # 점 개수
    #num_points = np.random.randint(30, 150)

    # 진한거 반 흐린거 반
    if x % 2 == 0:
        num_points = np.random.randint(42, 5555)  
    else:
        num_points = np.random.randint(6000, 10000)   

    # 점 찍기
    for _ in range(num_points):
        a = np.random.randint(0, h)
        b = np.random.randint(0, w)

        # 자연스러운 흐릿함을 위해 점 색 조절 - 배경색 +- 10 , 0이상 255 이하
        noise_value = np.clip(background_color + np.random.randint(-10, 10), 0, 255)
        logo[a, b] = noise_value

    # 크기 조절
    re = cv2.resize(logo, (200, 200))
    # 차원 조절
    seriesDF = re.reshape(1, -1) # 1열로 만들기

    # 재프린팅 필요 여부 ---------------------------------------------------------------------------
    reprint_label = 1 if num_points >= num_noise else 0  

    # 데이터프레임 저장
    new = pd.DataFrame(seriesDF)
    new['brand'] = 'Dessert39'
    new['num_noise'] = num_points
    new['reprint'] = reprint_label  # 재프린팅 라벨 추가
    dataDF = pd.concat([dataDF, new], ignore_index=True) # 데이터 추가

    ## 시각화
    #cv2.imshow('a', re)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()

print(dataDF.head())

# ------------------------------------------------------------------
#RandomForestClassifier
#픽셀 수가 많고(10,000개), 비선형적인 패턴이 있을 가능성이 높기 때문
#로고의 "흐릿함"이라는 애매한 개념은 단순한 직선 경계로는 분류하기 어려움
# ------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. X, y 나누기
X = dataDF.drop(columns=['brand', 'num_noise', 'reprint'])
y = dataDF['reprint']

# 2. 훈련/테스트 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 3. 모델 선택 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 테스트셋 예측
y_pred = model.predict(X_test)

# 5. 성능 평가
print("정확도 (Accuracy):", accuracy_score(y_test, y_pred))
print("\n 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['No Reprint', 'Reprint']))

'''
precision	내가 예측한 것 중 맞은 비율
recall	    실제 정답 중 얼마나 잘 맞췄는지
f1-score	precision과 recall의 조화 평균
support	    정답의 개수
'''

print(dataDF['reprint'].value_counts())


# -------------------------------------------------------
# 새로운 이미지 1장 예측 함수
# -------------------------------------------------------
def predict_one_image(model, image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("이미지 로드 실패:", image_path)
        return

    img = cv2.resize(img, (100, 100))
    flat = img.reshape(1, -1)  # 1차원 벡터로 변환
    flat = flat / 255.0        # 정규화 (선택적)

    pred = model.predict(flat)[0]
    print("\n예측 결과:", "재프린팅 필요" if pred == 1 else "재프린팅 불필요")
    return pred

# -------------------------------------------------------
# 실제 이미지 예측 실행
# -------------------------------------------------------
predict_one_image(model, './data/damage.jpg')




#-------------------------------------------------------------------
import joblib
joblib.dump(model, './cgi-bin/reprint_model.pkl')
#-------------------------------------------------------------------
