# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:22:02 2025

@author: Choe JongHyeon
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# (선택) GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

# 1. 2차원 입력 데이터 생성 (3개 클래스)
np.random.seed(42)

N = 1000
X = np.random.randn(N, 2)

# 선형결합 3개 만들어서 argmax로 클래스 지정 (약간 비선형 섞어도 됨)
logits1 = 0.8 * X[:, 0] - 0.2 * X[:, 1]
logits2 = -0.3 * X[:, 0] + 0.9 * X[:, 1]
logits3 = 0.1 * X[:, 0] - 0.8 * X[:, 1]

logits = np.stack([logits1, logits2, logits3], axis=1)
y = np.argmax(logits, axis=1)  # label: 0, 1, 2

print("X shape:", X.shape)
print("y shape:", y.shape, "classes:", np.unique(y))

# 2. MLP 모델 정의 (다중 클래스)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),  # 은닉층: 뉴런 16개
    keras.layers.Dense(3, activation='softmax')                   # 출력층: 클래스 수(3개)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # y가 정수 레이블(0,1,2)일 때
    metrics=['accuracy']
)

# 3. 학습
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    verbose=0  # 진행 표시 줄 보기 원하면 1로
)

print("최종 train accuracy:", history.history['accuracy'][-1])

# 4. 결정 경계 그리기
#   - X 범위 기준으로 grid 생성
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# 각 grid point에 대해 softmax 출력 → argmax로 클래스 예측
probs = model.predict(grid, verbose=0)
pred_classes = np.argmax(probs, axis=1)
Z = pred_classes.reshape(xx.shape)

# 5. 플롯
plt.figure(figsize=(6, 5))
# 결정 영역
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5,0.5,1.5,2.5])

# 실제 데이터 산점도
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

plt.title("3-Class MLP Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.show()
