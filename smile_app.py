# Streamlitを使って画像をアップロードし、笑顔を検出するアプリケーション
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

def detect_smiles(image, model):
    # 画像をグレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔検出用のカスケード分類器を読み込む
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = gray_image[y:y+h, x:x+w]

        # 顔の画像をPyTorch用に変換
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        face_tensor = transform(face_roi).unsqueeze(0)

        # 笑顔を検出
        with torch.no_grad():
            smile_prob = F.softmax(model(face_tensor), dim=1)[:, 1].item()

        if smile_prob > 0.5:
            cv2.putText(image, f"Smile: {smile_prob:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


# 学習済みモデルをロードする前に、モデルアーキテクチャを定義
class SmileModel(torch.nn.Module):
    def __init__(self):
        super(SmileModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 64 * 10 * 10)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# 事前学習済みモデルをロードするためのコードを追加
pretrained_model_path = 'pretrained_model.pth'
model = SmileModel()
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

st.title("Smile Detector")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    detected_image = detect_smiles(image, model)
    st.image(detected_image, caption="Detected smiles.", use_column_width=True)


# 最後に、Streamlitアプリケーションを実行して動作を確認しましょう。
# streamlit run smile_app.py

