# !pip install gradio openpyxl
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gradio as gr

# 엑셀 파일 업로드
from google.colab import files
uploaded = files.upload()

# 데이터 로딩 및 전처리
df = pd.read_excel("Validation.xlsx")

df[['사람문장1', '사람문장2', '사람문장3']] = df[['사람문장1', '사람문장2', '사람문장3']].fillna('')
df['문장'] = df['사람문장1'] + ' ' + df['사람문장2'] + ' ' + df['사람문장3']

# ---------- 감정 표현 치환 사전 ----------
emotion_synonyms = {
    "화나": ["짜증나", "열받아", "불쾌해"],
    "기뻐": ["행복해", "즐거워", "좋아"],
    "불안": ["초조해", "긴장돼", "걱정돼"],
    "슬퍼": ["우울해", "마음이 아파", "눈물나"],
    "무서워": ["두려워", "겁이나", "불안해"],
    "너무": ["정말", "매우", "되게"]
}

# ---------- 증강 함수 ----------
def augment_text(text, shuffle_prob=0.5, synonym_prob=0.7):
    words = text.split()
    if len(words) > 3 and random.random() < shuffle_prob:
        random.shuffle(words)
        text = " ".join(words)
    for key, replacements in emotion_synonyms.items():
        if key in text and random.random() < synonym_prob:
            text = re.sub(key, random.choice(replacements), text)
    return text

# ---------- 증강 데이터 생성 ----------
augmented_df = df.copy()
augmented_df['문장'] = df['문장'].apply(lambda x: augment_text(x))

# 원본 + 증강 합치기
full_df = pd.concat([df[['문장', '감정_대분류']], augmented_df[['문장', '감정_대분류']]], ignore_index=True)

# ---------- 라벨 인코딩 ----------
le = LabelEncoder()
y = le.fit_transform(full_df['감정_대분류'])

# ---------- 벡터화 ----------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(full_df['문장'])

# ---------- 학습 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
model.fit(X_train, y_train)

# ---------- 예측 함수 ----------
def predict_emotion(text):
    vec = vectorizer.transform([text])
    if vec.nnz == 0:
        return "판별 불가"
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

# ---------- Gradio UI ----------
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, label="한글 문장 입력"),
    outputs=gr.Label(label="예측된 감정 대분류"),
    title="한글 감정 예측기 (증강 학습 버전)",
    description="문장을 입력하면 감정 대분류를 예측합니다. 예: '회사에서 실수해서 스트레스 받아.'"
)

iface.launch(share=True)
