import os
import cv2
import numpy as np
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk, messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import spacy

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================
#  TEXT-BASED EMOTION MODEL
# ==============================

print("🔹 Loading dataset...")
df = pd.read_csv('cleaned_dataset_medium_grained.csv')
df = df.sample(n=20000, random_state=42)

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

df['preprocessed_comment'] = df['text'].apply(preprocess)

encoder = LabelEncoder()
df['emotion_num'] = encoder.fit_transform(df['Emotion'])
class_names = sorted(df['Emotion'].unique().tolist())

X = df['preprocessed_comment']
y = df['emotion_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

v = TfidfVectorizer(max_features=10000)
X_train_cv = v.fit_transform(X_train)
X_test_cv = v.transform(X_test)

# Train three models
print("🔹 Training Text Models...")
LR_model = LogisticRegression(max_iter=300, random_state=42)
LR_model.fit(X_train_cv, y_train)

CMB_model = ComplementNB(alpha=0.3)
CMB_model.fit(X_train_cv, y_train)

XGB_model = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
    eval_metric='mlogloss', random_state=42
)
XGB_model.fit(X_train_cv, y_train)

best_w_lr, best_w_nb, best_w_xgb = 0.5, 0.25, 0.25

def predict_text_emotion(text):
    processed = [preprocess(text)]
    text_vc = v.transform(processed)
    proba_lr = normalize(LR_model.predict_proba(text_vc), norm='l1', axis=1)
    proba_nb = normalize(CMB_model.predict_proba(text_vc), norm='l1', axis=1)
    proba_xgb = normalize(XGB_model.predict_proba(text_vc), norm='l1', axis=1)
    combined = (best_w_lr * proba_lr) + (best_w_nb * proba_nb) + (best_w_xgb * proba_xgb)
    idx = np.argmax(combined, axis=1)[0]
    return class_names[idx], combined[0]


# ==============================
#  VIDEO-BASED EMOTION MODEL
# ==============================

print("🔹 Loading CNN model for video...")
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)), Dropout(0.25),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)), Dropout(0.25),
    Flatten(), Dense(1024, activation='relu'),
    Dropout(0.5), Dense(7, activation='softmax')
])

if os.path.exists('model.h5'):
    model.load_weights('model.h5')
else:
    messagebox.showwarning("Model Missing", "Trained CNN weights 'model.h5' not found!")

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

color_dict = {
    "Angry": (0, 0, 255), "Disgusted": (128, 0, 128),
    "Fearful": (255, 0, 0), "Happy": (0, 255, 0),
    "Neutral": (255, 255, 255), "Sad": (255, 0, 255),
    "Surprised": (0, 255, 255)
}


def start_video_detection():
    cap = cv2.VideoCapture(0)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]
            cv2.putText(frame, emotion, (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_dict[emotion], 2, cv2.LINE_AA)

        cv2.imshow('Video Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==============================
#  TKINTER GUI
# ==============================

root = tk.Tk()
root.title("🎭 Multimodal Emotion Detection System")
root.geometry("700x600")
root.configure(bg="#1e1e1e")

title = tk.Label(root, text="🎭 Emotion Detection (Text + Video)",
                 font=("Arial", 18, "bold"), fg="#00ffcc", bg="#1e1e1e")
title.pack(pady=20)

# Text Entry Box
text_entry = tk.Text(root, height=6, width=70, font=("Arial", 12))
text_entry.pack(pady=10)

def predict_text():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Error", "Please enter some text.")
        return
    emotion, proba = predict_text_emotion(text)
    top3 = np.argsort(proba)[::-1][:3]
    top_str = "\n".join([f"{class_names[i]}: {proba[i]:.3f}" for i in top3])
    result_label.config(
        text=f"🧠 Predicted Emotion: {emotion}\n\nTop 3:\n{top_str}",
        fg="#00ff88"
    )

# Buttons
predict_btn = ttk.Button(root, text="🔤 Analyze Text Emotion", command=predict_text)
predict_btn.pack(pady=10)

video_btn = ttk.Button(root, text="🎥 Start Video Detection", command=start_video_detection)
video_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Consolas", 12),
                        fg="#00ff88", bg="#1e1e1e", justify="left")
result_label.pack(pady=20)

note_label = tk.Label(root, text="Press 'q' to stop video feed",
                      font=("Arial", 10), fg="#aaa", bg="#1e1e1e")
note_label.pack(side="bottom", pady=10)

root.mainloop()
