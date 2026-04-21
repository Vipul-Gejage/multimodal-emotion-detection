import re
import joblib
from sentence_transformers import SentenceTransformer

# Load saved artifacts once (change paths accordingly)
MODEL_PATH = "lgbm_sbert.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load models
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Load SBERT model
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Text cleaning function (same as training)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile("["
                      u"\U0001F600-\U0001F64F"
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"
                      u"\U0001F1E0-\U0001F1FF"
                      "]+", flags=re.UNICODE)

def clean_text(t):
    t = str(t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r"[^A-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()


def predict_emotion_from_saved(text):
    cleaned = clean_text(text)
    emb = sbert_model.encode([cleaned], convert_to_numpy=True)
    emb_scaled = scaler.transform(emb)
    
    pred_label = clf.predict(emb_scaled)[0]
    pred_probs = clf.predict_proba(emb_scaled)[0]
    
    emotion = le.inverse_transform([pred_label])[0]
    
    print(f"\nInput Text: {text}")
    print(f"Predicted Emotion: {emotion}")
    print("Probabilities:")
    for emo, prob in zip(le.classes_, pred_probs):
        print(f"  {emo}: {prob:.4f}")
    
    return emotion

predict_emotion_from_saved("I felt completely empty after she walked away")
predict_emotion_from_saved("i went to office and saw that my crush already seating with my manger and both looking at me")
predict_emotion_from_saved("I didn't expect to see him here today")
predict_emotion_from_saved("I can't believe they blamed me for their mistake.")
predict_emotion_from_saved("I really miss him and it hurts")