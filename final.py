import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import re
import joblib
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import sounddevice as sd
import google.generativeai as genai 

# Config / paths
CNN_WEIGHTS = "model.h5"
HAAR_PATH = "haarcascade_frontalface_default.xml"
WAV2VEC_DIR = "wav2vec2_emotion_model1"

MODEL_PATH = "lgbm_sbert.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Gemini model 
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Utilities
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

# Gemini setup (DIRECT KEY)
GEMINI_API_KEY = "AIzaSyBDlcMUn0gs79wd8OMlNPPwKXEBBA5x0R4" 

if not GEMINI_API_KEY.strip():
    messagebox.showerror(
        "Gemini API Key Missing",
        "Please paste your Gemini API key into GEMINI_API_KEY in the code."
    )
    raise SystemExit

genai.configure(api_key=GEMINI_API_KEY)
try:
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    messagebox.showerror("Gemini Error", f"Failed to initialize Gemini model:\n{e}")
    raise SystemExit

# Text Model (SBERT + LightGBM)
safe_print("🔹 Loading saved text model artifacts...")

if not (os.path.exists(MODEL_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)):
    messagebox.showerror("Missing files", "One or more saved text model artifacts not found.")
    raise SystemExit

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
sbert_model = SentenceTransformer('all-mpnet-base-v2')

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile("["  # emojis
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


def predict_text_emotion_with_probs(text):
    if not text.strip():
        return "neutral", 0.0, np.zeros(len(le.classes_))
    cleaned = clean_text(text)
    emb = sbert_model.encode([cleaned], convert_to_numpy=True)
    emb_scaled = scaler.transform(emb)

    pred_label = clf.predict(emb_scaled)[0]
    pred_probs = clf.predict_proba(emb_scaled)[0]
    emotion = le.inverse_transform([pred_label])[0]
    return emotion.lower(), float(pred_probs[pred_label]), pred_probs

# Video CNN Model
safe_print("🔹 Preparing CNN model for video...")
video_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),
    Flatten(), Dense(1024, activation='relu'), Dropout(0.5),
    Dense(7, activation='softmax')
])

if os.path.exists(CNN_WEIGHTS):
    try:
        video_cnn.load_weights(CNN_WEIGHTS)
        safe_print("🔹 Loaded CNN weights.")
    except Exception as e:
        messagebox.showwarning("Warning", f"Could not load CNN weights: {e}")
else:
    messagebox.showwarning(
        "Warning",
        f"CNN weights '{CNN_WEIGHTS}' not found; video will run but predictions may be invalid."
    )

emotion_dict_vid = {
    0: "angry", 1: "disgusted", 2: "fearful",
    3: "happy", 4: "neutral", 5: "sad", 6: "surprised"
}

# Color map (BGR)
video_color_map = {
    "sad":       (255,   0, 255),  # magenta-ish
    "disgusted": (128,   0, 128),  # purple
    "angry":     (  0,   0, 255),  # red
    "happy":     (  0, 255,   0),  # green
    "fearful":   (255,   0,   0),  # blue-ish
    "surprised": (  0, 255, 255),  # yellow/cyan
    "neutral":   (255, 255, 255),  # white
}


def predict_frame_emotion(frame_gray):
    try:
        img = cv2.resize(frame_gray, (48, 48))
        arr = np.expand_dims(np.expand_dims(img, -1), 0).astype('float32') / 255.0
        logits = video_cnn.predict(arr, verbose=0)
        probs = logits[0]
        idx = int(np.argmax(probs))
        return emotion_dict_vid[idx], float(probs[idx]), probs
    except Exception:
        return "neutral", 0.0, np.zeros(7)

# Audio Model (wav2vec2)
safe_print("🔹 Loading audio model (wav2vec2)...")
if not os.path.exists(WAV2VEC_DIR):
    messagebox.showerror("Audio model missing", f"Audio model directory '{WAV2VEC_DIR}' not found.")
    raise SystemExit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_DIR)
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC_DIR).to(device)
wav2vec_model.eval()

label_map_audio = {
    0: "fear", 1: "angry", 2: "disgust", 3: "neutral",
    4: "sad", 5: "pleasant_surprise", 6: "happy"
}

audio_stream = None
audio_buffer = []
audio_fs = 16000


def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        safe_print(status)
    audio_buffer.append(indata.copy())


def start_audio_stream():
    global audio_stream, audio_buffer
    if audio_stream is not None:
        return
    audio_buffer = []
    audio_stream = sd.InputStream(
        samplerate=audio_fs,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    audio_stream.start()
    safe_print("🎙 Audio stream started")


def stop_audio_stream_and_get_audio():
    global audio_stream, audio_buffer
    if audio_stream is None:
        return None
    try:
        audio_stream.stop()
        audio_stream.close()
    except Exception as e:
        safe_print("Audio stop error:", e)
    audio_stream = None
    if not audio_buffer:
        return None
    audio = np.concatenate(audio_buffer, axis=0).squeeze()
    audio_buffer = []
    safe_print("🎙 Audio stream stopped, length samples:", len(audio))
    return audio


def predict_audio_emotion_and_probs(audio_np, sampling_rate=16000):
    inputs = processor(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    with torch.no_grad():
        out = wav2vec_model(input_values, attention_mask=attention_mask)
        logits = out.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    label = label_map_audio.get(idx, "neutral")
    confidence = float(probs[idx])
    return label, confidence, probs

# Fusion logic 
def normalize_label(label):
    lab = label.lower()
    if "ang" in lab or "anger" in lab:
        return "anger"
    if "disgust" in lab:
        return "disgust"
    if "fear" in lab or "scared" in lab:
        return "fear"
    if "surpris" in lab or "pleasant_surprise" in lab:
        return "surprise"
    if "happy" in lab or "joy" in lab:
        return "happy"
    if "sad" in lab or "sorrow" in lab:
        return "sad"
    if "neutral" in lab or lab.strip() == "":
        return "neutral"
    return lab


def fuse_modalities_dynamic(modalities):
    """
    modalities: dict like {"text": (label, conf), "video": (...), "audio": (...)}
    Uses only keys that exist in the dict.
    """
    if not modalities:
        return "neutral", {}

    from collections import Counter

    normalized = {k: normalize_label(v[0]) for k, v in modalities.items()}
    votes = list(normalized.values())
    cnt = Counter(votes)
    most_common, count = cnt.most_common(1)[0]

    if len(votes) == 1:
        # Only one modality -> just use it
        return most_common, modalities

    # Majority vote (2/3, 2/2, 3/3)
    if count >= 2:
        return most_common, modalities

    # Otherwise fall back to highest confidence among given modalities
    best_mod = max(modalities.keys(), key=lambda k: modalities[k][1])
    best_label = normalized[best_mod]
    return best_label, modalities

# Gemini-based chatbot

# Map fused emotion to a tone guideline 
TONE_HINTS = {
    "happy": (
        "Use an overall upbeat, optimistic, and light tone. Be encouraging and a bit playful."
    ),
    "sad": (
        "Use a very gentle, validating, and comforting tone. Focus on empathy and reassurance."
    ),
    "anger": (
        "Use a calm, grounding tone. Acknowledge frustration but stay steady and de-escalating."
    ),
    "fear": (
        "Use a soothing, reassuring tone. Emphasize safety, clarity, and step-by-step support."
    ),
    "disgust": (
        "Use a composed, understanding tone. Acknowledge discomfort without judgment."
    ),
    "surprise": (
        "Use a curious, warm, slightly energetic tone. Match the surprise with positive curiosity."
    ),
    "neutral": (
        "Use a balanced, neutral, friendly tone. Neither overly cheerful nor overly serious."
    ),
}

def generate_bot_response(user_text, fused_emotion, details):
    """
    Use ONLY Gemini for the reply.
    Fused emotion is the ground-truth for tone and sentiment polarity.
    The model must NOT emotionally react to the literal text if it conflicts with fused_emotion.
    """
    fused_lower = fused_emotion.lower()

    # Build small summary of modalities for context
    lines = []
    for name, (lbl, conf) in details.items():
        lines.append(f"- {name.capitalize()} prediction: {lbl} (confidence {conf:.3f})")
    modality_block = "\n".join(lines) if lines else "(no modalities)"

    tone_hint = TONE_HINTS.get(
        fused_lower,
        "Use a balanced, neutral, friendly tone."
    )

    # Polarity rules: hard constraints for the model
    if fused_lower in ["happy", "surprise", "pleasant_surprise"]:
        polarity_rules = (
            "Overall emotional polarity MUST be POSITIVE.\n"
            "- Do NOT describe the user as sad, depressed, low, exhausted, hopeless, overwhelmed, or struggling.\n"
            "- Avoid phrases like 'difficult feelings', 'feeling low', 'hurting', 'in pain', 'really down', etc.\n"
            "- Focus instead on encouragement, appreciation, motivation, or light support with a positive framing.\n"
        )
    elif fused_lower in ["sad", "anger", "fear", "disgust"]:
        polarity_rules = (
            "Overall emotional polarity MUST reflect a NEGATIVE state that you respond to supportively.\n"
            "- It is okay to acknowledge struggle, pain, worry, or frustration.\n"
            "- Do NOT act overly cheerful or say that everything is great.\n"
            "- Avoid lines that sound like generic positivity ignoring their emotion.\n"
        )
    else:  # neutral
        polarity_rules = (
            "Overall emotional polarity MUST be NEUTRAL.\n"
            "- Avoid strong emotional language (very happy, very sad, furious, terrified, etc.).\n"
            "- Stay factual, clear, and lightly friendly.\n"
        )

    prompt = (
        "You are an empathetic, safe, and helpful chatbot inside a desktop app that does "
        "multimodal emotion detection (text, video, audio).\n\n"
        "The app has already detected the user's current emotion using a trained model. "
        "You MUST treat this detected emotion as the ground truth for their emotional state. "
        "Do NOT try to re-infer or override the emotion based on the wording of the text.\n\n"
        f"Detected fused emotion (from the model): {fused_emotion}\n"
        "Modalities used:\n"
        f"{modality_block}\n\n"
        f"Tone guideline based on fused emotion:\n{tone_hint}\n\n"
        "Strict sentiment/polarity rules (you MUST obey these even if the text sounds opposite):\n"
        f"{polarity_rules}\n"
        "User message (this may sound emotionally different, but the fused emotion above is more reliable):\n"
        f"{user_text}\n\n"
        "Instructions (very important):\n"
        "- Let the fused emotion and the polarity rules control your emotional tone and word choice.\n"
        "- Still answer the user's literal question or topic correctly and logically.\n"
        "- Do NOT say things like 'you seem sad/angry/happy' unless that matches the fused emotion.\n"
        "- Do NOT describe an emotional state that conflicts with the fused emotion or polarity rules.\n"
        "- Respond in 2–4 short sentences.\n"
        "- Be kind, non-judgmental, and conversational.\n"
    )

    resp = gemini_model.generate_content(prompt)
    text = getattr(resp, "text", "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty response.")
    return text

# Tkinter UI
root = tk.Tk()
root.title("🎭 Emotion-Aware Chatbot (Multimodal + Gemini)")
root.geometry("1000x720")
root.configure(bg="#e5ecff")  

style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass

ACCENT = "#3b82f6"        
ACCENT_DARK = "#2563eb"
BG_MAIN = "#f5f7fb"
CARD_BG = "#ffffff"
PANEL_BG = "#ffffff"
BUBBLE_USER = "#dbeafe"
BUBBLE_BOT = "#f3f4f6"
BUBBLE_SYS = "#fef9c3"
TEXT_DARK = "#111827"
MUTED = "#6b7280"

style.configure("TFrame", background=BG_MAIN)
style.configure("Header.TFrame", background=BG_MAIN)
style.configure("ChatCard.TLabelframe", background=CARD_BG, borderwidth=0, foreground=TEXT_DARK)
style.configure("ChatCard.TLabelframe.Label",
                background=CARD_BG, foreground=TEXT_DARK, font=("Segoe UI", 10, "bold"))
style.configure("SideCard.TLabelframe", background=CARD_BG, borderwidth=0, foreground=TEXT_DARK)
style.configure("SideCard.TLabelframe.Label",
                background=CARD_BG, foreground=TEXT_DARK, font=("Segoe UI", 10, "bold"))
style.configure("TLabel", background=BG_MAIN, foreground=TEXT_DARK, font=("Segoe UI", 10))
style.configure("Status.TLabel", background="#dbeafe", foreground="#1e3a8a", font=("Segoe UI", 9))
style.configure("Accent.TButton",
                font=("Segoe UI", 10, "bold"),
                padding=8,
                relief="flat")
style.map("Accent.TButton",
          background=[("active", ACCENT_DARK), ("!disabled", ACCENT)],
          foreground=[("!disabled", "#f9fafb")])
style.configure("Ghost.TButton",
                font=("Segoe UI", 10),
                padding=6,
                relief="flat",
                background=BG_MAIN)
style.map("Ghost.TButton",
          foreground=[("!disabled", TEXT_DARK)],
          background=[("active", "#e5e7eb")])

header = ttk.Frame(root, style="Header.TFrame", padding=(20, 16, 20, 8))
header.pack(fill="x")

title_lbl = ttk.Label(
    header,
    text="✨ Emotion-Aware Chatbot",
    font=("Segoe UI", 20, "bold"),
    background=BG_MAIN,
    foreground=TEXT_DARK
)
title_lbl.pack(side="left", anchor="w")

subtitle_lbl = ttk.Label(
    header,
    text="Multimodal Emotion • Gemini 2.5 Flash",
    font=("Segoe UI", 9),
    background=BG_MAIN,
    foreground=MUTED
)
subtitle_lbl.pack(side="left", padx=(10, 0))

engine_lbl = ttk.Label(
    header,
    text=f"Engine: {GEMINI_MODEL_NAME}",
    font=("Segoe UI", 9, "italic"),
    background=BG_MAIN,
    foreground="#059669"
)
engine_lbl.pack(side="right")

content = ttk.Frame(root, style="TFrame", padding=(20, 8, 20, 10))
content.pack(fill="both", expand=True)
content.columnconfigure(0, weight=3)
content.columnconfigure(1, weight=2)
content.rowconfigure(0, weight=1)
content.rowconfigure(1, weight=0)

chat_card = ttk.Labelframe(content, text="💬 Conversation", padding=12, style="ChatCard.TLabelframe")
chat_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 10))
chat_card.rowconfigure(0, weight=1)
chat_card.columnconfigure(0, weight=1)

chat_display = ScrolledText(
    chat_card,
    state="disabled",
    wrap="word",
    font=("Segoe UI", 10),
    bg=PANEL_BG,
    fg=TEXT_DARK,
    relief="flat",
    borderwidth=0
)
chat_display.grid(row=0, column=0, sticky="nsew")

chat_display.tag_config("user", background=BUBBLE_USER, foreground=TEXT_DARK,
                        lmargin1=8, lmargin2=8, rmargin=80, spacing1=4, spacing3=6)
chat_display.tag_config("bot", background=BUBBLE_BOT, foreground=TEXT_DARK,
                        lmargin1=40, lmargin2=40, rmargin=8, spacing1=4, spacing3=6)
chat_display.tag_config("system", background=BUBBLE_SYS, foreground="#78350f",
                        lmargin1=8, lmargin2=8, rmargin=8, spacing1=4, spacing3=6)

def append_chat(role, text):
    chat_display.config(state="normal")
    if role == "user":
        chat_display.insert(tk.END, "You:\n", "user")
        chat_display.insert(tk.END, text.strip() + "\n\n", "user")
    elif role == "bot":
        chat_display.insert(tk.END, "Bot:\n", "bot")
        chat_display.insert(tk.END, text.strip() + "\n\n", "bot")
    elif role == "system":
        chat_display.insert(tk.END, text.strip() + "\n\n", "system")
    chat_display.config(state="disabled")
    chat_display.see(tk.END)

def stream_bot_message(text, delay=15):
    """Typewriter-style streaming of bot response."""
    text = text.strip()
    if not text:
        return

    chat_display.config(state="normal")
    chat_display.insert(tk.END, "Bot:\n", "bot")
    chat_display.config(state="disabled")
    chat_display.see(tk.END)

    def _type(i=0):
        if i >= len(text):
            chat_display.config(state="normal")
            chat_display.insert(tk.END, "\n\n", "bot")
            chat_display.config(state="disabled")
            chat_display.see(tk.END)
            return
        chat_display.config(state="normal")
        chat_display.insert(tk.END, text[i], "bot")
        chat_display.config(state="disabled")
        chat_display.see(tk.END)
        root.after(delay, lambda: _type(i + 1))

    _type()

input_frame = ttk.Frame(content, style="TFrame")
input_frame.grid(row=1, column=0, sticky="ew", padx=(0, 12))
input_frame.columnconfigure(0, weight=1)
input_frame.columnconfigure(1, weight=0)

msg_entry = tk.Text(
    input_frame,
    height=3,
    font=("Segoe UI", 10),
    wrap="word",
    bg="#ffffff",
    fg=TEXT_DARK,
    relief="solid",
    borderwidth=1,
    insertbackground="#111827"
)
msg_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

send_btn = ttk.Button(
    input_frame,
    text="➤ Send",
    style="Accent.TButton"
)
send_btn.grid(row=0, column=1, sticky="nsew")

right_col = ttk.Frame(content, style="TFrame")
right_col.grid(row=0, column=1, rowspan=2, sticky="nsew")
right_col.rowconfigure(3, weight=1)

mod_card = ttk.Labelframe(right_col, text="🎚 Modalities", padding=12, style="SideCard.TLabelframe")
mod_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))

video_btn = ttk.Button(
    mod_card, text="🎥 Start Video Stream", style="Accent.TButton"
)
video_btn.grid(row=0, column=0, sticky="ew", pady=(2, 4))

audio_btn = ttk.Button(
    mod_card, text="🎙 Start Mic", style="Accent.TButton"
)
audio_btn.grid(row=1, column=0, sticky="ew", pady=(2, 4))

reset_modal_btn = ttk.Button(
    mod_card, text="♻ Reset Modalities", style="Ghost.TButton"
)
reset_modal_btn.grid(row=2, column=0, sticky="ew", pady=(8, 2))

summary_card = ttk.Labelframe(right_col, text="✨ Latest Detection", padding=12, style="SideCard.TLabelframe")
summary_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))

badge_font = ("Segoe UI", 10, "bold")
badge_text = tk.Label(
    summary_card,
    text="🔤 Text: —",
    font=badge_font,
    bg="#e0f2fe",
    fg="#1e3a8a",
    padx=10,
    pady=6,
    anchor="w"
)
badge_text.pack(fill="x", pady=3)

badge_video = tk.Label(
    summary_card,
    text="🎥 Video: —",
    font=badge_font,
    bg="#fef3c7",
    fg="#92400e",
    padx=10,
    pady=6,
    anchor="w"
)
badge_video.pack(fill="x", pady=3)

badge_audio = tk.Label(
    summary_card,
    text="🎙 Audio: —",
    font=badge_font,
    bg="#dcfce7",
    fg="#166534",
    padx=10,
    pady=6,
    anchor="w"
)
badge_audio.pack(fill="x", pady=3)

fused_card = ttk.Labelframe(right_col, text="🎯 Fused Emotion", padding=12, style="SideCard.TLabelframe")
fused_card.grid(row=2, column=0, sticky="ew")

fused_label = tk.Label(
    fused_card,
    text="—",
    font=("Segoe UI", 18, "bold"),
    bg="#eff6ff",
    fg="#1d4ed8",
    padx=12,
    pady=10,
    anchor="center"
)
fused_label.pack(fill="x")

footer_label = ttk.Label(
    right_col,
    text="Start video/mic as long as you want.\nSend = stop, fuse, and stream a Gemini reply.",
    font=("Segoe UI", 9),
    foreground=MUTED,
    background=BG_MAIN
)
footer_label.grid(row=3, column=0, sticky="s", pady=(10, 0))

status_bar = ttk.Label(
    root,
    text="Ready.",
    style="Status.TLabel",
    anchor="w",
    padding=6
)
status_bar.pack(side="bottom", fill="x")


def set_status(text):
    status_bar.config(text=text)

# State
root.last_text = ("neutral", 0.0)
root.last_video = ("neutral", 0.0)
root.last_audio = ("neutral", 0.0)

# Flags: whether video/audio are actually available for current fusion
root.video_available = False
root.audio_available = False

video_running = threading.Event()
video_thread = None


def update_badges():
    t_lbl, t_conf = root.last_text
    badge_text.config(text=f"🔤 Text: {t_lbl} ({t_conf:.2f})" if t_conf > 0 else "🔤 Text: —")

    if root.video_available:
        v_lbl, v_conf = root.last_video
        badge_video.config(text=f"🎥 Video: {v_lbl} ({v_conf:.2f})")
    else:
        badge_video.config(text="🎥 Video: —")

    if root.audio_available:
        a_lbl, a_conf = root.last_audio
        badge_audio.config(text=f"🎙 Audio: {a_lbl} ({a_conf:.2f})")
    else:
        badge_audio.config(text="🎙 Audio: —")

# Video stream (runs until Send or 'q')
def run_video_stream():
    global video_thread
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root.after(0, lambda: (
            messagebox.showerror("Camera Error", "Could not open camera."),
            set_status("Camera error."),
            video_running.clear(),
            video_btn.config(text="🎥 Start Video Stream")
        ))
        return

    if os.path.exists(HAAR_PATH):
        facecasc = cv2.CascadeClassifier(HAAR_PATH)
    else:
        facecasc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    from collections import Counter
    window_buffer = []

    set_status("Video streaming... press 'q' in window or click button to stop.")

    try:
        while video_running.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                emo, conf, probs = predict_frame_emotion(roi)
                window_buffer.append((emo, conf))

                color = video_color_map.get(emo.lower(), (59, 130, 246))

                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), color, 2)
                cv2.putText(frame, emo, (x + 10, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            cv2.imshow("Video Emotion Stream (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_running.clear()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        if window_buffer:
            from collections import Counter
            counts = Counter([w[0] for w in window_buffer])
            most_common, _ = counts.most_common(1)[0]
            confs = [c for (l, c) in window_buffer if l == most_common]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            root.last_video = (most_common, avg_conf)
            root.video_available = True
        else:
            root.last_video = ("neutral", 0.0)
            root.video_available = False

        root.after(0, lambda: (
            update_badges(),
            video_btn.config(text="🎥 Start Video Stream"),
            set_status("Video stream stopped.")
        ))

# UI callbacks
def toggle_video():
    global video_thread
    if not video_running.is_set():
        video_running.set()
        video_btn.config(text="⏹ Stop Video Stream")
        set_status("Starting video stream...")
        video_thread = threading.Thread(target=run_video_stream, daemon=True)
        video_thread.start()
    else:
        video_running.clear()
        set_status("Stopping video stream...")

def toggle_audio():
    global audio_stream
    if audio_stream is None:
        try:
            start_audio_stream()
            audio_btn.config(text="⏹ Stop Mic")
            set_status("Mic recording... will stop on Send or when you click again.")
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not start mic.\n{e}")
            set_status("Audio error.")
    else:
        audio = stop_audio_stream_and_get_audio()
        if audio is not None and len(audio) > 0:
            try:
                lbl, conf, probs = predict_audio_emotion_and_probs(audio, sampling_rate=audio_fs)
                root.last_audio = (lbl, conf)
                root.audio_available = True
                append_chat("system", f"🎙 Audio emotion captured: {lbl} (conf={conf:.3f})")
            except Exception as e:
                root.audio_available = False
                safe_print("Audio prediction error:", e)
                messagebox.showerror("Audio Error", f"Error during audio prediction.\n{e}")
        else:
            root.audio_available = False
        audio_btn.config(text="🎙 Start Mic")
        update_badges()
        set_status("Mic stopped.")

def reset_modalities():
    global audio_stream
    if audio_stream is not None:
        stop_audio_stream_and_get_audio()
        audio_stream = None
    if video_running.is_set():
        video_running.clear()
    root.last_text = ("neutral", 0.0)
    root.last_video = ("neutral", 0.0)
    root.last_audio = ("neutral", 0.0)
    root.video_available = False
    root.audio_available = False
    fused_label.config(text="—")
    update_badges()
    append_chat("system", "♻ Modalities reset to neutral.")
    set_status("Modalities reset.")

def on_send():
    global video_thread, audio_stream

    user_text = msg_entry.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Empty message", "Please type something to send.")
        return

    append_chat("user", user_text)
    msg_entry.delete("1.0", tk.END)
    set_status("Analyzing emotions and generating Gemini response...")

    def _job():
        global video_thread

        # Stop video if running
        if video_running.is_set():
            video_running.clear()
            if video_thread is not None:
                video_thread.join(timeout=5)

        # Stop mic if running
        if audio_stream is not None:
            audio = stop_audio_stream_and_get_audio()
            if audio is not None and len(audio) > 0:
                try:
                    lbl_a, conf_a, probs_a = predict_audio_emotion_and_probs(audio, sampling_rate=audio_fs)
                    root.last_audio = (lbl_a, conf_a)
                    root.audio_available = True
                except Exception as e:
                    root.audio_available = False
                    safe_print("Audio prediction error:", e)
            root.after(0, lambda: audio_btn.config(text="🎙 Start Mic"))

        # Text emotion 
        txt_lbl, txt_conf, txt_probs = predict_text_emotion_with_probs(user_text)
        root.last_text = (txt_lbl, txt_conf)

        # Build modalities dict based on availability
        modalities = {"text": (txt_lbl, txt_conf)}
        if root.video_available:
            modalities["video"] = root.last_video
        if root.audio_available:
            modalities["audio"] = root.last_audio

        fused, details = fuse_modalities_dynamic(modalities)

        # After using them for this turn, clear video/audio availability
        root.video_available = False
        root.audio_available = False

        # Analysis block
        analysis_lines = ["🧪 Emotion analysis:"]
        for name, (lbl, conf) in details.items():
            analysis_lines.append(f"• {name.capitalize()} → {lbl} (conf={conf:.3f})")
        analysis_lines.append(f"• Fused result → {fused}")
        analysis_text = "\n".join(analysis_lines)

        try:
            bot_text = generate_bot_response(user_text, fused, details)
        except Exception as e:
            bot_text = (
                "I ran into an error while talking to Gemini:\n"
                f"{e}\n\n"
                "Please check your internet connection and API key."
            )

        def _update_ui():
            fused_label.config(text=fused.upper())
            update_badges()
            append_chat("system", analysis_text)
            stream_bot_message(bot_text)
            set_status("Response generated.")

        root.after(0, _update_ui)

    threading.Thread(target=_job, daemon=True).start()

def on_enter_key(event):
    # Enter sends
    if event.state & 0x0001:  
        return
    on_send()
    return "break"

send_btn.config(command=on_send)
msg_entry.bind("<Return>", on_enter_key)
video_btn.config(command=toggle_video)
audio_btn.config(command=toggle_audio)
reset_modal_btn.config(command=reset_modalities)

# Initial UI state
update_badges()
append_chat("system", "👋 Hi! I’m your multimodal, Gemini-powered chatbot.\n"
                      "Capture video/audio if you want, then send me a message.")
set_status("Ready to chat.")

root.mainloop()

