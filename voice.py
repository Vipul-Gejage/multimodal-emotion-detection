import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ------------ Load Model -------------
model_dir = "wav2vec2_emotion_model1"   # your downloaded folder
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
model.eval()

# ------------ Label Map -------------
label_map = {
    0: "fear",
    1: "angry",
    2: "disgust",
    3: "neutral",
    4: "sad",
    5: "pleasant_surprise",
    6: "happy"
}

# ------------ Recording Function -------------
def record_audio(duration=5, fs=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("🎤 Done.")
    return audio.squeeze()

# ------------ Predict Function -------------
def predict(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred = torch.argmax(logits, dim=-1).item()
    return label_map[pred]

# ------------ Main -------------
audio = record_audio()
print("Predicted Emotion:", predict(audio))
