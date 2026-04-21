# System Architecture & Design

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Tkinter GUI)                 │
│                   ✨ Modern, Professional Design                │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├─────────────────────┬──────────────────────────────────────┐
       │                     │                                      │
       ▼                     ▼                                      ▼
   ┌────────┐            ┌────────┐                          ┌──────────┐
   │  TEXT  │            │ VIDEO  │                          │  AUDIO   │
   │ INPUT  │            │ STREAM │                          │ STREAM   │
   └────┬───┘            └────┬───┘                          └────┬─────┘
        │                     │                                    │
        ▼                     ▼                                    ▼
   ┌─────────────────┐  ┌──────────────────┐            ┌──────────────────┐
   │  TEXT CLEANING  │  │  FACE DETECTION  │            │ AUDIO BUFFERING  │
   │  • URL removal  │  │  • Haar Cascade  │            │  • 16kHz sampling│
   │  • Emoji remove │  │  • Crop 48×48    │            │  • Mono channel  │
   │  • Lemmatize    │  │  • Normalization │            │                  │
   └────────┬────────┘  └────────┬─────────┘            └────────┬─────────┘
            │                    │                               │
            ▼                    ▼                               ▼
   ┌──────────────────────┐  ┌─────────────┐         ┌──────────────────┐
   │ SBERT EMBEDDINGS     │  │  CNN MODEL  │         │  WAV2VEC2 MODEL  │
   │ • all-mpnet-base-v2  │  │ • 4-layer   │         │ • Transformer    │
   │ • 768-dim vectors    │  │ • Conv2D(32,│         │ • Pre-trained    │
   │                      │  │    64, 128) │         │ • Fine-tuned     │
   └────────┬─────────────┘  │ • Dense out │         └────────┬──────────┘
            │                 │ • Softmax   │                  │
            │                 └─────┬───────┘                  │
            │                       │                          │
            ▼                       ▼                          ▼
   ┌──────────────────────┐  ┌─────────────┐         ┌──────────────────┐
   │  LIGHTGBM            │  │ PREDICTION  │         │  PREDICTION      │
   │ CLASSIFIER           │  │ • Emotion   │         │ • Emotion        │
   │ • Trained on 16K     │  │ • Confidence│         │ • Confidence     │
   │ • 7-class output     │  │   score     │         │   score          │
   │                      │  │             │         │                  │
   └────────┬─────────────┘  └─────┬───────┘         └────────┬──────────┘
            │                      │                          │
            └──────────┬───────────┴──────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   EMOTION FUSION ENGINE          │
        │                                  │
        │  Input: 3 emotion predictions    │
        │         (label, confidence)      │
        │                                  │
        │  Algorithm:                      │
        │  1. Normalize labels             │
        │  2. Majority voting              │
        │  3. Confidence weighting         │
        │  4. Output: Fused emotion        │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │   GEMINI API INTEGRATION         │
        │                                  │
        │  Input:                          │
        │  • User message                  │
        │  • Fused emotion                 │
        │  • Modality details              │
        │  • Tone guidelines               │
        │  • Polarity rules                │
        │                                  │
        │  Processing:                     │
        │  • LLM inference                 │
        │  • Emotion-aware response        │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │   RESPONSE DISPLAY               │
        │   • Typewriter effect            │
        │   • Analysis badges              │
        │   • Chat history                 │
        └──────────────────────────────────┘
```

---

## 📊 Data Flow Pipeline

### **Text Emotion Detection Flow**
```
User Text
    ↓
[Regex Cleaning]
  • Remove URLs (http://, www.)
  • Remove @mentions
  • Remove HTML tags
  • Remove emojis
  • Remove special characters
  • Lowercase conversion
    ↓
[SBERT Encoding]
  • Input: cleaned text
  • Model: all-mpnet-base-v2
  • Output: 768-dim vector
    ↓
[Feature Scaling]
  • StandardScaler
  • Fitted on training data
    ↓
[LightGBM Inference]
  • Input: scaled embedding
  • Forward pass
  • Output: 7-class probabilities
    ↓
[Label Decoding]
  • Inverse transform
  • Get emotion name
    ↓
[Confidence Extraction]
  • Get probability for predicted class
  • Return: (emotion, confidence)
```

### **Video Emotion Detection Flow**
```
Webcam Frame
    ↓
[Haar Cascade Detection]
  • Detect faces in frame
  • Extract face region (x, y, w, h)
    ↓
[Face Crop & Resize]
  • Crop: gray[y:y+h, x:x+w]
  • Resize: 48×48 pixels
  • Normalize: divide by 255
    ↓
[Expand Dimensions]
  • Add batch dimension: (1, 48, 48, 1)
  • Convert to float32
    ↓
[CNN Forward Pass]
  • Conv2D(32) → ReLU → Conv2D(64) → ReLU
  • MaxPool(2×2) → Dropout(0.25)
  • Conv2D(128) → MaxPool → Conv2D(128) → MaxPool
  • Dropout(0.25)
  • Flatten → Dense(1024) → ReLU → Dropout(0.5)
  • Dense(7) → Softmax
  • Output: [p0, p1, p2, p3, p4, p5, p6]
    ↓
[Prediction & Confidence]
  • argmax(probabilities) → emotion_idx
  • max(probabilities) → confidence
    ↓
[Temporal Smoothing]
  • Add to window buffer (last 30 frames)
  • Count emotion occurrences
  • Return most common emotion
```

### **Audio Emotion Detection Flow**
```
Microphone Stream
    ↓
[Real-time Audio Capture]
  • Sampling rate: 16 kHz
  • Channels: 1 (mono)
  • Callback function collects chunks
  • Stored in audio_buffer
    ↓
[Audio Concatenation]
  • Combine all buffer chunks
  • Create continuous audio array
  • Length: variable (depends on duration)
    ↓
[Wav2Vec2 Preprocessing]
  • Processor.feature_extractor
  • Convert to mel-spectrogram features
  • Padding to max_length
  • Return: input_values, attention_mask
    ↓
[Wav2Vec2 Model Inference]
  • encoder.wav2vec_model()
  • Input: feature tensor
  • Hidden states: transformer layers
  • Output: logits (7-class)
    ↓
[Softmax & Confidence]
  • Apply softmax to logits
  • argmax → emotion_idx
  • max → confidence
    ↓
[Emotion Mapping]
  • 0→fear, 1→angry, 2→disgust
  • 3→neutral, 4→sad
  • 5→pleasant_surprise, 6→happy
    ↓
[Return: (emotion, confidence)]
```

---

## 🔄 Fusion Algorithm

### **Multimodal Fusion Strategy**

```
┌─────────────────────────────────────────────────────┐
│  Input: Dictionary of available modalities          │
│  {                                                  │
│    "text": ("happy", 0.85),                         │
│    "video": ("happy", 0.72),                        │
│    "audio": ("angry", 0.68)    # ← MINORITY         │
│  }                                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Step 1: Normalize  │
        │ Label Names        │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Normalized:                        │
        │ "happy" → "happy"                  │
        │ "happy" → "happy"                  │
        │ "angry" → "anger"                  │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Step 2: Majority Vote              │
        │ Count("happy") = 2                 │
        │ Count("anger") = 1                 │
        │                                    │
        │ Maximum count: 2                   │
        │ Is 2 >= 2? YES ✓                   │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Step 3: Return Fused Result        │
        │ Emotion: "happy"                   │
        │ Confidence: 0.79 (avg of 0.85, 0.72) │
        └────────────────────────────────────┘
```

### **Fallback Logic**

If **no majority** (e.g., 1 text, 1 video, 1 audio all different):
```
1. Find modality with highest confidence
2. Use that emotion as the final result
3. Example: audio (0.78) > text (0.65) > video (0.61)
   → Output: audio's emotion
```

---

## 🤖 Gemini Integration

### **Prompt Engineering Pipeline**

```
┌─────────────────────────────────────────────────────────────┐
│  Input Components                                           │
├─────────────────────────────────────────────────────────────┤
│  1. User Message:        "I'm feeling lost today"           │
│  2. Fused Emotion:       "sad"                              │
│  3. Text Confidence:     0.82                               │
│  4. Video Confidence:    0.65                               │
│  5. Audio Confidence:    0.78                               │
│  6. Modality Block:      "Text→sad, Video→sad, Audio→sad"  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ Step 1: Select Tone Hint           │
        │ TONE_HINTS["sad"] =                │
        │ "Use gentle, validating,           │
        │  comforting tone..."               │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Step 2: Set Polarity Rules         │
        │ Emotion="sad" → NEGATIVE polarity  │
        │ Rules:                             │
        │ ✓ OK to acknowledge struggle       │
        │ ✗ NOT generic positivity           │
        │ ✗ NOT overly cheerful              │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Step 3: Build System Prompt        │
        │ "You are an empathetic chatbot..." │
        │ • Role definition                  │
        │ • Multimodal context               │
        │ • Emotion ground truth             │
        │ • Strict constraints               │
        │ • Instructions (2-4 sentences)     │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Step 4: Call Gemini API            │
        │ • Send: full prompt                │
        │ • Model: gemini-2.5-flash          │
        │ • Receive: text response           │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │ Output: Bot Response               │
        │ "I hear you. Feeling lost can be   │
        │  overwhelming, but you don't have  │
        │  to face it alone. What's on your  │
        │  mind?"                            │
        └────────────────────────────────────┘
```

---

## 📁 File Organization

```
final.py (1000+ lines)
├── Imports & Config (lines 1-30)
├── Text Model Setup (lines 31-90)
│   ├── SBERT loading
│   ├── LightGBM classifier
│   ├── Text preprocessing regex
│   └── predict_text_emotion_with_probs()
├── Video Model Setup (lines 91-150)
│   ├── CNN architecture
│   ├── Haar Cascade
│   ├── predict_frame_emotion()
├── Audio Model Setup (lines 151-210)
│   ├── Wav2Vec2 processor
│   ├── Audio streaming callback
│   └── predict_audio_emotion_and_probs()
├── Fusion Engine (lines 211-280)
│   ├── normalize_label()
│   └── fuse_modalities_dynamic()
├── Gemini Integration (lines 281-380)
│   ├── TONE_HINTS dict
│   ├── Polarity rules
│   └── generate_bot_response()
└── Tkinter GUI (lines 381-1000+)
    ├── Style configuration
    ├── UI layout
    ├── Video stream handler
    ├── Audio stream handler
    ├── Send handler
    └── Event bindings
```

---

## 🔌 Model Connections

```
Text Model Chain:
Raw Text → Regex Clean → SBERT Encode → Scale → LightGBM → Emotion

Video Model Chain:
Frame → Face Detect → Crop → Resize → Normalize → CNN → Softmax → Emotion

Audio Model Chain:
Audio Stream → Collect → Concatenate → Processor → Wav2Vec2 → Softmax → Emotion

Fusion Chain:
(Text, Video, Audio) → Normalize → Vote → Confidence → Fused Emotion

Response Chain:
(User Message, Fused Emotion) → Prompt Build → Gemini → Response → Display
```

---

## 🎯 Threading Model

```
Main Thread (UI Thread)
├── Tkinter event loop
├── Chat display updates
├── Button callbacks
└── Status bar updates

Background Threads
├── Video capture thread
│   ├── Continuous frame reading
│   ├── Face detection & prediction
│   └── Display in OpenCV window
├── Audio processing thread
│   ├── Stream callback (separate)
│   └── Buffer accumulation
└── Send handler thread
    ├── Stop video/audio if running
    ├── Emotion prediction
    ├── Fusion calculation
    ├── Gemini API call
    └── UI update via root.after()
```

---

## 📈 Performance Characteristics

| Component | Latency | Throughput | GPU/CPU |
|-----------|---------|-----------|---------|
| Text embedding | 50ms | 20/sec | CPU |
| LightGBM | 5ms | 200/sec | CPU |
| Video CNN | 30ms/frame | 30fps | GPU |
| Wav2Vec2 | 200ms/sec | 1/sec | GPU |
| Gemini API | 2-4s | 1 req/5s | Network |
| Total pipeline | 2-4s | 0.25/sec | Mixed |

---

## 🔐 Security Boundaries

```
User Input
    ↓
[Validation & Sanitization]
  • Text: regex cleaning removes malicious chars
  • Video: only processes face regions (no storage)
  • Audio: temporary buffer only, not stored
    ↓
[Local Processing]
  • All ML models run locally
  • No raw data sent to servers
    ↓
[API Boundary]
  • Only user message + emotion sent to Gemini
  • API response only → display
  • No caching of personal data
    ↓
[Output Display]
  • Text only in UI
  • No screenshots or recordings
```

---

## 🚀 Deployment Architecture

### **Standalone Desktop (Current)**
- Single machine execution
- Local models (CNN, LightGBM, Wav2Vec2)
- Network: Gemini API calls only
- No database or backend

### **Future: Web/Cloud Version**
```
Client (Browser/Mobile)
    ↓
[REST API Server]
    ├── Flask/FastAPI
    ├── Authentication
    └── Rate limiting
    ↓
[Backend Services]
    ├── Model serving (TF Serving)
    ├── Audio processing
    └── Gemini integration
    ↓
[Database]
    ├── User sessions
    └── Emotion analytics
```

