# Contributing to Multimodal Emotion Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 🤝 How to Contribute

### **Types of Contributions**

1. **Bug Reports** 🐛
   - Found a bug? Please open an issue with:
     - Clear description of the bug
     - Steps to reproduce
     - Expected vs actual behavior
     - Screenshots/logs if applicable
     - Your environment (OS, Python version, hardware)

2. **Feature Requests** ✨
   - Have an idea? Share it!
     - Clear description of use case
     - Why this feature would be useful
     - Possible implementation approach
     - Any relevant references

3. **Code Improvements** 💻
   - Optimizations
   - Refactoring
   - Performance improvements
   - Code quality enhancements

4. **Documentation** 📚
   - Clarifications in README
   - Additional examples
   - Tutorials
   - Architecture documentation

5. **Model Improvements** 🧠
   - Better fusion algorithms
   - Model training optimizations
   - New emotion categories
   - Enhanced accuracy

## 🚀 Getting Started

### **1. Fork & Clone**
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR-USERNAME/multimodal-emotion-detection.git
cd multimodal-emotion-detection
```

### **2. Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### **3. Set Up Development Environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### **4. Make Changes**
- Write clean, readable code
- Follow PEP 8 style guide
- Add comments for complex logic
- Test your changes thoroughly

### **5. Commit with Clear Messages**
```bash
git add .
git commit -m "feat: add emotion intensity feature"
# or
git commit -m "fix: improve face detection in low light"
# or
git commit -m "docs: clarify fusion algorithm"
```

**Commit types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code formatting
- `refactor:` Code restructuring
- `perf:` Performance improvement
- `test:` Adding tests

### **6. Push & Create Pull Request**
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title
- Description of changes
- Motivation & context
- Testing performed
- Screenshots (if UI changes)

## 📋 Code Standards

### **Python Style**
```python
# Good
def predict_emotion_from_text(text, model, scaler):
    """
    Predict emotion from input text.
    
    Args:
        text (str): Input text to analyze
        model: Trained LightGBM classifier
        scaler: Fitted feature scaler
    
    Returns:
        tuple: (emotion_label, confidence_score)
    """
    cleaned = clean_text(text)
    embedding = encode_text(cleaned)
    scaled = scaler.transform(embedding)
    prediction = model.predict(scaled)
    return prediction

# Avoid
def pred_em(t, m, s):
    c = clean_text(t)
    e = encode_text(c)
    sc = s.transform(e)
    p = m.predict(sc)
    return p
```

### **Comments & Docstrings**
```python
# ✓ Good: Explains WHY, not WHAT
# The majority voting threshold is 2/3 because with 3 modalities,
# 2 matching predictions indicate strong consensus
if count >= 2:
    return most_common, modalities

# ✗ Avoid: Redundant with code
count = cnt.most_common(1)[0]  # Get the count
```

### **Variable Naming**
```python
# ✓ Clear
video_emotion, video_confidence = predict_video_emotion(frame)

# ✗ Unclear
v_e, v_c = pred_v_em(f)
```

## 🧪 Testing

### **Before Submitting PR:**
1. ✅ Test your changes manually
2. ✅ Check for import errors
3. ✅ Verify no regressions
4. ✅ Test edge cases
5. ✅ Check documentation matches code

### **Testing Checklist**
```python
# Test text emotion detection
python -c "from final import predict_text_emotion_with_probs; print(predict_text_emotion_with_probs('I am very happy'))"

# Test video detection (manual: start app, test camera)
python final.py

# Test audio detection (manual: start app, test mic)
python final.py
```

## 📖 Documentation

### **Update README if:**
- Adding new features
- Changing installation process
- Modifying configuration
- Updating models or dependencies

### **Update ARCHITECTURE if:**
- Changing system design
- Modifying data flow
- Updating model architectures
- Changing API integration

### **Code Comments:**
- Use clear, concise English
- Explain algorithm/approach, not just code
- Reference papers/sources for ML logic
- Mark TODOs with dates and owner

```python
# Good comment
def fuse_modalities_dynamic(modalities):
    """
    Fuse multi-modal emotion predictions using majority voting.
    
    This approach is based on the observation that when ≥2 of 3 modalities
    agree, they provide strong evidence for the true emotion state.
    Reference: "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2018)
    """
```

## 🎯 PR Review Process

### **What We Look For**
1. ✅ Code quality (clean, readable, documented)
2. ✅ Tests & verification
3. ✅ Documentation updates
4. ✅ No breaking changes
5. ✅ Performance (no major slowdowns)
6. ✅ Security (no exposed keys, safe inputs)

### **Common Feedback**
- Naming clarity
- Missing edge case handling
- Documentation gaps
- Performance concerns
- Security issues

### **Addressing Feedback**
- Reply to comments
- Make requested changes
- Force-push to same branch
- Don't create new PRs for iterations

## 📦 Dependencies & Versions

### **Adding New Dependencies**

Before adding a new package:
1. Check if alternative exists in current stack
2. Verify compatibility with existing versions
3. Test thoroughly
4. Update requirements.txt with pinned version
5. Update README dependencies section
6. Explain why in PR description

```bash
# Check before installing
pip show <package-name>  # Check if already installed
pip search <keyword>  # Find alternatives (if available)
```

## 🔐 Security

### **Never Commit**
- API keys or credentials
- Passwords or tokens
- Personal information
- Sensitive data

### **Environment Variables**
```bash
# Use .env file (in .gitignore)
GEMINI_API_KEY=your_key_here

# Load in code
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
```

## 🐛 Known Issues & TODOs

Check the Issues tab for:
- [ ] Low-light facial detection improvement
- [ ] Add emotion intensity levels
- [ ] Multi-person detection
- [ ] Offline LLM fallback
- [ ] Performance optimization for real-time audio


## ✨ Thank You!

Every contribution makes this project better. We appreciate:
- Your time & effort
- Detailed bug reports
- Thoughtful suggestions
- Quality code improvements

Happy contributing! 🎉

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
