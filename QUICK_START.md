# Quick Start Guide - Polymorphic Malware Detection

## 🚀 Get Started in 3 Steps

### Step 1: Train the Model (One-time)

```bash
python Model/train_randomforest.py
```

**Expected output:**
```
✅ TRAINING COMPLETE!
Accuracy: 0.9294
ROC-AUC: 0.9779
```

**Time:** ~2-3 seconds

---

### Step 2: Start the API Server

```bash
python Model/api_server.py
```

**Server will run on:** `http://localhost:8000`

**Check health:**
```bash
curl http://localhost:8000/health
```

---

### Step 3: Scan Files

**Option A: Using Test Client**
```bash
python Model/test_api_client.py path/to/file.exe
```

**Option B: Using cURL**
```bash
curl -X POST "http://localhost:8000/scan" -F "file=@file.exe"
```

**Option C: Using Python**
```python
import requests

with open('file.exe', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/scan',
        files={'file': ('file.exe', f)}
    )
    result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Malware probability: {result['malware_probability']:.2%}")
```

---

## 📊 What You Get

### Model Performance
- **Accuracy:** 92.94%
- **ROC-AUC:** 97.79%
- **False Positive Rate:** 4.5%
- **False Negative Rate:** 13%

### API Response Example

```json
{
  "filename": "sample.exe",
  "prediction": "malware",
  "confidence": 0.95,
  "malware_probability": 0.95,
  "benign_probability": 0.05,
  "timestamp": "2024-01-01T12:00:00",
  "features_extracted": 29,
  "file_size": 102400
}
```

---

## 🔧 Configuration Options

### Enable SMOTE (More Aggressive Class Balancing)

Edit `Model/train_randomforest.py`:
```python
USE_SMOTE = True
```

### Enable PCA (Dimensionality Reduction)

Edit `Model/train_randomforest.py`:
```python
USE_PCA = True
PCA_COMPONENTS = 20
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `Model/train_randomforest.py` | Train the model |
| `Model/api_server.py` | API server |
| `Model/test_api_client.py` | Test client |
| `Model/feature_extractor.py` | Feature extraction |
| `Model/randomforest_malware_detector.pkl` | Trained model (generated) |
| `Model/feature_list.json` | Feature metadata (generated) |

---

## 🎯 Features Used (29 Total)

### General (10 features)
- File size, virtual size
- Imports, exports count
- Debug, relocations, resources, signature, TLS presence

### Header (17 features)
- COFF: timestamp, machine, characteristics
- Optional: subsystem, DLL characteristics, versions, sizes

### Section (2 features)
- Section count and entries

---

## ⚡ Performance

- **Training time:** ~2 seconds
- **Prediction time:** <0.1 seconds per file
- **Dataset:** 24,562 samples
- **Features:** 29 metadata features

---

## 🔍 How It Works

1. **Upload PE file** → API receives file
2. **Extract features** → 29 metadata features extracted
3. **Predict** → Random Forest model classifies
4. **Return result** → Prediction + confidence scores

---

## 💡 Tips

### For Best Results:
- Use the full dataset (already configured)
- Keep class weighting enabled
- Monitor false positive/negative rates
- Retrain periodically with new samples

### For Production:
- Add authentication
- Implement rate limiting
- Run in sandboxed environment
- Use HTTPS
- Add logging and monitoring

---

## 🐛 Troubleshooting

### Model not found
```bash
# Train the model first
python Model/train_randomforest.py
```

### API won't start
```bash
# Check if port 8000 is available
# Or change port in api_server.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Feature extraction error
```bash
# Make sure file is a valid PE file (.exe, .dll, .sys)
# Check that pefile is installed:
pip install pefile
```

---

## 📚 More Information

- Full documentation: `Model/README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Dataset info: `dataset/feature.py`

---

## ✅ Success Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Trained model (`python Model/train_randomforest.py`)
- [ ] Started API server (`python Model/api_server.py`)
- [ ] Tested with sample file (`python Model/test_api_client.py file.exe`)
- [ ] Verified predictions are working

---

**You're all set! 🎉**

The system is ready to detect polymorphic malware with 92.94% accuracy and 97.79% ROC-AUC!

