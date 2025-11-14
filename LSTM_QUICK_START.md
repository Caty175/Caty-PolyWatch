# 🚀 LSTM Behavioral Model - Quick Start

## ⚡ Quick Commands

### Train the Model

```bash
python Model/train_lstm_behavioral.py
```

**Output:**
- 📦 Model components → `Model/components/`
- 📊 Visualizations → `Model/output/`

---

### Make Predictions

```bash
python Model/predict_lstm_behavioral.py
```

---

## 📁 New Folder Structure

```
Model/
├── components/              # 📦 Model artifacts (.h5, .pkl, .json)
│   ├── lstm_behavioral_malware_detector.h5
│   ├── behavioral_scaler.pkl
│   ├── lstm_model_metadata.json
│   └── best_model_checkpoint.h5
│
└── output/                  # 📊 Training outputs (.png)
    ├── training_history.png
    └── roc_curve.png
```

---

## 🎯 Why This Structure?

### ✅ Benefits

1. **Organized**: Components separate from outputs
2. **Deployment Ready**: Copy `components/` folder to deploy
3. **Clean**: Easy to find files
4. **Professional**: Industry-standard structure
5. **Version Control**: Git-friendly organization

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 78.63% |
| **ROC-AUC** | 0.8592 |
| **Precision (Malware)** | 97% |
| **Recall (Malware)** | 64% |
| **False Positive Rate** | 2.6% |

**Confusion Matrix:**
```
Predicted:     Benign  Malware
Actual Benign:   111      3     ← Only 3 false positives!
Actual Malware:   53     95     ← 95 malware detected
```

---

## 🔧 Usage Examples

### Example 1: Train Model

```bash
python Model/train_lstm_behavioral.py
```

**Output:**
```
📁 Output directories:
   - Components: Model/components
   - Outputs: Model/output

...training...

✅ TRAINING COMPLETE!
Accuracy: 0.7863
ROC-AUC: 0.8592

📁 Saved artifacts:
   📦 Components folder:
      - Model: lstm_behavioral_malware_detector.h5
      - Scalers: behavioral_scaler.pkl
      - Metadata: lstm_model_metadata.json
      - Checkpoint: best_model_checkpoint.h5

   📊 Output folder:
      - Training history: training_history.png
      - ROC curve: roc_curve.png
```

---

### Example 2: Single Prediction

```python
from Model.predict_lstm_behavioral import BehavioralMalwareDetector
import pandas as pd

# Load model (automatically finds components folder)
detector = BehavioralMalwareDetector()

# Load data
data = pd.read_csv('dataset/Malware_Analysis_kaggle.csv')

# Predict
result = detector.predict(data.iloc[0])

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Output:**
```
🔧 Loading LSTM behavioral malware detector...
   ✅ Model loaded from Model/components/lstm_behavioral_malware_detector.h5
   ✅ Scalers loaded from Model/components/behavioral_scaler.pkl
   ✅ Metadata loaded from Model/components/lstm_model_metadata.json

Prediction: benign
Confidence: 83.11%
```

---

### Example 3: Batch Prediction

```python
from Model.predict_lstm_behavioral import BehavioralMalwareDetector
import pandas as pd

detector = BehavioralMalwareDetector()
data = pd.read_csv('dataset/Malware_Analysis_kaggle.csv')

# Predict batch
results = detector.predict_batch(data.head(100))

# Count malware
malware_count = sum(1 for r in results if r['prediction'] == 'malware')
print(f"Detected {malware_count} malware out of 100 samples")
```

---

## 🚀 Deployment

### Option 1: Copy Components Folder

```bash
# Copy to deployment location
cp -r Model/components /path/to/deployment/

# On deployment system
from Model.predict_lstm_behavioral import BehavioralMalwareDetector

detector = BehavioralMalwareDetector(
    components_dir="/path/to/deployment/components"
)
```

---

### Option 2: Package as ZIP

```bash
cd Model
zip -r lstm_model.zip components/

# Transfer lstm_model.zip to deployment system
# Extract and use
```

---

## 📋 File Descriptions

### Components Folder

| File | Description | Size |
|------|-------------|------|
| `lstm_behavioral_malware_detector.h5` | Trained model | ~800 KB |
| `behavioral_scaler.pkl` | Feature normalization | ~50 KB |
| `lstm_model_metadata.json` | Configuration & metrics | ~15 KB |
| `best_model_checkpoint.h5` | Best model checkpoint | ~800 KB |

### Output Folder

| File | Description | Size |
|------|-------------|------|
| `training_history.png` | Training curves | ~200 KB |
| `roc_curve.png` | ROC curve | ~100 KB |

---

## 🎓 Key Features

### API Calls (261 features)
- Process creation, memory manipulation
- Code injection, registry modification
- Network activity, file operations

### Static Behavioral (59 features)
- File operations (created, deleted, read, written)
- DLL loading patterns
- Registry, network, command line activity

---

## 🔍 Troubleshooting

### Issue: "Model not found"

```bash
# Train the model first
python Model/train_lstm_behavioral.py
```

---

### Issue: "ModuleNotFoundError"

```bash
# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

---

### Issue: Memory error

```python
# Edit train_lstm_behavioral.py
BATCH_SIZE = 16  # Reduce from 32
```

---

## 📚 Documentation

- **Detailed Guide**: `Model/LSTM_BEHAVIORAL_README.md`
- **Folder Structure**: `Model/FOLDER_STRUCTURE.md`
- **Model Comparison**: `MODEL_COMPARISON.md`
- **Training Summary**: `TRAINING_SUMMARY.md`

---

## ✅ Quick Checklist

- [ ] Install dependencies
- [ ] Train model (`python Model/train_lstm_behavioral.py`)
- [ ] Check `Model/components/` for model files
- [ ] Check `Model/output/` for visualizations
- [ ] Test predictions (`python Model/predict_lstm_behavioral.py`)

---

## 🎉 You're Ready!

**Model trained and organized!**

- ✅ 78.63% accuracy
- ✅ 97% precision (very few false alarms)
- ✅ Perfect for polymorphic malware detection
- ✅ All files organized in `components/` and `output/`

**Start detecting polymorphic malware!** 🚀

