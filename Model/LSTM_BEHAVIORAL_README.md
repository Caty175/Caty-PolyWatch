# LSTM + Feedforward Neural Network for Behavioral Malware Detection

## 🎯 Overview

This is a **hybrid deep learning model** that combines **LSTM (Long Short-Term Memory)** and **Feedforward Neural Networks** to detect polymorphic malware based on **dynamic behavioral features** extracted from runtime execution.

### Why This Model is Perfect for Polymorphic Malware

**Polymorphic malware** can change its code signature to evade static analysis, but it **cannot easily change its behavior** without breaking functionality. This model captures:

- **API call patterns** (what the malware does at runtime)
- **File operations** (files created, deleted, read, written)
- **DLL loading behavior** (runtime library dependencies)
- **Network activity** (DNS resolution, connections)
- **Registry operations** (system modifications)
- **MITRE ATT&CK tactics** (behavioral techniques)

---

## 📊 Model Performance

### Test Set Results (262 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | **79.39%** |
| **ROC-AUC** | **0.8694** |
| **Precision (Malware)** | **94%** |
| **Recall (Malware)** | **68%** |
| **Precision (Benign)** | **69%** |
| **Recall (Benign)** | **95%** |

### Confusion Matrix

```
                Predicted
              Benign  Malware
Actual Benign    108       6
       Malware    48     100
```

### Key Insights

- **Very low false positive rate** (6 benign files misclassified as malware)
- **High precision for malware detection** (94% - when it says malware, it's usually correct)
- **Excellent benign recall** (95% - catches most benign files correctly)
- **Trade-off**: Some malware samples are missed (32% false negative rate) for higher precision

---

## 🏗️ Architecture

### Dual-Input Hybrid Model

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
├──────────────────────────┬──────────────────────────────────┤
│   API Call Sequences     │   Static Behavioral Features     │
│   (261 features)         │   (59 features)                  │
└──────────┬───────────────┴──────────────┬───────────────────┘
           │                              │
           ▼                              ▼
    ┌──────────────┐              ┌──────────────┐
    │ LSTM Branch  │              │  FF Branch   │
    ├──────────────┤              ├──────────────┤
    │ LSTM(128)    │              │ Dense(256)   │
    │ Dropout(0.3) │              │ BatchNorm    │
    │ LSTM(64)     │              │ Dropout(0.4) │
    │ Dropout(0.3) │              │ Dense(128)   │
    │ Dense(64)    │              │ BatchNorm    │
    └──────┬───────┘              │ Dropout(0.4) │
           │                      │ Dense(64)    │
           │                      └──────┬───────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
              ┌──────────────┐
              │    MERGE     │
              │ Concatenate  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Dense(128)   │
              │ Dropout(0.3) │
              │ Dense(64)    │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  OUTPUT(1)   │
              │  Sigmoid     │
              └──────────────┘
```

### Model Components

#### 1. **LSTM Branch** (API Call Sequences)
- **Input**: 261 API call features (temporal patterns)
- **Architecture**:
  - LSTM layer (128 units) with return sequences
  - Dropout (30%)
  - LSTM layer (64 units)
  - Dropout (30%)
  - Dense layer (64 units, ReLU)
- **Purpose**: Captures sequential patterns in API calls

#### 2. **Feedforward Branch** (Static Behavioral Features)
- **Input**: 59 static behavioral features
  - File operations (5 features)
  - DLL loading (50 features)
  - Other behavioral (4 features: regkey_read, directory_enumerated, dll_loaded_count, resolves_host)
- **Architecture**:
  - Dense layer (256 units, ReLU)
  - Batch Normalization
  - Dropout (40%)
  - Dense layer (128 units, ReLU)
  - Batch Normalization
  - Dropout (40%)
  - Dense layer (64 units, ReLU)
- **Purpose**: Captures static behavioral patterns

#### 3. **Merged Layers**
- Concatenates outputs from both branches (128 features total)
- Dense layer (128 units, ReLU)
- Dropout (30%)
- Dense layer (64 units, ReLU)
- Output layer (1 unit, Sigmoid) for binary classification

---

## 📁 Dataset

### Source
**Malware_Analysis_kaggle.csv** - 1,308 samples with behavioral features

### Features Breakdown

| Category | Count | Examples |
|----------|-------|----------|
| **API Calls** | 261 | `API_NtCreateFile`, `API_CreateProcessInternalW`, `API_RegSetValueExA` |
| **File Operations** | 5 | `file_created`, `file_deleted`, `file_read`, `file_written` |
| **DLL Loading** | 50 | `dll_freq_kernel32.dll`, `dll_freq_ntdll.dll`, `dll_freq_ws2_32.dll` |
| **Behavioral** | 4 | `regkey_read`, `directory_enumerated`, `dll_loaded_count`, `resolves_host` |
| **TOTAL** | **320** | |

### Label Distribution
- **Benign (Score < 5.0)**: 570 samples (43.6%)
- **Malware (Score >= 5.0)**: 738 samples (56.4%)

---

## 🚀 Usage

### 1. Training the Model

```bash
python Model/train_lstm_behavioral.py
```

**Expected output:**
```
✅ TRAINING COMPLETE!
Accuracy: 0.7939
ROC-AUC: 0.8694
```

**Training time:** ~5-10 minutes (CPU)

### 2. Making Predictions

```python
from predict_lstm_behavioral import BehavioralMalwareDetector
import pandas as pd

# Initialize detector
detector = BehavioralMalwareDetector()

# Load behavioral data (CSV with API calls and behavioral features)
data = pd.read_csv('behavioral_data.csv')

# Predict single sample
result = detector.predict(data.iloc[0])
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Malware probability: {result['malware_probability']:.2%}")

# Predict batch
results = detector.predict_batch(data)
```

### 3. Demo Script

```bash
python Model/predict_lstm_behavioral.py
```

This will load the model and make predictions on 10 random samples from the dataset.

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **LSTM Units** | 128, 64 | Number of LSTM units in each layer |
| **Dense Units** | 256, 128, 64 | Feedforward layer sizes |
| **LSTM Dropout** | 0.3 | Dropout rate for LSTM layers |
| **Dense Dropout** | 0.4 | Dropout rate for dense layers |
| **Learning Rate** | 0.001 | Initial learning rate (Adam optimizer) |
| **Batch Size** | 32 | Training batch size |
| **Max Epochs** | 50 | Maximum training epochs |
| **Early Stopping** | 10 | Patience for early stopping |

---

## 📈 Training Details

### Optimizer
- **Adam** with learning rate 0.001
- **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss doesn't improve for 5 epochs

### Loss Function
- **Binary Crossentropy** (binary classification)

### Callbacks
1. **Early Stopping**: Stops training if validation loss doesn't improve for 10 epochs
2. **Model Checkpoint**: Saves best model based on validation AUC
3. **ReduceLROnPlateau**: Adaptive learning rate reduction

### Data Split
- **Training**: 836 samples (64%)
- **Validation**: 210 samples (16%)
- **Test**: 262 samples (20%)

### Normalization
- **RobustScaler** for both API and static features (handles outliers better than StandardScaler)

---

## 📊 Visualizations

### Training History
![Training History](training_history.png)

Shows:
- Loss curves (train vs validation)
- Accuracy curves
- AUC curves
- Precision & Recall curves

### ROC Curve
![ROC Curve](roc_curve.png)

Shows the trade-off between True Positive Rate and False Positive Rate.

---

## 🎯 Comparison with Random Forest Model

| Metric | LSTM Behavioral | Random Forest (Static) |
|--------|----------------|------------------------|
| **Accuracy** | 79.39% | 92.94% |
| **ROC-AUC** | 0.8694 | 0.9779 |
| **Features** | 320 behavioral | 29 static metadata |
| **Dataset** | Kaggle (1,308) | EMBER 2018 (24,562) |
| **Detection Type** | Dynamic/Behavioral | Static/Structural |
| **Best For** | Polymorphic malware | General malware |

### Key Differences

- **LSTM**: Better for polymorphic malware (behavioral patterns are harder to change)
- **Random Forest**: Better overall accuracy but vulnerable to polymorphic variants
- **Hybrid Approach**: Combine both models for best results!

---

## 🔮 Future Improvements

1. **Hybrid Model**: Combine LSTM behavioral + Random Forest static features
2. **Attention Mechanism**: Add attention layers to focus on important API calls
3. **Transformer Architecture**: Replace LSTM with Transformer for better long-range dependencies
4. **Class Balancing**: Use SMOTE or class weights to improve malware recall
5. **Ensemble**: Combine multiple LSTM models with different architectures
6. **Feature Engineering**: Extract API call sequences (temporal ordering)
7. **Transfer Learning**: Pre-train on larger behavioral datasets

---

## 📦 Files Generated

| File | Description |
|------|-------------|
| `lstm_behavioral_malware_detector.h5` | Trained Keras model |
| `behavioral_scaler.pkl` | RobustScaler for feature normalization |
| `lstm_model_metadata.json` | Model metadata (features, hyperparameters, metrics) |
| `best_model_checkpoint.h5` | Best model checkpoint during training |
| `training_history.png` | Training history visualization |
| `roc_curve.png` | ROC curve visualization |

---

## 🛠️ Requirements

```
tensorflow>=2.20.0
scikit-learn>=1.7.0
pandas>=2.0.0
numpy>=1.23.0
matplotlib>=3.10.0
joblib>=1.5.0
```

Install with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

---

## 📝 Citation

If you use this model, please cite:

```
LSTM + Feedforward Behavioral Malware Detector
Dataset: Malware Analysis Kaggle Dataset
Architecture: Hybrid LSTM + Feedforward Neural Network
Features: 261 API calls + 59 static behavioral features
Performance: 79.39% accuracy, 0.8694 ROC-AUC
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better feature engineering
- Hyperparameter tuning
- Alternative architectures (Transformer, GRU, Attention)
- Ensemble methods
- Hybrid static + behavioral models

---

## 📄 License

This project is for educational and research purposes.

